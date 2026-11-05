"""Lightweight BOLD preprocessing and ROI extraction (ANTsPy + nilearn).

Per subject (once):
    MNI152 template --SyN--> skull-stripped session-1 MPRAGE.
    Both sessions use this one anatomical anchor; this avoids the thick-slice
    2D T1 of session 2 (problems_solutions.md #5).

Per run:
    1. rigid motion correction (ANTs BOLDRigid) -> 6 motion params + FD
    2. session-1 T1 --rigid--> mean EPI
    3. atlas labels MNI -> T1 -> EPI (genericLabel interpolation)
    4. masks: thalamus L/R (eroded 1 vox), V1, WM + CSF (eroded 2 vox), brain
    5. mean ROI time series, confound regression (motion + derivatives, WM,
       CSF, global), detrend and 0.008 Hz high-pass
    6. QC images, then (optionally) delete the 350 MB BOLD file

Known simplifications (stated in the README): no fieldmap distortion
correction, no slice-timing correction, Harvard-Oxford (FSL MNI152) labels
on the nilearn MNI152-2009 template.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.transform import Rotation

from . import config, qc

log = logging.getLogger(__name__)

# label value -> name (same encoding as qc.MASK_STYLE)
LABELS = {1: "thal_L", 2: "thal_R", 3: "v1", 4: "wm", 5: "csf"}


# --------------------------------------------------------------------------
# Templates / atlas (cached under derived/anat/_mni)
# --------------------------------------------------------------------------
def mni_assets(anat_dir: Path) -> dict[str, Path]:
    """Write MNI template, brain mask and combined label image to disk (cached)."""
    import nibabel as nib
    from nilearn import datasets, image

    d = anat_dir / "_mni"
    d.mkdir(parents=True, exist_ok=True)
    out = {"template": d / "mni_template.nii.gz", "mask": d / "mni_brainmask.nii.gz",
           "labels": d / "mni_labels.nii.gz"}
    if all(p.exists() for p in out.values()):
        return out

    datasets.load_mni152_template(resolution=1).to_filename(out["template"])
    datasets.load_mni152_brain_mask(resolution=1).to_filename(out["mask"])

    sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm", verbose=0)
    cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm", verbose=0)
    sub_img, cort_img = image.load_img(sub.maps), image.load_img(cort.maps)
    s, c = np.asarray(sub_img.dataobj), np.asarray(cort_img.dataobj)
    si, ci = sub.labels.index, cort.labels.index
    lab = np.zeros(s.shape, dtype=np.int16)
    lab[np.isin(s, [si("Left Cerebral White Matter"), si("Right Cerebral White Matter")])] = 4
    lab[np.isin(s, [si("Left Lateral Ventricle"), si("Right Lateral Ventricle")])] = 5
    lab[(c == ci("Intracalcarine Cortex")) & (lab == 0)] = 3
    lab[s == si("Left Thalamus")] = 1
    lab[s == si("Right Thalamus")] = 2
    nib.Nifti1Image(lab, sub_img.affine).to_filename(out["labels"])
    return out


# --------------------------------------------------------------------------
# Anatomy (per subject)
# --------------------------------------------------------------------------
def skull_strip(t1, mni_brain_mask):
    """Template-based skull strip of a full-head T1; returns a binary ANTs mask.

    1. Register ANTs' *full-head* MNI152 (with skull) to the T1 (quick SyN).
       Registering a brain-only template to a full head would inflate the brain
       mask to the size of the head, which happened in an earlier version.
    2. Warp the MNI brain mask into the T1 and refine it by intensity: keep
       voxels above the 30th percentile inside it (brain tissue), erode, keep the
       largest component, dilate back (``get_mask(cleanup=3)``). Then dilate by
       2 mm to bring back sulcal CSF, and fill holes.
    """
    import ants

    head = ants.image_read(ants.get_data("mni"))
    reg = ants.registration(fixed=t1, moving=head, type_of_transform="antsRegistrationSyNQuick[s]")
    m0 = ants.apply_transforms(fixed=t1, moving=mni_brain_mask, transformlist=reg["fwdtransforms"],
                               interpolator="nearestNeighbor")
    m0 = ants.iMath(m0, "MD", 1)
    vals = t1.numpy()[m0.numpy() > 0]
    tissue = ants.get_mask(t1 * m0, low_thresh=float(np.percentile(vals, 30)),
                           high_thresh=float(vals.max()) + 1, cleanup=3)
    return ants.iMath(ants.iMath(tissue, "MD", 2) * m0, "FillHoles")


def register_anat(subject: str, t1_path: Path, paths: config.Paths) -> dict:
    """Skull-strip the session-1 T1 and register the MNI template to it (SyN).

    Returns a dict with the T1-brain path, the MNI->T1 transform list and the
    MNI label image path. Results are cached in ``derived/anat/sub-XXX``.
    """
    import ants

    mni = mni_assets(paths.anat)
    d = paths.anat / f"sub-{subject}"
    d.mkdir(parents=True, exist_ok=True)
    prefix = str(d / f"sub-{subject}_mni2t1_")
    t1b_path = d / f"sub-{subject}_T1w_brain.nii.gz"
    fwd = [prefix + "1Warp.nii.gz", prefix + "0GenericAffine.mat"]
    result = {"t1_brain": t1b_path, "fwdtransforms": fwd, "labels_mni": mni["labels"]}
    if t1b_path.exists() and all(Path(f).exists() for f in fwd):
        return result

    log.info("sub-%s: anatomical registration (MNI -> T1, SyN)", subject)
    tpl = ants.image_read(str(mni["template"]))
    tpl_mask = ants.image_read(str(mni["mask"]))
    t1 = ants.image_read(str(t1_path))
    t1 = ants.resample_image(t1, (1.0, 1.0, 1.0), use_voxels=False, interp_type=0)
    t1 = ants.n4_bias_field_correction(t1)

    t1b = t1 * skull_strip(t1, tpl_mask)
    ants.image_write(t1b, str(t1b_path))

    ants.registration(fixed=t1b, moving=tpl, type_of_transform="SyN", outprefix=prefix)

    # QC: template warped with the *saved* transforms, edges on the subject T1
    warped = ants.apply_transforms(fixed=t1b, moving=tpl, transformlist=fwd)
    wpath = d / f"sub-{subject}_mni_in_t1.nii.gz"
    ants.image_write(warped, str(wpath))
    qc.plot_edges(t1b_path, wpath, paths.qc / f"sub-{subject}" / f"sub-{subject}_mni_to_t1.png",
                  title=f"sub-{subject}: T1 (session 1) + warped MNI edges")
    return result


# --------------------------------------------------------------------------
# Run-level helpers
# --------------------------------------------------------------------------
def motion_params(transform_lists: list) -> np.ndarray:
    """6 rigid parameters (3 translations mm, 3 rotations rad) per volume."""
    import ants

    out = np.zeros((len(transform_lists), 6))
    for k, tl in enumerate(transform_lists):
        if isinstance(tl, str) or not tl:  # "NA" for empty volumes
            continue
        p = np.asarray(ants.read_transform(tl[0]).parameters, dtype=float)
        rot = Rotation.from_matrix(p[:9].reshape(3, 3)).as_euler("xyz")
        out[k] = np.r_[p[9:12], rot]
    return out


def _erode(mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return mask
    er = ndimage.binary_erosion(mask, iterations=n)
    return er if er.sum() > 0 else mask  # never erode a mask to nothing


def build_confounds(mp: np.ndarray, wm: np.ndarray, csf: np.ndarray,
                    glob: np.ndarray | None) -> np.ndarray:
    dmp = np.vstack([np.zeros((1, mp.shape[1])), np.diff(mp, axis=0)])
    cols = [mp, dmp, wm[:, None], csf[:, None]]
    if glob is not None:
        cols.append(glob[:, None])
    return np.hstack(cols)


def process_run(row: pd.Series, anat: dict, paths: config.Paths,
                keep_bold: bool = False) -> Path:
    """Full per-run fMRI pipeline -> ``derived/roi/<stem>.parquet`` (+ QC)."""
    import ants
    from nilearn import signal

    out_parquet = paths.roi / f"{row.stem}.parquet"
    if out_parquet.exists():
        return out_parquet
    bold_path = paths.raw / row.bold
    if not bold_path.exists():
        raise FileNotFoundError(bold_path)

    work = paths.roi / "nii" / row.stem
    work.mkdir(parents=True, exist_ok=True)
    qc_dir = paths.qc / f"sub-{row.subject}" / f"ses-{row.session}"

    log.info("%s: motion correction", row.stem)
    bold = ants.image_read(str(bold_path))
    n_vols = bold.shape[3]
    ref = ants.get_average_of_timeseries(bold)
    mc = ants.motion_correction(bold, fixed=ref, type_of_transform="BOLDRigid")
    moco = mc["motion_corrected"]
    fd = np.nan_to_num(np.asarray(mc["FD"], dtype=float))
    mp = motion_params(mc["motion_parameters"])
    del bold
    mean_epi = ants.get_average_of_timeseries(moco)

    log.info("%s: T1 -> EPI registration and atlas warp", row.stem)
    t1b = ants.image_read(str(anat["t1_brain"]))
    # Fixed image = bias-corrected, skull-stripped mean EPI; mutual information
    # with a coarse-to-fine schedule (EPI and T1 have opposite CSF contrast).
    epi_mask = ants.get_mask(mean_epi)
    epi_brain = ants.n4_bias_field_correction(mean_epi, mask=epi_mask) * epi_mask
    reg = ants.registration(
        fixed=epi_brain, moving=t1b, type_of_transform="Rigid", aff_metric="mattes",
        aff_sampling=64, aff_random_sampling_rate=0.5,
        aff_iterations=(2000, 1000, 500, 100), aff_shrink_factors=(8, 4, 2, 1),
        aff_smoothing_sigmas=(4, 2, 1, 0))
    labels_mni = ants.image_read(str(anat["labels_mni"]))
    lab = ants.apply_transforms(fixed=mean_epi, moving=labels_mni,
                                transformlist=reg["fwdtransforms"] + list(anat["fwdtransforms"]),
                                interpolator="genericLabel").numpy().astype(np.int16)
    t1_epi = ants.apply_transforms(fixed=mean_epi, moving=t1b, transformlist=reg["fwdtransforms"])

    brain = ants.get_mask(mean_epi).numpy().astype(bool)
    epi = mean_epi.numpy()
    signal_ok = brain & (epi > 0.2 * np.median(epi[brain]))
    thal_any = np.isin(lab, [1, 2])
    coverage = float((thal_any & signal_ok).sum() / max(thal_any.sum(), 1))
    # expected thalamus voxels in EPI grid = MNI 1 mm voxels / EPI voxel volume
    lab_mni = labels_mni.numpy()
    vox_vol = float(np.prod(mean_epi.spacing))
    n_thal_expected = float(np.isin(lab_mni, [1, 2]).sum() / vox_vol)

    masks = {
        "thal_L": _erode(lab == 1, config.THAL_ERODE_VOX) & brain,
        "thal_R": _erode(lab == 2, config.THAL_ERODE_VOX) & brain,
        "v1": (lab == 3) & brain,
        "wm": _erode(lab == 4, config.CONFOUND_ERODE_VOX) & brain,
        "csf": _erode(lab == 5, config.CONFOUND_ERODE_VOX) & brain,
    }
    masks["thal"] = masks["thal_L"] | masks["thal_R"]

    data = moco.numpy()  # x, y, z, t
    ts = {k: data[m].mean(0) for k, m in masks.items()}
    glob = data[brain].mean(0)
    del data, moco

    conf = build_confounds(mp, ts["wm"], ts["csf"], glob if config.USE_GSR else None)
    roi_cols = ["thal", "thal_L", "thal_R", "v1"]
    raw_sig = np.column_stack([ts[c] for c in roi_cols])
    clean = signal.clean(raw_sig, confounds=conf, detrend=True, standardize=False,
                         high_pass=config.HIGH_PASS, t_r=config.TR)

    df = pd.DataFrame(clean, columns=roi_cols)
    df.insert(0, "tr", np.arange(n_vols))
    df["global_raw"] = glob
    df["fd"] = fd
    for k, v in (("subject", row.subject), ("session", row.session), ("task", row.task),
                 ("run", row.run), ("stem", row.stem)):
        df[k] = v
    df.to_parquet(out_parquet, index=False)

    # ---- QC images (NIfTI written by ANTs keeps the world coordinates) ----
    lab_out = np.zeros_like(lab)
    for val, name in LABELS.items():
        lab_out[masks[name]] = val
    mean_path, lab_path, t1_path = (work / "mean_epi.nii.gz", work / "labels_epi.nii.gz",
                                    work / "t1_in_epi.nii.gz")
    ants.image_write(mean_epi, str(mean_path))
    ants.image_write(mean_epi.new_image_like(lab_out.astype("float32")), str(lab_path))
    ants.image_write(t1_epi, str(t1_path))
    nvox = {k: int(m.sum()) for k, m in masks.items()}
    qc.plot_mask_overlay(
        mean_path, lab_path, qc_dir / f"{row.stem}_desc-masks.png",
        title=(f"{row.stem} | thal L/R {nvox['thal_L']}/{nvox['thal_R']} vox, "
               f"V1 {nvox['v1']}, coverage {coverage:.2f}"))
    qc.plot_edges(mean_path, t1_path, qc_dir / f"{row.stem}_desc-epi2t1.png",
                  title=f"{row.stem}: mean EPI + T1 edges")
    qc.plot_roi_timeseries(df, qc_dir / f"{row.stem}_desc-timeseries.png", title=row.stem)
    summary = {"stem": row.stem, "subject": row.subject, "session": row.session,
               "n_vols": int(n_vols), "n_thal_vox": nvox["thal"],
               "n_thal_expected": n_thal_expected, "coverage": coverage,
               "fd_mean": float(fd.mean()), **{f"nvox_{k}": v for k, v in nvox.items()}}
    (paths.roi / "qc_summary").mkdir(exist_ok=True)
    (paths.roi / "qc_summary" / f"{row.stem}.json").write_text(json.dumps(summary, indent=1))
    if coverage < config.COVERAGE_MIN:
        log.warning("%s: thalamus coverage %.2f < %.2f", row.stem, coverage, config.COVERAGE_MIN)

    if not keep_bold:
        bold_path.unlink(missing_ok=True)
        log.info("%s: deleted raw BOLD (%s)", row.stem, bold_path.name)
    return out_parquet


def process_subject(subject: str, runs: pd.DataFrame, paths: config.Paths,
                    keep_bold: bool = False) -> list[Path]:
    """Anatomy once, then every run of the subject. Errors are logged per run."""
    sub_runs = runs[runs.subject == subject]
    todo = [r for _, r in sub_runs.iterrows() if not (paths.roi / f"{r.stem}.parquet").exists()]
    if not todo:
        return [paths.roi / f"{r.stem}.parquet" for _, r in sub_runs.iterrows()]
    t1 = paths.raw / sub_runs.iloc[0].t1
    if not t1.exists():
        log.error("sub-%s: missing T1 %s, skipping subject", subject, t1)
        return []
    anat = register_anat(subject, t1, paths)
    out = []
    for r in todo:
        try:
            out.append(process_run(r, anat, paths, keep_bold=keep_bold))
        except Exception:
            log.exception("%s: fMRI processing failed", r.stem)
    return out
