"""Record discovery (via eegdash) and file download (anonymous S3).

eegdash indexes the *raw* 5 kHz BrainVision EEG of ds007216. We use that index
as the authoritative list of recordings (subject / session / task / run) and
derive from each record the paths of the companion files we actually need:

* the cleaned EEG derivative  ``derivatives/.../<stem>_eeg_Bergen_CWreg.set/.fdt``
  (gradient + BCG corrected with the Bergen toolbox, CWL-regressed, 250 Hz);
* the raw BOLD run             ``sub-X/ses-Y/func/<stem>_bold.nii.gz``;
* the session-1 MPRAGE         ``sub-X/ses-001/anat/sub-X_ses-001_T1w.nii.gz``.

The cleaned derivatives are not in the eegdash index, so they are fetched
directly from the public OpenNeuro bucket.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

_ENTITY_RE = re.compile(
    r"sub-(?P<subject>[^_/]+)_ses-(?P<session>[^_/]+)_task-(?P<task>[^_/]+)_run-(?P<run>[^_/]+)_eeg"
)


# --------------------------------------------------------------------------
# Record discovery
# --------------------------------------------------------------------------
def _records_from_eegdash(subjects: list[str]) -> list[str]:
    from eegdash import EEGDash

    recs = EEGDash().find({"dataset": config.DATASET, "subject": {"$in": list(subjects)}},
                          limit=1000)  # API maximum; ds007216 has <200 runs
    return [r["bidspath"] for r in recs]


def _records_from_s3(subjects: list[str]) -> list[str]:
    """Fallback: list the cleaned EEG derivatives directly on S3."""
    fs = _s3()
    paths = []
    for s in subjects:
        paths += fs.glob(f"{config.S3_ROOT}/derivatives/sub-{s}/ses-*/eeg/*{config.EEG_DERIV_SUFFIX}.set")
    return paths


def query_runs(subjects: list[str], limit_runs: int | None = None,
               source: str = "eegdash") -> pd.DataFrame:
    """Return one row per EEG-fMRI run with entity columns and relative file paths.

    Parameters
    ----------
    subjects : list of str
        Zero-padded subject labels, e.g. ``["001", "002"]``.
    limit_runs : int | None
        Keep at most this many runs per subject (sorted, GradCPT first). Used
        for smoke tests.
    source : {"eegdash", "s3"}
        Where to discover recordings. Falls back to ``"s3"`` if the eegdash
        metadata API is unreachable.
    """
    try:
        paths = _records_from_eegdash(subjects) if source == "eegdash" else _records_from_s3(subjects)
    except Exception as err:  # network / API outage -> fall back to bucket listing
        log.warning("eegdash query failed (%s); falling back to S3 listing", err)
        paths = _records_from_s3(subjects)

    rows = []
    for p in paths:
        if Path(p).name.startswith("._"):  # macOS metadata files exist in the bucket
            continue
        m = _ENTITY_RE.search(p)
        if m is None:
            continue
        rows.append(m.groupdict())
    if not rows:
        raise RuntimeError(f"No recordings found for subjects {subjects}")

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    df = df[df.subject.isin(subjects)]
    df["stem"] = ("sub-" + df.subject + "_ses-" + df.session + "_task-" + df.task + "_run-" + df.run)
    sub_ses = "sub-" + df.subject + "/ses-" + df.session
    df["eeg_set"] = "derivatives/" + sub_ses + "/eeg/" + df.stem + config.EEG_DERIV_SUFFIX + ".set"
    df["eeg_fdt"] = df.eeg_set.str.replace(r"\.set$", ".fdt", regex=True)
    df["bold"] = sub_ses + "/func/" + df.stem + "_bold.nii.gz"
    df["bold_json"] = sub_ses + "/func/" + df.stem + "_bold.json"
    df["t1"] = "sub-" + df.subject + "/ses-001/anat/sub-" + df.subject + "_ses-001_T1w.nii.gz"

    # GradCPT first so that --limit-runs 1 picks the task run (most behaviourally interesting)
    df["_task_order"] = (df.task != "GradCPT").astype(int)
    df = df.sort_values(["subject", "_task_order", "session", "run"]).drop(columns="_task_order")
    if limit_runs is not None:
        df = df.groupby("subject", group_keys=False).head(limit_runs)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Paths / download
# --------------------------------------------------------------------------
def _s3():
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def local_path(paths: config.Paths, rel: str) -> Path:
    return paths.raw / rel


def files_for_runs(runs: pd.DataFrame, include_bold: bool = True,
                   include_t1: bool = True) -> list[str]:
    files: list[str] = []
    for _, r in runs.iterrows():
        files += [r.eeg_set, r.eeg_fdt]
        if include_bold:
            files += [r.bold, r.bold_json]
    if include_t1:
        files += sorted(set(runs.t1))
    return list(dict.fromkeys(files))  # dedupe, keep order


def download(rel_paths: list[str], paths: config.Paths, dry_run: bool = False,
             skip_existing_rois: bool = True) -> pd.DataFrame:
    """Download files from the public bucket into ``paths.raw`` (skips cached files).

    BOLD files whose ROI time series have already been extracted are skipped as
    well (they are deleted after extraction to save disk), unless
    ``skip_existing_rois`` is False.

    Returns a table of (file, size_MB, status).
    """
    fs = _s3()
    report = []
    for rel in rel_paths:
        dst = local_path(paths, rel)
        if rel.endswith("_bold.nii.gz") and skip_existing_rois:
            stem = Path(rel).name.replace("_bold.nii.gz", "")
            if (paths.roi / f"{stem}.parquet").exists():
                report.append((rel, 0.0, "roi-done"))
                continue
        if dst.exists():
            report.append((rel, dst.stat().st_size / 1e6, "cached"))
            continue
        src = f"{config.S3_ROOT}/{rel}"
        try:
            size = fs.info(src)["size"] / 1e6
        except FileNotFoundError:
            log.warning("missing on S3: %s", rel)
            report.append((rel, 0.0, "missing"))
            continue
        if dry_run:
            report.append((rel, size, "planned"))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        log.info("downloading %s (%.1f MB)", rel, size)
        fs.get(src, str(tmp))
        tmp.replace(dst)
        report.append((rel, size, "downloaded"))
    return pd.DataFrame(report, columns=["file", "size_MB", "status"])


def load_runs(paths: config.Paths, subjects: list[str],
              limit_runs: int | None = None) -> pd.DataFrame:
    """Run table for ``subjects``: reuse ``derived/runs.csv`` if it covers them, else query."""
    cache = paths.derived / "runs.csv"
    old = pd.read_csv(cache, dtype=str) if cache.exists() else None
    if old is not None and set(subjects) <= set(old.subject):
        df_all = old
    else:
        df_all = query_runs(subjects)
        if old is not None:  # merge with previously cached subjects
            df_all = pd.concat([old[~old.subject.isin(subjects)], df_all], ignore_index=True)
        df_all.to_csv(cache, index=False)
    df = df_all[df_all.subject.isin(subjects)]
    if limit_runs is not None:
        df = df.groupby("subject", group_keys=False).head(limit_runs)
    return df.reset_index(drop=True)
