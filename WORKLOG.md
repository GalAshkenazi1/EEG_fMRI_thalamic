# Work log: thalamic activity from EEG (ds007216)

This file records what has been built, why, what has been checked on real data, the problems found along the way and how they were fixed, and what is still open.
For the *design* of the project, see [README.md](README.md). For the original reasoning, see [deliberates/](deliberates/).

Last updated: 2026-11-05 (added full-dataset estimates).

---

## 1. Status at a glance

| Area | State |
|---|---|
| Environment (`.venv`, eegdash 0.9.1) | Done |
| Unit tests (`tests/`, 13 tests) | Passing |
| Step 1: records + download | Done, checked on real data |
| Step 2: EEG preprocessing + features | Done, checked on real data (sub-001) |
| Step 3: fMRI ROI extraction | Done. A **registration bug was found and fixed**. sub-001 and sub-002 regenerated and visually OK |
| Step 4: model table | Done, checked on real data (2 subjects, 438 windows) |
| Step 5: models, null, controls | Done; runs end to end on the smoke sample (scores meaningless with n = 2) |
| Figures + notebook | Done; 11 figures written, notebook executes cleanly via nbconvert |
| Full run (24 subjects, Linux) | Not started; `run_all.sh` is ready |

---

## 2. Environment

* **Removed the stale global eegdash.** The global Python 3.13 had an editable `eegdash 0.4.1` that pointed to a dev checkout that no longer exists. It was removed with `pip uninstall -y eegdash`, and no leftover `.pth`/editable finder remains.
* **Created a project venv** (`.venv`, Python ≥ 3.11) with `requirements.txt`:
  * `eegdash[features]==0.9.1`, which pulls in braindecode 2.x and mne
  * `antspyx`, `nilearn`, `nibabel`, `s3fs`
  * `scikit-learn`, `scipy`, `pandas`, `pyarrow`
  * `matplotlib`, `seaborn`, `pytest`, `jupyter`/`nbconvert`
* **Installed the project itself** with `pip install -e .` (package `thalamic` under `src/`, configured in `pyproject.toml`).
* **Windows notes:**
  * `qc.py` and `plots.py` force the `Agg` matplotlib backend. The Windows Python has no working Tcl/Tk, so the default backend crashes.
  * Long runs need the PC to stay awake. During the smoke run a temporary stay-awake request (`SetThreadExecutionState`) was used. Alternative: `powercfg /change standby-timeout-ac 0`.

---

## 3. Code layout

```
src/thalamic/
  config.py    all constants + path handling (THAL_DATA_DIR / THAL_OUTPUT_DIR / CLI args)
  cli.py       shared argparse: --data-dir --output-dir --subjects --limit-runs --n-jobs -v
  records.py   run discovery (eegdash API, S3 fallback), path building, anonymous S3 download
  eeg.py       load cleaned .set, preprocess, TR markers, CWL RMS, TR-locked windows, PSD QC
  features.py  eegdash FeatureExtractor definition + column helpers
  fmri.py      MNI assets, skull strip, MNI->T1 SyN, motion correction, T1->EPI, masks, ROI signals
  qc.py        mask overlay / edge overlay / ROI time-series figures, HTML QC index
  align.py     EEG-fMRI alignment, scrubbing, HRF convolution, labels (float + binary)
  model.py     LOSO CV, models, scores, circular-shift null, controls/ablations, save/load
  plots.py     result figures (palette from the dataviz guidelines)
scripts/       01_download 02_eeg_features 03_fmri_roi 04_build_table 05_model
tests/         test_align.py, test_model.py
notebooks/     thalamus_from_eeg.ipynb
run_smoke.ps1  Windows smoke run (2 subjects, 1 run each, 20 null shifts)
run_all.sh     Linux full run (all subjects, N_JOBS / N_NULL / KEEP_BOLD env vars)
```

Every script takes `--subjects` and `--limit-runs`, and caches its outputs, so re-running skips finished work.

---

## 4. What each step does, and what was checked

### Step 1: records + download (`records.py`, `scripts/01_download.py`)
* `EEGDash().find({"dataset": "ds007216", ...}, limit=1000)` lists every run. The API caps `limit` at 1000; a larger value returns HTTP 422.
  If the API is unreachable, the code falls back to a glob over the public S3 bucket.
* The BIDS entities (subject, session, task, run) are parsed from `bidspath`. Each run row gets its file paths:
  * cleaned EEG `derivatives/.../*_eeg_Bergen_CWreg.set/.fdt`
  * BOLD `.nii.gz` + `.json`
  * the session-1 T1w
* Downloads are anonymous via `s3fs`. They write to a `.part` file and rename it when complete, and skip files that are already present.
  A BOLD file is also skipped if its ROI parquet already exists, because the BOLD is no longer needed.
* By default only the EEG and T1s are downloaded. BOLD is fetched just in time in step 3 and deleted after use. `--dry-run` prints the total size first.
* **Checked:** run discovery and download work for sub-001/002.

### Step 2: EEG (`eeg.py`, `features.py`, `scripts/02_eeg_features.py`)
* **Loading:** the cleaned derivative `.set` (250 Hz, 36 ch: 31 EEG + ECG + CWL1-4) is read with `mne.io.read_raw_eeglab`. ECG is typed as `ecg` and the CWL channels as `misc`.
* **TR markers:** `T  1` annotations, matched after whitespace normalization. Usable TRs = `min(markers, volumes) - 2`.
* **Preprocessing:**
  * keep the 31 scalp channels and apply the standard_1020 montage
  * FIR band-pass 1-45 Hz
  * average reference
  * **No notch filter:** line noise is at 60 Hz (Philadelphia), above the 45 Hz low-pass. This is documented in the `preprocess_raw` docstring, and `config.NOTCH_FREQ` re-enables a notch.
* **CWL RMS per TR** is computed for scrubbing only. The CWL channels are never used as features.
* **Windows:** one non-overlapping 2 s window per TR marker, built with braindecode `create_windows_from_events`.
* **Features:** eegdash `FeatureExtractor`, 8 per channel:
  * 5 relative band powers: delta, theta, alpha, beta, gamma
  * spectral entropy
  * line length
  * variance

  31 × 8 = **248 columns**, named `<feature>_<channel>`.
* **Checked on sub-001 GradCPT run 1:**
  * 240 windows × 248 features, no NaNs
  * the PSD QC shows a clear alpha peak, plus small residual bumps at the 13.5 Hz slice frequency and its harmonic (27 Hz)

### Step 3: fMRI (`fmri.py`, `qc.py`, `scripts/03_fmri_roi.py`)
Per subject, once:
* Skull-strip the session-1 T1, then register the MNI152 template to it (SyN).

Per run:
1. Rigid motion correction (`BOLDRigid`), giving 6 motion parameters and FD.
2. Rigid registration T1 → mean EPI.
3. Harvard-Oxford labels warped MNI → T1 → EPI (`genericLabel` interpolation):
   * thalamus L/R, eroded 1 voxel
   * V1 (intracalcarine)
   * WM and lateral ventricles, each eroded 2 voxels
4. Coverage check, then the mean ROI time series.
5. `nilearn.signal.clean` with confounds: motion + derivatives, WM, CSF, global signal; detrend; 0.008 Hz high-pass.
6. QC PNGs (masks, EPI↔T1 edges, time series) and a JSON summary.
7. The raw BOLD is deleted.

**Checked on sub-001 (after the fix in §5.1):**
* Coverage 1.00, mean FD 0.10 mm.
* Thalamus 544 / 546 voxels (L/R), V1 834, WM 12,124, CSF 79. sub-002: coverage 1.00, FD 0.06 mm, thalamus 633/664, V1 1054, CSF 40.
* The mask overlay shows the thalamus on both sides of the third ventricle, V1 along the calcarine region, and WM inside white matter.
* The T1 edges match the EPI ventricles and cortex.

### Step 4: model table (`align.py`, `scripts/04_build_table.py`)
1. Trim the EEG features and the ROI signals to the same usable TRs (the TR indices must match).
2. Log-transform variance and line length, then z-score every feature within the run.
3. Scrub TRs with FD > 0.5 mm or CWL RMS > median + 5·MAD. They are linearly interpolated for the convolution, then dropped.
4. Convolve each feature causally with the SPM canonical HRF ("convolve forward, don't shift back").
5. Drop the first 8 TRs (16 s HRF warm-up).
6. Targets: `thal`, `v1`, and `thal_nov1` (thalamus with V1 regressed out).
7. **Two label types for every target, both always stored:**
   * `float`: `<roi>_z`, z-scored within subject.
   * `binary`: `<roi>_active`, 1 if above the subject's 75th percentile, else 0.

**Checked on the smoke sample:**
* sub-001 has 216 usable windows: 240 TRs − 2 trimmed at the end − 8 HRF warm-up − 14 scrubbed. sub-002 has 222. That makes 438 in total.
* 248 features, no NaNs.
* Binary "active" rate 0.25 / 0.252 per subject.
* For sub-001 alone, the strongest single features are central variance and beta power (|r| ≈ 0.24 with `thal_z`). This is not cross-validated, so it is only a sanity check.

### Step 5: models (`model.py`, `scripts/05_model.py`)
* Leave-one-subject-out CV, with the scaler fitted inside each fold.
* Models:
  * **float labels:** RidgeCV, scored by Pearson r.
  * **binary labels:** logistic regression (C = 0.01, balanced class weights), scored by ROC-AUC.
  * **non-linear comparison:** HistGradientBoosting.
* Statistics:
  * a per-subject score, plus a one-sided Wilcoxon test against chance
  * a **circular-shift null**: the target is shifted within each run, which keeps its autocorrelation. 200 shifts in the full run, 20 in the smoke run.
* Analyses:
  * thalamus with the linear model (main) and with HGB
  * V1 control
  * thalamus minus V1
  * no-alpha ablation
  * a per-task breakdown
* Outputs:
  * `results.json`, per-subject scores, out-of-fold predictions, coefficients, the null distribution
  * figures: per-subject scores vs null, null histogram, controls, predicted vs actual, coefficient topomaps

**Smoke-run results (2 subjects, 20 null shifts):**
* The pipeline, figures and `results.json` are all produced.
* With n = 2, each LOSO fold trains on a single subject and tests on the other, so the numbers are noise:
  * thalamus AUC ≈ 0.43
  * V1 AUC ≈ 0.65
  * null mean ≈ 0.49

  They only show that the plumbing works. Real numbers need the full run.

### Unit tests (`tests/`)
13 tests:
* usable-TR trimming
* HRF causality (peak at 4-6 s)
* interpolation and scrubbing
* `align_run` and labels
* LOSO folds never share a subject
* recovery of a planted signal
* the null on random data is centred at 0.5
* circular-shift properties

All pass.

---

## 5. Problems found and how they were fixed

### 5.1 Masks in the wrong place (the main one)
* **Symptom:** in the first sub-001 QC image, the thalamus mask sat at brainstem level and the WM mask covered the temporal lobes and cerebellum. The MNI → T1 QC looked acceptable, but the EPI-with-T1-edges QC showed the T1 too large and shifted.
* **Investigation:** I tried several T1 → EPI registration setups on the saved mean EPI:
  * default rigid
  * rigid with a bias-corrected, skull-stripped EPI as the fixed image
  * rigid with a robust MI schedule
  * affine

  Correlation was useless as a score, because EPI and T1 have opposite CSF contrast. The edge overlays showed the real cause: the "skull-stripped" T1 still contained the **scalp and neck**.
* **Root cause:** the old skull strip affine-registered the nilearn MNI template to the full-head T1. That template is *brain only*, so the registration stretched the brain to fit the whole head, and the warped brain mask (dilated a further 3 mm) became a head mask. The leftover scalp then dominated the T1 → EPI registration.
* **Fix**, in `fmri.py`:
  * **New `skull_strip()`:**
    1. Register ANTs' **full-head** MNI152 (`ants.get_data("mni")`, with skull) to the T1 with quick SyN.
    2. Warp the MNI brain mask across.
    3. Refine it by intensity: keep tissue above the 30th percentile, erode, keep the largest component, dilate back (`get_mask(cleanup=3)`).
    4. Dilate 2 mm (to bring back sulcal CSF), intersect with the warped mask, and fill holes.

    Checked visually: the mask follows the brain surface.
  * **T1 → EPI registration:** the fixed image is now the N4-corrected, skull-stripped mean EPI. The registration uses Mattes MI with a coarse-to-fine schedule (shrink 8/4/2/1, smoothing 4/2/1/0 vox, 2000/1000/500/100 iterations, 50% sampling).
  * The invalid sub-001 outputs (ROI parquet, NIfTIs, summary, anatomy) were deleted and regenerated. The new QC looks correct (see §4, step 3).
  * **Note for the Linux run:** `ants.get_data("mni")` downloads a ~10 MB template into `~/.antspy` the first time, so it needs internet once.

### 5.2 Smaller issues
| Issue | Fix |
|---|---|
| `FeatureExtractor.__call__` rejected a private kwarg in an ad-hoc test | Tested through the real `extract_features` on braindecode windows instead |
| eegdash `find(limit=5000)` returned HTTP 422 | `limit=1000`, the API maximum |
| nilearn `add_contours(alpha=…)` warning | Use `transparency=` |
| Mask-overlay legend text invisible (black on black) | `labelcolor="white"` |
| Windows matplotlib crashed (no Tcl/Tk) | `Agg` backend everywhere |
| sub-002 checked after the fix | Coverage 1.00, mean FD 0.06 mm, thalamus 633/664 voxels; masks and EPI↔T1 alignment look correct |
| EEG PSD QC images lost when the bad fMRI QC folders were deleted; step 2 did not regenerate them because features were cached, so the notebook failed | Step 2 now regenerates a missing PSD image even when the features are cached |
| Controls bar chart: value labels floated mid-axis for bars below chance | Labels now form a fixed column right of the plot area |
| eegdash lists *raw* EEG runs, so step 3 would have downloaded and processed BOLD for 5 runs that have no cleaned EEG (~50 min wasted) | Step 3 skips runs with no EEG-feature file (warning logged) |
| The S3 bucket contains a macOS `._…_bold.nii.gz` metadata file (sub-015 ses-002) | `records.query_runs` ignores `._*` files |
| Background run stopped when the previous session ended / the PC slept overnight | Rerun (the steps are resumable); keep the PC awake during long runs |

---

## 6. Known caveats / open points
* **The CSF confound mask is small** (79 voxels for sub-001 after the 2-voxel erosion of the lateral ventricles). It works, but `CONFOUND_ERODE_VOX` could be 1 for CSF. Left unchanged for now, so that both smoke subjects use the same settings.
* The "expected" thalamus voxel count in the QC summary (~2,575) is the uneroded atlas volume. The observed ~1,090 voxels after a 1-voxel erosion at 2 mm is consistent with that.
* There is a small ventral misfit near the temporal poles, from EPI distortion; no fieldmap correction is applied. The thalamus is not affected.
* **Runs without cleaned EEG:** all 24 subjects have cleaned EEG (an earlier note that sub-022 was missing was wrong). Five BOLD runs have no cleaned EEG:
  * sub-003 ses-001 GradCPT
  * all 4 runs of sub-023 ses-002

  Step 2 skips them with a warning, and step 3 now skips their BOLD as well (see §5.2).
* Not implemented yet (stretch goal): the thalamic response around GradCPT errors.

---

## 7. Full-dataset estimates

### 7.1 Number of windows
Sources:
* Volume counts were read from the header of every BOLD file on S3, streaming only the first bytes of each.
* "Has cleaned EEG" was checked against the derivatives listing.

The `.fdt` sizes overestimate, because the EEG recording runs longer than the scan (sub-001 GradCPT has 289 TRs of EEG but only 240 volumes).

| | GradCPT | ExperienceSampling | Total |
|---|---|---|---|
| BOLD runs on OpenNeuro | 47 | 140 | 187 |
| Runs with cleaned EEG (usable) | 45 | 137 | **182** |
| Volumes per run (min / median / max) | 240 / 255 / 255 | 239 / 307 / 483 | |
| Usable volumes | ≈ 11,400 | ≈ 42,300 | ≈ 53,700 |
| Windows after end-trim (−2) and HRF warm-up (−8) per run | ≈ 10,950 | ≈ 40,940 | ≈ 51,900 |
| **Windows after scrubbing** (≈ 4.8 %, as in the smoke run) | **≈ 10,400** | **≈ 39,000** | **≈ 49,400** |

* That is about **2,050 windows per subject** (≈ 68 min of data), or about 49k rows × 248 features in the model table (≈ 100 MB as float64).
* With the binary label, about 12,350 windows are "active" (25 % per subject by construction).
* The scrub rate is extrapolated from 2 runs, so expect anything between about 47k and 51k windows.

### 7.2 Can the full run be done on this Windows machine?
**This machine:** Intel i5-9300H laptop CPU (4 cores / 8 threads), 7.9 GB RAM, 79 GB free on C:. The pipeline uses no GPU (ANTs and scikit-learn are CPU-only), so the Linux box's GPU does not matter; its cores do.

Measured here, extrapolated to 182 runs / 24 subjects:

| Step | Measured | Full-run estimate (here) | Limiting factor |
|---|---|---|---|
| 1. Download EEG + T1 | ~37 MB/s | ~3 min (4.9 GB) | network |
| 2. EEG features | 12 s per 240-TR run | ~45 min with 1 job, ~15 min with `--n-jobs 4` | CPU |
| 3a. Anatomy (skull strip + SyN), once per subject | ~3-3.5 min | ~1.4 h | CPU |
| 3b. BOLD download (streamed) | ~8 s per run | ~30 min (66 GB transferred, never stored) | network |
| 3c. Motion correction | ~9.3 min per 240-volume run (≈ 2.3 s/volume, all 8 threads) | **~34 h** for 53.7k volumes | CPU; **the bottleneck** |
| 3d. T1→EPI registration, masks, cleaning | ~20 s per run | ~1 h | CPU |
| 4. Build table | seconds | < 5 min | |
| 5. Models + 200-shift null, both label modes | per fold fit, 48k rows: logistic 0.2 s, ridge 2 s, HGB classifier 40 s, HGB regressor 1 s | ~1-1.5 h with `--n-jobs 4` (about 9,800 linear fits + 48 HGB fits) | CPU |
| **Total** | | **≈ 38-40 h of continuous running** | |

**Disk:** fine.
* About 4.2 GB cleaned EEG + 0.7 GB T1s + ~1 GB derived NIfTIs/QC + one BOLD at a time (≤ 0.5 GB).
* That comes to under 10 GB of the 79 GB free.
* The 68 GB of BOLD is never stored all at once, because each run is deleted after extraction.

**RAM:** OK only with one fMRI job.
* A 240-volume motion correction used ~0.85 GB.
* The longest run (483 volumes, ~1.4 GB as float32) should peak at about 2-3 GB.
* Running 2 subjects in parallel (`--n-jobs 2` in step 3) could exceed 7.9 GB, and parallel jobs would not speed things up anyway, because 1 job already uses all 8 threads.

**Verdict: possible here, but the Linux machine is the better choice.**
* Steps 1, 2, 4 and 5 are cheap enough to run anywhere (≈ 2-3 h in total).
* Step 3 (motion correction) is the problem: about 1.5 days of a laptop CPU at 100 %. That needs:
  * sleep disabled for the whole run
  * mains power
  * good cooling, because thermal throttling on a laptop would stretch the estimate further

  The pipeline is resumable (finished runs are cached), so an interruption costs at most the run in progress.
* On a 16-core Linux machine, step 3 with `FMRI_JOBS=4` (4 subjects at a time, 4 ANTs threads each) should take roughly **8-10 h**, and the whole pipeline about half a day.
* **A hybrid also works:**
  1. Run steps 1-2 here.
  2. Run step 3 on Linux, or split it by subject across both machines with `--subjects`.
  3. Copy `data/derived/roi/` back.
  4. Run steps 4-5 on either machine.
* If staying on Windows, use `powercfg /change standby-timeout-ac 0` and start with:
  ```powershell
  python scripts/03_fmri_roi.py --subjects all --n-jobs 1
  ```

## 8. Next steps
1. ~~Finish step 3 for sub-002 and check its QC images.~~ Done.
2. ~~Run steps 4 and 5 on the smoke sample.~~ Done.
3. ~~Execute the notebook.~~ Done.
4. Optional before the full run: reduce the CSF erosion to 1 voxel (the CSF masks are 40-80 voxels).
5. Run `run_all.sh` on the Linux machine for all subjects. Then review `outputs/qc/index.html` for flagged runs and fill in the README results.
