# Thalamic activity from scalp EEG (ds007216)

A mock project: can classical machine learning on **scalp-EEG features** predict, window by window, how active the **thalamus** is? Thalamic activity is measured by simultaneous fMRI.
The goal is to show a careful, leakage-free pipeline and an honest interpretation, not a scientific claim. Above-chance leave-one-subject-out performance is the target.

* **Data:** [OpenNeuro ds007216](https://openneuro.org/datasets/ds007216), simultaneous EEG-fMRI, 24 healthy adults × 2 sessions, GradCPT + resting experience sampling, TR = 2 s.
* **EEG:** discovered with [eegdash](https://eegdash.org), windowed with braindecode, features from the **eegdash feature bank**.
* **fMRI:** light Python pipeline (ANTsPy + nilearn).
* **Models:** ridge / logistic regression and gradient boosting, with leave-one-subject-out CV, a circular-shift null, a V1 control region and an alpha ablation.

![pipeline](outputs/figures/pipeline.png)

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_download.py` | Finds every run through the eegdash metadata API and downloads the **cleaned EEG derivatives** and session-1 T1s from the public OpenNeuro bucket. `--dry-run` reports the size first. |
| 2 | `scripts/02_eeg_features.py` | Loads the cleaned EEG, keeps the 31 scalp channels, band-passes 1-45 Hz and sets an average reference. Cuts **one 2-s window per scanner volume marker** and extracts eegdash features (31 ch × 8 = 248). Also writes an EEG PSD QC figure per run. |
| 3 | `scripts/03_fmri_roi.py` | For each BOLD run, downloaded just in time and deleted afterwards: motion correction, MNI → T1 (SyN) → EPI registration, thalamus / V1 / WM / CSF masks, confound regression. Writes QC images plus `outputs/qc/index.html`. |
| 4 | `scripts/04_build_table.py` | Trims both streams to the usable TRs, scrubs, **HRF-convolves the EEG features**, and adds both label types. |
| 5 | `scripts/05_model.py` | LOSO models, null distribution, controls and ablations, figures. |
| — | `notebooks/thalamus_from_eeg.ipynb` | Narrative walkthrough built from the cached outputs. |

### Key design decisions (see `deliberates/`)
* **Cleaned EEG from the derivatives.** eegdash indexes only the raw 5 kHz EEG, which still contains MR-gradient and BCG artifacts. The dataset's `derivatives/` folder has these removed (Bergen toolbox + carbon-wire-loop regression, 250 Hz). eegdash is still used for record discovery, windowing and feature extraction.
* **No notch filter.** The site is in Philadelphia, so line noise is at 60 Hz, which the 45 Hz low-pass already removes (documented in `eeg.preprocess_raw`; `config.NOTCH_FREQ` re-enables a notch).
* **TR-locked, non-overlapping windows.** Window *k* = `[marker_k, marker_k + 2 s)`. Usable TRs = `min(markers, volumes) - 2`, which handles the phantom or missing end-of-run markers described in the dataset README.
* **Convolve forward, don't shift back.** Each EEG feature time series is convolved with the SPM canonical HRF and then compared with the BOLD at the same TR.
* **One anatomical anchor.** Both sessions are registered via the high-resolution session-1 MPRAGE, not the 4-mm 2D T1 of session 2. The thalamus mask is eroded by 1 voxel.
* **Two labeling options** (both are always stored):
  * `float`: `thal_z`, the thalamic BOLD z-scored within subject.
  * `binary`: `thal_active`, 1 if `thal_z` is above the subject's 75th percentile, else 0. The split is 25/75, so models use class weights and are scored by AUC.
  * Select with `05_model.py --label float|binary|both`.
* **Validation:**
  * Leave-one-subject-out CV (both sessions of a subject stay in the same fold), with scaling fit inside each fold.
  * 200 within-run circular shifts of the target as a null.
  * Wilcoxon test across subjects.
  * **V1 control**, **thalamus minus V1**, **no-alpha** ablation, per-task breakdown.

### Adding features
Edit `src/thalamic/features.py::build_feature_extractor()`. You can add eegdash built-ins (`signal_hjorth_*`, `spectral_slope`, `complexity_*`, connectivity…) or your own functions decorated with `eegdash.features.univariate_feature`. Downstream code discovers the `<feature>_<channel>` columns automatically.

## Running

```bash
# environment (Python >= 3.11)
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q
```

**Small local smoke test** (2 subjects, 1 run each, about 0.7 GB, Windows):
```powershell
./run_smoke.ps1
```

**Full run** (all 24 subjects; designed for a Linux box with plenty of disk and cores):
```bash
THAL_DATA_DIR=/big/disk/thal N_JOBS=16 ./run_all.sh
```

* Data goes to `$THAL_DATA_DIR`, defaulting to `./data`. Outputs go to `$THAL_OUTPUT_DIR`, defaulting to `./outputs`.
* The cleaned EEG and T1s are about 5 GB. The ~65 GB of raw BOLD is streamed: each run is deleted after its ROI time series are extracted, unless `--keep-bold` / `KEEP_BOLD=1` is set.
* Every step caches its outputs and can be re-run or resumed.
* `--subjects 001 002` and `--limit-runs N` work on every script.

**Check the masks:** open `outputs/qc/index.html`. It shows, for every run, the mean EPI with the thalamus (L/R), V1, WM and CSF masks as filled contours, EPI ↔ T1 and MNI ↔ T1 edge overlays, ROI/FD time series, and EEG PSDs. Runs with low thalamus coverage or high motion are flagged in red.

## Results

*To be filled in after the full run on all 24 subjects. The smoke run on 2 subjects only tests the plumbing.*
Figures are written to `outputs/figures/`: per-subject scores vs. null, null distribution, controls/ablations, predicted vs. actual time course, and coefficient topomaps.

## Limitations
* **EEG cannot measure the thalamus directly.** The model learns scalp-visible *correlates* of thalamic BOLD: rhythms (alpha, spindle-range power), arousal and vigilance, and global state changes. The V1 control and the thalamus-minus-V1 target exist to show how much of the effect is thalamus-specific.
* Alpha power tracks vigilance and drowsiness and may explain much of the signal (see the no-alpha ablation).
* Subcortical BOLD has a lower SNR than cortical BOLD.
* The fMRI preprocessing is deliberately light:
  * no fieldmap distortion correction, no slice timing;
  * Harvard-Oxford atlas masks rather than subject-specific segmentation;
  * a scanner software upgrade affects the last 1.5 subjects (handled only by the LOSO design).
* MR-gradient/BCG artifact removal was done upstream by the dataset authors. Residual slice-frequency energy (13.5 Hz harmonics) is visible in the PSD QC.
* The sample is small and homogeneous: 24 healthy young adults from one area.
* Not yet implemented (stretch goal): an attention-relevance check of the thalamic signal around GradCPT errors.

## Layout
```
src/thalamic/   config, records (eegdash + S3), eeg, features, fmri, qc, align, model, plots, cli
scripts/        01-05 pipeline steps
tests/          unit tests (alignment, HRF, labels, LOSO, null, planted-signal recovery)
notebooks/      thalamus_from_eeg.ipynb
deliberates/    original plan and problem/solution notes
```
