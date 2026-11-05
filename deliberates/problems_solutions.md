Breakdown of every technical challenge identified in the dataset and plan, paired with its direct, practical solution.

---

### 1. Desynchronized TR Markers at Run Ends

* **The Problem:** Functional runs were stopped manually, causing mismatch at the very end of resting-state runs. The EEG recording received 1–2 phantom scanner markers (`T 1`) after scanning stopped, or missed a marker. If aligned naively by sample index, the alignment between EEG and fMRI drifts.
* **The Solution:** **Trim the tail.** Delete the last 2 TRs (4 seconds) of both EEG and fMRI across all runs. Align forward from the first validated `T 1` marker, which is confirmed to match the first fMRI volume across all subjects.

---

### 2. Scanner Artifact Leakage into EEG Bands

* **The Problem:** The MR gradient artifact is orders of magnitude larger than neural EEG. Standard template subtraction (AAS) leaves residual spikes at the slice/volume excitation frequencies (harmonics around 13.5 Hz, 27 Hz, etc.). These land directly inside alpha, beta, and low-gamma bands, risking models learning scanner pulses rather than brain dynamics.
* **The Solution:**
* Prioritize checking `derivatives/` to see if professionally cleaned EEG is already available.
* If processing from raw, inspect power spectra around gradient harmonic frequencies.
* Use the 4 Carbon Wire Loop (CWL) channels as motion regressors or for motion-scrubbing (drop time windows where CWL variance spikes).



---

### 3. Powerline Noise Frequency

* **The Problem:** Data was collected in Philadelphia, USA, where the grid frequency is 60 Hz (not European 50 Hz). A notch filter set to 50 Hz does nothing.
* **The Solution:** Notch filter at **60 Hz** (and harmonics at 120 Hz if analyzing frequencies above 45 Hz). Since your bandpass is 1–45 Hz, standard bandpass filtering already rejects 60 Hz, but confirm your high-cutoff roll-off does not distort low-gamma (30–45 Hz).

---

### 4. Hemodynamic Blur vs. Naive Time Shifting

* **The Problem:** Shifting BOLD back by 4–6 seconds assumes a single fixed delay. In reality, the hemodynamic response function (HRF) acts as a low-pass filter with a 10–12 second span. In 4-second overlapping windows, consecutive samples share heavy autocorrelation, producing blurry regression targets.
* **The Solution:** **Convolve forward, do not shift backward.**
* Extract continuous time series of your EEG spectral features.
* Convolve the feature curves with a canonical hemodynamic response function (HRF).
* Downsample the convolved EEG features directly to the fMRI TR grid (every 2 seconds).
* Use **non-overlapping windows** (step size = window length) to avoid artificial sample autocorrelation.



---

### 5. Thalamic Mask Registration Quality (Session 2 T1w Scan)

* **The Problem:** Session 1 collected high-resolution 0.8 mm isotropic 3D T1w images, while Session 2 used a fast 2D T1w scan with coarse 4.0 mm slice thickness. Registering an anatomical atlas directly to Session 2 risks severe partial volume effects (bleeding signal from ventricles or internal capsule into the thalamus mask).
* **The Solution:**
* If using preprocessed outputs (e.g., fMRIPrep), use the unified subject T1w template derived from Session 1 to warp all functional runs into standard MNI space, extracting the Harvard-Oxford thalamus mask in MNI space.
* If running registration manually, coregister Session 2 functional runs to Session 1’s 3D T1w image, rather than relying on Session 2’s thick-slice 2D scan.
* Erode the thalamic mask by 1 voxel to minimize boundary contamination from CSF and white matter.



---

### 6. Scanner Software Upgrade Discrepancy

* **The Problem:** Subjects 1–22 (and sub-023 ses-1) were scanned on Syngo MR E11 with nonlinear gradient correction disabled. Sub-023 ses-2 and sub-024 used Syngo MR XA30 with nonlinear gradient correction enabled. This introduces slight spatial geometric differences at the edges of the field of view across the cohort.
* **The Solution:** Ensure B0 fieldmap unwarping (susceptibility distortion correction) is applied consistently using the acquired field maps for every run. Because Leave-One-Subject-Out CV isolates subjects into test folds, cross-subject variations won't leak, but standardizing EPI pre-alignment through standard BIDS pipelines handles this discrepancy cleanly.

---

### Summary Checklist for a Lean Project Scope

| Focus Area | Recommended Action |
| --- | --- |
| **Data Intake** | Check `derivatives/` first. Use existing preprocessed BOLD and cleaned EEG if present. |
| **Synchronization** | Align from the first TR marker; drop the final 2 TRs from every run. |
| **Grid Filter** | Set notch to 60 Hz (Philadelphia site). |
| **HRF Alignment** | Convolve EEG feature trajectories with canonical HRF; resample to the 2 s TR. |
| **Sampling** | Use non-overlapping windows (e.g., 2 s or 4 s with no step overlap) to prevent inflated CV metrics. |
| **Anatomy** | Rely on Session 1 high-res T1w for spatial normalization across both sessions; erode thalamus mask. |
| **CWL Channels** | Use for motion thresholding/scrubbing rather than direct ML features. |
| **Behavioral Task** | Keep the GradCPT behavioral correlation check as a stretch goal; focus first on methodological validation and control regions. |