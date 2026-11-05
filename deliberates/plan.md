# Plan: Predicting Thalamic Activity from EEG (ds007216)

## Goal
Build a small, classical ML model that predicts whether the thalamus is active from EEG features in each time window. The project should show four things: careful EEG preprocessing, sound fMRI target construction, subject-level validation, and honest limits on what EEG can claim about deep structures.

Important(!): This is a mock project, i'ts purpose is to show capabilities and not deliver scientific results. For that reason we keep it simple, wnating to show above-chance detection.

## Deliberates

1. Data
Source: OpenNeuro ds007216, simultaneous EEG-fMRI in BIDS format, 24 subjects with two sessions each.
Tasks: GradCPT and resting-state experience sampling. The plan treats all samples the same regardless of task, so task is a covariate to report, not a split to model.
Check first: whether the derivatives/ folder already contains preprocessed EEG or fMRI. The paper describes raw BrainVision data at 5 kHz, so I expect the EEG to be raw. Confirm before writing a pipeline.

2. EEG preprocessing
If the EEG is raw, the pipeline is:

Load with eegdash, keeping the 31 cortical channels and dropping the four carbon-wire-loop (CWL) channels and the ECG channel from the feature set.
Remove the MR gradient artifact (average artifact subtraction) and the ballistocardiogram artifact. The CWL channels are useful as references for this step, even though they are not features.
Notch at 60 Hz, depending on the acquisition site.
Bandpass 1–45 Hz.
Re-reference to the average.
Downsample to around 250 Hz.
The artifact correction is the step most likely to determine whether the features are trustworthy, so document it in detail.

3. Windows and features
Windows: fixed-length windows, for example 4 s with 2 s step. Each window is one sample.
Features, per channel: relative band power in delta, theta, alpha, beta, and gamma. Add spectral entropy and line-length or variance as simple amplitude measures. This keeps the spatial resolution you asked for: 31 channels × about 7 features gives roughly 200 columns.
Optional: alpha asymmetry or channel-pair coherence later, if the baseline works.
Not in the plan yet: deep learning, and feature sets that depend on all channels jointly.

4. fMRI target
This is the hardest part, so I want to be precise about it.

The thalamus is not detected per sample. It is a fixed anatomical region. Each sample's label comes from the thalamic BOLD time series within that window, shifted by the hemodynamic lag (about 4–6 s).

The steps are:

Register the thalamus from an atlas (Harvard-Oxford subcortical, MNI space) into each subject's native EPI space, using the T1 image that matches each session. Session 1 has an MPRAGE. Session 2 has a fast 2D T1, so the registration quality will differ and should be checked visually.
Extract the mean BOLD in the thalamic mask for each TR.
Regress motion parameters, the mean white-matter and CSF signal (CompCor), and the global signal.
Trim the extra or missing TR markers at the ends of resting runs, as the dataset documentation describes.
Z-score within subject across all that subject's TRs, so "thalamic activity" is relative to that person's own distribution.
Shift by the hemodynamic lag and average over the window to get one value per sample.

5. Labels: binary, partial, or float
My recommendation is to keep the float as the primary target and derive the categories from it:

Primary target: the continuous z-score. It keeps all the information, suits regression, and avoids an arbitrary cutoff.
Binary label for the classifier: "active" if the z-score is above the subject's 75th percentile, "non-active" otherwise. The within-subject threshold keeps the classes balanced for each person.
Skip the three-level "partial" class. The middle band would be the least reliable region of the signal, and it adds a boundary you'd have to defend. A float plus a binary cutoff gives you the same story with less justification needed.
Report results for both. If the regression is weak but the classifier is clearly above chance, say so.

6. Models
Baseline: ridge regression on the float target, and logistic regression on the binary target.
Comparison: a random forest or gradient boosting model, to check whether non-linear interactions add anything.
Stop there. This is the classical-ML scope you specified.

7. Validation
Leave-one-subject-out CV. Keep both sessions of a subject in the same fold.
Feature standardization inside each fold, to avoid leakage.
Null comparison: phase-randomized or time-shuffled thalamic BOLD should score near chance.
Control region: repeat the analysis with a non-attention region (for example, primary visual cortex). If the thalamus does about as well, the model is probably learning generic signal, such as alpha power, arousal, or motion.
Alpha-removed model: report performance with alpha features dropped, since alpha tracks vigilance and drowsiness and could explain much of the signal.

8. Attention-relevance check
Before presenting the thalamus as attention-related, verify that its BOLD differs between GradCPT commission errors and hits, or correlates with the attention-related rating dimensions. If it doesn't, describe the target as "thalamic activity" and drop the attention claim.

9. Deliverable
A repository with a README, a notebook that runs top to bottom, and a short limitations section.
Figures: predicted vs. actual thalamic signal, per-channel feature importance, the null comparison, and the control-region comparison.
Limitations to state explicitly: EEG cannot measure the thalamus directly. The model learns scalp-visible correlates of thalamic activity, such as rhythms, arousal, and global state changes. Subcortical BOLD has a lower signal-to-noise ratio than cortical BOLD. The sample is small and mostly from one university area.

## Open decisions
Confirm whether the EEG is raw or preprocessed.
Confirm the thalamus is covered in the EPI field of view for all sessions.
Choose the window length and step size.
Decide whether to include the attention-relevance check or leave it as a stretch goal.