"""EEG <-> fMRI alignment, HRF convolution, scrubbing and label construction.

EEG features are already on the TR grid (one 2-s window per scanner marker).
Rather than shifting the BOLD backwards by a fixed lag, each EEG feature time
series is **convolved forward with the canonical HRF**, so it is directly
comparable with the BOLD sample at the same TR (problems_solutions.md #4).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .eeg import n_usable_trs

log = logging.getLogger(__name__)

KEYS = ["subject", "session", "task", "run", "stem"]


# --------------------------------------------------------------------------
# Feature transforms
# --------------------------------------------------------------------------
def _zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def prepare_features(X: pd.DataFrame) -> np.ndarray:
    """Log-transform amplitude features, then z-score each column within the run.

    The within-run z-score uses no label information, so it does not leak
    across the LOSO folds. It removes run- and subject-level offsets (e.g.
    impedance and cap-fit differences), which matter a lot across subjects.
    """
    arr = X.to_numpy(dtype=float).copy()
    for j, col in enumerate(X.columns):
        feat = col.rsplit("_", 1)[0]
        if feat.startswith(config.LOG_FEATURE_PREFIXES):
            arr[:, j] = np.log(np.clip(arr[:, j], 1e-30, None))
    return _zscore(arr)


def scrub_flags(fd: np.ndarray, cwl_rms: np.ndarray) -> np.ndarray:
    """TRs to drop: FD above threshold or CWL RMS outliers (median + k * MAD)."""
    fd = np.asarray(fd, dtype=float)
    cwl = np.asarray(cwl_rms, dtype=float)
    med = np.median(cwl)
    mad = 1.4826 * np.median(np.abs(cwl - med))
    cwl_bad = cwl > med + config.CWL_MAD_K * mad if mad > 0 else np.zeros_like(cwl, bool)
    return (fd > config.FD_THRESH) | cwl_bad


def interpolate_bad(X: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Linear interpolation over flagged rows (keeps the series continuous for convolution)."""
    if not bad.any() or bad.all():
        return X
    idx = np.arange(len(X))
    good = ~bad
    out = X.copy()
    for j in range(X.shape[1]):
        out[bad, j] = np.interp(idx[bad], idx[good], X[good, j])
    return out


def hrf_kernel(tr: float = config.TR) -> np.ndarray:
    """SPM canonical HRF sampled at the TR, normalised to unit sum."""
    from nilearn.glm.first_level import spm_hrf

    h = spm_hrf(tr, oversampling=1, time_length=32.0)
    return h / h.sum()


def hrf_convolve(X: np.ndarray, tr: float = config.TR) -> np.ndarray:
    """Causal convolution of each column with the HRF, same length as the input."""
    h = hrf_kernel(tr)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    return np.column_stack([np.convolve(X[:, j], h)[:n] for j in range(X.shape[1])])


# --------------------------------------------------------------------------
# Run alignment
# --------------------------------------------------------------------------
def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of y after OLS on [1, x]."""
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def align_run(eeg_run: pd.DataFrame, roi_run: pd.DataFrame,
              feature_cols: list[str]) -> pd.DataFrame:
    """Trim both streams to the usable TRs, scrub, HRF-convolve, drop warm-up."""
    eeg_run = eeg_run.sort_values("tr").reset_index(drop=True)
    roi_run = roi_run.sort_values("tr").reset_index(drop=True)
    n_use = n_usable_trs(len(eeg_run), len(roi_run))
    eeg_run, roi_run = eeg_run.iloc[:n_use], roi_run.iloc[:n_use]
    if not (eeg_run.tr.to_numpy() == roi_run.tr.to_numpy()).all():
        raise ValueError(f"{eeg_run.stem.iloc[0]}: TR index mismatch between EEG and BOLD")

    X = prepare_features(eeg_run[feature_cols])
    bad = scrub_flags(roi_run.fd.to_numpy(), eeg_run.cwl_rms.to_numpy())
    X = interpolate_bad(X, bad)
    Xc = hrf_convolve(X)

    keep = ~bad & (np.arange(n_use) >= config.HRF_WARMUP_TRS)
    thal, v1 = roi_run.thal.to_numpy(), roi_run.v1.to_numpy()
    meta = pd.DataFrame({
        **{k: eeg_run[k].iloc[0] for k in KEYS},
        "tr": eeg_run.tr.to_numpy()[keep],
        "fd": roi_run.fd.to_numpy()[keep],
        "thal": thal[keep],
        "v1": v1[keep],
        "thal_nov1": residualize(thal, v1)[keep],  # thalamus-specific part
    })
    out = pd.concat([meta, pd.DataFrame(Xc[keep], columns=feature_cols)], axis=1)
    out.attrs["n_scrubbed"] = int(bad[config.HRF_WARMUP_TRS:].sum())
    return out


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
def make_labels(df: pd.DataFrame, roi: str, mode: str,
                percentile: float = config.ACTIVE_PERCENTILE) -> pd.Series:
    """Window labels for an ROI.

    mode="float"  -> z-score of the ROI signal within subject (across all runs).
    mode="binary" -> 1 ("active") if that z-score is above the subject's
                     ``percentile``-th percentile, else 0 ("inactive").
    """
    z = df.groupby("subject")[roi].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    if mode == "float":
        return z.rename(f"{roi}_z")
    if mode == "binary":
        thr = z.groupby(df["subject"]).transform(lambda s: np.percentile(s, percentile))
        return (z > thr).astype(int).rename(f"{roi}_active")
    raise ValueError(f"mode must be 'float' or 'binary', got {mode!r}")


def label_column(roi: str, mode: str) -> str:
    return f"{roi}_z" if mode == "float" else f"{roi}_active"


def build_table(eeg_df: pd.DataFrame, roi_df: pd.DataFrame,
                feature_cols: list[str]) -> pd.DataFrame:
    """Align every run present in both streams and add both label types per ROI."""
    stems = sorted(set(eeg_df.stem) & set(roi_df.stem))
    missing = sorted(set(eeg_df.stem) ^ set(roi_df.stem))
    if missing:
        log.warning("runs present in only one stream (skipped): %s", missing)
    parts, n_scrub = [], 0
    for stem in stems:
        part = align_run(eeg_df[eeg_df.stem == stem], roi_df[roi_df.stem == stem], feature_cols)
        n_scrub += part.attrs.get("n_scrubbed", 0)
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    for roi in config.ROIS:
        table[label_column(roi, "float")] = make_labels(table, roi, "float")
        table[label_column(roi, "binary")] = make_labels(table, roi, "binary")
    log.info("model table: %d windows, %d runs, %d subjects, %d scrubbed TRs",
             len(table), len(stems), table.subject.nunique(), n_scrub)
    return table
