"""Central configuration: paths, acquisition constants and analysis parameters.

Paths are machine-agnostic. The data root is resolved, in priority order, from
an explicit ``--data-dir`` CLI argument, the ``THAL_DATA_DIR`` environment
variable, or ``<repo>/data``. This lets the same code run on a small Windows dev
box (smoke sample) and on a large Linux box (all 24 subjects).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
DATASET = "ds007216"
S3_ROOT = f"openneuro.org/{DATASET}"  # public, anonymous access
EEG_DERIV_SUFFIX = "_eeg_Bergen_CWreg"  # gradient + BCG (Bergen) + CWL-regressed EEG
ALL_SUBJECTS = [f"{i:03d}" for i in range(1, 25)]

# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------
TR = 2.0  # s
EEG_SFREQ = 250.0  # Hz, sampling rate of the cleaned derivative EEG
TR_MARKER = "T  1"  # scanner volume marker in the EEG event stream
N_TRIM_END = 2  # drop the last TRs of every run (phantom/missing end markers)
# Multiband EPI: 81 slices / MB factor 3 = 27 excitations per 2 s TR -> 13.5 Hz.
SLICE_FREQ = 81 / 3 / TR

EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
    "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz", "FC1", "FC2", "CP1", "CP2",
    "FC5", "FC6", "CP5", "CP6", "TP9", "TP10", "POz",
]  # 31 cortical channels
CWL_CHANNELS = ["CWL1", "CWL2", "CWL3", "CWL4"]  # carbon-wire loops: QC/scrubbing only
ECG_CHANNEL = "ECG"

# --------------------------------------------------------------------------
# EEG preprocessing / features
# --------------------------------------------------------------------------
L_FREQ = 1.0
H_FREQ = 45.0
# Line noise is 60 Hz (Philadelphia, USA). With H_FREQ = 45 Hz the FIR low-pass
# already attenuates 60 Hz strongly, so no notch is applied. Set this to 60.0
# if H_FREQ is ever raised above ~55 Hz.
NOTCH_FREQ: float | None = None

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
# Feature-name prefixes that get a log transform before modeling (heavy-tailed
# amplitude measures). Matching is done on the feature part of the column name.
LOG_FEATURE_PREFIXES = ("variance", "line_length")

# --------------------------------------------------------------------------
# fMRI
# --------------------------------------------------------------------------
HIGH_PASS = 0.008  # Hz
USE_GSR = True  # regress the global signal (plan.md section 4)
THAL_ERODE_VOX = 1
CONFOUND_ERODE_VOX = 2
COVERAGE_MIN = 0.9  # QC flag: fraction of thalamus mask inside the EPI brain mask
QC_FD_MEAN_MAX = 0.3  # mm, QC flag on mean framewise displacement

# --------------------------------------------------------------------------
# Alignment / labels
# --------------------------------------------------------------------------
HRF_WARMUP_TRS = 8  # drop first 16 s after HRF convolution
FD_THRESH = 0.5  # mm, scrub TRs above this
CWL_MAD_K = 5.0  # scrub TRs with CWL RMS > median + k * MAD (per run)
ACTIVE_PERCENTILE = 75  # binary label: "active" if above the subject's percentile
ROIS = ("thal", "v1", "thal_nov1")  # thal_nov1 = thalamus with V1 regressed out

# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------
N_NULL = 200
RANDOM_STATE = 0


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    data: Path
    outputs: Path

    @property
    def raw(self) -> Path:  # mirror of the S3 BIDS tree
        return self.data / "raw" / DATASET

    @property
    def derived(self) -> Path:
        return self.data / "derived"

    @property
    def eeg_features(self) -> Path:
        return self.derived / "eeg_features"

    @property
    def anat(self) -> Path:
        return self.derived / "anat"

    @property
    def roi(self) -> Path:
        return self.derived / "roi"

    @property
    def table(self) -> Path:
        return self.derived / "model_table.parquet"

    @property
    def qc(self) -> Path:
        return self.outputs / "qc"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def results(self) -> Path:
        return self.outputs / "results"

    def makedirs(self) -> "Paths":
        for p in (self.raw, self.eeg_features, self.anat, self.roi, self.qc,
                  self.figures, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(data_dir: str | Path | None = None,
              output_dir: str | Path | None = None) -> Paths:
    data = Path(data_dir or os.environ.get("THAL_DATA_DIR") or PROJECT_ROOT / "data")
    out = Path(output_dir or os.environ.get("THAL_OUTPUT_DIR") or PROJECT_ROOT / "outputs")
    return Paths(data=data.expanduser().resolve(), outputs=out.expanduser().resolve()).makedirs()
