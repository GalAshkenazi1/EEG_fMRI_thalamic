"""EEG feature definitions (eegdash feature bank).

``build_feature_extractor`` is the **single place** where features are
declared. To add features later, add entries to the dict there, either
another eegdash built-in (``signal_hjorth_mobility``, ``spectral_slope``,
``complexity_*``, connectivity, ...) or a custom function decorated with
``eegdash.features.univariate_feature``. Nothing downstream hard-codes feature
names or counts: columns are named ``<feature>_<channel>`` by eegdash and are
discovered with :func:`feature_columns`.
"""

from __future__ import annotations

from functools import partial

import pandas as pd

from . import config

# Columns added by eegdash/braindecode metadata or by us, i.e. not features.
META_COLUMNS = {
    "i_dataset", "i_window_in_trial", "i_trial_in_dataset", "i_start_in_trial",
    "i_stop_in_trial", "target", "subject", "session", "task", "run", "stem",
    "tr", "cwl_rms",
}


def build_feature_extractor(sfreq: float = config.EEG_SFREQ):
    """Starter feature set: 8 per-channel features (31 ch -> 248 columns).

    * relative band power (delta, theta, alpha, beta, gamma), from a Welch PSD
      normalised to unit total power over 1-45 Hz
    * spectral entropy of the normalised PSD
    * line length and variance (simple amplitude measures; log-transformed later)
    """
    import eegdash.features as F

    return F.FeatureExtractor(
        {
            "spec": F.FeatureExtractor(
                {
                    "rel": F.FeatureExtractor(
                        {
                            "bandpow": partial(F.spectral_bands_power, bands=config.BANDS),
                            "entropy": F.spectral_entropy,
                        },
                        preprocessor=F.spectral_normalized_preprocessor,
                    ),
                },
                preprocessor=partial(
                    F.spectral_preprocessor,
                    fs=sfreq, nperseg=int(sfreq), f_min=config.L_FREQ, f_max=config.H_FREQ,
                ),
            ),
            "line_length": F.signal_line_length,
            "variance": F.signal_variance,
        }
    )


def extract(windows, n_jobs: int = 1) -> pd.DataFrame:
    """Run eegdash feature extraction on a windows dataset -> DataFrame."""
    from eegdash.features import extract_features

    feats = extract_features(windows, build_feature_extractor(), n_jobs=n_jobs)
    return feats.to_dataframe(include_metadata=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All feature columns, i.e. columns named ``<feature>_<channel>``."""
    chans = set(config.EEG_CHANNELS)
    return [c for c in df.columns
            if c not in META_COLUMNS and "_" in c and c.rsplit("_", 1)[1] in chans]


def split_feature_column(col: str) -> tuple[str, str]:
    """``"spec_rel_bandpow_alpha_O1"`` -> ``("spec_rel_bandpow_alpha", "O1")``."""
    feat, ch = col.rsplit("_", 1)
    return feat, ch


def short_feature_name(feat: str) -> str:
    """Readable label for plots: ``spec_rel_bandpow_alpha`` -> ``alpha``."""
    return feat.replace("spec_rel_bandpow_", "").replace("spec_rel_", "spectral_")
