"""EEG loading, preprocessing and TR-locked windowing.

Input is the *cleaned* derivative EEG of ds007216 (``*_eeg_Bergen_CWreg.set``):
MR-gradient and ballistocardiogram artifacts were removed upstream with the
Bergen fMRI toolbox and carbon-wire-loop (CWL) regression was applied; data are
at 250 Hz and still contain the scanner volume markers (``"T  1"``).

Windows are locked to those markers: window *k* spans ``[T_k, T_k + 2 s)``, i.e.
exactly one fMRI volume. Windows are therefore non-overlapping and live on the
fMRI TR grid directly (see ``deliberates/problems_solutions.md`` #1 and #4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from . import config

log = logging.getLogger(__name__)


def load_clean_eeg(set_path: str | Path) -> mne.io.BaseRaw:
    """Load a cleaned derivative ``.set`` file and assign channel types."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    types = {}
    if config.ECG_CHANNEL in raw.ch_names:
        types[config.ECG_CHANNEL] = "ecg"
    types.update({c: "misc" for c in config.CWL_CHANNELS if c in raw.ch_names})
    raw.set_channel_types(types, verbose="ERROR")
    missing = set(config.EEG_CHANNELS) - set(raw.ch_names)
    if missing:
        raise ValueError(f"{set_path}: missing EEG channels {sorted(missing)}")
    if not np.isclose(raw.info["sfreq"], config.EEG_SFREQ):
        log.warning("%s: sfreq %.1f, resampling to %.1f", set_path, raw.info["sfreq"], config.EEG_SFREQ)
        raw.resample(config.EEG_SFREQ, verbose="ERROR")
    return raw


def tr_marker_samples(raw: mne.io.BaseRaw) -> np.ndarray:
    """Sample indices of the scanner volume markers, sorted.

    Marker descriptions are whitespace-normalised because EEGLAB/MNE may keep or
    collapse the double space in ``"T  1"``.
    """
    target = " ".join(config.TR_MARKER.split())
    onsets = [a["onset"] for a in raw.annotations if " ".join(a["description"].split()) == target]
    samples = np.round(np.asarray(onsets) * raw.info["sfreq"]).astype(int)
    samples = np.sort(samples)
    if samples.size > 1:
        d = np.diff(samples) / raw.info["sfreq"]
        if not np.allclose(d, config.TR, atol=0.01):
            log.warning("irregular TR markers: diff range %.3f-%.3f s", d.min(), d.max())
    return samples


def n_usable_trs(n_markers: int, n_vols: int, n_trim: int = config.N_TRIM_END) -> int:
    """Number of TRs usable in both streams.

    The first marker always matches the first fMRI volume (dataset README).
    Runs can have 1-2 extra phantom markers at the end, or one marker too few.
    ``min(markers, volumes)`` handles both cases, and trimming the final
    ``n_trim`` TRs adds a safety margin (problems_solutions.md #1).
    """
    return max(0, min(n_markers, n_vols) - n_trim)


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Pick the 31 cortical channels, band-pass 1-45 Hz, average reference.

    Notes
    -----
    **No notch filter is applied.** The recording site is in Philadelphia, USA,
    so line noise is at 60 Hz (not 50 Hz). With a 45 Hz FIR low-pass the 60 Hz
    component is already in the stop band and strongly attenuated, so a notch
    would be redundant. If ``config.H_FREQ`` is ever raised towards or above
    60 Hz, set ``config.NOTCH_FREQ = 60.0`` to re-enable it.

    Gradient (AAS) and BCG correction are *not* done here: they were done
    upstream in the dataset's derivatives (Bergen toolbox + CWL regression).
    """
    raw = raw.copy().pick(config.EEG_CHANNELS)
    raw.set_montage("standard_1020", on_missing="warn", verbose="ERROR")
    if config.NOTCH_FREQ is not None:
        raw.notch_filter(config.NOTCH_FREQ, verbose="ERROR")
    raw.filter(config.L_FREQ, config.H_FREQ, fir_design="firwin", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def cwl_rms_per_tr(raw: mne.io.BaseRaw, starts: np.ndarray, n_samples: int) -> np.ndarray:
    """Mean RMS of the (high-passed) CWL channels in each TR window.

    The CWL loops pick up head motion inside the scanner; large values mark
    windows likely contaminated by motion artifacts. Used for scrubbing only,
    never as a model feature.
    """
    cwl = [c for c in config.CWL_CHANNELS if c in raw.ch_names]
    if not cwl:
        return np.zeros(len(starts))
    x = raw.copy().pick(cwl).filter(1.0, None, picks="all", verbose="ERROR").get_data()
    return np.array([np.sqrt(np.mean(x[:, s:s + n_samples] ** 2)) for s in starts])


def make_tr_windows(raw: mne.io.BaseRaw, starts: np.ndarray, description: dict):
    """Build a braindecode windows dataset with one 2-s window per TR marker.

    Parameters
    ----------
    raw : preprocessed Raw.
    starts : sample indices of the usable TR markers (already trimmed).
    description : recording-level metadata (subject, session, task, run, stem).

    Returns
    -------
    braindecode.datasets.BaseConcatDataset of EEG windows; window metadata
    ``i_trial_in_dataset`` equals the TR index.
    """
    from braindecode.datasets import BaseConcatDataset, RawDataset
    from braindecode.preprocessing import create_windows_from_events

    sf = raw.info["sfreq"]
    win = int(round(config.TR * sf))
    starts = starts[starts + win <= raw.n_times]  # guard against windows past the end
    raw = raw.copy()
    raw.set_annotations(mne.Annotations(onset=starts / sf, duration=0.0,
                                        description=["TR"] * len(starts)))
    ds = BaseConcatDataset([RawDataset(raw, description=description)])
    return create_windows_from_events(
        ds,
        mapping={"TR": 0},
        trial_start_offset_samples=0,
        trial_stop_offset_samples=win,
        window_size_samples=win,
        window_stride_samples=win,
        drop_last_window=False,
        preload=True,
    )


def plot_psd_qc(raw: mne.io.BaseRaw, out_png: str | Path, title: str) -> None:
    """Average PSD with slice-frequency harmonics marked (residual gradient check).

    After AAS, residual gradient energy appears as narrow peaks at the slice
    excitation frequency (13.5 Hz) and its harmonics; these sit inside the
    alpha/beta/gamma bands (problems_solutions.md #2).
    """
    psd = raw.compute_psd(fmin=0.5, fmax=60, n_fft=int(4 * raw.info["sfreq"]), verbose="ERROR")
    f, p = psd.freqs, psd.get_data().mean(0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(f, p, color="k", lw=1)
    for k in range(1, 5):
        fk = k * config.SLICE_FREQ
        if fk < 60:
            ax.axvline(fk, color="tab:red", ls="--", lw=0.8, alpha=0.7)
    ax.axvspan(config.H_FREQ, 60, color="0.9")
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)", title=title)
    ax.text(0.99, 0.95, "red: slice-freq harmonics\ngrey: outside 1-45 Hz band",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
