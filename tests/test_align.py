import numpy as np
import pandas as pd
import pytest

from thalamic import align, config
from thalamic.eeg import n_usable_trs


@pytest.mark.parametrize("markers, vols, expected", [
    (241, 240, 238),  # one phantom marker at the end
    (242, 240, 238),  # two phantom markers
    (239, 240, 237),  # one marker missing
    (240, 240, 238),  # perfectly aligned
    (1, 240, 0),
])
def test_n_usable_trs(markers, vols, expected):
    assert n_usable_trs(markers, vols) == expected


def test_hrf_convolution_length_and_causality():
    x = np.zeros(60)
    x[20] = 1.0
    y = align.hrf_convolve(x)[:, 0]
    assert y.shape == (60,)
    assert np.allclose(y[:20], 0)  # nothing before the impulse
    peak_s = (np.argmax(y) - 20) * config.TR
    assert 4 <= peak_s <= 6  # canonical HRF peaks ~5 s after the impulse
    assert np.isclose(align.hrf_kernel().sum(), 1.0)


def test_interpolate_and_scrub():
    X = np.arange(10, dtype=float)[:, None]
    bad = np.zeros(10, bool)
    bad[4] = True
    X2 = X.copy()
    X2[4] = 999
    assert np.isclose(align.interpolate_bad(X2, bad)[4, 0], 4.0)
    fd = np.r_[np.zeros(9), 1.0]
    cwl = np.r_[np.ones(8), 1.01, 50.0]
    flags = align.scrub_flags(fd, cwl)
    assert flags[-1] and not flags[:8].any()


def _toy_run(stem, subject, n=60, seed=0):
    rng = np.random.default_rng(seed)
    keys = dict(subject=subject, session="001", task="GradCPT", run="001", stem=stem)
    eeg = pd.DataFrame({**keys, "tr": np.arange(n + 1), "cwl_rms": 1.0 + 0.01 * rng.random(n + 1),
                        "variance_O1": rng.random(n + 1) + 0.5,
                        "spec_rel_bandpow_alpha_O1": rng.random(n + 1)})
    roi = pd.DataFrame({**keys, "tr": np.arange(n), "thal": rng.normal(size=n),
                        "v1": rng.normal(size=n), "fd": np.zeros(n), "global_raw": 0.0})
    return eeg, roi


def test_align_run_and_labels():
    feats = ["variance_O1", "spec_rel_bandpow_alpha_O1"]
    parts = []
    for i, sub in enumerate(["001", "002"]):
        eeg, roi = _toy_run(f"s{sub}", sub, seed=i)
        out = align.align_run(eeg, roi, feats)
        n_use = n_usable_trs(len(eeg), len(roi))
        assert len(out) == n_use - config.HRF_WARMUP_TRS
        assert out.tr.min() == config.HRF_WARMUP_TRS and out.tr.max() == n_use - 1
        assert not out[feats].isna().any().any()
        parts.append(out)
    df = pd.concat(parts, ignore_index=True)
    z = align.make_labels(df, "thal", "float")
    assert np.allclose(z.groupby(df.subject).mean(), 0, atol=1e-9)
    b = align.make_labels(df, "thal", "binary")
    rate = b.groupby(df.subject).mean()
    assert ((rate > 0.2) & (rate < 0.3)).all()  # ~25% "active" per subject
    with pytest.raises(ValueError):
        align.make_labels(df, "thal", "ternary")
