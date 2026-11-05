import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from thalamic import align, features, model


def _synthetic(n_sub=5, n_tr=200, signal=1.0, seed=0):
    """Feature = HRF-convolved latent drive (+ noise); target BOLD = same drive (+ noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sub):
        drive = np.convolve(rng.normal(size=n_tr), np.ones(5) / 5, mode="same")
        bold = align.hrf_convolve(drive)[:, 0]
        X = rng.normal(size=(n_tr, 6))
        X[:, 0] += signal * drive / drive.std()  # EEG feature tracks the neural drive
        Xc = align.hrf_convolve(X)
        df = pd.DataFrame(Xc, columns=[f"spec_rel_bandpow_alpha_{c}" if j == 0 else f"variance_{c}"
                                       for j, c in enumerate(["O1", "O2", "Pz", "Cz", "Fz", "F3"])])
        df["subject"] = f"{s:03d}"
        df["stem"] = f"run{s}"
        df["thal"] = bold + 0.3 * bold.std() * rng.normal(size=n_tr)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["thal_z"] = align.make_labels(df, "thal", "float")
    df["thal_active"] = align.make_labels(df, "thal", "binary")
    return df


def test_loso_folds_never_share_subjects():
    groups = np.repeat(["a", "b", "c"], 10)
    for tr, te in LeaveOneGroupOut().split(np.zeros(30), groups=groups):
        assert set(groups[tr]).isdisjoint(groups[te])


@pytest.mark.parametrize("mode", ["binary", "float"])
def test_planted_signal_is_recovered(mode):
    df = _synthetic(signal=1.0)
    cols = features.feature_columns(df)
    assert len(cols) == 6
    X, y = df[cols].to_numpy(), df[align.label_column("thal", mode)].to_numpy(float)
    g = df.subject.to_numpy()
    oof = model.loso_predict(X, y, g, model.make_model("linear", mode), mode)
    s = model.summarize(model.per_subject_scores(y, oof, g, mode), mode)
    assert s["mean"] > (0.7 if mode == "binary" else 0.4)


def test_null_is_centered_at_chance():
    df = _synthetic(signal=0.0, seed=1)
    cols = features.feature_columns(df)
    X, y = df[cols].to_numpy(), df["thal_active"].to_numpy(float)
    null = model.null_distribution(X, y, df.subject.to_numpy(), df.stem.to_numpy(),
                                   model.make_model("linear", "binary"), "binary", n_perm=20)
    assert abs(null.mean() - 0.5) < 0.05


def test_circular_shift_preserves_values_within_run():
    y = np.arange(20.0)
    runs = np.repeat(["r1", "r2"], 10)
    ys = model.circular_shift(y, runs, np.random.default_rng(0))
    for r in ("r1", "r2"):
        assert sorted(ys[runs == r]) == sorted(y[runs == r])
        assert not np.array_equal(ys[runs == r], y[runs == r])
