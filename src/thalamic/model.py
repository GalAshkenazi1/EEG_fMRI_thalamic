"""Leave-one-subject-out models, null distributions, controls and ablations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .align import label_column

log = logging.getLogger(__name__)

CHANCE = {"binary": 0.5, "float": 0.0}
METRIC = {"binary": "AUC", "float": "Pearson r"}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_model(kind: str, mode: str):
    """``kind`` in {"linear", "hgb"}; ``mode`` in {"float", "binary"}."""
    if kind == "linear" and mode == "float":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    if kind == "linear" and mode == "binary":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=2000))
    hgb = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
               l2_regularization=1.0, random_state=config.RANDOM_STATE)
    if kind == "hgb" and mode == "float":
        return HistGradientBoostingRegressor(**hgb)
    if kind == "hgb" and mode == "binary":
        return HistGradientBoostingClassifier(class_weight="balanced", **hgb)
    raise ValueError((kind, mode))


def _fit_predict(model, X, y, train, test, mode):
    m = clone(model).fit(X[train], y[train])
    pred = m.predict_proba(X[test])[:, 1] if mode == "binary" else m.predict(X[test])
    return test, pred, m


def loso_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model, mode: str,
                 n_jobs: int = 1, return_models: bool = False):
    """Out-of-fold predictions with leave-one-subject-out CV.

    The scaler lives inside the pipeline, so it is fit on training subjects
    only. Both sessions of a subject are always in the same fold.
    """
    folds = list(LeaveOneGroupOut().split(X, y, groups))
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(model, X, y, tr, te, mode)
                                  for tr, te in folds)
    oof = np.empty(len(y), dtype=float)
    for test, pred, _ in res:
        oof[test] = pred
    return (oof, [m for *_, m in res]) if return_models else oof


def score(y: np.ndarray, pred: np.ndarray, mode: str) -> float:
    if mode == "binary":
        return roc_auc_score(y, pred) if len(np.unique(y)) == 2 else np.nan
    if np.std(pred) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(y, pred)[0, 1])


def per_subject_scores(y, pred, groups, mode) -> pd.DataFrame:
    rows = [(g, score(y[groups == g], pred[groups == g], mode), int((groups == g).sum()))
            for g in np.unique(groups)]
    return pd.DataFrame(rows, columns=["subject", "score", "n"])


def summarize(per_sub: pd.DataFrame, mode: str) -> dict:
    s = per_sub["score"].dropna().to_numpy()
    chance = CHANCE[mode]
    try:
        p = float(stats.wilcoxon(s - chance, alternative="greater").pvalue) if len(s) > 1 else np.nan
    except ValueError:
        p = np.nan
    return {"metric": METRIC[mode], "chance": chance, "mean": float(np.mean(s)) if len(s) else np.nan,
            "sem": float(stats.sem(s)) if len(s) > 1 else np.nan, "n_subjects": int(len(s)),
            "n_above_chance": int((s > chance).sum()), "wilcoxon_p": p}


# --------------------------------------------------------------------------
# Null distribution
# --------------------------------------------------------------------------
def circular_shift(y: np.ndarray, run_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Circularly shift the target within each run (keeps its autocorrelation).

    Shifts are drawn away from zero (between 10% and 90% of the run length) so
    that the shifted target is decorrelated from the EEG at all HRF lags.
    """
    out = y.copy()
    for r in np.unique(run_ids):
        idx = np.flatnonzero(run_ids == r)
        n = len(idx)
        lo = max(1, n // 10)
        shift = rng.integers(lo, max(lo + 1, n - lo))
        out[idx] = np.roll(y[idx], shift)
    return out


def _null_one(X, y, groups, run_ids, model, mode, seed):
    rng = np.random.default_rng(seed)
    y_perm = circular_shift(y, run_ids, rng)
    oof = loso_predict(X, y_perm, groups, model, mode, n_jobs=1)
    return float(per_subject_scores(y_perm, oof, groups, mode)["score"].mean())


def null_distribution(X, y, groups, run_ids, model, mode, n_perm: int,
                      n_jobs: int = 1, seed: int = config.RANDOM_STATE) -> np.ndarray:
    seeds = np.random.SeedSequence(seed).generate_state(n_perm)
    vals = Parallel(n_jobs=n_jobs)(delayed(_null_one)(X, y, groups, run_ids, model, mode, int(s))
                                   for s in seeds)
    return np.asarray(vals)


# --------------------------------------------------------------------------
# Analysis driver
# --------------------------------------------------------------------------
@dataclass
class Analysis:
    name: str
    roi: str
    kind: str = "linear"
    drop_feature: str | None = None  # substring of columns to drop (ablation)


ANALYSES = [
    Analysis("thalamus (linear)", "thal"),
    Analysis("thalamus (HGB)", "thal", kind="hgb"),
    Analysis("V1 control", "v1"),
    Analysis("thalamus minus V1", "thal_nov1"),
    Analysis("thalamus, no alpha", "thal", drop_feature="_alpha_"),
]
MAIN = ANALYSES[0].name


def run_mode(table: pd.DataFrame, feature_cols: list[str], mode: str, n_null: int,
             n_jobs: int = 1) -> dict:
    """All analyses for one label mode. Returns summaries + per-subject + OOF + null."""
    groups = table["subject"].to_numpy()
    run_ids = table["stem"].to_numpy()
    out = {"summary": {}, "per_subject": [], "oof": None, "null": None, "coefs": None}

    for a in ANALYSES:
        cols = [c for c in feature_cols if not (a.drop_feature and a.drop_feature in c)]
        X = table[cols].to_numpy(dtype=float)
        y = table[label_column(a.roi, mode)].to_numpy(dtype=float)
        model = make_model(a.kind, mode)
        log.info("[%s] %s: %d features", mode, a.name, len(cols))
        if a.name == MAIN:
            oof, models = loso_predict(X, y, groups, model, mode, n_jobs, return_models=True)
            out["oof"] = table[["subject", "session", "task", "run", "stem", "tr"]].assign(
                y=y, pred=oof)
            out["coefs"] = _coef_table(models, cols, mode)
        else:
            oof = loso_predict(X, y, groups, model, mode, n_jobs)
        ps = per_subject_scores(y, oof, groups, mode).assign(analysis=a.name, mode=mode)
        out["per_subject"].append(ps)
        out["summary"][a.name] = summarize(ps, mode)
        log.info("[%s] %s: mean %s = %.3f (p=%.3g)", mode, a.name, METRIC[mode],
                 out["summary"][a.name]["mean"], out["summary"][a.name]["wilcoxon_p"])

        if a.name == MAIN:  # per-task breakdown of the main model's OOF predictions
            for task in sorted(table["task"].unique()):
                m = (table["task"] == task).to_numpy()
                pst = per_subject_scores(y[m], oof[m], groups[m], mode)
                name = f"thalamus (linear), {task} only"
                out["per_subject"].append(pst.assign(analysis=name, mode=mode))
                out["summary"][name] = summarize(pst, mode)

    if n_null > 0:
        X = table[feature_cols].to_numpy(dtype=float)
        y = table[label_column("thal", mode)].to_numpy(dtype=float)
        log.info("[%s] null distribution: %d circular shifts", mode, n_null)
        null = null_distribution(X, y, groups, run_ids, make_model("linear", mode), mode,
                                 n_null, n_jobs)
        real = out["summary"][MAIN]["mean"]
        out["null"] = null
        out["summary"][MAIN]["null_mean"] = float(null.mean())
        out["summary"][MAIN]["null_95"] = float(np.percentile(null, 95))
        out["summary"][MAIN]["perm_p"] = float((1 + (null >= real).sum()) / (1 + len(null)))
    out["per_subject"] = pd.concat(out["per_subject"], ignore_index=True)
    return out


def _coef_table(models, cols, mode) -> pd.DataFrame:
    """Fold-averaged standardized coefficients of the linear model."""
    coefs = []
    for m in models:
        est = m[-1]
        c = est.coef_[0] if mode == "binary" else est.coef_
        coefs.append(np.ravel(c))
    coefs = np.asarray(coefs)
    feat, ch = zip(*(c.rsplit("_", 1) for c in cols))
    return pd.DataFrame({"column": cols, "feature": feat, "channel": ch,
                         "coef": coefs.mean(0), "coef_sd": coefs.std(0)})


def save_results(results: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {mode: r["summary"] for mode, r in results.items()}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    pd.concat([r["per_subject"] for r in results.values()]).to_csv(
        out_dir / "per_subject_scores.csv", index=False)
    for mode, r in results.items():
        r["oof"].to_parquet(out_dir / f"oof_{mode}.parquet", index=False)
        r["coefs"].to_csv(out_dir / f"coefs_{mode}.csv", index=False)
        if r["null"] is not None:
            np.save(out_dir / f"null_{mode}.npy", r["null"])


def load_results(out_dir: Path) -> dict[str, dict]:
    summary = json.loads((out_dir / "results.json").read_text())
    per_sub = pd.read_csv(out_dir / "per_subject_scores.csv", dtype={"subject": str})
    res = {}
    for mode, s in summary.items():
        nf = out_dir / f"null_{mode}.npy"
        res[mode] = {"summary": s, "per_subject": per_sub[per_sub["mode"] == mode],
                     "oof": pd.read_parquet(out_dir / f"oof_{mode}.parquet"),
                     "coefs": pd.read_csv(out_dir / f"coefs_{mode}.csv"),
                     "null": np.load(nf) if nf.exists() else None}
    return res
