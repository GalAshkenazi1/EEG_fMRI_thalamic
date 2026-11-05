"""Result figures (static matplotlib PNGs for the README / notebook)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config
from .features import short_feature_name
from .model import CHANCE, MAIN, METRIC

# Reference palette (categorical slots in fixed order) + recessive ink/grid tokens.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_orange", [BLUE, "#f0efec", ORANGE])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "font.size": 9, "lines.linewidth": 2,
})


def _save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline(out) -> None:
    """Simple box-and-arrow schematic of the pipeline."""
    steps = [
        ("Cleaned EEG\n(Bergen + CWL, 250 Hz)", "eegdash query\n+ derivatives"),
        ("TR-locked 2 s windows\n31 ch, 1-45 Hz, avg ref", "braindecode"),
        ("eegdash features\n(rel. band power, entropy,\nline length, variance)", "FeatureExtractor"),
        ("HRF convolution\n+ within-run z-score", "align.py"),
        ("LOSO models\nridge / logistic / HGB", "model.py"),
    ]
    fmri = [("Raw BOLD", ""), ("Moco + T1/MNI reg.\n(ANTsPy)", ""),
            ("Thalamus / V1 masks\n+ confound clean", ""), ("Labels: float z\nor binary >P75", "")]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_axis_off()
    ax.grid(False)

    def box(x, y, text, colour):
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=colour, lw=1.5))

    for i, (t, _) in enumerate(steps):
        box(i * 2.4, 1.0, t, BLUE)
        if i:
            ax.annotate("", xy=(i * 2.4 - 0.95, 1.0), xytext=(i * 2.4 - 1.45, 1.0),
                        arrowprops=dict(arrowstyle="->", color=MUTED))
    for i, (t, _) in enumerate(fmri):
        box(i * 2.4, -0.4, t, ORANGE)
        if i:
            ax.annotate("", xy=(i * 2.4 - 0.95, -0.4), xytext=(i * 2.4 - 1.45, -0.4),
                        arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("", xy=(4 * 2.4, 0.55), xytext=(3 * 2.4 + 0.9, -0.2),
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(-1.2, 1.0, "EEG", color=BLUE, fontweight="bold", ha="right", va="center")
    ax.text(-1.2, -0.4, "fMRI", color=ORANGE, fontweight="bold", ha="right", va="center")
    ax.set_xlim(-1.6, 10.8)
    ax.set_ylim(-1.1, 1.7)
    _save(fig, out)


def plot_pred_vs_actual(oof: pd.DataFrame, per_subject: pd.DataFrame, mode: str, out,
                        subject: str | None = None) -> None:
    """Held-out prediction vs actual thalamic signal for one run of one subject."""
    ps = per_subject[per_subject.analysis == MAIN].dropna(subset=["score"])
    if subject is None:
        subject = ps.sort_values("score").iloc[len(ps) // 2]["subject"] if len(ps) else oof.subject.iloc[0]
    d = oof[oof.subject == subject]
    stem = d.groupby("stem").size().idxmax()
    d = d[d.stem == stem].sort_values("tr")
    t = d.tr.to_numpy() * config.TR
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    if mode == "binary":
        ax.fill_between(t, 0, 1, where=d.y.to_numpy() > 0.5, color=ORANGE, alpha=0.18, lw=0,
                        transform=ax.get_xaxis_transform(), label="actual: active (>P75)")
        ax.plot(t, z(d.pred), color=BLUE, lw=1.5, label="predicted P(active), z")
    else:
        ax.plot(t, d.y, color=ORANGE, lw=1.5, label="actual thalamic BOLD (z)")
        ax.plot(t, z(d.pred) * d.y.std(), color=BLUE, lw=1.5, label="EEG prediction (scaled)")
    sc = ps.set_index("subject").score.get(subject, np.nan)
    ax.set(xlabel="time (s)", ylabel="z",
           title=f"Held-out subject sub-{subject} ({stem.split('_', 2)[2]}): "
                 f"{METRIC[mode]} = {sc:.2f} (subject-level, all runs)")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    _save(fig, out)


def plot_subject_scores(per_subject: pd.DataFrame, summary: dict, mode: str, out) -> None:
    """Per-subject held-out score (dots) for the main model, with chance and null band."""
    ps = per_subject[per_subject.analysis == MAIN].sort_values("score")
    s = summary[MAIN]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(ps))
    if "null_95" in s:
        ax.axhspan(CHANCE[mode], s["null_95"], color=MUTED, alpha=0.15, lw=0,
                   label="null (circular shift), chance to 95th pct.")
    ax.axhline(CHANCE[mode], color=INK2, lw=1, ls="--", label="chance")
    ax.scatter(x, ps.score, s=48, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3,
               label="held-out subject")
    ax.axhline(s["mean"], color=BLUE, lw=1.5, alpha=0.7)
    ax.text(len(ps) - 0.5, s["mean"], f" mean {s['mean']:.3f}", color=INK2, va="bottom", ha="right")
    ax.set_xticks(x, ["sub-" + v for v in ps.subject], rotation=90)
    ax.set(ylabel=METRIC[mode],
           title=f"Thalamus ({mode} labels), LOSO: {s['n_above_chance']}/{s['n_subjects']} "
                 f"subjects above chance, Wilcoxon p = {s['wilcoxon_p']:.3g}")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    _save(fig, out)


def plot_null(null: np.ndarray, summary: dict, mode: str, out) -> None:
    s = summary[MAIN]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(null, bins=30, color=MUTED, alpha=0.6, edgecolor="white", linewidth=1)
    ax.axvline(s["mean"], color=BLUE, lw=2)
    ax.text(s["mean"], ax.get_ylim()[1] * 0.92, f"  observed {s['mean']:.3f}\n  p = {s['perm_p']:.3g}",
            color=INK, va="top")
    ax.set(xlabel=f"mean held-out {METRIC[mode]}", ylabel="count",
           title=f"Null: {len(null)} within-run circular shifts of the target")
    _save(fig, out)


def plot_controls(summary: dict, mode: str, out) -> None:
    """Main model vs control region and ablations (mean ± SEM across subjects)."""
    names = [n for n in summary if "only" not in n]
    names += [n for n in summary if "only" in n]
    means = [summary[n]["mean"] for n in names]
    sems = [summary[n]["sem"] for n in names]
    colours = [BLUE if n.startswith("thalamus") else ORANGE if n.startswith("V1") else AQUA
               for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, np.array(means) - CHANCE[mode], left=CHANCE[mode], height=0.6, color=colours,
            xerr=sems, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    ax.axvline(CHANCE[mode], color=INK2, lw=1, ls="--")
    for yi, m, n in zip(y, means, names):
        p = summary[n]["wilcoxon_p"]
        # value column just right of the plot area, so labels never float mid-axis
        ax.text(1.01, yi, f"{m:.3f}  (p={p:.2g})", va="center", color=INK2, fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=f"mean held-out {METRIC[mode]} (± SEM)",
           title=f"Controls and ablations ({mode} labels)")
    _save(fig, out)


def plot_coef_topomaps(coefs: pd.DataFrame, mode: str, out) -> None:
    """One scalp map per feature type of the fold-averaged linear coefficients."""
    import mne

    info = mne.create_info(config.EEG_CHANNELS, config.EEG_SFREQ, "eeg")
    info.set_montage("standard_1020")
    feats = list(dict.fromkeys(coefs.feature))
    vmax = np.abs(coefs.coef).max()
    ncol = min(4, len(feats))
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow + 0.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        c = coefs[coefs.feature == f].set_index("channel").reindex(config.EEG_CHANNELS).coef
        im, _ = mne.viz.plot_topomap(c.to_numpy(), info, axes=ax, show=False, cmap=DIVERGING,
                                     vlim=(-vmax, vmax), contours=0, sensors=True)
        ax.set_title(short_feature_name(f), fontsize=9)
    for ax in axes[len(feats):]:
        ax.set_axis_off()
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="standardized coefficient")
    fig.suptitle(f"Linear model weights (fold average, {mode} labels): "
                 "orange = higher thalamic activity", fontsize=10)
    _save(fig, out)


def make_all(results: dict[str, dict], fig_dir: Path) -> list[Path]:
    fig_dir = Path(fig_dir)
    made = [fig_dir / "pipeline.png"]
    plot_pipeline(made[0])
    for mode, r in results.items():
        s, ps = r["summary"], r["per_subject"]
        jobs = [
            (f"pred_vs_actual_{mode}.png", lambda o: plot_pred_vs_actual(r["oof"], ps, mode, o)),
            (f"subject_scores_{mode}.png", lambda o: plot_subject_scores(ps, s, mode, o)),
            (f"controls_{mode}.png", lambda o: plot_controls(s, mode, o)),
            (f"topomaps_{mode}.png", lambda o: plot_coef_topomaps(r["coefs"], mode, o)),
        ]
        if r.get("null") is not None:
            jobs.append((f"null_{mode}.png", lambda o: plot_null(r["null"], s, mode, o)))
        for name, fn in jobs:
            fn(fig_dir / name)
            made.append(fig_dir / name)
    return made
