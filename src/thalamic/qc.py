"""Visual quality control: mask placement, registration and ROI signals.

Every fMRI run gets PNGs under ``outputs/qc/sub-XXX/ses-YYY/`` plus a single
``outputs/qc/index.html`` page so all runs can be eyeballed in one scroll.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import image, plotting

from . import config

# label value -> (name, colour). Same encoding as fmri.LABELS.
MASK_STYLE = {
    1: ("thalamus L", "#e41a1c"),
    2: ("thalamus R", "#ff7f00"),
    3: ("V1 (intracalcarine)", "#4daf4a"),
    4: ("white matter", "#377eb8"),
    5: ("CSF (lat. ventricles)", "#984ea3"),
}


def _mask_world_extent(label_img, values=(1, 2)):
    data = np.asarray(label_img.dataobj)
    ijk = np.argwhere(np.isin(data, values))
    if ijk.size == 0:
        return None, None
    xyz = image.coord_transform(ijk[:, 0], ijk[:, 1], ijk[:, 2], label_img.affine)
    xyz = np.column_stack(xyz)
    return xyz.mean(0), (xyz[:, 2].min(), xyz[:, 2].max())


def plot_mask_overlay(mean_epi: str | Path, labels: str | Path, out_png: str | Path,
                      title: str) -> None:
    """Mean EPI with all ROI/confound masks as filled contours.

    Top: orthogonal views centred on the thalamus. Bottom: 8 axial slices
    spanning the thalamus. Lets you check by eye that each mask sits where it
    should (thalamus medial to the internal capsule, V1 along the calcarine
    sulcus, WM/CSF inside white matter/ventricles).
    """
    bg = image.load_img(str(mean_epi))
    lab = image.load_img(str(labels))
    centre, (zmin, zmax) = _mask_world_extent(lab)
    fig = plt.figure(figsize=(13, 7.5))
    ax_top = fig.add_axes([0.0, 0.42, 1.0, 0.52])
    ax_bot = fig.add_axes([0.0, 0.0, 1.0, 0.40])
    cut = tuple(centre) if centre is not None else None
    d1 = plotting.plot_anat(bg, display_mode="ortho", cut_coords=cut, axes=ax_top,
                            title=title, dim=-0.3, draw_cross=False, annotate=True)
    zs = list(np.linspace(zmin - 4, zmax + 4, 8)) if centre is not None else 8
    d2 = plotting.plot_anat(bg, display_mode="z", cut_coords=zs, axes=ax_bot, dim=-0.3,
                            annotate=False)
    for val, (_, colour) in MASK_STYLE.items():
        m = image.math_img(f"(img == {val}).astype('int8')", img=lab)
        if np.asarray(m.dataobj).sum() == 0:
            continue
        for d in (d1, d2):
            d.add_contours(m, levels=[0.5], colors=[colour], filled=True, transparency=0.45)
            d.add_contours(m, levels=[0.5], colors=[colour], linewidths=0.8)
    handles = [plt.Line2D([0], [0], color=c, lw=6, alpha=0.6) for _, c in MASK_STYLE.values()]
    fig.legend(handles, [n for n, _ in MASK_STYLE.values()], loc="lower center",
               ncol=5, frameon=False, fontsize=9, labelcolor="white")  # black figure bg
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100, facecolor="black")
    plt.close(fig)


def plot_edges(background: str | Path, edges: str | Path, out_png: str | Path,
               title: str, cut_coords=None) -> None:
    """Background image with the edges of a second image overlaid (registration check)."""
    d = plotting.plot_anat(str(background), display_mode="ortho", title=title,
                           cut_coords=cut_coords, dim=-0.2, draw_cross=False)
    d.add_edges(str(edges), color="r")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    d.savefig(str(out_png), dpi=100)
    d.close()


def plot_roi_timeseries(df: pd.DataFrame, out_png: str | Path, title: str) -> None:
    """Cleaned thalamus / V1 time series, raw global signal and FD."""
    t = df["tr"].to_numpy() * config.TR
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for col, c in (("thal", "#e41a1c"), ("v1", "#4daf4a")):
        z = (df[col] - df[col].mean()) / (df[col].std() + 1e-12)
        axes[0].plot(t, z, color=c, lw=1, label=col)
    axes[0].set_ylabel("cleaned BOLD (z)")
    axes[0].legend(loc="upper right", fontsize=8)
    g = df["global_raw"]
    axes[1].plot(t, (g - g.mean()) / (g.std() + 1e-12), color="0.3", lw=1)
    axes[1].set_ylabel("global (raw, z)")
    axes[2].plot(t, df["fd"], color="k", lw=1)
    axes[2].axhline(config.FD_THRESH, color="tab:red", ls="--", lw=0.8)
    axes[2].set(ylabel="FD (mm)", xlabel="time (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def write_index(qc_dir: str | Path, summary_dir: str | Path) -> Path:
    """Build ``qc/index.html`` listing every QC image, flagging problem runs in red."""
    qc_dir, summary_dir = Path(qc_dir), Path(summary_dir)
    rows = [json.loads(p.read_text()) for p in sorted(summary_dir.glob("*.json"))]
    parts = [
        "<html><head><meta charset='utf-8'><title>ds007216 QC</title><style>",
        "body{font-family:sans-serif;background:#111;color:#ddd;margin:20px}",
        "img{max-width:49%;margin:2px;border:1px solid #333}",
        ".bad{color:#ff5555;font-weight:bold}.ok{color:#7fd17f}",
        "table{border-collapse:collapse}td,th{padding:3px 8px;border-bottom:1px solid #333}",
        "</style></head><body><h1>ds007216 fMRI QC</h1>",
        f"<p>Flags: thalamus coverage &lt; {config.COVERAGE_MIN} or mean FD &gt; "
        f"{config.QC_FD_MEAN_MAX} mm.</p>",
    ]
    if rows:
        df = pd.DataFrame(rows)
        parts.append("<table><tr><th>run</th><th>vols</th><th>thal vox (exp.)</th>"
                     "<th>coverage</th><th>mean FD</th><th>status</th></tr>")
        for r in df.itertuples():
            bad = r.coverage < config.COVERAGE_MIN or r.fd_mean > config.QC_FD_MEAN_MAX
            parts.append(
                f"<tr><td><a href='#{r.stem}'>{html.escape(r.stem)}</a></td><td>{r.n_vols}</td>"
                f"<td>{r.n_thal_vox} ({r.n_thal_expected:.0f})</td><td>{r.coverage:.2f}</td>"
                f"<td>{r.fd_mean:.3f}</td><td class='{'bad' if bad else 'ok'}'>"
                f"{'CHECK' if bad else 'ok'}</td></tr>")
        parts.append("</table>")
    # per-subject anatomy images first, then each run
    for sub_dir in sorted(qc_dir.glob("sub-*")):
        parts.append(f"<h2>{sub_dir.name}</h2>")
        for png in sorted(sub_dir.glob("*.png")):
            parts.append(f"<img src='{png.relative_to(qc_dir).as_posix()}' title='{png.name}'>")
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            stems = sorted({p.name.split("_desc-")[0] for p in ses_dir.glob("*_desc-*.png")})
            for stem in stems:
                parts.append(f"<h3 id='{stem}'>{stem}</h3>")
                for png in sorted(ses_dir.glob(f"{stem}_desc-*.png")):
                    parts.append(f"<img src='{png.relative_to(qc_dir).as_posix()}' "
                                 f"title='{png.name}'>")
    parts.append("</body></html>")
    out = qc_dir / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
