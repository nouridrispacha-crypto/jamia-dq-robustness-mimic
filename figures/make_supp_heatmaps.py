# -*- coding: utf-8 -*-
"""Figure 6 and Supplementary Figures 9-10: change in each co-primary metric,
per model and DQ dimension, at c = 0.3 (N = 3 500, means over 500 replicates).

Figure 6 is Scenario S2 (main text), Supplementary Figure 9 is S1, 10 is S3.

Each cell carries two encodings:
  number  the absolute change of the metric (c = 0.3 minus clean data), with
          its real sign, in the units of the manuscript's Delta values;
  colour  the relative loss, in % of the clean-data value, on one logarithmic
          scale shared by every panel and all three figures.
For R^2 the relative loss is the share of explained variance lost, so 100% or
more means worse than predicting the mean. Grey marks an improvement. No cell is
framed: the tested contrasts are reported in Table 3, not on the heatmap.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")
PAPER = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
IMG = PAPER / "img"
sys.path.insert(0, str(PAPER))
import analyse_stats_v2 as V2  # noqa: E402

MINUS = "\u2212"
DIMS = ["completeness", "coherence", "validity", "precision", "uniqueness"]
DLAB = [d.capitalize() for d in DIMS]
LAB = {"LASSO": "LASSO", "DecisionTree": "Decision tree", "RandomForest": "Random forest",
       "XGBoost": "XGBoost", "MLP1": "MLP", "CoxPH": "CoxPH", "RandomSurvivalForest": "RSF",
       "XGBoostSurv": "XGBoostSurv", "DeepSurv1": "DeepSurv"}
# (row, column, title, config, metric, +1 if a rise is a degradation, decimals)
PANELS = [
    (0, 0, "In-hospital mortality: AUROC\n(lower is worse)", V2.BINARY, "auc", -1, 3),
    (0, 1, "In-hospital mortality: Brier score\n(higher is worse)", V2.BINARY, "brier", +1, 3),
    (1, 0, "ICU length of stay: RMSE\n(higher is worse)", V2.CONTINUOUS, "rmse", +1, 3),
    (1, 1, "ICU length of stay: R$^2$\n(lower is worse)", V2.CONTINUOUS, "r2", -1, 3),
    (2, 0, "Time to death: C-index\n(lower is worse)", V2.SURVIVAL, "c_index", -1, 3),
    (2, 1, "Time to death: IBS\n(higher is worse)", V2.SURVIVAL, "ibs", +1, 3),
]
# rows grouped by model family: tree-based first, then the other models
GROUPS = {
    "binary": [("Tree-\nbased", ["DecisionTree", "RandomForest", "XGBoost"]), ("Linear and\nneural", ["LASSO", "MLP1"])],
    "continuous": [("Tree-\nbased", ["DecisionTree", "RandomForest", "XGBoost"]), ("Linear and\nneural", ["LASSO", "MLP1"])],
    "survival": [("Tree-\nbased", ["RandomSurvivalForest", "XGBoostSurv"]), ("Linear and\nneural", ["CoxPH", "DeepSurv1"])],
}
FIGS = [("S2", "figure6_heatmap_s2"), ("S1", "suppfig9_heatmap_s1"), ("S3", "suppfig10_heatmap_s3")]
NORM = LogNorm(vmin=1, vmax=300)
CMAP = plt.get_cmap("YlOrRd")
IMPROVE = "#dfe6ee"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})

_cache = {}


def spread(cfg, dim, scen, metric, level, model):
    """2.5th-97.5th percentile of the CHANGE from clean data, paired by replicate."""
    d = _cache[cfg.file_map[dim][scen]]
    d = d[d["model"] == model]
    key = "bootstrap" if "bootstrap" in d.columns else d.columns[0]
    a = d[d["p"] == 0.0].set_index(key)[metric]
    b = d[d["p"] == level].set_index(key)[metric]
    delta = (b - a).dropna()
    return (delta.quantile(0.025), delta.quantile(0.975))


def means(cfg, dim, scen, metric, level):
    f = cfg.file_map[dim][scen]
    if f not in _cache:
        d = pd.read_csv(f, on_bad_lines="skip").rename(columns={"pollution": "p"})
        d["p"] = d["p"].round(2)
        _cache[f] = d[(d["N"] == cfg.n_fixed) & d["model"].isin(cfg.models)]
    d = _cache[f]
    return d[d["p"] == level].groupby("model")[metric].mean()


def fmt(v, decimals):
    s = f"{v:+.1f}" if abs(v) >= 10 else f"{v:+.{decimals}f}"
    return s.replace("-", MINUS)


def panel(ax, cfg, metric, sign, decimals, scen, ylabels):
    groups = GROUPS[cfg.name]
    models = [m for _, ms in groups for m in ms]
    assert sorted(models) == sorted(cfg.models)
    n_m = len(models)
    change = np.zeros((n_m, len(DIMS)))
    loss_pct = np.zeros((n_m, len(DIMS)))
    for j, dim in enumerate(DIMS):
        c0 = means(cfg, dim, scen, metric, 0.0)
        c3 = means(cfg, dim, scen, metric, 0.3)
        change[:, j] = [(c3 - c0)[m] for m in models]
        loss_pct[:, j] = [sign * (c3 - c0)[m] / abs(c0[m]) * 100 for m in models]
    dim_of = {j: d for j, d in enumerate(DIMS)}
    ax.imshow(np.clip(loss_pct, NORM.vmin, NORM.vmax), cmap=CMAP, norm=NORM, aspect="auto")
    for i in range(n_m):
        for j in range(len(DIMS)):
            p = loss_pct[i, j]
            if p <= -1:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color=IMPROVE, zorder=2))
            col = "white" if p >= 40 else "#1f1f1f"
            ax.text(j, i, fmt(change[i, j], decimals), ha="center", va="center",
                    fontsize=8.0, zorder=3, color=col)
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels(DLAB, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([LAB[m] for m in models] if ylabels else [])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(DIMS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_m, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    # separator between model families, and family brackets on the left column
    start = 0
    for k, (gname, ms) in enumerate(groups):
        end = start + len(ms)
        if k > 0:
            ax.axhline(start - 0.5, color="#111111", lw=1.6, zorder=5)
        if ylabels:
            ax.annotate("", xy=(-0.47, 1 - (start + 0.1) / n_m), xytext=(-0.47, 1 - (end - 0.1) / n_m),
                        xycoords="axes fraction", arrowprops=dict(arrowstyle="-", lw=1.0, color="#555555"),
                        annotation_clip=False)
            ax.text(-0.50, 1 - (start + end) / 2 / n_m, gname, transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=7.6, color="#555555", linespacing=1.0)
        start = end
    return change, loss_pct


if __name__ == "__main__":
    for scen, name in FIGS:
        fig, axes = plt.subplots(3, 2, figsize=(7.4, 9.0),
                                 gridspec_kw={"hspace": 0.58, "wspace": 0.06,
                                              "height_ratios": [5, 5, 4]})
        for r, c, title, cfg, metric, sign, decimals in PANELS:
            panel(axes[r, c], cfg, metric, sign, decimals, scen, ylabels=(c == 0))
            axes[r, c].set_title(title, fontweight="bold", pad=6, fontsize=9.5)
        fig.subplots_adjust(left=0.23, right=0.98, top=0.95, bottom=0.14)
        cax = fig.add_axes([0.22, 0.045, 0.66, 0.012])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
        cb.set_ticks([1, 2, 5, 10, 20, 50, 100, 300])
        cb.set_ticklabels(["\u22641%", "2%", "5%", "10%", "20%", "50%", "100%", "\u2265300%"])
        cb.outline.set_visible(False)
        cb.set_label("Colour: loss as a percentage of the clean-data value\n"
                     "(for R$^2$, 100% or more = no better than predicting the mean)", fontsize=8.5)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
            fig.savefig(IMG / f"{name}.{ext}", bbox_inches="tight", **kw)
        plt.close(fig)
        print("wrote", name)
