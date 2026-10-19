# -*- coding: utf-8 -*-
"""Supplementary Figure 10: the ten dimension contrasts of each model (mortality, AUROC,
test-time degradation at c = 0.3), as median paired differences with their 2.5th-97.5th
percentile range; and contrasts_all.csv, every contrast of the design.

A contrast is reliably separable when its Holm-adjusted p < 0.05 and its interval excludes
zero; with 500 replicates the p-value alone separates almost every pair, so the figure shows
the interval, which is what the rule of Table 3 uses.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PAPER = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
sys.path.insert(0, str(PAPER))
import make_s2_grid as G  # noqa: E402
V2 = G.V2

LAB = {"LASSO": "LASSO", "DecisionTree": "Decision tree", "RandomForest": "Random forest",
       "XGBoost": "XGBoost", "MLP1": "MLP", "CoxPH": "CoxPH", "RandomSurvivalForest": "RSF",
       "XGBoostSurv": "XGBoostSurv", "DeepSurv1": "DeepSurv"}
SHORT = {"completeness": "Completeness", "coherence": "Coherence", "validity": "Validity",
         "precision": "Precision", "uniqueness": "Uniqueness"}
TASK = {"binary": "In-hospital mortality", "continuous": "ICU length of stay", "survival": "Time to death"}
SCEN = {"S2": "test-time (S2)", "S1": "training-time (S1)", "S3": "joint (S3)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})


def blended(ax):
    """x in axes coordinates, y in data coordinates, to pin the share at the right edge."""
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transAxes, ax.transData)


def table(cfg, scenario, metric):
    loss = V2.build_loss_table(cfg, scenario, metric, dict(cfg.metrics.values())[metric])
    d = V2.pairwise_contrasts(cfg, loss).copy()
    d["separable"] = (d["p_holm"] < 0.05) & d["ci_excludes_zero"]
    counts = [_agreeing(loss, r["model"], r["dim_A"], r["dim_B"], r["median_diff"])
              for _, r in d.iterrows()]
    d["n_agree"] = [c[0] for c in counts]
    d["n_rep"] = [c[1] for c in counts]
    d["share"] = d["n_agree"] / d["n_rep"]
    return d


def _agreeing(loss, model, a, b, median_diff):
    """How many replicates give the difference the sign of its median, and how many there
    are: the quantity the percentile interval encodes, counted directly."""
    c3 = loss[(loss["level"].astype(str) == "0.3") & (loss["model"] == model)]
    piv = c3.pivot_table(index="replicate", columns="dimension", values="loss", observed=True)
    if a not in piv.columns or b not in piv.columns:
        return (0, 0)
    diff = (piv[a] - piv[b]).dropna()
    return (int((diff > 0).sum() if median_diff > 0 else (diff < 0).sum()), int(len(diff)))



def _symfmt(v, _pos):
    """Compact labels for the compressed axis: 0, +/-1, and powers beyond that.

    Mathtext sets its own minus, so the plain labels carry U+2212 rather than a hyphen;
    otherwise the small ticks and the large ones would not show the same sign.
    """
    if v == 0:
        return "0"
    if abs(v) < 10:
        return ("−" if v < 0 else "") + f"{abs(v):g}"
    exponent = int(round(np.log10(abs(v))))
    return ("$-" if v < 0 else "$") + f"10^{{{exponent}}}$"


def figure(cfg, scenario, out, only=None, scale=1.0):
    """One panel per model and co-primary metric; rows share their x axis.

    `only` keeps a single metric, and `scale` enlarges the type: the main-text version is
    printed at nearly its native size on a landscape page, the supplementary ones are
    reduced, so they cannot carry the same font sizes."""
    metrics = [(met, lab) for _, c, met, lab in G.COLS if c is cfg]
    if only:
        metrics = [(met, lab) for met, lab in metrics if met == only]
    n = len(cfg.models)
    fig, axes = plt.subplots(len(metrics), n, figsize=(2.0 * n, 3.4 * len(metrics) + 0.6),
                             squeeze=False)
    for row, (metric, mlabel) in enumerate(metrics):
        full = table(cfg, scenario, metric)
        # one pair order per metric, so that rows line up across models
        order = (full.assign(size=full["median_diff"].abs()).groupby(["dim_A", "dim_B"])["size"]
                 .mean().sort_values().index.tolist())
        for col, m in enumerate(cfg.models):
            ax = axes[row, col]
            d = full[full["model"] == m].copy()
            d["order"] = [order.index((a, b)) for a, b in zip(d.dim_A, d.dim_B)]
            d = d.sort_values("order").reset_index(drop=True)
            for k, r in d.iterrows():
                c_ = "#B2182B" if r["separable"] else "#9AA0A6"
                ax.plot([r["pct2_5"], r["pct97_5"]], [k, k], color=c_, lw=1.5, solid_capstyle="round")
                ax.plot(r["median_diff"], k, "o", ms=3.6, color=c_, mec="white", mew=0.4, zorder=3)
                # the share sits at the end the contrast points to, so it reads with the effect
                right = r["median_diff"] > 0
                ax.annotate(f"{r['n_agree']:.0f}",
                            xy=(r["pct97_5"] if right else r["pct2_5"], k),
                            xytext=(4 if right else -4, 0), textcoords="offset points",
                            ha="left" if right else "right", va="center",
                            fontsize=5.8 * scale, color=c_)
            ax.axvline(0, color="0.35", lw=0.7, ls=(0, (2, 2)))
            span = d[["pct2_5", "pct97_5"]].abs().to_numpy().max()
            if span > 3:                      # the neural model needs a compressed axis
                ax.set_xscale("symlog", linthresh=0.1)
                # the default locator labels every decade, ten of them across two inches,
                # and they overlap into illegibility: keep zero, the unit decade and the
                # outermost one the data reach
                top = max(1, int(np.ceil(np.log10(span))))
                ax.set_xticks([-10 ** top, -1, 0, 1, 10 ** top])
                ax.xaxis.set_minor_locator(mticker.NullLocator())
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(_symfmt))
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels([f"{SHORT[a]} $-$ {SHORT[b]}" for a, b in zip(d.dim_A, d.dim_B)]
                               if col == 0 else [], fontsize=6.5 * scale)
            ax.set_title(LAB[m], fontsize=8 * scale, fontweight="bold")
            ax.tick_params(length=2, labelsize=6.5 * scale)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.margins(x=0.28)
            ax.grid(axis="x", color="0.93", lw=0.5)
            ax.set_axisbelow(True)
        axes[row, 0].set_ylabel(mlabel, fontsize=8.5 * scale, fontweight="bold")
    fig.supxlabel("Median paired difference in loss between the two degradations", fontsize=8 * scale)
    fig.tight_layout()
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(PAPER / "img" / f"{out}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote", out)


def csv(path=PAPER / "contrasts_all.csv"):
    rows = []
    for scenario in ("S2", "S1", "S3"):
        for _, cfg, metric, mlabel in G.COLS:
            d = table(cfg, scenario, metric)
            d.insert(0, "metric", mlabel.replace("$^2$", "2"))
            d.insert(0, "task", TASK[cfg.name])
            d.insert(0, "scenario", scenario)
            rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    cols = ["scenario", "task", "metric", "model", "dim_A", "dim_B", "n_pairs", "median_diff",
            "pct2_5", "pct97_5", "n_agree", "n_rep", "share", "wilcoxon_stat", "p_raw", "p_holm",
            "ci_excludes_zero", "separable"]
    out[cols].to_csv(path, index=False)
    print("wrote", path, len(out), "contrasts,", int(out["separable"].sum()), "separable")


if __name__ == "__main__":
    figure(V2.BINARY, "S2", "figure6_contrasts_main", only="auc", scale=1.25)
    for i, cfg in enumerate((V2.BINARY, V2.CONTINUOUS, V2.SURVIVAL)):
        figure(cfg, "S2", f"suppfig{10 + i}_contrasts_{cfg.name}")
    csv()
