# -*- coding: utf-8 -*-
"""Combined Scenario S2 figure: 5 rows (DQ dimensions) x 3 outcomes x 2 co-primary metrics.

One colour per model role, shared across outcomes, so a single legend serves the
whole figure: decision tree; random forest / RSF; XGBoost / XGBoostSurv;
LASSO / CoxPH; MLP / DeepSurv. Bands are the 2.5th-97.5th percentile range across
bootstrap replicates, as in the other figures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PAPER
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")
PAPER = Path(PAPER)
IMG = PAPER / "img"
sys.path.insert(0, str(PAPER))
import analyse_stats_v2 as V2  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
                     "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})

DIMS = ["completeness", "coherence", "validity", "precision", "uniqueness"]
TAB = plt.cm.tab10.colors
ROLE = {  # role -> (colour, marker, linestyle, label)
    "dt":   (TAB[1], "s", "--", "Decision tree"),
    "rf":   (TAB[2], "^", "-",  "Random forest / RSF"),
    "xgb":  (TAB[3], "D", "--", "XGBoost / XGBoostSurv"),
    "lin":  (TAB[0], "o", "-",  "LASSO / CoxPH"),
    "nn":   (TAB[4], "v", "-",  "MLP / DeepSurv"),
}
MODEL_ROLE = {"DecisionTree": "dt", "RandomForest": "rf", "XGBoost": "xgb", "LASSO": "lin", "MLP1": "nn",
              "RandomSurvivalForest": "rf", "XGBoostSurv": "xgb", "CoxPH": "lin", "DeepSurv1": "nn"}
BRIER_NULL = 0.098
COLS = [  # (outcome header, cfg, metric, metric label)
    ("In-hospital mortality", V2.BINARY, "auc", "AUROC"),
    ("In-hospital mortality", V2.BINARY, "brier", "Brier score"),
    ("ICU length of stay", V2.CONTINUOUS, "rmse", "RMSE"),
    ("ICU length of stay", V2.CONTINUOUS, "r2", "R$^2$"),
    ("Time to death", V2.SURVIVAL, "c_index", "C-index"),
    ("Time to death", V2.SURVIVAL, "ibs", "IBS"),
]
_cache = {}

SHOW = ("most",)   # labels drawn on the figure; "least" is reported in the table
SHADE = {"all_most": "#F6D2CD", "some_most": "#FBEAE7", "all_least": "#DFF0DF"}
_lead_cache = {}
_model_cache = {}


def model_sets(cfg, scenario, metric):
    """Per model, the (most, least) damaging dimensions under the strict rule of Table 3."""
    key = (cfg.name, scenario, metric)
    if key not in _model_cache:
        _lead_cache.pop(key, None)
        leaders(cfg, scenario, metric)
    return _model_cache[key]


def vs_mean_of_others(cfg, loss):
    """Per model and dimension, the median over replicates of the loss minus the MEAN loss
    of the other four degraded dimensions, at c = 0.3. Positive means the dimension costs
    more than the average of the others."""
    c3 = loss[loss["level"].astype(str) == "0.3"]
    out = {}
    for m in cfg.models:
        piv = c3[c3["model"] == m].pivot_table(index="replicate", columns="dimension",
                                               values="loss", observed=True)
        out[m] = {}
        for d in DIMS:
            others = [c for c in DIMS if c != d and c in piv.columns]
            out[m][d] = (float((piv[d] - piv[others].mean(axis=1)).median())
                         if d in piv.columns and others else np.nan)
    return out


def leaders(cfg, scenario, metric):
    """Dimension status from reliably separable contrasts at c = 0.3 (Holm p < 0.05 and
    2.5-97.5 percentile interval excluding zero), per model, as in Table 3:
    most damaging = at least one separable win, no loss, and a loss above the mean of the
    other four degraded dimensions; least = the reverse. The requirement on the mean keeps
    a single mild competitor from making a dimension leading on its own."""
    key = (cfg.name, scenario, metric)
    if key in _lead_cache:
        return _lead_cache[key]
    direction = dict(cfg.metrics.values())[metric]
    loss = V2.build_loss_table(cfg, scenario, metric, direction)
    con = V2.pairwise_contrasts(cfg, loss)
    sep = con[(con["p_holm"] < 0.05) & con["ci_excludes_zero"]]
    gap = vs_mean_of_others(cfg, loss)
    most, least = {}, {}
    for m in cfg.models:
        s_m = sep[sep["model"] == m]
        wins = {d: 0 for d in DIMS}
        losses = {d: 0 for d in DIMS}
        for _, row in s_m.iterrows():
            hi, lo = (row["dim_A"], row["dim_B"]) if row["median_diff"] > 0 else (row["dim_B"], row["dim_A"])
            wins[hi] += 1
            losses[lo] += 1
        most[m] = {d for d in DIMS if wins[d] > 0 and losses[d] == 0 and gap[m][d] > 0}
        least[m] = {d for d in DIMS if losses[d] > 0 and wins[d] == 0 and gap[m][d] < 0}
    _model_cache[key] = (most, least)
    trees = [m for m in cfg.models if MODEL_ROLE[m] in ("dt", "rf", "xgb")]
    other = [m for m in cfg.models if m not in trees]
    status = {}
    for d in DIMS:
        entry = {}
        for kind, sets in (("most", most), ("least", least)):
            fam = [name for name, grp in (("trees", trees), ("linear & neural", other))
                   if grp and all(d in sets[m] for m in grp)]
            if len(fam) == 2:
                entry[kind] = "all"
            elif fam:
                entry[kind] = fam[0]
        if entry:
            status[d] = entry
    _lead_cache[key] = status
    return status



def read(path):
    if path not in _cache:
        d = pd.read_csv(path, on_bad_lines="skip").rename(columns={"pollution": "p"})
        d["p"] = d["p"].round(2)
        _cache[path] = d
    return _cache[path]


def main(out_name="figure_s2_grid", figsize=(7.5, 8.6), scenario="S2", shade=True):
    fig, axes = plt.subplots(5, 6, figsize=figsize, sharex=True, squeeze=False)
    for r, dim in enumerate(DIMS):
        for c, (_, cfg, metric, mlabel) in enumerate(COLS):
            ax = axes[r, c]
            d = read(cfg.file_map[dim][scenario])
            d = d[(d["N"] == 3500) & (d["p"] <= 0.3 + 1e-9)]
            for m in cfg.models:
                col, mk, ls, _ = ROLE[MODEL_ROLE[m]]
                g = d[d["model"] == m].groupby("p")[metric]
                mean, lo, hi = g.mean(), g.quantile(0.025), g.quantile(0.975)
                if metric == "r2":
                    mean, lo, hi = mean.clip(-1, 1), lo.clip(-1, 1), hi.clip(-1, 1)
                ax.fill_between(mean.index, lo.values, hi.values, color=col, alpha=0.10, lw=0)
                ax.plot(mean.index, mean.values, color=col, ls=ls, lw=1.2, marker=mk, ms=2.8,
                        mec="white", mew=0.3, zorder=3)
            if metric == "brier":
                ax.axhline(BRIER_NULL, color="0.3", lw=0.7, ls=(0, (1, 1.5)))
            if metric == "r2":
                ax.axhline(0, color="0.3", lw=0.7, ls=(0, (1, 1.5)))
                ax.set_ylim(-1.05, 1.0)
            if shade:
                st = {k: v for k, v in leaders(cfg, scenario, metric).get(dim, {}).items()
                      if k in SHOW}
                if st:
                    most = "most" in st
                    ax.add_patch(Rectangle((-0.02, -0.02), 1.04, 1.04, transform=ax.transAxes,
                                           fill=False, clip_on=False, zorder=6,
                                           edgecolor="#4A4A4A" if most else "#7A7A7A",
                                           lw=0.9 if most else 0.8,
                                           ls="-" if most else (0, (3, 2))))
                    lines = [f"{k}:\n{st[k]}" if len(st[k]) > 6 else f"{k}: {st[k]}"
                             for k in ("most", "least") if k in st]
                    for j, txt in enumerate(lines):
                        bold = txt.startswith("most")
                        ax.text(0.98, 0.97 - 0.13 * j, txt, transform=ax.transAxes, ha="right",
                                va="top", fontsize=5.6, fontweight="normal",
                                color="#333333" if bold else "#555555", zorder=7,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            ax.set_xlim(0, 0.3)
            ax.set_xticks([0, 0.1, 0.2, 0.3])
            ax.set_xticklabels(["0", ".1", ".2", ".3"])
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
            ax.grid(True, color="0.92", lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(mlabel, fontweight="bold", pad=7)
            if c == 0:
                ax.set_ylabel(dim.capitalize(), fontweight="bold", fontsize=8)
            if r == 4:
                ax.set_xlabel("$c$", labelpad=1)
    # shared y range per metric column (RMSE and R2 excepted)
    for c, (_, _, metric, _) in enumerate(COLS):
        if metric in ("rmse", "r2"):
            continue
        lo = min(axes[r, c].get_ylim()[0] for r in range(5))
        hi = max(axes[r, c].get_ylim()[1] for r in range(5))
        for r in range(5):
            axes[r, c].set_ylim(lo, hi)
    fig.tight_layout(h_pad=0.5, w_pad=0.35, rect=(0, 0, 1, 0.925))
    # outcome headers spanning their two metric columns
    for k, name in enumerate(["In-hospital mortality", "ICU length of stay", "Time to death"]):
        a, b = axes[0, 2 * k].get_position(), axes[0, 2 * k + 1].get_position()
        x = (a.x0 + b.x1) / 2
        fig.text(x, 0.938, name, ha="center", va="bottom", fontsize=9, fontweight="bold")
        fig.add_artist(Line2D([a.x0, b.x1], [0.934, 0.934], color="0.35", lw=0.8,
                              transform=fig.transFigure))
    handles = [Line2D([0], [0], color=col, ls=ls, lw=1.6, marker=mk, ms=4, mec="white", mew=0.4,
                      label=lab) for col, mk, ls, lab in ROLE.values()]
    if shade:
        from matplotlib.patches import Patch
        handles += [Patch(facecolor="white", edgecolor="#4A4A4A", lw=0.9,
                          label="most damaging (all models, or the family named)")]
    # the dotted reference line had no key: a reader saw it on ten of the thirty panels
    # with nothing to identify it, and had to go down into the caption
    handles += [Line2D([0], [0], color="0.3", lw=0.7, ls=(0, (1, 1.5)),
                       label="no-information Brier score; $R^2 = 0$")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.955), handlelength=2.2, columnspacing=1.0)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(IMG / f"{out_name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote", IMG / f"{out_name}.png")


if __name__ == "__main__":
    main()
