# -*- coding: utf-8 -*-
"""Print-sized figures for the manuscript and its supplement.

The earlier grids were drawn five panels wide at 27 inches and shrunk to the
16 cm text width, a factor of about 0.23: labels printed near 4 pt and curves
under 1 pt. Here every grid has one row per DQ dimension and one column per
metric or scenario, and is drawn close to its printed size, so text prints at
roughly 8-9 pt and curves at about 1.5 pt.

Data sources
  c <= 0.3 figures : analyse_stats_v2 file maps (coherence without rule R4)
  c <= 0.5 annex   : analyse_stats file maps (original runs), coherence excluded
  S2-MD            : JAMIA_V2 composite run

Y scales are shared within a metric across dimensions, as before, so the
dimensions stay visually comparable; RMSE and MAE keep one scale per panel
(the MLP under precision would flatten every other panel), and R^2 is fixed
to [-1, 1] with a reference at 0.
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

warnings.filterwarnings("ignore")
PAPER = Path(PAPER)
IMG = PAPER / "img"
sys.path.insert(0, str(PAPER))
import analyse_stats_v2 as V2      # noqa: E402
import analyse_stats as ORIG       # noqa: E402

# The standardised S2-MD run of 6 October, which is what Figure 2 and Table S8 report.
# This pointed at the September run under JAMIA_V2 while the figure was drawn from the
# standardised one, so the code as archived did not reproduce the published figure.
S2MD_CSV = PAPER / "rerun_standardisation/s2md/Results/S2MD_duplast/S2MD_duplast_P4_observe_test.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 11.5,
    "axes.labelsize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 11, "pdf.fonttype": 42, "axes.linewidth": 0.8,
})

DIMS = ["completeness", "coherence", "validity", "precision", "uniqueness"]
DIMS4 = ["completeness", "validity", "precision", "uniqueness"]
MODELS = {
    "binary": ["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"],
    "continuous": ["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"],
    "survival": ["CoxPH", "RandomSurvivalForest", "XGBoostSurv", "DeepSurv1"],
}
# legend grouped by model family, tree-based first (colours and markers unchanged)
FAMILIES = {
    "binary": [("Tree-based", ["DecisionTree", "RandomForest", "XGBoost"]), ("Linear and neural", ["LASSO", "MLP1"])],
    "continuous": [("Tree-based", ["DecisionTree", "RandomForest", "XGBoost"]), ("Linear and neural", ["LASSO", "MLP1"])],
    "survival": [("Tree-based", ["RandomSurvivalForest", "XGBoostSurv"]), ("Linear and neural", ["CoxPH", "DeepSurv1"])],
}
LABEL = {"LASSO": "LASSO", "DecisionTree": "DT", "RandomForest": "RF", "XGBoost": "XGBoost",
         "MLP1": "MLP", "CoxPH": "CoxPH", "RandomSurvivalForest": "RSF",
         "XGBoostSurv": "XGBoostSurv", "DeepSurv1": "DeepSurv"}
MARK = {"LASSO": "o", "DecisionTree": "s", "RandomForest": "^", "XGBoost": "D", "MLP1": "v",
        "CoxPH": "o", "RandomSurvivalForest": "s", "XGBoostSurv": "D", "DeepSurv1": "v"}
LS = {"LASSO": "-", "DecisionTree": "--", "RandomForest": "-", "XGBoost": "--", "MLP1": "-",
      "CoxPH": "-", "RandomSurvivalForest": "--", "XGBoostSurv": "--", "DeepSurv1": "-"}
BRIER_NULL = 0.098   # p(1-p) with the cohort mortality prevalence 1103/10000

_cache = {}


def read(path):
    path = Path(path)
    if path not in _cache:
        d = pd.read_csv(path, on_bad_lines="skip").rename(columns={"pollution": "p"})
        d["p"] = d["p"].round(2)
        _cache[path] = d
    return _cache[path]


def grid(outcome, rows, cols, out, *, figsize, cap=0.3, n=3500, share=(), fixed=None,
         refs=None, vline=None):
    """rows: [(label, {scenario: csv})]; cols: [(title, metric, scenario)]."""
    fixed, refs = fixed or {}, refs or {}
    palette = plt.cm.tab10.colors
    color = {m: palette[i] for i, m in enumerate(MODELS[outcome])}
    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=figsize, sharex=True, squeeze=False)

    for r, (rlabel, files) in enumerate(rows):
        for c, (title, metric, scen) in enumerate(cols):
            ax = axes[r, c]
            d = read(files[scen])
            d = d[(d["N"] == n) & (d["p"] <= cap + 1e-9)]
            for m in MODELS[outcome]:
                g = d[d["model"] == m].groupby("p")[metric]
                mean, lo, hi = g.mean(), g.quantile(0.025), g.quantile(0.975)
                ax.fill_between(mean.index, lo.values, hi.values, color=color[m], alpha=0.12, lw=0)
                ax.plot(mean.index, mean.values, color=color[m], ls=LS[m], lw=1.9,
                        marker=MARK[m], ms=5, mec="white", mew=0.6, zorder=3)
            if metric in refs:
                ax.axhline(refs[metric], color="0.25", lw=1.0, ls=(0, (1, 1.6)), zorder=2)
            if vline is not None:
                ax.axvline(vline, color="0.35", lw=1.1, ls=(0, (4, 3)), zorder=2)
            ax.set_xlim(0, cap)
            ax.set_xticks(np.round(np.arange(0, cap + 1e-9, 0.1), 1))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.grid(True, color="0.9", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(title, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(rlabel, fontweight="bold")
            if r == nr - 1:
                ax.set_xlabel("Corruption level $c$")

    for metric in share:
        axs = [axes[r, c] for r in range(nr) for c, col in enumerate(cols) if col[1] == metric]
        lo = min(a.get_ylim()[0] for a in axs)
        hi = max(a.get_ylim()[1] for a in axs)
        for a in axs:
            a.set_ylim(lo, hi)
    for metric, lim in fixed.items():
        for r in range(nr):
            for c, col in enumerate(cols):
                if col[1] == metric:
                    axes[r, c].set_ylim(*lim)

    handles, is_title = [], []
    for fam, ms in FAMILIES[outcome]:
        handles.append(Line2D([], [], ls="none", label=f"{fam}:"))
        is_title.append(True)
        for m in ms:
            handles.append(Line2D([0], [0], color=color[m], ls=LS[m], lw=2.2, marker=MARK[m], ms=6,
                                  mec="white", mew=0.6, label=LABEL[m]))
            is_title.append(False)
    fig.tight_layout(h_pad=0.9, w_pad=1.0, rect=(0, 0, 1, 0.97))
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                     bbox_to_anchor=(0.5, 0.975), handlelength=2.2, columnspacing=1.0,
                     handletextpad=0.5, fontsize=10)
    for txt, handle, title in zip(leg.get_texts(), leg.legend_handles, is_title):
        if title:
            txt.set_fontweight("bold")
            txt.set_color("0.3")
            handle.set_visible(False)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(IMG / f"{out}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {out}: {nr} x {nc} panels, {figsize[0]} x {figsize[1]} in")


def rows_for(cfg, dims):
    return [(d.capitalize(), cfg.file_map[d]) for d in dims]


if __name__ == "__main__":
    B, C, S = V2.BINARY, V2.CONTINUOUS, V2.SURVIVAL
    MAIN = (7.6, 8.8)          # 5 x 2, printed at text width
    TRIPLE = (8.6, 8.8)        # 5 x 3
    SCEN3 = [("S1 (training-time)", None, "S1"), ("S2 (test-time)", None, "S2"), ("S3 (joint)", None, "S3")]

    # ── main text ──
    grid("binary", rows_for(B, DIMS), [("AUROC", "auc", "S2"), ("Brier score", "brier", "S2")],
         "figure2_binary_combined", figsize=MAIN, share=("auc", "brier"), refs={"brier": BRIER_NULL})
    grid("continuous", rows_for(C, DIMS), [("RMSE", "rmse", "S2"), ("$\mathbf{R}^\mathbf{2}$", "r2", "S2")],
         "figure3_continuous_combined", figsize=MAIN, fixed={"r2": (-1, 1)}, refs={"r2": 0.0})
    grid("survival", rows_for(S, DIMS), [("C-index", "c_index", "S2"), ("IBS", "ibs", "S2")],
         "figure4_survival_combined", figsize=MAIN, share=("c_index", "ibs"))
    grid("binary", [("", {"S2": S2MD_CSV})],
         [("AUROC", "auc", "S2"), ("Brier score", "brier", "S2")],
         "figure5_s2md_p4observed", figsize=(7.6, 3.0))

    # ── supplementary: training-time and joint, one figure per scenario ──
    for scen, tag in (("S1", "s1"), ("S3", "s3")):
        grid("binary", rows_for(B, DIMS), [("AUROC", "auc", scen), ("Brier score", "brier", scen)],
             f"figureS9_binary_{tag}", figsize=MAIN, share=("auc", "brier"), refs={"brier": BRIER_NULL})
        grid("continuous", rows_for(C, DIMS), [("RMSE", "rmse", scen), ("$\mathbf{R}^\mathbf{2}$", "r2", scen)],
             f"figureS10_continuous_{tag}", figsize=MAIN, fixed={"r2": (-1, 1)}, refs={"r2": 0.0})
        grid("survival", rows_for(S, DIMS), [("C-index", "c_index", scen), ("IBS", "ibs", scen)],
             f"figureS11_survival_{tag}", figsize=MAIN, share=("c_index", "ibs"))

    # ── supplementary: F1-macro, cohort sizes, MAE (dimension rows, scenario columns) ──
    grid("binary", rows_for(B, DIMS), [(t, "f1_macro", s) for t, _, s in SCEN3],
         "figure_classif_f1", figsize=TRIPLE, share=("f1_macro",))
    grid("binary", rows_for(B, DIMS), [(t, "auc", s) for t, _, s in SCEN3],
         "figS2_binary_N1500", figsize=TRIPLE, n=1500, share=("auc",))
    grid("binary", rows_for(B, DIMS), [(t, "auc", s) for t, _, s in SCEN3],
         "figS5_binary_N7000", figsize=TRIPLE, n=7000, share=("auc",))
    grid("continuous", rows_for(C, DIMS), [(t, "mae", s) for t, _, s in SCEN3],
         "figure7_continuous_mae", figsize=TRIPLE)

    sys.exit(0)  # the c = 0.5 annex below was removed from the manuscript
    # ── annex: four dimensions up to c = 0.5, original runs, coherence excluded ──
    A4 = (7.6, 7.2)
    grid("binary", rows_for(ORIG.BINARY, DIMS4), [("AUROC", "auc", "S2"), ("Brier score", "brier", "S2")],
         "figure_annex_c05_binary", figsize=A4, cap=0.5, vline=0.3, share=("auc", "brier"),
         refs={"brier": BRIER_NULL})
    grid("continuous", rows_for(ORIG.CONTINUOUS, DIMS4), [("RMSE", "rmse", "S2"), ("$\mathbf{R}^\mathbf{2}$", "r2", "S2")],
         "figure_annex_c05_continuous", figsize=A4, cap=0.5, vline=0.3, fixed={"r2": (-1, 1)},
         refs={"r2": 0.0})
    grid("survival", rows_for(ORIG.SURVIVAL, DIMS4), [("C-index", "c_index", "S2"), ("IBS", "ibs", "S2")],
         "figure_annex_c05_survival", figsize=A4, cap=0.5, vline=0.3, share=("c_index", "ibs"))
