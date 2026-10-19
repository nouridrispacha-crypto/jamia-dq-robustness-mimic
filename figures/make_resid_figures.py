# -*- coding: utf-8 -*-
"""Supplementary Figure 8: residual QQ-plots and histograms of the mixed model,
English labels. Refits the model of analyse_stats_v2 on its saved loss tables
for the three cases shown, and checks the skewness against the saved
diagnostics before writing.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as spstats
from scipy.stats import skew

warnings.filterwarnings("ignore")
PAPER = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
sys.path.insert(0, str(PAPER))
import analyse_stats_v2 as V2  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
CASES = [("continuous", "S1", "r2", "Continuous outcome, R$^2$, S1 (training-time)"),
         ("binary", "S1", "brier", "Binary outcome, Brier score, S1 (training-time)"),
         ("binary", "S2", "auc", "Binary outcome, AUROC, S2 (test-time)")]

for outcome, scen, metric, label in CASES:
    res_dir = PAPER / f"{outcome}_stats_results_v2"
    loss = pd.read_csv(res_dir / f"loss_{outcome}_{scen}_{metric}.csv")
    loss["dimension"] = pd.Categorical(loss["dimension"], categories=[V2.REFERENCE_DIMENSION] +
                                       [d for d in loss["dimension"].unique() if d != V2.REFERENCE_DIMENSION])
    loss["level"] = pd.Categorical(loss["level"].map(lambda v: f"{float(v):.1f}"), categories=["0.1", "0.2", "0.3"])
    result = V2.fit_mixed_model(loss)
    resid = np.asarray(result.resid)
    resid = resid[np.isfinite(resid)]
    sk = float(skew(resid))
    saved = pd.read_csv(res_dir / f"residdiag_{outcome}_{scen}_{metric}.csv")["skew"].iloc[0]
    assert abs(sk - saved) < 0.01, (outcome, scen, metric, sk, saved)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    (osm, osr), (slope, intercept, _) = spstats.probplot(resid, dist="norm")
    axes[0].scatter(osm, osr, s=6, color="#33506b", alpha=0.5, lw=0)
    axes[0].plot(osm, slope * osm + intercept, color="#b8452f", lw=1.6)
    axes[0].set_title("Normal QQ-plot of residuals")
    axes[0].set_xlabel("Theoretical normal quantiles")
    axes[0].set_ylabel("Ordered residuals")
    axes[1].hist(resid, bins=60, color="#33506b", alpha=0.85)
    axes[1].set_title("Histogram of residuals")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{label}; skewness {sk:.2f}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = PAPER / "img" / f"suppfig8_resid_{outcome}_{metric}_{scen.lower()}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}: skew {sk:.2f} (saved {saved:.2f})")
