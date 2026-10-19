"""Supplementary Figure 5 (residual diagnostics) on the standardised arm.

make_resid_figures.py reads the binary loss tables from binary_stats_results_v2, which
date from 12 September and predate the standardisation. The binary panels are therefore
drawn on superseded losses; the continuous one is not, that chain already standardised.

The figure shows three cases chosen for what they are -- the most skewed overall, the most
skewed continuous, and a well-behaved comparison -- so the standardised skewness decides
which they are:

    most skewed overall      binary Brier S1  3.34  ->  binary Brier S2  2.18
    most skewed continuous   continuous R2 S1 1.95  ->  unchanged
    well-behaved comparison  binary AUROC S2  0.52  ->  unchanged (0.28)

One case changes, so one file name changes with it, and the manuscript's \\includegraphics
has to follow. Nothing here touches the manuscript.
"""
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as spstats
from scipy.stats import skew

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PAPER = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
sys.path.insert(0, str(PAPER))
import analyse_stats_v2 as V2                                          # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

# (sortie, scenario, metrique, dossier des pertes, intitule)
CASES = [
    ("continuous", "S1", "r2", "continuous_stats_results_v2",
     "Continuous outcome, R$^2$, S1 (training-time)"),
    ("binary", "S2", "brier", "binary_stats_results_v2_scaled",
     "Binary outcome, Brier score, S2 (test-time)"),
    ("binary", "S2", "auc", "binary_stats_results_v2_scaled",
     "Binary outcome, AUROC, S2 (test-time)"),
    # le cas que la figure montrait avant : regenere sur les memes donnees, pour que le
    # fichier vers lequel le manuscrit pointe encore ne reste pas celui de septembre
    ("binary", "S1", "brier", "binary_stats_results_v2_scaled",
     "Binary outcome, Brier score, S1 (training-time)"),
]

STAMP = f"{datetime.now():%Y%m%d_%H%M}"
for f in (PAPER / "img").glob("suppfig8_resid_*.png"):
    shutil.copy(f, f.with_name(f"{f.stem}_backup_{STAMP}.png"))

for outcome, scen, metric, folder, label in CASES:
    res_dir = PAPER / folder
    loss = pd.read_csv(res_dir / f"loss_{outcome}_{scen}_{metric}.csv")
    others = [d for d in loss["dimension"].unique() if d != V2.REFERENCE_DIMENSION]
    loss["dimension"] = pd.Categorical(loss["dimension"],
                                       categories=[V2.REFERENCE_DIMENSION] + others)
    loss["level"] = pd.Categorical(loss["level"].map(lambda v: f"{float(v):.1f}"),
                                   categories=["0.1", "0.2", "0.3"])
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
    print("ecrit {:38s} skew {:.2f} (diagnostic enregistre {:.2f}) <- {}".format(
        out.name, sk, saved, folder))
