# -*- coding: utf-8 -*-
"""Supplementary Table S6: does the leading degradation survive at lower corruption levels?

For every outcome, co-primary metric and scenario, and at each corruption level, the
dimension most often on the damaging side of a reliably separable pair (Holm-adjusted
p < 0.05 within the ten pairs of a model, and 2.5-97.5 percentile interval of the paired
differences excluding zero), with the number of reliably separable pairs in parentheses.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

PAPER = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
sys.path.insert(0, str(PAPER))
import make_s2_grid as G  # noqa: E402
V2 = G.V2
DIMS = G.DIMS
LEVELS = ["0.1", "0.2", "0.3"]
ROWS = [("Mortality", V2.BINARY, [("AUROC", "auc"), ("Brier score", "brier")]),
        ("Length of stay", V2.CONTINUOUS, [("RMSE", "rmse"), ("R\\textsuperscript{2}", "r2")]),
        ("Time to death", V2.SURVIVAL, [("C-index", "c_index"), ("IBS", "ibs")])]


def leader(cfg, sc, met, level):
    """(dimension most often on the damaging side, number of reliably separable pairs)."""
    L = V2.build_loss_table(cfg, sc, met, dict(cfg.metrics.values())[met])
    L = L[L["level"].astype(str) == level]
    count = {d: 0 for d in DIMS}
    nsep = 0
    for m in cfg.models:
        d = L[L["model"] == m].pivot_table(index="replicate", columns="dimension",
                                          values="loss", observed=True)
        if not all(x in d.columns for x in DIMS):
            continue
        ps, info = [], []
        for x, y in combinations(DIMS, 2):
            diff = (d[x] - d[y]).dropna()
            ps.append(wilcoxon(diff)[1])
            info.append((x, y, float(diff.median()), np.percentile(diff, [2.5, 97.5])))
        for (x, y, md, (lo, hi)), ph in zip(info, multipletests(ps, method="holm")[1]):
            if ph < 0.05 and (lo > 0 or hi < 0):
                nsep += 1
                count[x if md > 0 else y] += 1
    best = max(count, key=lambda k: count[k])
    return best.capitalize(), nsep


def latex():
    lines, stable = [], 0
    for task, cfg, mets in ROWS:
        first = True
        for mlabel, met in mets:
            for sc in ("S1", "S2", "S3"):
                cells = [leader(cfg, sc, met, lv) for lv in LEVELS]
                stable += len({c[0] for c in cells}) == 1
                lines.append(f"    {task if first else ''} & {mlabel if sc == 'S1' else ''} & {sc} & "
                             + " & ".join(f"{d} ({n})" for d, n in cells) + " \\\\")
                first = False
            lines.append("    \\addlinespace[2pt]")
        lines[-1] = "    \\addlinespace[3pt]"
    print(f"stable rows: {stable}/18", file=sys.stderr)
    body = "\n".join(lines[:-1])
    return r"""\begin{table}[htbp]
\caption*{Supplementary Table S6. Dimension whose degradation was most often on the
damaging side of a reliably separable pair ($N=3\,500$), at each corruption level, with
the number of reliably separable pairs in parentheses: out of fifty for the binary and
continuous outcomes (five models $\times$ ten dimension pairs) and forty for the censored
time-to-event outcome (four models $\times$ ten pairs). A pair is reliably separable when
its Holm-adjusted $p<0.05$ and its 2.5th--97.5th percentile interval excludes zero.}
\label{tab:lowc}
\centering
\singlespacing
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lllccc}
    \toprule
    Task & Metric & Scenario & $c=0.1$ & $c=0.2$ & $c=0.3$ \\
    \midrule
""" + body + r"""
    \bottomrule
\end{tabular}
\end{table}"""


if __name__ == "__main__":
    out = PAPER / "table_s6.tex"
    out.write_text(latex(), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
