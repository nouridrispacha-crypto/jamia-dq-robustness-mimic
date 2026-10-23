# -*- coding: utf-8 -*-
"""Supplementary Tables S9-S17: every result of the main grid, one table per scenario and
task: for each model and each DQ dimension, the mean of the two co-primary metrics at every
corruption level (c = 0, 0.1, 0.2, 0.3), with the 2.5th-97.5th percentile range across the
500 bootstrap replicates.

Rows are the five degradations, repeated for each model; columns are the corruption levels
of the first metric, then of the second. Written to table_perf_<scenario>_<task>.tex.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PAPER
import sys
from pathlib import Path

PAPER = Path(PAPER)
sys.path.insert(0, str(PAPER))
import make_s2_grid as G  # noqa: E402

LAB = {"LASSO": "LASSO", "DecisionTree": "Decision tree", "RandomForest": "Random forest",
       "XGBoost": "XGBoost", "MLP1": "MLP", "CoxPH": "CoxPH", "RandomSurvivalForest": "RSF",
       "XGBoostSurv": "XGBoostSurv", "DeepSurv1": "DeepSurv"}
ORDER = {"binary": ["DecisionTree", "RandomForest", "XGBoost", "LASSO", "MLP1"],
         "continuous": ["DecisionTree", "RandomForest", "XGBoost", "LASSO", "MLP1"],
         "survival": ["RandomSurvivalForest", "XGBoostSurv", "CoxPH", "DeepSurv1"]}
TASK = {"binary": "in-hospital mortality", "continuous": "ICU length of stay",
        "survival": "time to death"}
PAIR = {"binary": ("auc", "AUROC", "brier", "Brier score"),
        "continuous": ("rmse", "RMSE", "r2", "R\\textsuperscript{2}"),
        "survival": ("c_index", "C-index", "ibs", "IBS")}
DEC = {"auc": 3, "brier": 3, "rmse": 2, "r2": 2, "c_index": 3, "ibs": 3}
SCEN = {"S2": "test-time (S2)", "S1": "training-time (S1)", "S3": "joint (S3)"}
LEVELS = [0.0, 0.1, 0.2, 0.3]
START = {"S2": 9, "S1": 12, "S3": 15}          # S9-S11 test-time, S12-S14 training, S15-S17 joint


def num(v, dec):
    t = f"{v:.{dec}f}"
    return "0" + t[2:] if t == f"{-0.0:.{dec}f}" else t


def short(v, dec):
    """Interval bounds without the leading zero, to keep eight columns on one page."""
    t = num(v, dec)
    return t.replace("0.", ".", 1) if t.startswith(("0.", "-0.")) else t


def cells(cfg, model, dim, scenario, metric):
    d = G.read(cfg.file_map[dim][scenario])
    d = d[(d["N"] == 3500) & (d["model"] == model)]
    dec = DEC[metric]
    out = []
    for c in LEVELS:
        x = d[d["p"] == c][metric]
        out.append("--" if x.empty else
                   f"{num(x.mean(), dec)} ({short(x.quantile(0.025), dec)}, {short(x.quantile(0.975), dec)})")
    return out


def table(scenario, cfg):
    m1, l1, m2, l2 = PAIR[cfg.name]
    num = START[scenario] + ["binary", "continuous", "survival"].index(cfg.name)
    body = []
    for model in ORDER[cfg.name]:
        body.append(f"    \\multicolumn{{9}}{{l}}{{\\textbf{{{LAB[model]}}}}} \\\\")
        for dim in G.DIMS:
            row = cells(cfg, model, dim, scenario, m1) + cells(cfg, model, dim, scenario, m2)
            body.append(f"    \\quad {dim.capitalize()} & " + " & ".join(row) + " \\\\")
        body.append("    \\addlinespace[3pt]")
    return (r"""\begin{landscape}
\begin{table}[htbp]
\caption*{Supplementary Table S""" + str(num) + r""". """ + TASK[cfg.name].capitalize() + r""" under """ +
            SCEN[scenario] + r""" degradation ($N=3\,500$): mean of each co-primary metric at every
corruption level, with the 2.5th--97.5th percentile range across the 500 bootstrap
replicates in parentheses, written without their leading zero. The column $c=0$ is the
uncorrupted reference of the same replicates and is identical across degradations by
construction.}
\centering
\singlespacing
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{l cccc cccc}
    \toprule
    & \multicolumn{4}{c}{""" + l1 + r"""} & \multicolumn{4}{c}{""" + l2 + r"""} \\
    \cmidrule(lr){2-5} \cmidrule(lr){6-9}
    Degradation & $c=0$ & $c=0.1$ & $c=0.2$ & $c=0.3$ & $c=0$ & $c=0.1$ & $c=0.2$ & $c=0.3$ \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
\end{tabular}
\end{table}
\end{landscape}""")


if __name__ == "__main__":
    for sc in ("S2", "S1", "S3"):
        for cfg in (G.V2.BINARY, G.V2.CONTINUOUS, G.V2.SURVIVAL):
            out = PAPER / f"table_perf_{sc}_{cfg.name}.tex"
            out.write_text(table(sc, cfg), encoding="utf-8")
            print("wrote", out.name)
