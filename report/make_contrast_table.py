# -*- coding: utf-8 -*-
"""Table 3: most and least damaging degradations by outcome, metric and scenario.

Same rule as the framed panels of the S2 grid (make_s2_grid.leaders): for each
model, a degradation is most damaging when it is reliably separable (Holm p <
0.05 and 2.5-97.5 percentile interval excluding zero) from at least one other,
never on the less damaging side, and costs more than the mean of the other four
degraded dimensions; least damaging is the reverse. A cell names the degradation
with "all" when this holds for every model, otherwise with the family for which
it holds for every model of that family.
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

ROWS = [("Mortality", G.V2.BINARY, "auc", "AUROC"), ("Mortality", G.V2.BINARY, "brier", "Brier score"),
        ("Length of stay", G.V2.CONTINUOUS, "rmse", "RMSE"),
        ("Length of stay", G.V2.CONTINUOUS, "r2", "R\\textsuperscript{2}"),
        ("Time to death", G.V2.SURVIVAL, "c_index", "C-index"), ("Time to death", G.V2.SURVIVAL, "ibs", "IBS")]
SCEN = [("S2", "Test-time (S2)"), ("S1", "Training-time (S1)"), ("S3", "Joint (S3)")]
ORDER = {"all": 0, "trees": 1, "linear & neural": 2}
FAMILY = {"all": "all", "trees": "trees", "linear & neural": "linear and neural"}


def cell(status, kind):
    items = sorted(((v[kind], d) for d, v in status.items() if kind in v), key=lambda t: (ORDER[t[0]], G.DIMS.index(t[1])))
    if not items:
        return "--"
    parts = [f"{d} ({FAMILY[f]})" for f, d in items]
    txt = "; ".join(parts)
    return txt[0].upper() + txt[1:]


def latex():
    lines = []
    for outcome, cfg, metric, mlabel in ROWS:
        cells = []
        for sc, _ in SCEN:
            st = G.leaders(cfg, sc, metric)
            cells.append(f"\\textbf{{Most:}} {cell(st, 'most')}\\newline \\textit{{Least:}} {cell(st, 'least')}")
        task = outcome if metric in ("auc", "rmse", "c_index") else ""
        lines.append(f"    {task} & {mlabel} & " + " & ".join(cells) + " \\\\")
        if metric in ("brier", "r2"):
            lines.append("    \\addlinespace[3pt]")
    body = "\n".join(lines)
    return r"""\begin{table}[htbp]
\caption{Most and least damaging degradations at $c=0.3$ ($N=3\,500$), by task,
co-primary metric and scenario.
\textit{The rule.} A degradation is most damaging for a model when three
conditions hold: (i) it costs reliably more than at least one other degradation,
that is, Holm-adjusted $p<0.05$ with the 2.5th--97.5th percentile interval of the
paired differences excluding zero; (ii) no other degradation reliably costs more
than it; and (iii) it costs more than the average of the other four. Least
damaging is the mirror image of the three. Condition (iii) stops a single mild
competitor from promoting a degradation on its own.
\textit{Reading a cell.} A degradation is named when the three conditions hold
for every model (all), or for every model of one family; a dash means that none
qualifies. Trees: decision tree, random forest and XGBoost for mortality and
length of stay, random survival forest and XGBoostSurv for time to death; linear
and neural: LASSO and the MLP, or CoxPH and DeepSurv.
\textit{Two cautions.} The rule is not a ranking by mean loss: a degradation with
a larger mean loss may fail to qualify when its effect varies more across
replicates, and two degradations that are not separable from each other can both
qualify. Absence from the table is an absence of demonstrated difference, not
evidence of equivalence. Supplementary Figures~10--12 plot every contrast with
its interval, and Supplementary Tables~S9--S17 give performance at every level
for each model and degradation.}
\label{tab:contrasts}
\centering
\singlespacing
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{>{\raggedright\arraybackslash}p{1.6cm}
                >{\raggedright\arraybackslash}p{1.3cm}
                >{\raggedright\arraybackslash}p{4.2cm}
                >{\raggedright\arraybackslash}p{4.2cm}
                >{\raggedright\arraybackslash}p{4.2cm}}
    \toprule
    Task & Metric & Test-time (S2) & Training-time (S1) & Joint (S3) \\
    \midrule
""" + body + r"""
    \bottomrule
\end{tabular}
\end{table}"""


if __name__ == "__main__":
    out = PAPER / "table3_contrasts.tex"
    out.write_text(latex(), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
