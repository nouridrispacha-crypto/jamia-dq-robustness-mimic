"""Supplementary Table S6, time-to-event column recomputed on the 90-day relaunch.

regen_s6 copied this column from the manuscript, because the survival chain was not being
changed then. It is now, so the column is rebuilt the way the caption describes it: the
median paired difference between c=0.3 and clean data in Scenario S2 at N=3500, with its
2.5th-97.5th percentile range, pooling the four survival models of the manuscript and the
five hundred bootstrap replicates.

The difference is reported in the metric's own direction, so a loss of discrimination is
negative, as the other columns are. The binary and continuous columns are a regression
guard: they must come out of this script byte for byte as they went in.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PAPER
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(PAPER)
SURV = P / "rerun_survival_j90/merged/survival_outcome"
STEM = {"Completeness": "mnar_top/outputs_mnar_top_s2/S2_test_only",
        "Coherence": "Coherence/outputs_coherence_s2/S2_test_only",
        "Validity": "validity/outputs_validity_S2/S2_test_only",
        "Precision": "precision/outputs_precision_S2/S2_test_only",
        "Uniqueness": "uniqueness/outputs_uniqueness_S2/S2_test_only"}
MODELS = ["CoxPH", "RandomSurvivalForest", "XGBoostSurv", "DeepSurv1"]
ORDER = ["Coherence", "Completeness", "Validity", "Precision", "Uniqueness"]
COL_TTE = 3                                  # 4e colonne de valeurs : Delta C-index


def cell(dim):
    d = pd.read_csv(SURV / (STEM[dim] + ".csv"))
    d = d[(d["N"] == 3500) & d.model.isin(MODELS)]
    a = d[d.p.round(2) == 0.0].set_index(["model", "bootstrap"])["c_index"]
    b = d[d.p.round(2) == 0.3].set_index(["model", "bootstrap"])["c_index"]
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    k = a.index.intersection(b.index)
    assert len(k) == 2000, (dim, len(k))
    v = (b.loc[k] - a.loc[k]).to_numpy()
    return float(np.median(v)), *(float(x) for x in np.percentile(v, [2.5, 97.5]))


def tex(x):
    return ("$+{:.3f}$" if x > 0 else "$-{:.3f}$" if x < 0 else "$0.000$").format(abs(x))


tex_path = P / "manuscript.tex"
s = tex_path.read_text(encoding="utf-8")
i = s.index("\\label{tab:effectsizes}")
a = s.rindex("\\begin{table}", 0, i)
b = s.index("\\end{table}", i)
blk = s[a:b]
before = blk

print("colonne Delta C-index, S2, c=0.3 contre c=0, N=3500, 4 modeles x 500 replicats")
for dim in ORDER:
    m = re.search(r"(?m)^(\s*" + dim + r"\s*&\s*)(.*?)\\\\", blk)
    assert m, dim
    cells = [c.strip() for c in m.group(2).split("&")]
    assert len(cells) == 4, (dim, cells)
    med, lo, hi = cell(dim)
    new = list(cells)
    new[COL_TTE] = "{} [{}, {}]".format(tex(med), tex(lo), tex(hi))
    print("  {:14s} {:28s} -> {}".format(dim, cells[COL_TTE], new[COL_TTE]))
    blk = blk[:m.start()] + m.group(1) + " & ".join(new) + " \\\\" + blk[m.end():]

# Garde-fou : seules les cellules de la 4e colonne ont change.
for dim in ORDER:
    o = re.search(r"(?m)^(\s*" + dim + r"\s*&\s*)(.*?)\\\\", before)
    n = re.search(r"(?m)^(\s*" + dim + r"\s*&\s*)(.*?)\\\\", blk)
    oc = [c.strip() for c in o.group(2).split("&")]
    nc = [c.strip() for c in n.group(2).split("&")]
    for j in range(3):
        assert oc[j] == nc[j], "colonne {} de {} alteree".format(j, dim)

s = s[:a] + blk + s[b:]
assert s.count("{") == s.count("}")
assert s.rstrip().endswith("\\end{document}")
nocom = re.sub(r"(?m)^\s*%.*$", "", s)
labels = set(re.findall(r"\\label\{([^}]*)\}", nocom))
used = (set(re.findall(r"\\hyperref\[([^\]]*)\]", nocom))
        | set(re.findall(r"\\ref\{([^}]*)\}", nocom)))
assert not (used - labels), sorted(used - labels)
shutil.copy(tex_path,
            P / "backups_j90" / f"manuscript_before_s6j90_{datetime.now():%H%M%S}.tex")
tex_path.write_text(s, encoding="utf-8")
print("\ncolonnes binaires et continue intactes ; colonne survie reecrite")
