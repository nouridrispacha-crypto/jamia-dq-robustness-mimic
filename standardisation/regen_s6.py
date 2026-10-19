"""Supplementary Table S6, binary columns recomputed on the standardised arm.

No new simulation: the paired c=0.3 minus c=0 differences come from the substituted inputs
the figures and Table 3 already use. The continuous and time-to-event columns are copied
from the manuscript unchanged, since those chains always standardised.
"""
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
IN = Path(__file__).resolve().parent / "scaled_inputs"
KEY = {"Coherence": "coherence", "Completeness": "completeness", "Validity": "validity",
       "Precision": "precision", "Uniqueness": "uniqueness"}
MODELS = ["LASSO", "MLP1", "DecisionTree", "RandomForest", "XGBoost"]
ORDER = ["Coherence", "Completeness", "Validity", "Precision", "Uniqueness"]


def cell(dim, metric):
    d = pd.read_csv(IN / KEY[dim] / "S2.csv")
    lvl = "pollution" if "pollution" in d.columns else "p"
    d = d[(d["N"] == 3500) & d.model.isin(MODELS)]
    a = d[d[lvl].round(2) == 0.0].set_index(["model", "bootstrap"])[metric]
    b = d[d[lvl].round(2) == 0.3].set_index(["model", "bootstrap"])[metric]
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    k = a.index.intersection(b.index)
    assert len(k) == 2500, (dim, metric, len(k))
    v = (b.loc[k] - a.loc[k]).to_numpy()
    med = float(np.median(v))
    lo, hi = (float(x) for x in np.percentile(v, [2.5, 97.5]))
    return med, lo, hi


def tex(x):
    return ("$+{:.3f}$" if x > 0 else "$-{:.3f}$" if x < 0 else "$0.000$").format(abs(x))


tex_path = P / "manuscript.tex"
shutil.copy(tex_path, P / f"manuscript_backup_before_s6_{datetime.now():%Y%m%d_%H%M}.tex")
s = tex_path.read_text(encoding="utf-8")
i = s.index("\\label{tab:effectsizes}")
a = s.rindex("\\begin{table}", 0, i)
b = s.index("\\end{table}", i)
blk = s[a:b]

changed = 0
for dim in ORDER:
    m = re.search(r"(?m)^(\s*" + dim + r"\s*&\s*)(.*?)\\\\", blk)
    assert m, dim
    cells = [c.strip() for c in m.group(2).split("&")]
    assert len(cells) == 4, (dim, cells)
    new = list(cells)
    for j, metric in ((0, "auc"), (1, "brier")):
        med, lo, hi = cell(dim, metric)
        new[j] = "{} [{}, {}]".format(tex(med), tex(lo), tex(hi))
        if " ".join(new[j].split()) != " ".join(cells[j].split()):
            changed += 1
    row = m.group(1) + " & ".join(new) + " \\\\"
    blk = blk[:m.start()] + row + blk[m.end():]
    print(f"  {dim:14s} AUROC {new[0]:32s} Brier {new[1]}")

s = s[:a] + blk + s[b:]
assert s.count("{") == s.count("}")
assert s.rstrip().endswith("\\end{document}")
nocom = re.sub(r"(?m)^\s*%.*$", "", s)
labels = set(re.findall(r"\\label\{([^}]*)\}", nocom))
assert not (set(re.findall(r"\\ref\{([^}]*)\}", nocom)) - labels)
# the continuous and time-to-event columns must be untouched
t6 = s[s.index("\\label{tab:effectsizes}"):]
t6 = t6[:t6.index("\\end{table}")]
for probe in ("$+0.153$", "$+0.093$", "$+0.099$", "$+0.034$", "$+0.025$"):
    assert probe in t6, "colonne continue alteree : {}".format(probe)
tex_path.write_text(s, encoding="utf-8")
print(f"\n{changed} cellules binaires reecrites ; colonnes continue et survie intactes")
