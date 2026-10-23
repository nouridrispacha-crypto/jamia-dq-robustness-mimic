"""Rebuild Supplementary Table S5 and its footnote from the refitted models.

regen_s5_j90 refitted all twenty converging models and reproduced every binary and continuous
value the manuscript prints, so only the survival rows move. Three of them enter the seven
most asymmetric and two leave, which reorders the table rather than editing cells, and the
footnote's quoted range changes with the membership.

The table is rewritten from the ranking file. The binary and continuous values are a
regression guard: each one that stays in the table must keep the number the manuscript has.
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

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(PAPER)
T = pd.read_csv(Path(__file__).resolve().parent / "resid_all_j90/skewness_all.csv")
PUBLISHED = {("Binary", "S2", "Brier"): 2.18,
             ("Continuous", "S1", "R\\textsuperscript{2}"): 1.95,
             ("Continuous", "S1", "RMSE"): 1.47,
             ("Binary", "S2", "F1-macro"): 1.31,
             ("Binary", "S1", "AUROC"): 0.66}

top = T.head(7)
for _, r in top.iterrows():
    key = (r["outcome"], r["scenario"], r["metric"])
    if key in PUBLISHED:
        assert abs(r["skew"] - PUBLISHED[key]) < 0.005, (key, r["skew"])
print("garde-fou : les valeurs non-survie du top 7 sont celles du manuscrit")

rows = "\n".join(
    "    {} & {} & {} & {}\\,{:03d} & {:.2f} \\\\".format(
        r["outcome"], r["scenario"], r["metric"],
        int(r["n"]) // 1000, int(r["n"]) % 1000, r["skew"])
    for _, r in top.iterrows())

rest = T.iloc[7:]
lo, hi = rest.loc[rest["skew"].idxmin()], rest.loc[rest["skew"].idxmax()]


def name(r):
    met = r["metric"].replace("\\textsuperscript{2}", "\\textsuperscript{2}")
    return "{} {}, {}".format(r["outcome"].lower().replace("time-to-event", "time-to-event"),
                              met, r["scenario"])


foot = ("    \\multicolumn{{5}}{{p{{13cm}}}}{{\\footnotesize The thirteen models not listed "
        "had\n    skewness between ${:.2f}$ ({}) and ${:.2f}$ ({}).\n"
        "    The continuous R\\textsuperscript{{2}} model under Scenario S2 did not\n"
        "    converge (singular matrix).}} \\\\").format(
            lo["skew"], name(lo), hi["skew"], name(hi))

tex = P / "manuscript.tex"
s = tex.read_text(encoding="utf-8")
i = s.index("Supplementary Table S5. Residual skewness")
a = s.rindex("\\begin{table}", 0, i)
b = s.index("\\end{table}", i)
blk = s[a:b]

m0 = blk.index("\\midrule") + len("\\midrule")
m1 = blk.index("\\bottomrule")
old_rows = blk[m0:m1]
new_blk = blk[:m0] + "\n" + rows + "\n    " + blk[m1:]

f0 = new_blk.index("\\multicolumn{5}")
f1 = new_blk.index("\\end{tabular}")
old_foot = new_blk[f0:f1]
new_blk = new_blk[:f0] + foot.strip()[4:] + "\n" + new_blk[f1:]

s = s[:a] + new_blk + s[b:]
assert s.count("{") == s.count("}")
assert s.rstrip().endswith("\\end{document}")
nocom = re.sub(r"(?m)^\s*%.*$", "", s)
labels = set(re.findall(r"\\label\{([^}]*)\}", nocom))
used = (set(re.findall(r"\\hyperref\[([^\]]*)\]", nocom))
        | set(re.findall(r"\\ref\{([^}]*)\}", nocom)))
assert not (used - labels), sorted(used - labels)
chk = s[s.index("Supplementary Table S5. Residual skewness"):]
chk = chk[:chk.index("\\end{table}")]
assert chk.count("\\\\") >= 8, chk.count("\\\\")
assert "Time-to-event & S2 & IBS" in chk and "2.81" in chk

shutil.copy(tex, P / "backups_j90" / f"manuscript_before_s5j90_{datetime.now():%H%M%S}.tex")
tex.write_text(s, encoding="utf-8")
print("\nTableau S5 reconstruit")
for line in rows.splitlines():
    print("   " + " ".join(line.split()))
print("\npied de page : entre {:.2f} ({}) et {:.2f} ({})".format(
    lo["skew"], name(lo), hi["skew"], name(hi)))
