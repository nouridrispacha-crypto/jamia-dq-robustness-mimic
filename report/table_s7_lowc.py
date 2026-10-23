"""Supplementary Table S7 recomputed for the 90-day survival relaunch.

Follows regen_s7, which did the same for the standardised binary arm, with two changes. The
labels are already the outcome-type ones that script installed, so the entry assertion looks
for those instead of the generator's. And only the six survival rows may move: the twelve
binary and continuous rows are a regression guard, since the binary arm still comes from the
standardised inputs and the continuous arm from its own unchanged source.

The lead-in sentence counts how many of the eighteen rows keep the same leader at all three
corruption levels. That count is recomputed, not assumed, because two survival rows change
from a stable leader to one that moves with the level.
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(PAPER)
sys.path.insert(0, str(P))
IN = Path(__file__).resolve().parent / "scaled_inputs"

import analyse_stats_v2 as V2
import make_s2_grid as G

for dim in V2.BINARY.file_map:
    for scen in V2.BINARY.file_map[dim]:
        p = IN / dim / f"{scen}.csv"
        assert p.exists(), p
        V2.BINARY.file_map[dim][scen] = p
for c in (G._cache, G._lead_cache, G._model_cache):
    c.clear()
print("binaire -> standardise | survie -> {}".format(
    Path(V2.SURVIVAL.file_map["coherence"]["S2"]).parents[2].name))

import make_table_s6 as S7

S7.ROWS = [("Binary", V2.BINARY, [("AUROC", "auc"), ("Brier score", "brier")]),
           ("Continuous", V2.CONTINUOUS, [("RMSE", "rmse"),
                                          ("R\\textsuperscript{2}", "r2")]),
           ("Time-to-event", V2.SURVIVAL, [("C-index", "c_index"), ("IBS", "ibs")])]
new = S7.latex()
assert "Binary & AUROC" in new and "Mortality" not in new, "le renommage n'a pas pris"


def data_rows(text):
    return [" ".join(r.split()) for r in text.split("\\\\")
            if r.count("&") == 5 and "Metric" not in r]


tex = P / "manuscript.tex"
s = tex.read_text(encoding="utf-8")
i = s.index("\\label{tab:lowc}")
a = s.rindex("\\begin{table}", 0, i)
b = s.index("\\end{table}", i) + len("\\end{table}")
old = s[a:b]
assert "Binary & AUROC" in old, "la S7 du manuscrit n'a pas les etiquettes par type"

old_rows, new_rows = data_rows(old), data_rows(new)
assert len(old_rows) == len(new_rows) == 18, (len(old_rows), len(new_rows))

drift = [k for k in range(12) if old_rows[k] != new_rows[k]]
if drift:
    print("ARRET : {} ligne(s) binaires ou continues ont bouge".format(len(drift)))
    for k in drift:
        print("  avant : {}".format(old_rows[k][:120]))
        print("  apres : {}".format(new_rows[k][:120]))
    sys.exit(1)
print("garde-fou : les 12 lignes binaires et continues sont inchangees\n")

print("lignes survie")
for k in range(12, 18):
    tag = "   " if old_rows[k] == new_rows[k] else " ->"
    print("{} avant : {}".format(tag, old_rows[k][:130]))
    if old_rows[k] != new_rows[k]:
        print("   apres : {}".format(new_rows[k][:130]))


def stable(rows):
    n = 0
    for r in rows:
        cells = [c.strip() for c in r.split("&")]
        names = [re.sub(r"\s*\(\d+\)", "", c).strip() for c in cells[3:6]]
        if names[0] == names[1] == names[2] and names[0]:
            n += 1
    return n


was, now = stable(old_rows), stable(new_rows)
print("\nlignes gardant le meme leader aux trois niveaux : {} -> {} sur 18".format(was, now))

body = new[new.index("\\begin{tabular}"):new.index("\\end{tabular}") + len("\\end{tabular}")]
oldbody = old[old.index("\\begin{tabular}"):old.index("\\end{tabular}") + len("\\end{tabular}")]
s = s.replace(oldbody, body, 1)

WORD = {11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "all eighteen"}
m = re.search(r"(\w+) of the eighteen rows keep the same leader", s)
assert m, "la phrase de comptage est introuvable"
if m.group(1) != WORD[now]:
    s = s.replace(m.group(0),
                  "{} of the eighteen rows keep the same leader".format(WORD[now]), 1)
    print("phrase du corps : '{}' -> '{}'".format(m.group(1), WORD[now]))
else:
    print("phrase du corps inchangee : {}".format(m.group(1)))

assert s.count("{") == s.count("}")
assert s.rstrip().endswith("\\end{document}")
nocom = re.sub(r"(?m)^\s*%.*$", "", s)
labels = set(re.findall(r"\\label\{([^}]*)\}", nocom))
used = (set(re.findall(r"\\hyperref\[([^\]]*)\]", nocom))
        | set(re.findall(r"\\ref\{([^}]*)\}", nocom)))
assert not (used - labels), sorted(used - labels)
shutil.copy(tex, P / "backups_j90" / f"manuscript_before_s7j90_{datetime.now():%H%M%S}.tex")
tex.write_text(s, encoding="utf-8")
print("\nTable S7 remplacee")
