"""Supplementary Table S7 recomputed on the standardised binary arm.

No new simulation: make_table_s6 reads through V2.build_loss_table, so repointing
BINARY.file_map at the substituted inputs is enough. Task labels are brought onto the
outcome-type naming Table 3 now uses, and the claim in the lead-in text about how many
rows keep their leader is re-counted rather than assumed.
"""
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
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
print("BINARY.file_map repointe sur le bras standardise")

import make_table_s6 as S7

S7.ROWS = [("Binary", V2.BINARY, [("AUROC", "auc"), ("Brier score", "brier")]),
           ("Continuous", V2.CONTINUOUS, [("RMSE", "rmse"),
                                          ("R\\textsuperscript{2}", "r2")]),
           ("Time-to-event", V2.SURVIVAL, [("C-index", "c_index"), ("IBS", "ibs")])]
new = S7.latex()
assert "Binary & AUROC" in new and "Mortality" not in new, "le renommage n'a pas pris"

# ── how many of the eighteen rows keep the same leader at all three levels?
rows = [r for r in new.split("\\\\") if r.count("&") == 5 and "Metric" not in r]
assert len(rows) == 18, len(rows)
same = 0
for r in rows:
    cells = [c.strip() for c in r.split("&")]
    names = [re.sub(r"\s*\(\d+\)", "", c).strip() for c in cells[3:6]]
    if names[0] == names[1] == names[2] and names[0]:
        same += 1
print(f"lignes gardant le meme leader aux trois niveaux : {same} / 18")

# ── splice into the manuscript
tex = P / "manuscript.tex"
shutil.copy(tex, P / f"manuscript_backup_before_s7_{datetime.now():%Y%m%d_%H%M}.tex")
s = tex.read_text(encoding="utf-8")
i = s.index("\\label{tab:lowc}")
a = s.rindex("\\begin{table}", 0, i)
b = s.index("\\end{table}", i) + len("\\end{table}")
old = s[a:b]
assert "Mortality & AUROC" in old
# keep the manuscript's own S7 caption number, which the generator still calls S6
body = new[new.index("\\begin{tabular}"):new.index("\\end{tabular}") + len("\\end{tabular}")]
oldbody = old[old.index("\\begin{tabular}"):old.index("\\end{tabular}") + len("\\end{tabular}")]
s = s.replace(oldbody, body, 1)

# ── the lead-in sentence states the count; make it match what was just recomputed
WORD = {14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "all eighteen",
        13: "thirteen", 12: "twelve"}
m = re.search(r"(\w+) of the eighteen rows keep the same leader", s)
assert m, "la phrase de comptage est introuvable"
if m.group(1) != WORD[same]:
    s = s.replace(m.group(0), f"{WORD[same]} of the eighteen rows keep the same leader", 1)
    print(f"phrase du corps : '{m.group(1)}' -> '{WORD[same]}'")
else:
    print(f"phrase du corps inchangee : {m.group(1)}")

t7 = s[s.index("\\label{tab:lowc}"):]
t7 = t7[:t7.index("\\end{table}")]
for gone in ("Mortality &", "Length of stay &", "Time to death &"):
    assert gone not in t7, gone
assert s.count("{") == s.count("}")
assert s.rstrip().endswith("\\end{document}")
nocom = re.sub(r"(?m)^\s*%.*$", "", s)
labels = set(re.findall(r"\\label\{([^}]*)\}", nocom))
assert not (set(re.findall(r"\\ref\{([^}]*)\}", nocom)) - labels)
tex.write_text(s, encoding="utf-8")
print("\nTable S7 remplacee, etiquettes par type")
print("\n--- lignes binaires ---")
for r in rows[:6]:
    print("   " + " ".join(r.split())[:110])
