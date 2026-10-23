"""Table 3 recomputed with the standardised binary arm and the 90-day survival relaunch.

Two sources differ from analyse_stats_v2's own file maps. The binary arm is the standardised
re-run under scaled_inputs, repointed at run time as regen_s7 and regen_figures do, because
the module deliberately keeps the unstandardised map so full_grid can compare the two. The
survival arm is the 90-day landmark relaunch, which the module now points at directly.

The binary cells are a regression guard, not an output: they must come back exactly as the
manuscript already has them, since nothing about the binary arm changed. If they move, this
script writes nothing and says which cell moved, because that would mean the standardised
source no longer reproduces the table it produced on 6 October.

Only the body between midrule and bottomrule is spliced. The caption, the label, the column
widths and the task labels are the manuscript's.
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
print("binaire   -> bras standardise (scaled_inputs)")
print("survie    -> {}".format(
    Path(V2.SURVIVAL.file_map["coherence"]["S2"]).parents[2].name))
print("continu   -> inchange\n")

import make_contrast_table as T3

LABELS = {"Mortality": "Binary", "Length of stay": "Continuous",
          "Time to death": "Time-to-event"}
new = T3.latex()
for a, b in LABELS.items():
    assert new.count(a) == 1, a
    new = new.replace(a, b)


def body_of(text):
    return text[text.index(r"\midrule") + len(r"\midrule"):text.index(r"\bottomrule")]


def rows_of(text):
    return [" ".join(r.replace(r"\addlinespace[3pt]", "").split())
            for r in body_of(text).split(r"\\") if "&" in r]


def cells(row):
    out = {}
    for k, col in enumerate(row.split("&")[2:5]):
        for kind, pat in (("most", r"\\textbf\{Most:\}(.*?)(?:\\newline|$)"),
                          ("least", r"\\textit\{Least:\}(.*)")):
            m = re.search(pat, col)
            txt = (m.group(1) if m else "").strip()
            out[(kind, ["S2", "S1", "S3"][k])] = frozenset(
                x.strip().rstrip(".").lower() for x in txt.split(";")
                if x.strip() and x.strip() != "--")
    return out


tex = P / "manuscript.tex"
man = tex.read_text(encoding="utf-8")
i = man.index("label{tab:contrasts}")
t0, t1 = man.rindex(r"\begin{table}", 0, i), man.index(r"\end{table}", i)
blk = man[t0:t1]

old_rows, new_rows = rows_of(blk), rows_of(new)
assert len(old_rows) == len(new_rows) == 6, (len(old_rows), len(new_rows))
LAB = ["binaire AUROC", "binaire Brier", "continu RMSE", "continu R2",
       "survie C-index", "survie IBS"]

drift = []
for k in range(4):                        # binaire et continu : garde-fou
    a, b = cells(old_rows[k]), cells(new_rows[k])
    for key in a:
        if a[key] != b[key]:
            drift.append((LAB[k], key, sorted(a[key]), sorted(b[key])))
if drift:
    print("ARRET : {} cellule(s) hors survie ont bouge, rien n'est ecrit".format(len(drift)))
    for lab, key, a, b in drift:
        print("  {} {} {}".format(lab, key[0], key[1]))
        print("    manuscrit : {}".format(", ".join(a) or "--"))
        print("    recalcule : {}".format(", ".join(b) or "--"))
    sys.exit(1)
print("garde-fou : les 24 cellules binaires et continues sont inchangees\n")

print("cellules survie qui changent")
for k in (4, 5):
    a, b = cells(old_rows[k]), cells(new_rows[k])
    for key in sorted(a, key=lambda x: (x[1], x[0])):
        if a[key] != b[key]:
            print("  {:14s} {:5s} {:3s}".format(LAB[k], key[1], key[0]))
            print("     avant : {}".format(", ".join(sorted(a[key])) or "--"))
            print("     apres : {}".format(", ".join(sorted(b[key])) or "--"))

# Seules les deux lignes survie sont transplantees. Le generateur ordonne les entrees d'une
# cellule autrement que le manuscrit, si bien que reprendre tout le corps reecrirait les
# lignes binaires et continues sans changer ce qu'elles disent.
ANCHOR = "Time-to-event & C-index"
ob, nb = body_of(blk), body_of(new)
assert ob.count(ANCHOR) == 1 and nb.count(ANCHOR) == 1, ANCHOR
old_surv, new_surv = ob[ob.index(ANCHOR):], nb[nb.index(ANCHOR):]
assert "IBS" in old_surv and "IBS" in new_surv
out = man.replace(blk, blk.replace(old_surv, new_surv, 1), 1)
assert out != man, "rien n'a change"

# Les quatre lignes non-survie doivent rester mot pour mot celles du manuscrit.
kept = rows_of(out[out.rindex(r"\begin{table}", 0, out.index("label{tab:contrasts}")):
                   out.index(r"\end{table}", out.index("label{tab:contrasts}"))])
assert len(kept) == 6, len(kept)
for k in range(4):
    assert kept[k] == old_rows[k], "la ligne {} a ete reecrite".format(LAB[k])
print("\nlignes binaires et continues : inchangees mot pour mot")
assert out.count("{") == out.count("}")
assert out.rstrip().endswith(r"\end{document}")
for gone in ("Mortality &", "Length of stay &", "Time to death &"):
    assert gone not in out, gone
nocom = re.sub(r"(?m)^\s*%.*$", "", out)
labels = set(re.findall(r"\\label\{([^}]*)\}", nocom))
used = (set(re.findall(r"\\hyperref\[([^\]]*)\]", nocom))
        | set(re.findall(r"\\ref\{([^}]*)\}", nocom)))
assert not (used - labels), sorted(used - labels)

shutil.copy(tex, P / "backups_j90" / f"manuscript_before_t3j90_{datetime.now():%H%M%S}.tex")
tex.write_text(out, encoding="utf-8")
print("\nTableau 3 remplace")
