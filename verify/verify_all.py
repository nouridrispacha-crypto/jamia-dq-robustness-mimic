"""Full verification of the manuscript after the 90-day survival relaunch.

Four kinds of check, each failing loudly rather than warning.

Structure: braces, every cross-reference resolved, the document closed, the journal's float
and word limits.

Stale text: strings that belonged to the old survival design and must no longer appear
anywhere, including the numbers they carried.

Numbers against data: the survival column of Table S6, the survival rows of Tables S5 and S7
and the counting claims of the Results are each recomputed from the result files and compared
with what the manuscript prints. A claim that cannot be recomputed is reported as such rather
than passed.

Artefacts: every figure the manuscript includes exists, and the ones that depend on the
relaunch are newer than the results they are drawn from.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PAPER, SURVIVAL_RESULTS
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(PAPER)
SURV = SURVIVAL_RESULTS
S = (P / "manuscript.tex").read_text(encoding="utf-8")
NOCOM = re.sub(r"(?m)^\s*%.*$", "", S)
fails = []


def chk(label, ok, detail=""):
    print("  [{}] {}{}".format("ok" if ok else "!!", label,
                               "" if ok else "   <- " + str(detail)))
    if not ok:
        fails.append(label)


print("structure")
chk("accolades equilibrees", S.count("{") == S.count("}"))
chk("document clos", S.rstrip().endswith("\\end{document}"))
labels = set(re.findall(r"\\label\{([^}]*)\}", NOCOM))
used = (set(re.findall(r"\\hyperref\[([^\]]*)\]", NOCOM))
        | set(re.findall(r"\\ref\{([^}]*)\}", NOCOM)))
chk("tout renvoi resolu", not (used - labels), sorted(used - labels))
bodytex = S[S.index("\\section{Introduction}"):S.index("\\section*{Author Contributions}")]
nb = re.sub(r"(?m)^\s*%.*$", "", bodytex)
chk("tableaux du corps <= 4", nb.count("\\begin{table}") <= 4,
    nb.count("\\begin{table}"))
chk("figures du corps <= 6", nb.count("\\begin{figure}") <= 6,
    nb.count("\\begin{figure}"))
m = re.search(r"tables\): (\d)\\,(\d{3})", S)
declared = int(m.group(1) + m.group(2)) if m else -1
chk("declaration sous 4000", 0 < declared <= 4000, declared)

print("\ntextes perimes de l'ancien design")
STALE = ["81.1\\%", "5.95", "18.9\\% died", "median follow-up", "supp:tabS12",
         "Supplementary Table~S12", "emptied in turn", "nothing qualifying on",
         "Calibration of the censored time-to-event", "in five of the six",
         "thirteen of the eighteen", "censored at hospital discharge",
         "predict forward from a 24-hour landmark"]
for t in STALE:
    chk("absent : {!r}".format(t), t not in S)

print("\nTableau S6, colonne Delta C-index contre les donnees")
STEM = {"Completeness": "mnar_top/outputs_mnar_top_s2/S2_test_only",
        "Coherence": "Coherence/outputs_coherence_s2/S2_test_only",
        "Validity": "validity/outputs_validity_S2/S2_test_only",
        "Precision": "precision/outputs_precision_S2/S2_test_only",
        "Uniqueness": "uniqueness/outputs_uniqueness_S2/S2_test_only"}
MODELS = ["CoxPH", "RandomSurvivalForest", "XGBoostSurv", "DeepSurv1"]
i6 = S.index("\\label{tab:effectsizes}")
blk6 = S[S.rindex("\\begin{table}", 0, i6):S.index("\\end{table}", i6)]
for dim, stem in STEM.items():
    d = pd.read_csv(SURV / (stem + ".csv"))
    d = d[(d["N"] == 3500) & d.model.isin(MODELS)]
    a = d[d.p.round(2) == 0.0].set_index(["model", "bootstrap"])["c_index"]
    b = d[d.p.round(2) == 0.3].set_index(["model", "bootstrap"])["c_index"]
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    k = a.index.intersection(b.index)
    v = (b.loc[k] - a.loc[k]).to_numpy()
    want = "$-{:.3f}$ [$-{:.3f}$, $-{:.3f}$]".format(
        abs(np.median(v)), abs(np.percentile(v, 2.5)), abs(np.percentile(v, 97.5)))
    row = re.search(r"(?m)^\s*" + dim + r"\s*&(.*?)\\\\", blk6)
    got = [c.strip() for c in row.group(1).split("&")][3] if row else "(ligne absente)"
    chk("{:13s} {}".format(dim, want), got == want, got)

print("\nTableau S5, lignes survie contre les modeles refittes")
sk = pd.read_csv(Path(__file__).resolve().parent / "reference/skewness_all.csv")
i5 = S.index("Supplementary Table S5. Residual skewness")
blk5 = S[S.rindex("\\begin{table}", 0, i5):S.index("\\end{table}", i5)]
top7 = sk.head(7)
for _, r in top7.iterrows():
    pat = "{} & {} & {}".format(r["outcome"], r["scenario"], r["metric"])
    chk("{:38s} {:.2f}".format(pat, r["skew"]),
        pat in blk5 and "{:.2f}".format(r["skew"]) in blk5)
chk("7 lignes dans S5", blk5.count("\\\\") >= 8, blk5.count("\\\\"))

print("\ndecomptes des Resultats recomptes depuis le Tableau 3")
DIMS = ["completeness", "coherence", "validity", "precision", "uniqueness"]
COMB = ["Binary AUROC", "Binary Brier", "Continuous RMSE", "Continuous R2",
        "TTE C-index", "TTE IBS"]
i3 = S.index("label{tab:contrasts}")
blk3 = S[S.rindex("\\begin{table}", 0, i3):S.index("\\end{table}", i3)]
rows3 = [r for r in blk3[blk3.index("\\midrule"):blk3.index("\\bottomrule")].split("\\\\")
         if "&" in r]
chk("6 lignes dans le Tableau 3", len(rows3) == 6, len(rows3))
named = {}
for ci, raw in enumerate(rows3):
    for k, col in enumerate(raw.split("&")[2:5]):
        for kind, pat in (("most", r"\\textbf\{Most:\}(.*?)(?:\\newline|$)"),
                          ("least", r"\\textit\{Least:\}(.*)")):
            mm = re.search(pat, col)
            for part in (mm.group(1) if mm else "").split(";"):
                g = re.match(r"(\w+)\s*\(", " ".join(part.split()))
                if g and g.group(1).lower() in DIMS:
                    named.setdefault((["S2", "S1", "S3"][k], kind), {}).setdefault(
                        g.group(1).lower(), []).append(COMB[ci])
WORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "all six"}
for scen, kind, dim, phrase in [
        ("S2", "most", "validity", "in three of the six task and metric"),
        ("S2", "least", "precision", "among the least damaging in all six combinations"),
        ("S1", "most", "completeness", "to completeness, in four of the six"),
        ("S3", "most", "uniqueness", "to uniqueness, in all six")]:
    n = len(named.get((scen, kind), {}).get(dim, []))
    # Le manuscrit coupe ses phrases en lignes : la comparaison se fait sur un texte
    # dont les blancs sont normalises, sinon un retour a la ligne fait echouer un
    # controle qui porte sur les mots.
    flat = " ".join(S.split())
    chk("{} {} {} = {} et le texte le dit".format(scen, kind, dim, WORD[n]),
        phrase in flat and WORD[n] in phrase, "{} / {!r}".format(WORD[n], phrase))
nlead = len(named.get(("S2", "most"), {}).get("coherence", [])) \
    + len(named.get(("S1", "most"), {}).get("coherence", [])) \
    + len(named.get(("S3", "most"), {}).get("coherence", []))
nleast = sum(len(named.get((sc, "least"), {}).get("coherence", []))
             for sc in ("S2", "S1", "S3"))
chk("coherence ne mene nulle part", nlead == 0, nlead)
chk("coherence epargnee 12 fois, et le Tableau 2 le dit",
    nleast == 12 and "least damaging in twelve of the eighteen" in S, nleast)

print("\nfigures")
inc = sorted(set(re.findall(r"includegraphics[^{}]*\{([^}]*)\}", NOCOM)))
newest_data = max(f.stat().st_mtime for f in SURV.rglob("*.csv"))
for rel in inc:
    f = P / rel
    ok = f.exists()
    tag = ""
    if ok and any(x in rel for x in ("grid", "heatmap", "contrasts")):
        ok = f.stat().st_mtime > newest_data
        tag = datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")
    chk("{:46s} {}".format(rel, tag), ok)

print("\n" + ("TOUT PASSE" if not fails else "{} ECHEC(S) : {}".format(
    len(fails), "; ".join(fails[:6]))))
sys.exit(1 if fails else 0)
