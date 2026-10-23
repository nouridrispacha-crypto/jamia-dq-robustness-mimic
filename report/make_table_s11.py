# -*- coding: utf-8 -*-
"""Supplementary Table S11: the preprocessing sensitivity, model by model and dimension
by dimension.

The original pipeline fits the hot-deck donor pools, the one-hot encoder and the
standardisation constants once per cohort, before the bootstrap loop, so the out-of-bag
predictor distribution takes part in building the preprocessing that is then applied to
it. runner_noleak.py repeats Scenario S2 at c = 0.3 with all three refitted inside every
replicate on the in-bag sample alone, on the same cohort, splits and seeds, so the two
arms are paired replicate by replicate.

Means are reported because the sentence in the Results is about how far the reported
values move, and the main-text figures report means.

The verdicts are recomputed on each arm by the rule of Table 3 and compared at the
granularity Table 3 uses, which is the model family and not the single model: a
degradation is named when the rule holds for every model, or for every model of one
family. Comparing per model instead would print disagreements that Table 3 never claims
either way. They are reported in the prose rather than as marks in the table, because a
mark per metric per row cannot say which family a verdict belongs to.

Both metrics are handled, with opposite loss directions: for AUROC higher is better, so
the loss is minus the metric; for the Brier score higher is worse, so the loss is the
metric itself. The clean reference cancels in a paired difference between two
degradations, so the verdicts follow from the raw metric without a clean arm.

Writes table_s11.tex, to be pasted into the supplement.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PAPER
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PAPER = Path(PAPER)
SRC = PAPER / "rerun_standardisation" / "noleak_compare.csv"

DIMS = ["completeness", "coherence", "validity", "precision", "uniqueness"]
LABEL = {d: d.capitalize() for d in DIMS}
MODELS = ["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"]
MNAME = {"LASSO": "LASSO", "DecisionTree": "Decision tree", "RandomForest": "Random forest",
         "XGBoost": "XGBoost", "MLP1": "MLP"}
TREES = ["DecisionTree", "RandomForest", "XGBoost"]
LINNEU = ["LASSO", "MLP1"]


def holm(ps):
    order = np.argsort(ps)
    out, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        out[i] = min(1.0, run)
    return out


def verdicts(d, col, higher_is_worse):
    """most/least damaging per model, by the rule of Table 3, from the raw metric.

    higher_is_worse selects the loss direction: the loss is the metric itself when a
    larger value is worse (Brier score), and minus the metric otherwise (AUROC).
    """
    s = 1.0 if higher_is_worse else -1.0
    res = {}
    for m, sub in d.groupby("model"):
        piv = sub.pivot_table(index="b", columns="dim", values=col)[DIMS]
        rows = []
        for i in range(len(DIMS)):
            for j in range(i + 1, len(DIMS)):
                a, b = DIMS[i], DIMS[j]
                diff = (s * (piv[a] - piv[b])).dropna()      # perte(a) - perte(b)
                _, p = wilcoxon(diff)
                rows.append({"a": a, "b": b, "med": diff.median(),
                             "lo": diff.quantile(0.025), "hi": diff.quantile(0.975), "p": p})
        t = pd.DataFrame(rows)
        t["holm"] = holm(t["p"].to_numpy())
        t["sep"] = (t["holm"] < 0.05) & ((t["lo"] > 0) | (t["hi"] < 0))
        wins = dict.fromkeys(DIMS, 0)
        loses = dict.fromkeys(DIMS, 0)
        for _, r in t[t["sep"]].iterrows():
            hi, lo = (r["a"], r["b"]) if r["med"] > 0 else (r["b"], r["a"])
            wins[hi] += 1
            loses[lo] += 1
        others = {x: [y for y in DIMS if y != x] for x in DIMS}
        gap = {x: float((s * (piv[x] - piv[others[x]].mean(axis=1))).median()) for x in DIMS}
        res[m] = ({x for x in DIMS if wins[x] > 0 and loses[x] == 0 and gap[x] > 0},
                  {x for x in DIMS if loses[x] > 0 and wins[x] == 0 and gap[x] < 0})
    return res


def families(res, idx):
    """Table 3's granularity: '(all)' when every model qualifies, else the family."""
    out = []
    allm = set.intersection(*[res[m][idx] for m in MODELS])
    for dim in sorted(allm):
        out.append((dim, "all"))
    for name, grp in (("trees", TREES), ("linear and neural", LINNEU)):
        for dim in sorted(set.intersection(*[res[m][idx] for m in grp]) - allm):
            out.append((dim, name))
    return out


def phrase(items):
    if not items:
        return "--"
    return "; ".join("{} ({})".format(LABEL[d], f) for d, f in items)


d = pd.read_csv(SRC)
d["dim"] = d["dim"].str.lower()
assert d["b"].nunique() == 500, d["b"].nunique()
assert set(d["model"]) == set(MODELS), sorted(set(d["model"]))
assert set(d["dim"]) == set(DIMS), sorted(set(d["dim"]))

mean = d.groupby(["model", "dim"])[["auc_pub", "auc_nl", "brier_pub", "brier_nl"]].mean()

# ---- le tableau numerique
body, worst_a, worst_b = [], 0.0, 0.0
for m in MODELS:
    for k, dim in enumerate(DIMS):
        r = mean.loc[(m, dim)]
        da, db = r.auc_nl - r.auc_pub, r.brier_nl - r.brier_pub
        worst_a, worst_b = max(worst_a, abs(da)), max(worst_b, abs(db))
        body.append("    {} & {} & {:.3f} & {:.3f} & {:+.3f} & {:.3f} & {:.3f} & {:+.3f} \\\\".format(
            MNAME[m] if k == 0 else "", LABEL[dim],
            r.auc_pub, r.auc_nl, da, r.brier_pub, r.brier_nl, db))
    if m != MODELS[-1]:
        body.append("    \\addlinespace[2pt]")

# ---- le classement des cinq degradations, modele par modele
#
# Un classement par moyenne peut s'inverser sur un ecart de 0,0001, ce qui ne dit rien. On
# retient donc, pour chaque inversion, l'ecart entre les deux degradations echangees sous
# chaque bras : c'est ce qui separe une reorganisation reelle de deux degradations qui
# n'etaient de toute facon pas distinguables.
SWAP_TIE = 0.002

rank_same, swaps = {}, {}
for metric, worse in (("auc", False), ("brier", True)):
    ok, found = True, []
    for m in MODELS:
        s = mean.loc[m]
        a = s["{}_pub".format(metric)].sort_values(ascending=worse)
        b = s["{}_nl".format(metric)].sort_values(ascending=worse)
        if list(a.index) == list(b.index):
            continue
        ok = False
        for i in range(len(DIMS) - 1):
            x, y = a.index[i], a.index[i + 1]
            if list(b.index).index(x) > list(b.index).index(y):
                found.append({"model": m, "lost": x, "won": y,
                              "rank": i, "n": len(DIMS),
                              "gap_pub": abs(a[x] - a[y]), "gap_nl": abs(b[x] - b[y])})
    rank_same[metric], swaps[metric] = ok, found

# ---- les verdicts, par famille, sur chaque bras
lines, changed = [], []
for metric, worse, mlabel in (("auc", False, "AUROC"), ("brier", True, "Brier score")):
    p = verdicts(d, "{}_pub".format(metric), worse)
    n = verdicts(d, "{}_nl".format(metric), worse)
    for idx, kind in ((0, "Most"), (1, "Least")):
        fp, fn = families(p, idx), families(n, idx)
        same = fp == fn
        if not same:
            changed.append((mlabel, kind.lower(), phrase(fp), phrase(fn)))
        lines.append("    {} & {} & {} & {} & {} \\\\".format(
            mlabel, kind, phrase(fp), phrase(fn), "identical" if same else "\\textbf{differs}"))

assert rank_same["auc"], "le classement AUROC bouge, la phrase des Resultats est fausse"
assert worst_a <= 0.010 + 5e-4, worst_a

# ---- la phrase sur les classements, tiree des donnees et non ecrite a la main
#
# Le classement AUROC ne bouge pour aucun modele. Celui du score de Brier bouge pour
# plusieurs, et il faut le dire : une seule de ces inversions porte sur la degradation la
# moins penalisante et depasse le seuil de quasi-egalite, les autres echangent deux
# degradations voisines au milieu du classement.
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _place(s):
    """Where in the ranking the inversion sits. Rank 0 is the least damaging end."""
    if s["rank"] == 0:
        return "the least damaging end"
    if s["rank"] == s["n"] - 2:
        return "the most damaging end"
    return "the middle of the ranking"


def _rank_sentence():
    if rank_same["brier"]:
        return ("Every model ranks the five degradations in the same order under both "
                "pipelines, in both metrics.")
    sw = swaps["brier"]
    real = [s for s in sw if max(s["gap_pub"], s["gap_nl"]) > SWAP_TIE]
    minor = [s for s in sw if s not in real]
    out = ["Every model ranks the five degradations in the same order under both pipelines "
           "in AUROC. In the Brier score {} of the five do not.".format(
               WORD[len({s["model"] for s in sw})])]
    for s in real:
        out.append("For {} the degradation at {} changes, {} giving way to {}, a mean Brier "
                   "score ${:.3f}$ apart in the primary analysis and ${:.3f}$ apart when refitted; this "
                   "is the one verdict change recorded below.".format(
                       MNAME[s["model"]], _place(s), s["lost"], s["won"],
                       s["gap_pub"], s["gap_nl"]))
    if minor:
        # Les ecarts minimes tiennent dans une seule phrase, bornee par le plus grand
        # d'entre eux et arrondie vers le haut pour que la borne enoncee reste vraie.
        clauses = ["for {} {} and {} exchange places at {}".format(
            MNAME[s["model"]], s["lost"], s["won"], _place(s)) for s in minor]
        worst = max(max(s["gap_pub"], s["gap_nl"]) for s in minor)
        bound = np.ceil(worst * 1000) / 1000
        joined = clauses[0] if len(clauses) == 1 else \
            ", and ".join([", ".join(clauses[:-1]), clauses[-1]])
        out.append("Otherwise {}, in {} less than ${:.3f}$ apart under either pipeline, so "
                   "neither order is distinguishable from the other.".format(
                       joined, "both cases" if len(minor) > 1 else "that case", bound))
    return " ".join(out)


rank_note = _rank_sentence()

if changed:
    mlabel, kind, was, now = changed[0]
    note = ("One verdict does not hold: among the {} damaging degradations in the {}, {} "
            "under the original pipeline and {} when the preprocessing is refitted in-bag "
            "only. Every most-damaging verdict is unchanged, in both metrics and both "
            "families.".format(kind, mlabel, was.lower(), now.lower()))
else:
    note = "Every verdict is unchanged, in both metrics and both families."
assert len(changed) == 1, [c[:2] for c in changed]

tex = r"""\phantomsection\label{supp:tabS11}
\subsection*{Supplementary Table S11. Preprocessing fitted on the in-bag sample alone}

The original pipeline builds the sex-conditioned donor pools, the one-hot encoder and the
standardisation constants once per cohort, before the bootstrap loop, and applies them to
both the in-bag and the out-of-bag set; the out-of-bag predictor distribution therefore
takes part in building the preprocessing applied to it. This analysis repeats Scenario S2
at $c=0.3$ with all three refitted inside every replicate on the in-bag sample alone. The
cohort, the stratified splits, the corruption seeds and the model constructors are those of
the main analysis, so the two pipelines are paired replicate by replicate and the only
difference between them is where the preprocessing is estimated.

Entries are means over the 500 replicates. Across the twenty-five cells the mean AUROC
moves by at most WORST_A and the mean Brier score by at most WORST_B. RANKNOTE The second
table recomputes the verdicts of Table~\ref{tab:contrasts} on each arm, at the granularity
Table~\ref{tab:contrasts} uses: a degradation is named when the rule holds for every model
(all), or for every model of one family. NOTE

\begin{table}[htbp]
\caption*{Supplementary Table S11. Scenario S2 at $c=0.3$ ($N=3\,500$, 500 bootstrap
replicates) with the preprocessing fitted once on the cohort, as in the primary
analysis, and refitted
on the in-bag sample alone. Entries are means; the difference is the in-bag-only arm minus
the original one.}
\label{tab:noleak}
\centering
\singlespacing
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{>{\raggedright\arraybackslash}p{2.2cm}
                >{\raggedright\arraybackslash}p{2.2cm}
                ccc|ccc}
    \toprule
    & & \multicolumn{3}{c|}{AUROC} & \multicolumn{3}{c}{Brier score} \\
    Model & Dimension & Original & In-bag & Difference & Original & In-bag & Difference \\
    \midrule
BODY
    \bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption*{Supplementary Table S11 (continued). Most and least damaging degradations for the
binary outcome under test-time degradation, by the rule of Table~\ref{tab:contrasts},
recomputed on each arm.}
\label{tab:noleakverdicts}
\centering
\singlespacing
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{>{\raggedright\arraybackslash}p{1.7cm}
                >{\raggedright\arraybackslash}p{1.0cm}
                >{\raggedright\arraybackslash}p{4.6cm}
                >{\raggedright\arraybackslash}p{4.6cm}
                >{\raggedright\arraybackslash}p{1.6cm}}
    \toprule
    Metric & Rule & Original & In-bag only & Comparison \\
    \midrule
VERDICTS
    \bottomrule
\end{tabular}
\end{table}
"""
tex = (tex.replace("BODY", "\n".join(body))
          .replace("VERDICTS", "\n".join(lines))
          .replace("WORST_A", "${:.3f}$".format(worst_a))
          .replace("WORST_B", "${:.3f}$".format(worst_b))
          .replace("RANKNOTE", rank_note)
          .replace("NOTE", note))
out = PAPER / "table_s11.tex"
out.write_text(tex, encoding="utf-8")

print("ecrit : {}".format(out.name))
print("ecart maximal | AUROC {:.4f} | Brier {:.4f}".format(worst_a, worst_b))
print("classement des degradations inchange : AUROC {} | Brier {}".format(
    rank_same["auc"], rank_same["brier"]))
for s in swaps["brier"]:
    print("  Brier, {:14s} rang {} : {} <-> {} (ecart publie {:.5f}, in-bag {:.5f})".format(
        s["model"], s["rank"], s["lost"], s["won"], s["gap_pub"], s["gap_nl"]))
print("\nverdicts par famille :")
for l in lines:
    print("  " + l.strip().rstrip("\\").replace(" & ", " | "))
print("\nnote :\n{}".format(note))
