"""Supplementary Table S9 Panel B on the standardised pipeline.

Panel A is untouched: it compares the sentinel with leaving the cell missing, for the three
tree models only, which are invariant to a per-column affine map.

Panel B is rebuilt from the two arms just run. The clean baseline is the standardised one
for LASSO and the MLP, and the published one for the trees, which standardisation leaves
alone -- measured at a mean difference of +7e-5 on 500 replicates, not assumed.
"""
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
H = Path(__file__).resolve().parent
MODELS = ["LASSO", "MLP1", "DecisionTree", "RandomForest", "XGBoost"]
PRETTY = {"LASSO": "LASSO", "MLP1": "MLP", "DecisionTree": "Decision tree",
          "RandomForest": "Random forest", "XGBoost": "XGBoost"}
DIMS = ["Completeness", "Validity"]

arms = {}
for enc in ("sentinel", "median"):
    d = pd.read_csv(H / f"s9std_{enc}.csv")
    d = d[(d.p.round(2) == 0.3) & d.model.isin(MODELS)]
    for dim in DIMS:
        g = d[d.dimension == dim]
        assert g.groupby("model").bootstrap.nunique().eq(500).all(), (enc, dim)
    arms[enc] = d
print("deux bras charges, 500 replicats par modele et dimension")

clean = pd.read_csv(H / "clean.csv")
clean = clean[(clean.arm == "scaled") & clean.model.isin(["LASSO", "MLP1"])][["model", "bootstrap", "auc", "brier"]]
pub = pd.read_csv(P / "consolidated_binary_outcome.csv")
pub = pub[(pub.scenario == "S2_test_only") & (pub.N == 3500)
          & (pub.pollution.round(2) == 0.0)
          & pub.model.isin(["DecisionTree", "RandomForest", "XGBoost"])]
pub = pub.drop_duplicates(subset=["model", "bootstrap"])[["model", "bootstrap", "auc", "brier"]]
base = pd.concat([clean, pub], ignore_index=True).set_index(["model", "bootstrap"])
for m in MODELS:
    assert len(base.loc[m]) == 500, m


def loss(enc, dim, model, metric):
    d = arms[enc]
    x = d[(d.dimension == dim) & (d.model == model)].set_index("bootstrap")[metric]
    c = base.loc[model, metric]
    k = x.index.intersection(c.index)
    assert len(k) == 500
    return (c.loc[k] - x.loc[k]) if metric == "auc" else (x.loc[k] - c.loc[k])


def holm(ps):
    o, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


cells, order, ps = {}, [], []
for m in MODELS:
    for metric in ("auc", "brier"):
        for dim in DIMS:
            s = loss("sentinel", dim, m, metric)
            a = loss("median", dim, m, metric)
            d = (a - s).to_numpy()
            cells[(m, metric, dim)] = dict(ref=float(np.median(s)), alt=float(np.median(a)),
                                           diff=float(np.median(d)),
                                           lo=float(np.percentile(d, 2.5)),
                                           hi=float(np.percentile(d, 97.5)))
            order.append((m, metric, dim))
            ps.append(wilcoxon(d).pvalue)
for k, p in zip(order, holm(np.array(ps))):
    cells[k]["sep"] = bool(p < 0.05 and (cells[k]["lo"] > 0 or cells[k]["hi"] < 0))


def f(v):
    return "${:.3f}$".format(v) if v >= 0 else "$-{:.3f}$".format(abs(v))


def sg(v):
    return "${:+.4f}$".format(v) if 0 < abs(v) < 0.0005 else "${:+.3f}$".format(v)


rows = []
for m in MODELS:
    for metric, mn in (("auc", "AUROC"), ("brier", "Brier")):
        out = []
        for dim in DIMS:
            c = cells[(m, metric, dim)]
            star = "$^{*}$" if c["sep"] else ""
            out += [f(c["ref"]), f(c["alt"]),
                    "{} [{},\\,{}]{}".format(sg(c["diff"]), sg(c["lo"]), sg(c["hi"]), star)]
        rows.append("    {} & {} & ".format(PRETTY[m] if metric == "auc" else "", mn)
                    + " & ".join(out) + r" \\")
body = "\n".join(rows)

tex = P / "manuscript.tex"
shutil.copy(tex, P / f"manuscript_backup_before_s9num_{datetime.now():%Y%m%d_%H%M}.tex")
s = tex.read_text(encoding="utf-8")
i = s.index(r"\textit{Panel B --- median imputation")
j = s.index(r"\end{tabular}", i)
blk = s[i:j]
a = blk.index(r"\midrule") + len(r"\midrule")
b = blk.rindex(r"\bottomrule")
s = s[:i] + blk[:a] + "\n" + body + "\n    " + blk[b:] + s[j:]

nsep = sum(c["sep"] for c in cells.values())
print(f"Panel B reecrit : {nsep} / 20 comparaisons separables")
for m in MODELS:
    c = cells[(m, "auc", "Completeness")]
    print(f"  {PRETTY[m]:14s} AUROC completeness : {c['ref']:.3f} -> {c['alt']:.3f}"
          f"  ({c['diff']:+.3f}){'*' if c['sep'] else ''}")

assert s.count("{") == s.count("}")
assert s.rstrip().endswith("\\end{document}")
tex.write_text(s, encoding="utf-8")
print("\nPanel A inchange ; Panel B sur le pipeline standardise")
