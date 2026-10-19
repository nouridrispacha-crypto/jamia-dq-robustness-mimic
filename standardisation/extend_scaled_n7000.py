"""Extend the substituted inputs to the N = 7000 cohort.

Same operation as extend_scaled_n1500.py, for Supplementary Figure 3. Only the LASSO and
MLP1 rows are replaced; the three tree models are invariant to a per-column affine map and
are left alone.

The sweep died twice, so its output is spread over several files: S2 over three (the three
dimensions that had finished, then uniqueness at 0.3, then coherence) and S3 over two (the
cell salvaged from the run killed on 14 October, then the resumed cells). They are
concatenated and de-duplicated on (dimension, level, replicate, model); the overlap comes
from cells that were restarted in full and must agree, which is checked before anything is
written.

f1_macro is not recorded by runner_full, so the replaced rows carry none. Supplementary
Figure 3 is an AUROC figure, so this does not affect it, but it does mean N = 7000 must
stay out of any F1 figure.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
IN = HERE / "scaled_inputs"
SWAP = ["LASSO", "MLP1"]
ALL_MODELS = ["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"]
LEVELS = [0.0, 0.1, 0.2, 0.3]
N = 7000
KEY = {"completeness": "Completeness", "coherence": "Coherence", "validity": "Validity",
       "precision": "Precision", "uniqueness": "Uniqueness"}
JOIN = ["dimension", "p", "bootstrap", "model"]

GROUPS = {
    "S1": ["scaled_N7000_S1.csv"],
    "S2": ["scaled_N7000_S2.csv", "scaled_N7000_S2_uniq03.csv", "scaled_N7000_S2_coh.csv"],
    "S3": ["scaled_N7000_S3_sauve_1cellule.csv", "scaled_N7000_S3_resume.csv"],
}

corr = {}
for scen, files in GROUPS.items():
    d = pd.concat([pd.read_csv(HERE / f) for f in files], ignore_index=True)
    if "arm" in d.columns:
        d = d[d["arm"] == "scaled"]
    d = d[d["model"].isin(SWAP)]
    # les cellules relancees en entier doivent redonner les memes valeurs
    dup = d[d.duplicated(subset=JOIN, keep=False)]
    if len(dup):
        g = dup.groupby(JOIN)["auc"].agg(["min", "max"])
        gap = (g["max"] - g["min"]).max()
        assert gap < 1e-12, (scen, gap)
        print("{} : {} replicats refaits, ecart auc max {:.1e}".format(scen, len(g), gap))
    d = d.drop_duplicates(subset=JOIN, keep="first")
    assert len(d) == 5 * 3 * 500 * 2, (scen, len(d))
    corr[scen] = d

clean = pd.read_csv(HERE / "scaled_clean_N{}.csv".format(N))
if "arm" in clean.columns:
    clean = clean[clean["arm"] == "scaled"]
clean = clean[clean["model"].isin(SWAP)].drop_duplicates(subset=["model", "bootstrap"])
assert len(clean) == 2 * 500, len(clean)
print("bras standardise N={} : {} lignes corrompues, {} propres\n".format(
    N, sum(len(v) for v in corr.values()), len(clean)))

stamp = datetime.now().strftime("%Y%m%d_%H%M")
for dim in KEY:
    for scen in ("S1", "S2", "S3"):
        path = IN / dim / "{}.csv".format(scen)
        shutil.copy(path, path.with_suffix(".csv.bak_n7000_{}".format(stamp)))
        pub = pd.read_csv(path)
        lvl = "pollution" if "pollution" in pub.columns else "p"

        drop = (pub.model.isin(SWAP) & (pub["N"] == N)
                & pub[lvl].round(2).isin(LEVELS))
        kept = pub[~drop].copy()

        new = corr[scen][corr[scen].dimension == KEY[dim]][
            ["model", "bootstrap", "p", "auc", "brier"]].rename(columns={"p": lvl})
        cl = clean[["model", "bootstrap", "auc", "brier"]].copy()
        cl[lvl] = 0.0
        rep = pd.concat([cl, new], ignore_index=True)
        rep["N"] = N
        for c in pub.columns:
            if c not in rep.columns:
                rep[c] = pub[c].iloc[0] if c in ("outcome_type", "dimension", "scenario") \
                    else np.nan
        rep = rep[pub.columns]
        out = pd.concat([kept, rep], ignore_index=True)

        chk = out[(out["N"] == N) & out.model.isin(ALL_MODELS)]
        for m in ALL_MODELS:
            got = sorted(chk[chk.model == m][lvl].round(2).unique())
            assert all(l in got for l in LEVELS), (dim, scen, m, got)
            for l in LEVELS:
                k = chk[(chk.model == m) & (chk[lvl].round(2) == l)].bootstrap.nunique()
                assert k == 500, (dim, scen, m, l, k)
        assert len(out[(out["N"] == N) & out.model.isin(SWAP)
                       & out[lvl].round(2).isin(LEVELS)]) == 2 * 4 * 500, (dim, scen)
        out.to_csv(path, index=False)
    print("  {} fait".format(dim))

# la substitution doit avoir bouge les chiffres, et seulement pour les deux modeles
s = pd.read_csv(IN / "completeness" / "S2.csv")
old = pd.read_csv(IN / "completeness" / "S2.csv.bak_n7000_{}".format(stamp))
print()
for m in SWAP + ["RandomForest"]:
    a = s[(s.N == N) & (s.model == m) & (s.pollution.round(2) == 0.3)].auc.mean()
    b = old[(old.N == N) & (old.model == m) & (old.pollution.round(2) == 0.3)].auc.mean()
    print("  {:14s} {:.5f} -> {:.5f}  {}".format(
        m, b, a, "remplace" if abs(a - b) > 1e-9 else "inchange (attendu)"))
