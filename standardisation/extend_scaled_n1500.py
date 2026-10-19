"""Extend the substituted inputs to the N = 1500 cohort.

build_scaled_inputs.py replaced the LASSO and MLP rows at N = 3500 only, because that is
what had been re-run at the time. The cohort sweep has since produced the standardised
N = 1500 arm, so Supplementary Figure 2 can be brought onto the same pipeline as the rest
of the paper. The three tree models are left alone: they are invariant to a per-column
affine map.

f1_macro stays empty on the replaced rows, exactly as at N = 3500, so Supplementary
Figure 1 still must not be regenerated from these files.
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
LEVELS = [0.0, 0.1, 0.2, 0.3]
N = 1500
KEY = {"completeness": "Completeness", "coherence": "Coherence", "validity": "Validity",
       "precision": "Precision", "uniqueness": "Uniqueness"}

corr = {}
for scen in ("S1", "S2", "S3"):
    d = pd.read_csv(HERE / "scaled_N{}_{}.csv".format(N, scen))
    d = d[d.arm == "scaled"] if "arm" in d else d
    d = d[d.model.isin(SWAP)]
    assert len(d) == 5 * 3 * 500 * 2, (scen, len(d))
    corr[scen] = d
clean = pd.read_csv(HERE / "scaled_clean_N{}.csv".format(N))
clean = clean[(clean.arm == "scaled") & clean.model.isin(SWAP)]
assert len(clean) == 2 * 500, len(clean)
print("bras standardise N={} : {} lignes corrompues, {} propres".format(
    N, sum(len(v) for v in corr.values()), len(clean)))

stamp = datetime.now().strftime("%Y%m%d_%H%M")
for dim in KEY:
    for scen in ("S1", "S2", "S3"):
        path = IN / dim / "{}.csv".format(scen)
        shutil.copy(path, path.with_suffix(".csv.bak_{}".format(stamp)))
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

        chk = out[(out["N"] == N) & out.model.isin(
            ["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"])]
        for m in ["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"]:
            got = sorted(chk[chk.model == m][lvl].round(2).unique())
            assert all(l in got for l in LEVELS), (dim, scen, m, got)
            for l in LEVELS:
                k = chk[(chk.model == m) & (chk[lvl].round(2) == l)].bootstrap.nunique()
                assert k == 500, (dim, scen, m, l, k)
        assert len(out[(out["N"] == N) & out.model.isin(SWAP)
                       & out[lvl].round(2).isin(LEVELS)]) == 2 * 4 * 500, (dim, scen)
        out.to_csv(path, index=False)
    print("  {} fait".format(dim))

# the substitution must have moved the numbers, and only for the two models
s = pd.read_csv(IN / "completeness" / "S2.csv")
old = pd.read_csv(IN / "completeness" / "S2.csv.bak_{}".format(stamp))
for m in SWAP + ["RandomForest"]:
    a = s[(s.N == N) & (s.model == m) & (s.pollution.round(2) == 0.3)].auc.mean()
    b = old[(old.N == N) & (old.model == m) & (old.pollution.round(2) == 0.3)].auc.mean()
    print("  {:14s} {:.5f} -> {:.5f}  {}".format(
        m, b, a, "remplace" if abs(a - b) > 1e-9 else "inchange (attendu)"))
