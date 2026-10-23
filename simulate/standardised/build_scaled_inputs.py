"""Write binary-outcome input files in the published schema with the standardised arm
substituted, so every figure script can be run unchanged by repointing BINARY.file_map.

Only LASSO and MLP1 rows are replaced. The three tree models are invariant to a per-column
affine map, measured on 500 clean replicates (decision tree: mean difference +7e-5), so
their published rows stand. Levels 0 to 0.3 are covered, which is what the figures plot.

f1_macro is left empty on the replaced rows: the re-run recorded only the two co-primary
metrics, so Supplementary Figure 1 (F1-macro) must not be regenerated from these files.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PAPER
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, PAPER)
import analyse_stats_v2 as V2

HERE = Path(__file__).resolve().parent
OUT = HERE / "scaled_inputs"
OUT.mkdir(exist_ok=True)
KEY = {"completeness": "Completeness", "coherence": "Coherence", "validity": "Validity",
       "precision": "Precision", "uniqueness": "Uniqueness"}
SWAP = ["LASSO", "MLP1"]
LEVELS = [0.0, 0.1, 0.2, 0.3]

# the standardised arm: corrupted levels per scenario, plus the clean condition
corr = {}
for scen in ("S1", "S2", "S3"):
    d = pd.read_csv(HERE / f"scaled_{scen}.csv")
    d = d[(d.arm == "scaled") & (d.model.isin(SWAP))]
    assert len(d) == 5 * 3 * 500 * 2, (scen, len(d))
    corr[scen] = d
clean = pd.read_csv(HERE / "clean.csv")
clean = clean[(clean.arm == "scaled") & (clean.model.isin(SWAP))]
assert len(clean) == 2 * 500, len(clean)
print("bras standardise charge : {} lignes corrompues, {} propres".format(
    sum(len(v) for v in corr.values()), len(clean)))

fmap = {}
for dim, scens in V2.BINARY.file_map.items():
    fmap[dim] = {}
    for scen, path in scens.items():
        pub = pd.read_csv(path)
        lvl = "pollution" if "pollution" in pub.columns else "p"
        n = V2.BINARY.n_fixed

        # rows to drop: the two swapped models at this cohort size and these levels
        drop = (pub.model.isin(SWAP) & (pub["N"] == n)
                & pub[lvl].round(2).isin(LEVELS))
        kept = pub[~drop].copy()

        # build the replacements in the published schema
        new = corr[scen][corr[scen].dimension == KEY[dim]][
            ["model", "bootstrap", "p", "auc", "brier"]].rename(columns={"p": lvl})
        cl = clean[["model", "bootstrap", "auc", "brier"]].copy()
        cl[lvl] = 0.0
        rep = pd.concat([cl, new], ignore_index=True)
        rep["N"] = n
        for c in pub.columns:
            if c not in rep.columns:
                rep[c] = pub[c].iloc[0] if c in ("outcome_type", "dimension", "scenario") \
                    else np.nan
        rep = rep[pub.columns]

        out = pd.concat([kept, rep], ignore_index=True)
        # every model the config uses must have all four levels at this cohort size
        chk = out[(out["N"] == n) & out.model.isin(V2.BINARY.models)]
        for m in V2.BINARY.models:
            got = sorted(chk[chk.model == m][lvl].round(2).unique())
            assert all(l in got for l in LEVELS), (dim, scen, m, got)
            for l in LEVELS:
                k = chk[(chk.model == m) & (chk[lvl].round(2) == l)].bootstrap.nunique()
                assert k == 500, (dim, scen, m, l, k)
        # and the swapped rows must really be the standardised ones
        probe = chk[(chk.model == "MLP1") & (chk[lvl].round(2) == 0.0)].auc.mean()
        assert abs(probe - 0.822) < 0.01, "la baseline MLP n'est pas le bras standardise: {:.3f}".format(probe)

        p = OUT / dim / f"{scen}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
        fmap[dim][scen] = p
        print(f"  {dim:14s}{scen}  {len(pub):6d} -> {len(out):6d} lignes "
              f"({int(drop.sum())} remplacees par {len(rep)})")

print("\nAUROC propre apres substitution, N=3500 :")
d = pd.read_csv(fmap["completeness"]["S2"])
lvl = "pollution" if "pollution" in d.columns else "p"
g = d[(d["N"] == 3500) & (d[lvl].round(2) == 0.0) & d.model.isin(V2.BINARY.models)]
for m, v in g.groupby("model").auc.mean().items():
    print(f"  {m:14s}{v:.3f}")
print("\nrepertoire : {}".format(OUT))
