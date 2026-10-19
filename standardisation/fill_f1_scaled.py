"""Write the F1-macro of the standardised arm into the substituted inputs.

regen_figures.py left Supplementary Figure 1 alone because the October re-run recorded only
auc and brier, so every row it substituted carries an empty f1_macro. runner_f1.py has now
produced those values on the same cohort, splits and seeds.

Only LASSO and MLP1 rows are empty -- the tree rows were never substituted, standardisation
leaving them unchanged -- and only at N = 1500 and N = 3500. Supplementary Figure 1 is drawn
at N = 3500, so that is what is filled here; N = 1500 carries no F1 figure and is left as it
is.

Before writing anything, each row's auc is checked against the F1 run's auc for the same
(dimension, level, replicate, model). The two runs share their seeds, so an exact match is
what proves the replicate indices line up; anything else and the join would be pairing
different replicates and the figure would be silently wrong.

The level column is named "pollution" in six of the fifteen files and "p" in the others,
as make_print_figures.read() already has to handle.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path(__file__).resolve().parent
IN = R / "scaled_inputs"
STAMP = f"{datetime.now():%Y%m%d_%H%M}"
DIMS = ["completeness", "coherence", "validity", "precision", "uniqueness"]
N = 3500

deg = pd.concat([pd.read_csv(R / f) for f in
                 ("f1_S1.csv", "f1_S2.csv", "f1_S3_complet.csv")], ignore_index=True)
deg["dim"] = deg["dimension"].str.lower()
clean = pd.read_csv(R / "f1_clean.csv")          # p = 0, commun a toutes les cellules
print("sources : {} lignes degradees, {} lignes propres".format(len(deg), len(clean)))

total, checked = 0, 0
for dim in DIMS:
    for scen in ("S1", "S2", "S3"):
        f = IN / dim / f"{scen}.csv"
        d = pd.read_csv(f)
        lev = "pollution" if "pollution" in d.columns else "p"

        src = pd.concat([
            clean.assign(**{lev: 0.0}),
            deg[(deg["dim"] == dim) & (deg["scenario"] == scen)].rename(columns={"p": lev}),
        ], ignore_index=True)[[lev, "bootstrap", "model", "auc", "f1_macro"]]

        need = d["f1_macro"].isna() & (d["N"] == N)
        j = d.loc[need, [lev, "bootstrap", "model", "auc"]].merge(
            src, on=[lev, "bootstrap", "model"], how="left", suffixes=("_in", "_f1"))
        assert len(j) == int(need.sum()), (f, len(j), int(need.sum()))
        assert j["f1_macro"].notna().all(), (f, int(j["f1_macro"].isna().sum()))
        gap = (j["auc_in"] - j["auc_f1"]).abs().max()
        assert gap == 0.0, (f, gap)        # memes graines : l'egalite doit etre exacte

        shutil.copy(f, f.with_suffix(f".csv.bak_f1_{STAMP}"))
        d.loc[need, "f1_macro"] = j["f1_macro"].to_numpy()
        d.to_csv(f, index=False)
        total += int(need.sum())
        checked += len(j)
        print("   {:13s} {} : {:5d} lignes remplies, ecart auc {:.1e}".format(
            dim, scen, int(need.sum()), gap))

print("\n{} lignes remplies, {} verifiees sur auc".format(total, checked))
reste = 0
for dim in DIMS:
    for scen in ("S1", "S2", "S3"):
        d = pd.read_csv(IN / dim / f"{scen}.csv")
        reste += int((d["f1_macro"].isna() & (d["N"] == N)).sum())
print("f1_macro encore absent a N={} : {}".format(N, reste))
