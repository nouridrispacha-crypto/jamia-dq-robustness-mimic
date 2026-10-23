"""Residual skewness of all 21 mixed models, on the sources the paper now uses.

Supplementary Table S5 ranks the seven most asymmetric of the twenty that converge, and its
footnote quotes the range of the thirteen it leaves out. Three survival models move into that
top seven and two leave it, so the ranking cannot be patched cell by cell: every model is
refitted and the table is rebuilt from the result.

The fitting code is the archived analyse_stats, whose specification matches the manuscript's
observation counts (37,500 for the binary arm, 30,000 for survival, three corruption levels).
Its file maps are replaced by the ones the paper now uses: the standardised binary arm, the
continuous arm unchanged, and the 90-day survival relaunch.

Nothing is written to the manuscript here. The values are printed and saved, so the table and
the sentence that depends on them can be reviewed before either is touched.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PAPER, ROOT
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(PAPER)
ANA = Path(ROOT / "analyse")
IN = Path(__file__).resolve().parent / "scaled_inputs"
OUT = Path(__file__).resolve().parent / "resid_all_j90"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(P))
import analyse_stats_v2 as V2                                   # cartes de fichiers reelles
sys.path.insert(0, str(ANA))
import analyse_stats as A                                       # modele mixte du manuscrit

# Le binaire passe au bras standardise ; les deux autres gardent leurs sources.
BIN = {d: {s: IN / d / f"{s}.csv" for s in m} for d, m in V2.BINARY.file_map.items()}
for dim, m in BIN.items():
    for s, p in m.items():
        assert p.exists(), p
A.BINARY.file_map.update(BIN)
A.CONTINUOUS.file_map.update({d: dict(m) for d, m in V2.CONTINUOUS.file_map.items()})
A.SURVIVAL.file_map.update({d: dict(m) for d, m in V2.SURVIVAL.file_map.items()})
print("binaire -> standardise | continu -> inchange | survie -> relance J90")
print("niveaux du modele mixte : {}\n".format(A.LEVELS))

NAME = {"binary": "Binary", "continuous": "Continuous", "survival": "Time-to-event"}
METRIC = {"auc": "AUROC", "brier": "Brier", "f1_macro": "F1-macro", "rmse": "RMSE",
          "r2": "R\\textsuperscript{2}", "c_index": "C-index", "ibs": "IBS"}

rows = []
for cfg in (A.BINARY, A.CONTINUOUS, A.SURVIVAL):
    for scen in ("S1", "S2", "S3"):
        for met in cfg.metrics:
            try:
                A.run_metric(cfg, scen, met, OUT)
            except Exception as exc:
                print("  ECHEC {} {} {} : {}".format(cfg.name, scen, met,
                                                     type(exc).__name__))
                continue
            f = OUT / "residdiag_{}_{}_{}.csv".format(cfg.name, scen, met)
            if not f.exists():
                print("  sans diagnostic : {} {} {}".format(cfg.name, scen, met))
                continue
            d = pd.read_csv(f).iloc[0]
            rows.append({"outcome": NAME[cfg.name], "scenario": scen,
                         "metric": METRIC[met], "n": int(d["n_resid"]),
                         "skew": float(d["skew"])})

t = pd.DataFrame(rows).sort_values("skew", key=lambda s: -s.abs()).reset_index(drop=True)
t.to_csv(OUT / "skewness_all.csv", index=False)
print("\n{} modeles ajustes\n".format(len(t)))
print("  {:14s} {:4s} {:16s} {:>8s} {:>8s}".format("outcome", "scen", "metrique", "n", "skew"))
for k, r in t.iterrows():
    mark = " <- top 7" if k < 7 else ""
    print("  {:14s} {:4s} {:16s} {:8d} {:8.2f}{}".format(
        r.outcome, r.scenario, r.metric, r["n"], r.skew, mark))

rest = t.iloc[7:]
print("\nles {} non listes vont de {:.2f} ({} {}, {}) a {:.2f} ({} {}, {})".format(
    len(rest), rest.skew.min(),
    rest.loc[rest.skew.idxmin(), "outcome"], rest.loc[rest.skew.idxmin(), "metric"],
    rest.loc[rest.skew.idxmin(), "scenario"], rest.skew.max(),
    rest.loc[rest.skew.idxmax(), "outcome"], rest.loc[rest.skew.idxmax(), "metric"],
    rest.loc[rest.skew.idxmax(), "scenario"]))
print("\necrit : {}".format(OUT / "skewness_all.csv"))
