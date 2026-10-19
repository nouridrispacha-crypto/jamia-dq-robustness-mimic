"""The binary grid on the standardised arm, recording every metric, not only the two
co-primary ones.

runner_full.py writes auc and brier; fit_all computes accuracy, f1_macro and auprc as well
and throws them away. Supplementary Figure 1 and three rows of Supplementary Table S5 rest
on f1_macro, so they could not be brought onto the standardised pipeline. This runner is
runner_full with a wider CSV writer -- same cohort, same seeds, same polluters, same model
constructors -- so the auc and brier columns it produces must reproduce runner_full's.

USAGE
    python runner_f1.py --scen S2 --levels 0.1,0.2,0.3 --out f1_S2.csv
    python runner_f1.py --scen S2 --dims Completeness --levels 0.0 --out f1_clean.csv
"""
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "polluter"))

import runner_full as RF                                              # noqa: E402

METRICS = ["accuracy", "f1_macro", "auc", "brier", "auprc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="S2")
    ap.add_argument("--dims", default=",".join(RF.DIMS))
    ap.add_argument("--levels", default="0.1,0.2,0.3")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3500)
    a = ap.parse_args()

    df, pools, preproc, mu, sd, n_cat = RF.setup(a.n)
    out = Path(a.out)
    new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["scenario", "dimension", "p", "bootstrap", "model"] + METRICS + ["N"])

    jobs = [(d, RF.DIMS[d], float(p)) for d in a.dims.split(",") for p in a.levels.split(",")]
    t0, done, total = time.time(), 0, len(jobs) * a.reps
    for dim, fn, p in jobs:
        for b in range(a.reps):
            r = RF.one_replicate(df, pools, preproc, mu, sd, n_cat, fn, p, b,
                                 a.scen, ("scaled",))["scaled"]
            for m, v in r.items():
                w.writerow([a.scen, dim, p, b, m] + [v.get(k) for k in METRICS] + [a.n])
            done += 1
            if b % 25 == 0:
                fh.flush()
                el = time.time() - t0
                print("{} {} p={} b={:03d} | {}/{} | {:.0f} min | reste ~{:.1f} h".format(
                    a.scen, dim, p, b, done, total, el / 60,
                    (total - done) * el / done / 3600), flush=True)
    fh.close()
    print("ecrit :", out, flush=True)


if __name__ == "__main__":
    main()
