"""What would the .632+ estimator have changed?

The published runs store only the combined .632 metric, so the in-bag and out-of-bag terms
are recomputed here for a few replicates and the three estimators compared:

    .632      0.368 * in-bag + 0.632 * OOB
    OOB       the out-of-bag term alone
    .632+     (1 - w) * in-bag + w * OOB,   w = 0.632 / (1 - 0.368 R)

with R the relative overfitting rate, computed on the error scale as Efron and Tibshirani
define it: R = (Err_oob - err_in) / (gamma - err_in), clipped to [0, 1]. For AUROC the error
is 1 - AUC and the no-information rate is 0.5; for the Brier score the no-information value
is the one the paper already reports for this cohort.

Two questions are answered separately, because they have different answers: what the
estimator does to the reported LEVELS, and what it does to the CONTRASTS between dimensions,
which is what every verdict rests on.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "polluter"))

import common_runner as CR                                            # noqa: E402
import runner_full as RF                                              # noqa: E402

BRIER_NULL = 0.098                      # no-information Brier for this cohort
MODELS = ["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"]


def w_hat(err_in, err_oob, gamma):
    """Efron-Tibshirani adaptive weight, on the error scale."""
    denom = gamma - err_in
    r = 0.0 if denom <= 0 else (err_oob - err_in) / denom
    r = float(np.clip(r, 0.0, 1.0))
    return 0.632 / (1 - 0.368 * r)


def one(df, pools, preproc, mu, sd, n_cat, fn, p, b):
    """In-bag and OOB metrics for one replicate, standardised arm, test-time degradation."""
    inbag, oob = CR.stratified_inbag_oob(df, CR.TARGET_COL, seed=10_000 + b)
    inbag = CR.impute_hf_random_draw_sex(inbag, CR.SEX_COL, CR.IMPUTE_COLS, pools, seed=20_000 + b)
    oob = CR.impute_hf_random_draw_sex(oob, CR.SEX_COL, CR.IMPUTE_COLS, pools, seed=30_000 + b)
    if p > 0 and fn is not None:
        y = oob[CR.TARGET_COL].copy()
        r = fn(oob, p, seed=50_000 + b)
        oob = r[0] if isinstance(r, tuple) else r
        oob[CR.TARGET_COL] = y
    inbag = CR.encode_nan_as_sentinel(CR.enforce_column_types(CR.ensure_schema(inbag)), CR.NUM_COLS)
    oob = CR.encode_nan_as_sentinel(CR.enforce_column_types(CR.ensure_schema(oob)), CR.NUM_COLS)
    _, Xtr, Xoob, ytr, yoob = CR.prepare_data_for_models(inbag, oob, preproc)
    Xtr = np.nan_to_num(Xtr, nan=0.0).astype(np.float64)
    Xoob = np.nan_to_num(Xoob, nan=0.0).astype(np.float64)
    Xtr[:, n_cat:] = (Xtr[:, n_cat:] - mu) / sd
    Xoob[:, n_cat:] = (Xoob[:, n_cat:] - mu) / sd
    S, O = Xtr.astype(np.float32), Xoob.astype(np.float32)
    # fit_all scores the second pair; calling it twice gives the OOB and the in-bag side
    return RF.fit_all(S, O, ytr, yoob), RF.fit_all(S, S, ytr, ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    df, pools, preproc, mu, sd, n_cat = RF.setup()
    CONDS = [("clean", None, 0.0),
             ("Completeness", RF.DIMS["Completeness"], 0.3),
             ("Validity", RF.DIMS["Validity"], 0.3)]

    rows = []
    for name, fn, p in CONDS:
        for b in range(a.reps):
            oob_m, in_m = one(df, pools, preproc, mu, sd, n_cat, fn, p, b)
            for m in MODELS:
                for met, gamma in (("auc", 0.5), ("brier", BRIER_NULL)):
                    o, i = oob_m[m][met], in_m[m][met]
                    ei = (1 - i) if met == "auc" else i      # error scale
                    eo = (1 - o) if met == "auc" else o
                    w = w_hat(ei, eo, gamma)
                    e632 = 0.368 * ei + 0.632 * eo
                    eplus = (1 - w) * ei + w * eo
                    back = (lambda e: 1 - e) if met == "auc" else (lambda e: e)
                    rows.append(dict(cond=name, b=b, model=m, metric=met, w=w,
                                     inbag=i, oob=o, v632=back(e632), vplus=back(eplus)))
        print("  {} fait".format(name), flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "compare_632plus.csv", index=False)

    print("\n=== NIVEAUX (condition propre) ===")
    g = d[(d.cond == "clean")].groupby(["metric", "model"])
    print("{:7s}{:14s}{:>9s}{:>9s}{:>9s}{:>9s}{:>8s}".format(
        "metr", "modele", "in-bag", "OOB", ".632", ".632+", "w"))
    for (met, m), gg in g:
        print("{:7s}{:14s}{:9.4f}{:9.4f}{:9.4f}{:9.4f}{:8.3f}".format(
            met, m, gg.inbag.mean(), gg.oob.mean(), gg.v632.mean(), gg.vplus.mean(), gg.w.mean()))

    print("\n=== CONTRASTE completude - validite, a c=0.3 (ce qui porte les verdicts) ===")
    print("{:7s}{:14s}{:>12s}{:>12s}{:>10s}".format("metr", "modele", ".632", ".632+", "ecart"))
    for met in ("auc", "brier"):
        for m in MODELS:
            def loss(cond, col):
                cl = d[(d.cond == "clean") & (d.model == m) & (d.metric == met)].set_index("b")[col]
                dg = d[(d.cond == cond) & (d.model == m) & (d.metric == met)].set_index("b")[col]
                k = cl.index.intersection(dg.index)
                s = 1 if met == "auc" else -1
                return (s * (cl.loc[k] - dg.loc[k]))
            c632 = float(np.median(loss("Completeness", "v632") - loss("Validity", "v632")))
            cplus = float(np.median(loss("Completeness", "vplus") - loss("Validity", "vplus")))
            print("{:7s}{:14s}{:12.4f}{:12.4f}{:10.4f}".format(met, m, c632, cplus, cplus - c632))
    print("\ndetail : compare_632plus.csv")


if __name__ == "__main__":
    main()
