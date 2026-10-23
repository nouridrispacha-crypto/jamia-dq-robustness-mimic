"""Does the .632 estimator favour or disfavour the test-time-versus-training-time finding?

The Results state that, averaged over the five degradations, test-time degradation cost more
than training-time for every model and metric. Under S2 the in-bag term cancels and the loss
is exactly 0.632 of the out-of-bag loss; under S1 the training data are themselves degraded,
the in-bag term does not cancel, and the loss is not a fixed rescaling. The two sides of the
comparison are therefore not computed the same way.

This measures the direction of that asymmetry: the same losses are recomputed from the
out-of-bag term alone, and the S2 - S1 gap under the two estimators is compared. A gap that
is larger under .632 than out-of-bag means the estimator flatters the finding; smaller means
the finding is conservative.
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
# Le bras brut est le pipeline binaire publie, sans copie locale.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'binary'))
sys.path.insert(0, str(HERE / "polluter"))

import common_runner as CR                                            # noqa: E402
import runner_full as RF                                              # noqa: E402

MODELS = ["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"]
DIMS = list(RF.DIMS)


def one(df, pools, preproc, mu, sd, n_cat, fn, p, b, train, test):
    inbag, oob = CR.stratified_inbag_oob(df, CR.TARGET_COL, seed=10_000 + b)
    inbag = CR.impute_hf_random_draw_sex(inbag, CR.SEX_COL, CR.IMPUTE_COLS, pools, seed=20_000 + b)
    oob = CR.impute_hf_random_draw_sex(oob, CR.SEX_COL, CR.IMPUTE_COLS, pools, seed=30_000 + b)
    if p > 0 and fn is not None:
        if train:
            y = inbag[CR.TARGET_COL].copy()
            r = fn(inbag, p, seed=40_000 + b)
            inbag = r[0] if isinstance(r, tuple) else r
            inbag[CR.TARGET_COL] = y
        if test:
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
    return RF.fit_all(S, O, ytr, yoob), RF.fit_all(S, S, ytr, ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    df, pools, preproc, mu, sd, n_cat = RF.setup()

    rows = []
    jobs = [("clean", None, 0.0, False, False)]
    for d in DIMS:
        jobs.append(("S1|" + d, RF.DIMS[d], 0.3, True, False))
        jobs.append(("S2|" + d, RF.DIMS[d], 0.3, False, True))

    for name, fn, p, tr, te in jobs:
        for b in range(a.reps):
            oob_m, in_m = one(df, pools, preproc, mu, sd, n_cat, fn, p, b, tr, te)
            for m in MODELS:
                for met in ("auc", "brier"):
                    rows.append(dict(cond=name, b=b, model=m, metric=met,
                                     inbag=in_m[m][met], oob=oob_m[m][met],
                                     v632=0.368 * in_m[m][met] + 0.632 * oob_m[m][met]))
        print("  {} fait".format(name), flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "sens_632_stage.csv", index=False)

    def loss(cond, model, met, col):
        cl = d[(d.cond == "clean") & (d.model == model) & (d.metric == met)].set_index("b")[col]
        dg = d[(d.cond == cond) & (d.model == model) & (d.metric == met)].set_index("b")[col]
        k = cl.index.intersection(dg.index)
        s = 1 if met == "auc" else -1
        return float(np.median(s * (cl.loc[k] - dg.loc[k])))

    print("\nperte moyenne sur les cinq degradations, a c=0.3")
    print("{:6s}{:14s}{:>9s}{:>9s}{:>8s}{:>10s}{:>10s}{:>9s}".format(
        "metr", "modele", "S1 .632", "S2 .632", "ecart", "S1 OOB", "S2 OOB", "ecart"))
    flip = 0
    for met in ("auc", "brier"):
        for m in MODELS:
            s1a = np.mean([loss("S1|" + x, m, met, "v632") for x in DIMS])
            s2a = np.mean([loss("S2|" + x, m, met, "v632") for x in DIMS])
            s1o = np.mean([loss("S1|" + x, m, met, "oob") for x in DIMS])
            s2o = np.mean([loss("S2|" + x, m, met, "oob") for x in DIMS])
            g632, goob = s2a - s1a, s2o - s1o
            mark = "" if (g632 > 0) == (goob > 0) else "   <<< LE SENS CHANGE"
            flip += (g632 > 0) != (goob > 0)
            print("{:6s}{:14s}{:9.4f}{:9.4f}{:8.4f}{:10.4f}{:10.4f}{:9.4f}{}".format(
                met, m, s1a, s2a, g632, s1o, s2o, goob, mark))
    print("\ncellules ou la conclusion S2>S1 change de sens : {} / 10".format(flip))
    print("un ecart .632 plus GRAND que l'ecart OOB = l'estimateur flatte le constat")
    print("\ndetail : sens_632_stage.csv")


if __name__ == "__main__":
    main()
