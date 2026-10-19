"""Full standardisation re-run of the binary task: the five models the paper reports, both
arms, N = 3500, corruption levels 0 to 0.3, scenarios S1, S2 and S3.

  raw     the published pipeline, numeric columns passed through unchanged.
  scaled  the numeric block standardised by means and standard deviations computed once on
          the uncorrupted sampled cohort, which is what the continuous and survival chains
          already do and what the Methods describe.

Model constructors are copied from JAMIA_V2/binary/common_runner.py, which is the code
analyse_stats_v2.py reads its results from. The raw arm must reproduce the published
numbers; --validate checks that before any standardised number is trusted.
"""
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "polluter"))

import common_runner as CR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch

from polluter.pollute_mnar_top import pollution_mnar_top
from polluter.pollute_validity import pollution_validity
from polluter.pollute_precision import pollution_precision
from polluter.pollute_uniqueness import pollution_uniqueness
from polluter.coherence_polluter import pollution_coherence

DIMS = {"Completeness": pollution_mnar_top, "Validity": pollution_validity,
        "Precision": pollution_precision, "Uniqueness": pollution_uniqueness,
        "Coherence": pollution_coherence}
LEVELS = [0.0, 0.1, 0.2, 0.3]
SCEN = {"S1": (True, False), "S2": (False, True), "S3": (True, True)}


WANT = None          # set from --models; None means all five


def want(m):
    return WANT is None or m in WANT


def fit_all(Xtr, Xoob, ytr, yoob):
    """The five models, constructors verbatim from the published runner."""
    res = {}

    if want("LASSO"):
        la = Lasso(alpha=0.001, max_iter=5000)
        la.fit(Xtr, ytr)
        res["LASSO"] = CR.combine_point632(
            CR.compute_metrics(ytr, (np.clip(la.predict(Xtr), 0, 1) > 0.5).astype(int),
                               np.clip(la.predict(Xtr), 0, 1)),
            CR.compute_metrics(yoob, (np.clip(la.predict(Xoob), 0, 1) > 0.5).astype(int),
                               np.clip(la.predict(Xoob), 0, 1)))

    if want("DecisionTree"):
        dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=10, random_state=0)
        dt.fit(Xtr, ytr)
        res["DecisionTree"] = CR.combine_point632(
            CR.compute_metrics(ytr, (dt.predict_proba(Xtr)[:, 1] > 0.5).astype(int),
                               dt.predict_proba(Xtr)[:, 1]),
            CR.compute_metrics(yoob, (dt.predict_proba(Xoob)[:, 1] > 0.5).astype(int),
                               dt.predict_proba(Xoob)[:, 1]))

    if want("RandomForest"):
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=10,
                                    n_jobs=-1, random_state=0)
        rf.fit(Xtr, ytr)
        res["RandomForest"] = CR.combine_point632(
            CR.compute_metrics(ytr, (rf.predict_proba(Xtr)[:, 1] > 0.5).astype(int),
                               rf.predict_proba(Xtr)[:, 1]),
            CR.compute_metrics(yoob, (rf.predict_proba(Xoob)[:, 1] > 0.5).astype(int),
                               rf.predict_proba(Xoob)[:, 1]))

    if want("XGBoost"):
        dtr, doob = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xoob, label=yoob)
        bst = xgb.train({"objective": "binary:logistic", "tree_method": "hist",
                         "device": "cuda"}, dtr, num_boost_round=100)
        p_tr, p_oob = bst.predict(dtr), bst.predict(doob)
        res["XGBoost"] = CR.combine_point632(
            CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
            CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob))

    if want("MLP1"):
        per_seed = []
        for seed in (0, 1, 2):
            m = CR.train_mlp(Xtr, ytr, (64,), seed)
            p_tr, p_oob = CR.predict_mlp(m, Xtr), CR.predict_mlp(m, Xoob)
            per_seed.append(CR.combine_point632(
                CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
                CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob)))
        res["MLP1"] = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def one_replicate(df, pools, preproc, mu, sd, n_cat, fn, p, b, scen, arms):
    do_train, do_test = SCEN[scen]
    inbag, oob = CR.stratified_inbag_oob(df, CR.TARGET_COL, seed=10_000 + b)
    inbag = CR.impute_hf_random_draw_sex(inbag, CR.SEX_COL, CR.IMPUTE_COLS,
                                        pools, seed=20_000 + b)
    oob = CR.impute_hf_random_draw_sex(oob, CR.SEX_COL, CR.IMPUTE_COLS,
                                      pools, seed=30_000 + b)
    if p > 0 and fn is not None:
        if do_train:
            y = inbag[CR.TARGET_COL].copy()
            r = fn(inbag, p, seed=40_000 + b)
            inbag = r[0] if isinstance(r, tuple) else r
            inbag[CR.TARGET_COL] = y
        if do_test:
            y = oob[CR.TARGET_COL].copy()
            r = fn(oob, p, seed=50_000 + b)
            oob = r[0] if isinstance(r, tuple) else r
            oob[CR.TARGET_COL] = y
    inbag = CR.enforce_column_types(CR.ensure_schema(inbag))
    oob = CR.enforce_column_types(CR.ensure_schema(oob))
    inbag = CR.encode_nan_as_sentinel(inbag, CR.NUM_COLS)
    oob = CR.encode_nan_as_sentinel(oob, CR.NUM_COLS)
    _, Xtr, Xoob, _, _ = CR.prepare_data_for_models(inbag, oob, preproc)
    ytr = inbag[CR.TARGET_COL].astype(int).to_numpy()
    yoob = oob[CR.TARGET_COL].astype(int).to_numpy()
    Xtr = np.nan_to_num(Xtr, nan=0.0).astype(np.float64)
    Xoob = np.nan_to_num(Xoob, nan=0.0).astype(np.float64)

    out = {}
    if "raw" in arms:
        out["raw"] = fit_all(Xtr.astype(np.float32), Xoob.astype(np.float32), ytr, yoob)
    if "scaled" in arms:
        S, O = Xtr.copy(), Xoob.copy()
        S[:, n_cat:] = (S[:, n_cat:] - mu) / sd
        O[:, n_cat:] = (O[:, n_cat:] - mu) / sd
        out["scaled"] = fit_all(S.astype(np.float32), O.astype(np.float32), ytr, yoob)
    return out


def setup(n=3500):
    df = pd.read_csv(HERE / "data" / "18_variables.csv")
    df = df.sample(n=n, random_state=42).reset_index(drop=True)
    prefit = CR.enforce_column_types(CR.ensure_schema(df.copy()))
    pools = CR.build_sex_conditioned_pools(prefit, CR.SEX_COL, CR.IMPUTE_COLS)
    preproc, Xfit, *_ = CR.prepare_data_for_models(prefit, prefit)
    n_cat = sum(len(c) for c in preproc.named_transformers_["cat"].categories_)
    blk = np.asarray(Xfit, dtype=np.float64)[:, n_cat:]
    mu, sd = np.nanmean(blk, axis=0), np.nanstd(blk, axis=0)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    return df, pools, preproc, mu, sd, n_cat


def validate(reps):
    """The raw arm must reproduce the published numbers for all five models."""
    PUB = pd.read_csv("C:/Users/IDRINO/Desktop/JAMIA_PAPER/consolidated_binary_outcome.csv")
    PUB = PUB[(PUB.scenario == "S2_test_only") & (PUB.N == 3500)]
    df, pools, preproc, mu, sd, n_cat = setup()
    print(f"validation du bras brut sur {reps} replicats\n")
    worst = {}
    for dim, pdim, fn in (("Completeness", "mnar_top", pollution_mnar_top),
                          ("Precision", "precision", pollution_precision),
                          ("Uniqueness", "uniqueness", pollution_uniqueness)):
        ref = PUB[(PUB.dimension == pdim) & (PUB.pollution.round(2) == 0.3)]
        ref = ref.set_index(["model", "bootstrap"])[["auc", "brier"]].sort_index()
        for b in range(reps):
            got = one_replicate(df, pools, preproc, mu, sd, n_cat, fn, 0.3, b,
                                "S2", ("raw",))["raw"]
            for m, v in got.items():
                if (m, b) not in ref.index:
                    continue
                e = max(abs(v["auc"] - ref.loc[(m, b), "auc"]),
                        abs(v["brier"] - ref.loc[(m, b), "brier"]))
                worst[m] = max(worst.get(m, 0.0), e)
        print(f"  {dim} fait")
    print(f"\n  {'modele':16s}{'ecart max au publie':>22s}")
    ok = True
    for m in sorted(worst):
        flag = "OK" if worst[m] < 1e-6 else ("proche" if worst[m] < 5e-3 else "DIFFERE")
        ok &= worst[m] < 5e-3
        print(f"  {m:16s}{worst[m]:22.2e}   {flag}")
    print("\n" + ("bras brut valide, le bras standardise est interpretable"
                  if ok else "ECHEC : ne pas se fier au bras standardise"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--scen", default="S2")
    ap.add_argument("--dims", default=",".join(DIMS))
    ap.add_argument("--levels", default="0.0,0.1,0.2,0.3")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--arms", default="raw,scaled")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=3500)
    ap.add_argument("--models", default=None)
    a = ap.parse_args()
    global WANT
    if a.models:
        WANT = set(a.models.split(","))
    if a.validate:
        sys.exit(0 if validate(a.validate) else 1)

    df, pools, preproc, mu, sd, n_cat = setup(a.n)
    arms = tuple(a.arms.split(","))
    out = Path(a.out or HERE / f"full_{a.scen}.csv")
    new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["arm", "scenario", "dimension", "p", "bootstrap", "model",
                    "auc", "brier", "N"])
    jobs = [(d, DIMS[d], float(p)) for d in a.dims.split(",") for p in a.levels.split(",")]
    t0, done = time.time(), 0
    total = len(jobs) * a.reps
    for dim, fn, p in jobs:
        for b in range(a.reps):
            r = one_replicate(df, pools, preproc, mu, sd, n_cat, fn, p, b, a.scen, arms)
            for arm, models in r.items():
                for m, v in models.items():
                    w.writerow([arm, a.scen, dim, p, b, m, v.get("auc"), v.get("brier"), a.n])
            done += 1
            if b % 25 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"{a.scen} {dim} p={p} b={b:03d} | {done}/{total} | "
                      f"{el/60:.0f} min | reste ~{(total-done)*el/done/3600:.1f} h",
                      flush=True)
    fh.close()
    print("ecrit :", out, flush=True)


if __name__ == "__main__":
    main()
