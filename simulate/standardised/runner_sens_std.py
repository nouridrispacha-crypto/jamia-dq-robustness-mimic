"""Encoding sensitivity arm, Scenario S2, binary outcome.

Two encodings of the values that corruption leaves missing or invalid:

  sentinel  the published encoding: per-column out-of-domain constants
            (-9999 for urine output, -99 for age, ...). Used here only to
            validate that this script reproduces the published numbers.

  mixed     NaN kept for the tree models, which split on it natively;
            in-bag median plus a missingness indicator for LASSO and the MLP,
            which cannot fit with NaN. The indicator preserves the MNAR signal
            the sentinel carried, so the only thing removed is the extreme
            magnitude -- which is what the criticism is about.

Everything else is held fixed: same sample, same stratified split seeds, same
hot-deck seeds, same corruption seeds, so each replicate is paired across arms.
Model constructors are copied verbatim from common_runner.
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
# Le bras brut est le pipeline binaire publie, sans copie locale.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'binary'))
# Les cinq mecanismes de degradation sont partages entre les issues : ils
# sont identiques, et une copie par issue laissait croire le contraire.
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "polluter"))

import common_runner as CR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
import xgboost as xgb
import torch

from polluter.pollute_mnar_top import pollution_mnar_top
from polluter.pollute_validity import pollution_validity

DIMS = {"Completeness": pollution_mnar_top, "Validity": pollution_validity}
TREE_MODELS = ("DecisionTree", "RandomForest", "XGBoost")


def prepare_keep_nan(df_train, df_eval, preprocessor):
    """prepare_data_for_models without its final nan_to_num guard."""
    ytr = df_train[CR.TARGET_COL].astype(int).to_numpy()
    yev = df_eval[CR.TARGET_COL].astype(int).to_numpy()
    Xtr_df = df_train.drop(columns=CR.ID_COLS + [CR.TARGET_COL], errors="ignore").copy()
    Xev_df = df_eval.drop(columns=CR.ID_COLS + [CR.TARGET_COL], errors="ignore").copy()
    for c in CR.CAT_COLS:
        if c in Xtr_df:
            Xtr_df[c] = Xtr_df[c].astype(str).fillna(CR.MISSING_TOKEN)
            Xev_df[c] = Xev_df[c].astype(str).fillna(CR.MISSING_TOKEN)
    Xtr = preprocessor.transform(Xtr_df).astype(np.float64)
    Xev = preprocessor.transform(Xev_df).astype(np.float64)
    return Xtr, Xev, ytr, yev


def median_arm(Xtr, Xoob):
    """In-bag median imputation plus indicator columns for the columns that can be
    missing. Medians come from the in-bag set only, so no information leaks."""
    mtr, moob = np.isnan(Xtr), np.isnan(Xoob)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)          # a wholly missing column
    Xtr_i = np.where(mtr, med, Xtr)
    Xoob_i = np.where(moob, med, Xoob)
    keep = mtr.any(axis=0) | moob.any(axis=0)        # drop constant indicators
    if keep.any():
        Xtr_i = np.hstack([Xtr_i, mtr[:, keep].astype(np.float64)])
        Xoob_i = np.hstack([Xoob_i, moob[:, keep].astype(np.float64)])
    return Xtr_i.astype(np.float32), Xoob_i.astype(np.float32)


MU = SD = None
N_CAT = 0


def standardise(X):
    """The real numeric predictors only. The one-hot columns come first and are left
    alone, and the median arm appends 0/1 missingness indicators after the predictors,
    which must not be scaled either."""
    if MU is None:
        return X
    Y = np.asarray(X, dtype=np.float64).copy()
    a, b = N_CAT, N_CAT + len(MU)
    assert Y.shape[1] >= b, (Y.shape, b)
    Y[:, a:b] = (Y[:, a:b] - MU) / SD
    return Y.astype(np.float32)


def fit_five(Xtr_t, Xoob_t, Xtr_l, Xoob_l, ytr, yoob):
    Xtr_t, Xoob_t = standardise(Xtr_t), standardise(Xoob_t)
    Xtr_l, Xoob_l = standardise(Xtr_l), standardise(Xoob_l)
    """The five models the paper reports. Trees get the first matrix pair, LASSO and
    the MLP the second. Constructors verbatim from common_runner."""
    res = {}

    lasso = Lasso(alpha=0.001, max_iter=5000)
    lasso.fit(Xtr_l, ytr)
    p_tr = np.clip(lasso.predict(Xtr_l), 0, 1)
    p_oob = np.clip(lasso.predict(Xoob_l), 0, 1)
    res["LASSO"] = CR.combine_point632(
        CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob))

    dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=10, random_state=0)
    dt.fit(Xtr_t, ytr)
    p_tr, p_oob = dt.predict_proba(Xtr_t)[:, 1], dt.predict_proba(Xoob_t)[:, 1]
    res["DecisionTree"] = CR.combine_point632(
        CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob))

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=10,
                                n_jobs=-1, random_state=0)
    rf.fit(Xtr_t, ytr)
    p_tr, p_oob = rf.predict_proba(Xtr_t)[:, 1], rf.predict_proba(Xoob_t)[:, 1]
    res[CR.RF_NAME] = CR.combine_point632(
        CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob))

    dtr, doob = xgb.DMatrix(Xtr_t, label=ytr), xgb.DMatrix(Xoob_t, label=yoob)
    booster = xgb.train({"objective": "binary:logistic", "tree_method": "hist",
                         "device": "cuda"}, dtr, num_boost_round=100)
    p_tr, p_oob = booster.predict(dtr), booster.predict(doob)
    res["XGBoost"] = CR.combine_point632(
        CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob))

    per_seed = []
    for seed in [0, 1, 2]:
        model = CR.train_mlp(Xtr_l, ytr, (64,), seed)
        p_tr, p_oob = CR.predict_mlp(model, Xtr_l), CR.predict_mlp(model, Xoob_l)
        per_seed.append(CR.combine_point632(
            CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
            CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob)))
    res["MLP1"] = {k: np.mean([m[k] for m in per_seed]) for k in per_seed[0]}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def one_replicate(df_sample, pools, preproc, pollute_fn, p, b, encoding):
    # The runner schema-enforces only AFTER pollution; doing it earlier changes the
    # dtypes the validity polluter selects on, and so changes which cells it touches.
    inbag, oob = CR.stratified_inbag_oob(df_sample, CR.TARGET_COL, seed=10_000 + b)
    inbag = CR.impute_hf_random_draw_sex(inbag, CR.SEX_COL, CR.IMPUTE_COLS,
                                         pools, seed=20_000 + b)
    oob = CR.impute_hf_random_draw_sex(oob, CR.SEX_COL, CR.IMPUTE_COLS,
                                       pools, seed=30_000 + b)
    if p > 0:                                   # S2: test set only
        y = oob[CR.TARGET_COL].copy()           # the runner restores the target
        out = pollute_fn(oob, p, seed=50_000 + b)
        oob = out[0] if isinstance(out, tuple) else out
        oob[CR.TARGET_COL] = y
    inbag = CR.enforce_column_types(CR.ensure_schema(inbag))
    oob = CR.enforce_column_types(CR.ensure_schema(oob))

    if encoding == "sentinel":
        inbag = CR.encode_nan_as_sentinel(inbag, CR.NUM_COLS)
        oob = CR.encode_nan_as_sentinel(oob, CR.NUM_COLS)
        Xtr, Xoob, ytr, yoob = prepare_keep_nan(inbag, oob, preproc)
        Xtr = np.nan_to_num(Xtr, nan=0.0).astype(np.float32)
        Xoob = np.nan_to_num(Xoob, nan=0.0).astype(np.float32)
        return fit_five(Xtr, Xoob, Xtr, Xoob, ytr, yoob)

    Xtr, Xoob, ytr, yoob = prepare_keep_nan(inbag, oob, preproc)
    Xtr_l, Xoob_l = median_arm(Xtr, Xoob)
    assert not np.isnan(Xtr_l).any() and not np.isnan(Xoob_l).any(), "NaN left after imputation"
    if encoding == "median":
        # symmetric arm: every model sees the same matrix, so the architecture
        # contrast is estimated with the information held constant
        return fit_five(Xtr_l, Xoob_l, Xtr_l, Xoob_l, ytr, yoob)
    return fit_five(Xtr.astype(np.float32), Xoob.astype(np.float32),
                    Xtr_l, Xoob_l, ytr, yoob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", choices=["sentinel", "mixed", "median"], required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--levels", default="0.3")
    ap.add_argument("--dims", default="Completeness,Validity")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    levels = [float(x) for x in a.levels.split(",")]
    wanted = [d.strip() for d in a.dims.split(",")]

    df = pd.read_csv(HERE / "data" / "18_variables.csv")
    df_sample = df.sample(n=3500, random_state=42).reset_index(drop=True)
    prefit = CR.enforce_column_types(CR.ensure_schema(df_sample.copy()))
    pools = CR.build_sex_conditioned_pools(prefit, CR.SEX_COL, CR.IMPUTE_COLS)
    preproc, Xfit, *_ = CR.prepare_data_for_models(prefit, prefit)
    global MU, SD, N_CAT
    N_CAT = sum(len(c) for c in preproc.named_transformers_["cat"].categories_)
    blk = np.asarray(Xfit, dtype=np.float64)[:, N_CAT:]
    MU = np.nanmean(blk, axis=0)
    SD = np.nanstd(blk, axis=0)
    SD = np.where((SD == 0) | ~np.isfinite(SD), 1.0, SD)
    print(f"standardisation : {blk.shape[1]} colonnes numeriques apres {N_CAT} one-hot, "
          f"rapport des ecarts-types {SD.max()/SD.min():.0f}x", flush=True)

    out = Path(a.out or HERE / f"sens_{a.encoding}.csv")
    new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["encoding", "dimension", "p", "bootstrap", "model", "auc", "brier"])

    t0 = time.time()
    for dim, fn in [(k, v) for k, v in DIMS.items() if k in wanted]:
        for p in levels:
            for b in range(a.reps):
                r = one_replicate(df_sample, pools, preproc, fn, p, b, a.encoding)
                for model, m in r.items():
                    w.writerow([a.encoding, dim, p, b, model,
                                m.get("auc", np.nan), m.get("brier", np.nan)])
                if b % 10 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"{dim} p={p} b={b:03d}/{a.reps} | {el/60:.1f} min",
                          flush=True)
    fh.close()
    print("written:", out)


if __name__ == "__main__":
    main()
