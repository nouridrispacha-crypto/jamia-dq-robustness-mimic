"""Standardisation arm for the binary task.

The binary pipeline feeds predictors on their original scale; the continuous and survival
pipelines standardise. Raw scales span a factor of ~10,000 even after clipping
physiologically impossible values, so the two scale-sensitive models may be handicapped.

Two arms, paired replicate by replicate, everything else held fixed:

  raw     the published pipeline: ColumnTransformer with ("num", "passthrough").
  scaled  the same, with the numeric block standardised by means and standard
          deviations computed ONCE on the uncorrupted sampled cohort, which is what
          the continuous and survival chains do and what the Methods describe.

The sentinel encoding is unchanged in both arms, so this is independent of
Supplementary Table S9. The scaler is fitted on the uncorrupted cohort, so the
out-of-domain sentinels become large negative z-scores and keep their flag role.

Only LASSO and MLP1 are refitted: the three tree models split on thresholds and are
invariant to a monotone per-column affine map. The decision tree is fitted in both arms
at c=0 anyway, to verify that invariance rather than assume it.
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
import torch

from polluter.pollute_mnar_top import pollution_mnar_top
from polluter.pollute_validity import pollution_validity
from polluter.pollute_precision import pollution_precision
from polluter.pollute_uniqueness import pollution_uniqueness
from polluter.coherence_polluter import pollution_coherence

DIMS = {
    "Completeness": pollution_mnar_top,
    "Validity": pollution_validity,
    "Precision": pollution_precision,
    "Uniqueness": pollution_uniqueness,
    "Coherence": pollution_coherence,
}


def numeric_block(preproc):
    """Where the numeric columns start in the transformed matrix. The ColumnTransformer
    is [("cat", OneHotEncoder, cat_cols), ("num", passthrough, num_cols)]."""
    cats = preproc.named_transformers_["cat"].categories_
    return sum(len(c) for c in cats)


def fit_models(Xtr, Xoob, ytr, yoob, with_tree):
    out = {}
    per_seed = []
    for seed in (0, 1, 2):
        m = CR.train_mlp(Xtr, ytr, (64,), seed)
        p_tr, p_oob = CR.predict_mlp(m, Xtr), CR.predict_mlp(m, Xoob)
        per_seed.append(CR.combine_point632(
            CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
            CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob)))
    out["MLP1"] = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]}

    la = Lasso(alpha=0.001, max_iter=5000)
    la.fit(Xtr, ytr)
    p_tr = np.clip(la.predict(Xtr), 0, 1)
    p_oob = np.clip(la.predict(Xoob), 0, 1)
    out["LASSO"] = CR.combine_point632(
        CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob))

    if with_tree:
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(Xtr, ytr)
        p_tr = dt.predict_proba(Xtr)[:, 1]
        p_oob = dt.predict_proba(Xoob)[:, 1]
        out["DecisionTree"] = CR.combine_point632(
            CR.compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
            CR.compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def one_replicate(df_sample, pools, preproc, mu, sd, n_cat, pollute_fn, p, b, with_tree):
    inbag, oob = CR.stratified_inbag_oob(df_sample, CR.TARGET_COL, seed=10_000 + b)
    inbag = CR.impute_hf_random_draw_sex(inbag, CR.SEX_COL, CR.IMPUTE_COLS,
                                        pools, seed=20_000 + b)
    oob = CR.impute_hf_random_draw_sex(oob, CR.SEX_COL, CR.IMPUTE_COLS,
                                      pools, seed=30_000 + b)
    if p > 0:                                    # S2: test partition only
        y = oob[CR.TARGET_COL].copy()
        res = pollute_fn(oob, p, seed=50_000 + b)
        oob = res[0] if isinstance(res, tuple) else res
        oob[CR.TARGET_COL] = y
    inbag = CR.enforce_column_types(CR.ensure_schema(inbag))
    oob = CR.enforce_column_types(CR.ensure_schema(oob))
    inbag = CR.encode_nan_as_sentinel(inbag, CR.NUM_COLS)
    oob = CR.encode_nan_as_sentinel(oob, CR.NUM_COLS)

    _, Xtr, Xoob, ytr, yoob = CR.prepare_data_for_models(inbag, oob, preproc)
    Xtr = np.nan_to_num(Xtr, nan=0.0).astype(np.float64)
    Xoob = np.nan_to_num(Xoob, nan=0.0).astype(np.float64)

    rows = {}
    rows["raw"] = fit_models(Xtr.astype(np.float32), Xoob.astype(np.float32),
                             ytr, yoob, with_tree)
    Str, Soob = Xtr.copy(), Xoob.copy()
    Str[:, n_cat:] = (Str[:, n_cat:] - mu) / sd
    Soob[:, n_cat:] = (Soob[:, n_cat:] - mu) / sd
    rows["scaled"] = fit_models(Str.astype(np.float32), Soob.astype(np.float32),
                                ytr, yoob, with_tree)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--dims", default="clean")
    ap.add_argument("--level", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(HERE / "data" / "18_variables.csv")
    df_sample = df.sample(n=3500, random_state=42).reset_index(drop=True)
    prefit = CR.enforce_column_types(CR.ensure_schema(df_sample.copy()))
    pools = CR.build_sex_conditioned_pools(prefit, CR.SEX_COL, CR.IMPUTE_COLS)
    preproc, Xfit, *_ = CR.prepare_data_for_models(prefit, prefit)

    # means and standard deviations computed ONCE on the uncorrupted sampled cohort
    n_cat = numeric_block(preproc)
    block = np.asarray(Xfit, dtype=np.float64)[:, n_cat:]
    mu = np.nanmean(block, axis=0)
    sd = np.nanstd(block, axis=0)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    print(f"bloc numerique : {block.shape[1]} colonnes apres {n_cat} colonnes one-hot")
    print(f"ecarts-types : min {sd.min():.4f}  max {sd.max():.1f}  "
          f"rapport {sd.max() / sd.min():.0f}x", flush=True)

    out = Path(a.out)
    new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["arm", "dimension", "p", "bootstrap", "model", "auc", "brier"])

    jobs = ([("clean", None, 0.0)] if a.dims == "clean"
            else [(d, DIMS[d], a.level) for d in a.dims.split(",")])
    t0 = time.time()
    for name, fn, p in jobs:
        for b in range(a.reps):
            r = one_replicate(df_sample, pools, preproc, mu, sd, n_cat,
                              fn, p, b, with_tree=(name == "clean"))
            for arm, models in r.items():
                for model, m in models.items():
                    w.writerow([arm, name, p, b, model,
                                m.get("auc", np.nan), m.get("brier", np.nan)])
            if b % 10 == 0:
                fh.flush()
                el = time.time() - t0
                done = b + 1
                print(f"{name} p={p} b={b:03d}/{a.reps} | {el/60:.1f} min "
                      f"| {el/done:.1f} s/rep | reste ~{(a.reps-done)*el/done/60:.0f} min",
                      flush=True)
    fh.close()
    print("ecrit :", out, flush=True)


if __name__ == "__main__":
    main()
