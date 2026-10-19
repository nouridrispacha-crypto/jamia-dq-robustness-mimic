"""Binary task with every preprocessing step fitted on the in-bag sample alone.

The published pipeline builds the hot-deck donor pools, the one-hot encoder and the
standardisation constants once per cohort, before the bootstrap loop, and then applies them
to both the in-bag and the out-of-bag set. The out-of-bag predictor distribution therefore
takes part in building the preprocessing that is applied to it. This runner does what a
reviewer would ask for instead, inside every replicate:

    split -> fit donor pools, encoder and standardisation on the in-bag -> apply to the OOB

Nothing else changes: the same cohort, the same stratified resampler and seeds, the same
polluters, the same five model constructors as runner_full.py, whose fit_all and
one_replicate this module reuses so the two differ in the preprocessing and nowhere else.

Note that under S1 and S3 the in-bag is corrupted before the constants are computed, so the
standardisation is itself estimated on corrupted data. That is the realistic pipeline and
the point of the exercise, but it means the scaling is no longer common across corruption
levels, unlike in the published runs.

USAGE
    python runner_noleak.py --scen S2 --levels 0.3 --reps 500 --out noleak_S2.csv
    python runner_noleak.py --compare 20      # same replicates, with and without the leak
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

import common_runner as CR                                            # noqa: E402
import runner_full as RF                                              # noqa: E402

DIMS, SCEN = RF.DIMS, RF.SCEN


def one_replicate_noleak(df, fn, p, b, scen, arms):
    """RF.one_replicate with the preprocessing re-fitted on the in-bag of this replicate."""
    do_train, do_test = SCEN[scen]
    inbag, oob = CR.stratified_inbag_oob(df, CR.TARGET_COL, seed=10_000 + b)

    # ---- donor pools from the in-bag only (published: once on the whole cohort)
    pre = CR.enforce_column_types(CR.ensure_schema(inbag.copy()))
    pools = CR.build_sex_conditioned_pools(pre, CR.SEX_COL, CR.IMPUTE_COLS)

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

    # ---- encoder fitted on the in-bag: preprocessor=None takes the fit_transform branch
    preproc, Xtr, Xoob, ytr, yoob = CR.prepare_data_for_models(inbag, oob, None)
    n_cat = sum(len(c) for c in preproc.named_transformers_["cat"].categories_)
    Xtr = np.nan_to_num(Xtr, nan=0.0).astype(np.float64)
    Xoob = np.nan_to_num(Xoob, nan=0.0).astype(np.float64)

    out = {}
    if "raw" in arms:
        out["raw"] = RF.fit_all(Xtr.astype(np.float32), Xoob.astype(np.float32), ytr, yoob)
    if "scaled" in arms:
        # ---- standardisation constants from the in-bag block only
        blk = Xtr[:, n_cat:]
        mu, sd = np.nanmean(blk, axis=0), np.nanstd(blk, axis=0)
        sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
        S, O = Xtr.copy(), Xoob.copy()
        S[:, n_cat:] = (S[:, n_cat:] - mu) / sd
        O[:, n_cat:] = (O[:, n_cat:] - mu) / sd
        out["scaled"] = RF.fit_all(S.astype(np.float32), O.astype(np.float32), ytr, yoob)
    return out


def compare(reps, scen="S2", level=0.3):
    """Both pipelines on the same replicates, so the difference is the preprocessing."""
    df, pools, preproc, mu, sd, n_cat = RF.setup()
    print(f"{reps} replicats, {scen}, c={level}, bras standardise\n")
    rows = []
    t0 = time.time()
    for dim, fn in DIMS.items():
        for b in range(reps):
            a = RF.one_replicate(df, pools, preproc, mu, sd, n_cat, fn, level, b,
                                 scen, ("scaled",))["scaled"]
            n = one_replicate_noleak(df, fn, level, b, scen, ("scaled",))["scaled"]
            for m in a:
                rows.append((dim, m, b, a[m]["auc"], n[m]["auc"],
                             a[m]["brier"], n[m]["brier"]))
        print(f"  {dim} fait ({(time.time()-t0)/60:.1f} min)", flush=True)
    d = pd.DataFrame(rows, columns=["dim", "model", "b", "auc_pub", "auc_nl",
                                    "brier_pub", "brier_nl"])
    d["d_auc"] = d.auc_nl - d.auc_pub
    d["d_brier"] = d.brier_nl - d.brier_pub
    print("\necart (sans fuite - publie), mediane sur les replicats\n")
    print("  {:14s}{:>12s}{:>12s}".format("modele", "d AUROC", "d Brier"))
    for m, g in d.groupby("model"):
        print("  {:14s}{:>12.5f}{:>12.5f}".format(m, g.d_auc.median(), g.d_brier.median()))
    print("\n  ecart absolu maximal : AUROC {:.5f}, Brier {:.5f}".format(
        d.d_auc.abs().max(), d.d_brier.abs().max()))
    d.to_csv(HERE / "noleak_compare.csv", index=False)
    print("\n  detail : noleak_compare.csv")
    print("  duree {:.1f} min pour {} replicats x 5 dimensions x 2 pipelines".format(
        (time.time() - t0) / 60, reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=int, default=0)
    ap.add_argument("--scen", default="S2")
    ap.add_argument("--dims", default=",".join(DIMS))
    ap.add_argument("--levels", default="0.3")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--arms", default="scaled")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=3500)
    ap.add_argument("--models", default=None)
    a = ap.parse_args()
    if a.models:
        RF.WANT = set(a.models.split(","))
    if a.compare:
        compare(a.compare, a.scen, float(a.levels.split(",")[0]))
        return

    df = pd.read_csv(HERE / "data" / "18_variables.csv")
    df = df.sample(n=a.n, random_state=42).reset_index(drop=True)
    arms = tuple(a.arms.split(","))
    out = Path(a.out or HERE / f"noleak_{a.scen}.csv")
    new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["arm", "scenario", "dimension", "p", "bootstrap", "model",
                    "auc", "brier", "N"])
    jobs = [(d, DIMS[d], float(p)) for d in a.dims.split(",") for p in a.levels.split(",")]
    t0, done, total = time.time(), 0, len(jobs) * a.reps
    for dim, fn, p in jobs:
        for b in range(a.reps):
            r = one_replicate_noleak(df, fn, p, b, a.scen, arms)
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
