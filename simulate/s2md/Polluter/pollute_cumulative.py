# -*- coding: utf-8 -*-
"""
CUMULATIVE COMPOSITE POLLUTER -- Scenario S2-MD, cumulative design
Compatible common_runner (signature: fn(df, density, seed) -> (df, row_idx, col_idx))

PRINCIPLE
---------
The five DQ dimensions are applied to the SAME test set, so that one patient
can carry several types of defect. A predictor cell carries at most one
cell-level defect; only uniqueness, which acts on whole records, may share a
record (and its cells) with later cell-level defects.

  budget      B = round(c * n * p) predictor cells, p = predictor columns
              dimension k is allotted round(w_k * B) cells (profile weights)

  1. uniqueness  (record level, first, on the clean test set)
                 records are overwritten by a copy of another record of the
                 whole test set, as in the single-dimension polluter
                 (exact / noisy / age-targeted); these cells stay available
  2. coherence   (cell pairs; most constrained, reserved first)
  3. completeness (severity-weighted, severity read on the clean data)
  4. precision   (noise / bias / unit scale, same parameters)
  5. validity    (non-numeric tokens, invalid codes, invalid sex; written
                  last because it puts strings into numeric columns)

  Cell-level dimensions 2-5 draw only among cells not yet taken by 2-5,
  so no cell receives two cell-level defects. The reservation order is fixed
  (most constrained first); it changes which cells a dimension receives, not
  how many.

ACCOUNTING
----------
A cell-level defect may fall on a record already copied by uniqueness, so the
share of DISTINCT predictor cells altered is slightly below c (about 2-3% of
the budget at c = 0.3). This is accepted and reported: LAST_STATS holds, for
the last call, the budget, the cells reached per dimension, the share of
distinct cells altered and the share of patients carrying two or more types
of defect.

The single-dimension mechanisms are reused unchanged: uniqueness and coherence
call their polluters directly; completeness, precision and validity reproduce
their per-cell rules on the free cells only.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import os.path as _op
sys.path.insert(0, _op.join(_op.dirname(_op.dirname(_HERE)), 'polluter'))

from pollute_uniqueness import pollution_uniqueness                       # noqa: E402
from coherence_polluter import apply_coherence_pollution_18vars           # noqa: E402
from coherence_polluter import NUM_COLS as COH_NUM_COLS                   # noqa: E402
from pollute_precision import (VITALS, LABS, DRUGS, ADDITIVE_BIAS,        # noqa: E402
                               MULTIPLICATIVE_BIAS)
from pollute_validity import (NUM_COLS as VAL_NUM_COLS, CAT_COLS as VAL_CAT_COLS,  # noqa: E402
                              BOOL_COLS, TYPING_REPLS, BOOL_OOD, GENDER_OOD)
from pollute_mixed import W_OBSERVE                                        # noqa: E402

DIMENSIONS = ["uniqueness", "coherence", "completeness", "precision", "validity"]
PROTECTED = {"subject_id", "hadm_id", "stay_id", "in_hospital_mortality"}
SEED_OFFSET = {"uniqueness": 505, "coherence": 202, "completeness": 101,
               "precision": 404, "validity": 303}
ALPHA = 5.0          # completeness severity separation, as pollute_mnar_top
VALIDITY_WEIGHTS = {"typing": 0.6, "bool_ood": 0.4}


def _precision_value(x: float, col: str, rng: np.random.Generator) -> float:
    """Per-cell rule of pollution_precision (same probabilities and magnitudes)."""
    if rng.random() < 0.6:
        cv = 0.05 if col in VITALS else 0.03 if col in LABS else 0.10 if col in DRUGS else 0.05
        sigma = cv * abs(x) if abs(x) > 0 else cv
        return x + rng.normal(0, sigma)
    if rng.random() < 0.75:
        if col in ADDITIVE_BIAS:
            return x + ADDITIVE_BIAS[col]
        if col in MULTIPLICATIVE_BIAS:
            return x * (1.0 + MULTIPLICATIVE_BIAS[col])
        return x                                 # no bias for this variable: discarded
    if col in DRUGS:
        return 0.0 if x != 0 else 1.0
    return x * rng.choice([0.1, 10.0])


def make_cumulative_polluter(weights: Optional[Dict[str, float]] = None,
                             profile_name: str = "P4_observe_cumulative"):
    weights = dict(W_OBSERVE if weights is None else weights)
    total = float(sum(weights.values()))
    weights = {d: float(weights.get(d, 0.0)) / total for d in DIMENSIONS}

    def pollution_cumulative(df: pd.DataFrame, density: float, seed: int = 42
                             ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        c = float(density)
        assert 0.0 <= c <= 1.0
        n = len(df)
        if n == 0 or c == 0.0:
            pollution_cumulative.LAST_STATS = {}
            return df.copy(), np.array([], dtype=int), np.array([], dtype=int)

        original_index = df.index
        clean = df.reset_index(drop=True)
        X = clean.copy()
        pred = [col for col in X.columns if col not in PROTECTED]
        num = [col for col in pred if pd.api.types.is_numeric_dtype(X[col])]
        for col in num:
            X[col] = X[col].astype("float64")
        colpos = {col: X.columns.get_loc(col) for col in X.columns}

        p = len(pred)
        B = int(round(c * n * p))
        target = {d: int(round(weights[d] * B)) for d in DIMENSIONS}
        reached = {d: 0 for d in DIMENSIONS}
        taken = np.zeros((n, X.shape[1]), dtype=bool)        # cell-level reservations
        cell_dim = np.full((n, X.shape[1]), "", dtype=object)
        rows_all: List[np.ndarray] = []
        cols_all: List[np.ndarray] = []

        # ---- 1. uniqueness: record level, first, donors from the whole test set
        uniq_rows = np.array([], dtype=int)
        if target["uniqueness"] > 0:
            X, r, cc = pollution_uniqueness(X, target["uniqueness"] / (n * p),
                                            seed=int(seed) + SEED_OFFSET["uniqueness"])
            X = X.reset_index(drop=True)
            for col in num:
                X[col] = X[col].astype("float64")
            reached["uniqueness"] = len(r)
            uniq_rows = np.unique(r)
            rows_all.append(np.asarray(r, int))
            cols_all.append(np.asarray(cc, int))

        # ---- 2. coherence: pairs, reserved first among cell-level dimensions
        coh_cols = [col for col in COH_NUM_COLS if col in X.columns and col not in PROTECTED]
        if target["coherence"] > 0 and coh_cols:
            p_coh = min(1.0, target["coherence"] / (n * len(coh_cols)))
            X, r, cc = apply_coherence_pollution_18vars(
                X, p=p_coh, seed=int(seed) + SEED_OFFSET["coherence"])
            X = X.reset_index(drop=True)
            r, cc = np.asarray(r, int), np.asarray(cc, int)
            taken[r, cc] = True
            cell_dim[r, cc] = "coherence"
            reached["coherence"] = len(r)
            rows_all.append(r)
            cols_all.append(cc)

        # ---- 3. completeness: severity-weighted among free cells
        if target["completeness"] > 0:
            sev = clean["norepinephrine"].to_numpy(dtype=float)
            sev = np.nan_to_num(sev, nan=0.0).astype(int)
            p_sev, p_non = 1.0 / (1.0 + np.exp(-ALPHA / 2)), 1.0 / (1.0 + np.exp(ALPHA / 2))
            p_row = np.where(sev == 1, p_sev, p_non)
            p_row = p_row / p_row.mean()
            pos = np.array([colpos[col] for col in pred])
            free = ~taken[:, pos] & ~X[pred].isna().to_numpy()
            W = p_row[:, None] * free
            k = min(target["completeness"], int(free.sum()))
            if k > 0 and W.sum() > 0:
                rng = np.random.default_rng(int(seed) + SEED_OFFSET["completeness"])
                flat = rng.choice(W.size, size=k, replace=False, p=(W / W.sum()).ravel())
                r, cl = flat // len(pred), flat % len(pred)
                cc = pos[cl]
                for ri, ci in zip(r, cc):
                    X.iat[ri, ci] = np.nan
                taken[r, cc] = True
                cell_dim[r, cc] = "completeness"
                reached["completeness"] = k
                rows_all.append(r.astype(int))
                cols_all.append(cc.astype(int))

        # ---- 4. precision: same per-cell rule, free finite numeric cells
        if target["precision"] > 0:
            rng = np.random.default_rng(int(seed) + SEED_OFFSET["precision"])
            prec_cols = [col for col in num if col in X.columns]
            vals = X[prec_cols].to_numpy(dtype=float)
            pos = np.array([colpos[col] for col in prec_cols])
            cand = np.argwhere(~taken[:, pos] & np.isfinite(vals))
            cand = cand[rng.permutation(len(cand))]
            r_out, c_out = [], []
            for ri, cj in cand:
                if len(r_out) >= target["precision"]:
                    break
                col = prec_cols[cj]
                x = vals[ri, cj]
                new_x = _precision_value(x, col, rng)
                if not np.isfinite(new_x) or new_x == x:
                    continue
                X.iat[ri, pos[cj]] = new_x
                r_out.append(ri)
                c_out.append(pos[cj])
            r_out, c_out = np.array(r_out, int), np.array(c_out, int)
            taken[r_out, c_out] = True
            cell_dim[r_out, c_out] = "precision"
            reached["precision"] = len(r_out)
            rows_all.append(r_out)
            cols_all.append(c_out)

        # ---- 5. validity: last (writes strings), free cells incl. sex
        if target["validity"] > 0:
            rng = np.random.default_rng(int(seed) + SEED_OFFSET["validity"])
            val_cols = ([col for col in VAL_NUM_COLS if col in X.columns and col not in PROTECTED]
                        + [col for col in VAL_CAT_COLS if col in X.columns])
            pos = np.array([colpos[col] for col in val_cols])
            cand = np.argwhere(~taken[:, pos])
            k = min(target["validity"], len(cand))
            pick = cand[rng.choice(len(cand), size=k, replace=False)] if k > 0 else np.empty((0, 2), int)
            types = rng.choice(list(VALIDITY_WEIGHTS), size=k, p=list(VALIDITY_WEIGHTS.values()))
            for col in {val_cols[j] for _, j in pick if val_cols[j] in num}:
                X[col] = X[col].astype(object)
            r_out, c_out = [], []
            for (ri, cj), t in zip(pick, types):
                col = val_cols[cj]
                if col in VAL_CAT_COLS:
                    v = GENDER_OOD[rng.integers(0, len(GENDER_OOD))]
                elif t == "bool_ood" and col in BOOL_COLS:
                    v = BOOL_OOD[rng.integers(0, len(BOOL_OOD))]
                else:
                    v = TYPING_REPLS[rng.integers(0, len(TYPING_REPLS))]
                X.iat[ri, pos[cj]] = v
                r_out.append(ri)
                c_out.append(pos[cj])
            r_out, c_out = np.array(r_out, int), np.array(c_out, int)
            taken[r_out, c_out] = True
            cell_dim[r_out, c_out] = "validity"
            reached["validity"] = len(r_out)
            rows_all.append(r_out)
            cols_all.append(c_out)

        # ---- diagnostics
        changed = np.zeros((n, p), dtype=bool)
        for j, col in enumerate(pred):
            before, after = clean[col], X[col]
            if col in num:
                a_num = pd.to_numeric(before, errors="coerce").to_numpy(dtype=float)
                b_num = pd.to_numeric(after, errors="coerce").to_numpy(dtype=float)
                b_str = after.map(lambda v: isinstance(v, str)).to_numpy()
                same = (a_num == b_num) | (np.isnan(a_num) & np.isnan(b_num) & ~b_str)
            else:
                same = (before.astype(str) == after.astype(str)).to_numpy()
            changed[:, j] = ~same
        types_per_row = np.zeros(n, dtype=int)
        for d in ("coherence", "completeness", "precision", "validity"):
            types_per_row += (cell_dim == d).any(axis=1)
        types_per_row[uniq_rows] += 1
        pollution_cumulative.LAST_STATS = {
            "budget_cells": B,
            "target": target,
            "reached": reached,
            "share_distinct_cells_altered": float(changed.mean()),
            "share_patients_2plus_types": float((types_per_row >= 2).mean()),
            "share_patients_any_defect": float((types_per_row >= 1).mean()),
        }

        X.index = original_index
        row_idx = np.concatenate(rows_all).astype(int) if rows_all else np.array([], int)
        col_idx = np.concatenate(cols_all).astype(int) if cols_all else np.array([], int)
        return X, row_idx, col_idx

    pollution_cumulative.APPROXIMATE_RATE = True
    pollution_cumulative.INTRODUCES_INVALID = True
    pollution_cumulative.PROFILE_NAME = profile_name
    pollution_cumulative.WEIGHTS = weights
    pollution_cumulative.LAST_STATS = {}
    return pollution_cumulative


pollution_cumulative_observed = make_cumulative_polluter(W_OBSERVE, "P4_observe_cumulative")
