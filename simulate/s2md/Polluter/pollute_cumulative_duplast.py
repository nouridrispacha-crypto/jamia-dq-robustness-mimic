# -*- coding: utf-8 -*-
"""
CUMULATIVE COMPOSITE POLLUTER, DUPLICATION LAST -- Scenario S2-MD
Compatible common_runner (signature: fn(df, density, seed) -> (df, row_idx, col_idx))

Same design as pollute_cumulative.py except for uniqueness, which now copies
records AFTER the cell-level defects, so that a duplicate reproduces its source
record as recorded, defects included, instead of becoming a new patient.

  budget      B = round(c * n * p) predictor cells, p = predictor columns
              dimension k is allotted round(w_k * B) cells (profile weights)

  1. uniqueness targets are drawn first, with the rule of the single-dimension
     polluter (target record, source record, exact / noisy / age-targeted),
     and the budget is counted on the clean test set; nothing is written yet.
     The target records are reserved: no cell-level defect falls on them.
  2. coherence, 3. completeness, 4. precision, 5. validity act on the other
     records, on distinct cells, in the fixed order of pollute_cumulative.py.
  6. uniqueness is applied: each target record is overwritten by a copy of its
     source record as it stands after steps 2-5, in the order of the draw.

A copied record therefore carries the defects of its source, and a patient can
carry several types of defect through a copy. No cell receives two cell-level
defects. LAST_STATS reports the same diagnostics as pollute_cumulative.py.
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

from coherence_polluter import apply_coherence_pollution_18vars           # noqa: E402
from coherence_polluter import NUM_COLS as COH_NUM_COLS                   # noqa: E402
from pollute_validity import (NUM_COLS as VAL_NUM_COLS, CAT_COLS as VAL_CAT_COLS,  # noqa: E402
                              BOOL_COLS, TYPING_REPLS, BOOL_OOD, GENDER_OOD)
from pollute_mixed import W_OBSERVE                                        # noqa: E402
from pollute_cumulative import (DIMENSIONS, PROTECTED, SEED_OFFSET, ALPHA,  # noqa: E402
                                VALIDITY_WEIGHTS, _precision_value)


def _draw_uniqueness(clean: pd.DataFrame, pred: List[str], target_cells: int, seed: int,
                     age_col: str = "age", age_quantile: float = 0.80):
    """Draw (target, source) pairs with the rule of pollution_uniqueness, counting the
    budget on a simulated copy of the clean data; nothing is written to the output."""
    rng = np.random.default_rng(int(seed))
    D = clean[pred].copy().reset_index(drop=True)
    N = len(D)
    budget = max(1, min(int(target_cells), N * len(pred)))
    if age_col in D.columns:
        thr = D[age_col].quantile(age_quantile)
        age_pool = D.index[D[age_col] >= thr].to_numpy()
    else:
        age_pool = np.array([], int)
    touched, pairs = set(), []
    changed, attempts = 0, 0
    while changed < budget and attempts < budget * 20:
        attempts += 1
        t = int(rng.integers(0, N))
        if t in touched:
            continue
        mode = rng.choice(["exact", "noisy", "age"])
        s = int(rng.choice(age_pool)) if mode == "age" and len(age_pool) > 0 else int(rng.integers(0, N))
        if s == t:
            continue
        before, after = D.iloc[t].values, D.iloc[s].values   # noisy: single-row SD is 0, no noise
        n_changed = int((before != after).sum())
        if n_changed == 0:
            continue
        D.iloc[t] = after
        touched.add(t)
        pairs.append((t, s))
        changed += n_changed
    return pairs


def make_cumulative_duplast_polluter(weights: Optional[Dict[str, float]] = None,
                                     profile_name: str = "P4_observe_cumulative_duplast",
                                     completeness_repair: float = 0.0):
    """completeness_repair: share of the completeness defects written back with the true
    value, before duplication, all other defects left untouched (perfect repair, an upper
    bound on what imputation could recover)."""
    weights = dict(W_OBSERVE if weights is None else weights)
    total = float(sum(weights.values()))
    weights = {d: float(weights.get(d, 0.0)) / total for d in DIMENSIONS}

    def pollution_cumulative_duplast(df: pd.DataFrame, density: float, seed: int = 42
                                     ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        c = float(density)
        assert 0.0 <= c <= 1.0
        n = len(df)
        if n == 0 or c == 0.0:
            pollution_cumulative_duplast.LAST_STATS = {}
            return df.copy(), np.array([], dtype=int), np.array([], dtype=int)

        original_index = df.index
        clean = df.reset_index(drop=True)
        X = clean.copy()
        pred = [col for col in X.columns if col not in PROTECTED]
        num = [col for col in pred if pd.api.types.is_numeric_dtype(X[col])]
        for col in num:
            X[col] = X[col].astype("float64")
        colpos = {col: X.columns.get_loc(col) for col in X.columns}
        predpos = np.array([colpos[col] for col in pred])

        p = len(pred)
        B = int(round(c * n * p))
        target = {d: int(round(weights[d] * B)) for d in DIMENSIONS}
        reached = {d: 0 for d in DIMENSIONS}
        taken = np.zeros((n, X.shape[1]), dtype=bool)
        cell_dim = np.full((n, X.shape[1]), "", dtype=object)
        rows_all: List[np.ndarray] = []
        cols_all: List[np.ndarray] = []

        # ---- 1. uniqueness: draw targets and sources on the clean data, reserve targets
        pairs = []
        if target["uniqueness"] > 0:
            pairs = _draw_uniqueness(clean, pred, target["uniqueness"],
                                     seed=int(seed) + SEED_OFFSET["uniqueness"])
        uniq_rows = np.array(sorted({t for t, _ in pairs}), dtype=int)
        is_target = np.zeros(n, dtype=bool)
        is_target[uniq_rows] = True
        taken[uniq_rows, :] = True
        free_rows = np.flatnonzero(~is_target)

        # ---- 2. coherence: pairs, on the non-target records only
        coh_cols = [col for col in COH_NUM_COLS if col in X.columns and col not in PROTECTED]
        if target["coherence"] > 0 and coh_cols and len(free_rows):
            p_coh = min(1.0, target["coherence"] / (len(free_rows) * len(coh_cols)))
            sub = X.iloc[free_rows].reset_index(drop=True)
            sub, r, cc = apply_coherence_pollution_18vars(
                sub, p=p_coh, seed=int(seed) + SEED_OFFSET["coherence"])
            r, cc = free_rows[np.asarray(r, int)], np.asarray(cc, int)
            for col in coh_cols:
                X.iloc[free_rows, colpos[col]] = sub[col].to_numpy()
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
            free = ~taken[:, predpos] & ~X[pred].isna().to_numpy()
            W = p_row[:, None] * free
            k = min(target["completeness"], int(free.sum()))
            if k > 0 and W.sum() > 0:
                rng = np.random.default_rng(int(seed) + SEED_OFFSET["completeness"])
                flat = rng.choice(W.size, size=k, replace=False, p=(W / W.sum()).ravel())
                r, cl = flat // len(pred), flat % len(pred)
                cc = predpos[cl]
                for ri, ci in zip(r, cc):
                    X.iat[ri, ci] = np.nan
                taken[r, cc] = True
                cell_dim[r, cc] = "completeness"
                reached["completeness"] = k
                if completeness_repair > 0:
                    # write the true value back into a share of the missing cells; the
                    # cells stay reserved, so every other defect is left where it is
                    rep = np.random.default_rng(int(seed) + 909).choice(
                        k, size=int(round(completeness_repair * k)), replace=False)
                    for ri, ci in zip(r[rep], cc[rep]):
                        X.iat[ri, ci] = clean.iat[ri, ci]
                        cell_dim[ri, ci] = "repaired"
                    reached["completeness"] = k - len(rep)
                    reached["completeness_repaired"] = len(rep)
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

        # ---- 5. validity: free cells incl. sex
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

        # ---- 6. uniqueness applied last: copy each source record as it now stands
        inherited = np.zeros(n, dtype=int)
        for t, s in pairs:
            for j, col in enumerate(pred):
                X.iat[t, predpos[j]] = X.iat[s, predpos[j]]
            cell_dim[t, :] = cell_dim[s, :]
        for t in uniq_rows:
            inherited[t] = int(sum((cell_dim[t] == d).any()
                                   for d in ("coherence", "completeness", "precision", "validity")))

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
        if len(uniq_rows):
            ur, uc = np.nonzero(changed[uniq_rows])
            rows_all.append(uniq_rows[ur])
            cols_all.append(predpos[uc])
            reached["uniqueness"] = int(changed[uniq_rows].sum())
        types_per_row = np.zeros(n, dtype=int)
        for d in ("coherence", "completeness", "precision", "validity"):
            types_per_row += (cell_dim == d).any(axis=1)
        types_per_row[uniq_rows] += 1
        pollution_cumulative_duplast.LAST_STATS = {
            "budget_cells": B,
            "target": target,
            "reached": reached,
            "n_duplicates": int(len(uniq_rows)),
            "share_duplicates_with_inherited_defects": float((inherited[uniq_rows] > 0).mean()) if len(uniq_rows) else 0.0,
            "share_distinct_cells_altered": float(changed.mean()),
            "share_patients_2plus_types": float((types_per_row >= 2).mean()),
            "share_patients_any_defect": float((types_per_row >= 1).mean()),
        }

        X.index = original_index
        row_idx = np.concatenate(rows_all).astype(int) if rows_all else np.array([], int)
        col_idx = np.concatenate(cols_all).astype(int) if cols_all else np.array([], int)
        return X, row_idx, col_idx

    pollution_cumulative_duplast.APPROXIMATE_RATE = True
    pollution_cumulative_duplast.INTRODUCES_INVALID = True
    pollution_cumulative_duplast.PROFILE_NAME = profile_name
    pollution_cumulative_duplast.WEIGHTS = weights
    pollution_cumulative_duplast.LAST_STATS = {}
    return pollution_cumulative_duplast


pollution_cumulative_duplast_observed = make_cumulative_duplast_polluter(W_OBSERVE)
