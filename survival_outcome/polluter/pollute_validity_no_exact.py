"""
VALIDITY POLLUTER — EXACT CELL-LEVEL BUDGET
Compatible COMMON_RUNNER

Corrupts exactly p fraction of eligible cells, without overlap.
Types:
1) typing   : non-numeric values in numeric columns
2) cat_ood  : unexpected categories in gender
3) bool_ood : invalid values for 0/1 variables
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from copy import deepcopy

# ============================================================
# CONSTANTES
# ============================================================

ID_COLS = ["subject_id", "hadm_id", "stay_id"]
TARGET_COL = "in_hospital_mortality"

CAT_COLS = ["gender"]

NUM_COLS = [
    "age", "hr_mean", "sbp_mean", "dbp_min", "rr_mean", "spo2_min",
    "wbc_min", "aniongap_min", "aniongap_max", "bun_min",
    "inr_min", "inr_max", "ptt_min", "urine_output",
    "dobutamine", "dopamine", "norepinephrine", "phenylephrine",
]

BOOL_COLS = ["dobutamine", "dopamine", "norepinephrine", "phenylephrine"]

PROTECTED = set(ID_COLS + [TARGET_COL])

# Valeurs invalides
TYPING_REPLS = np.array(["??", "N/A", "--", "abc", "1O.O"], dtype=object)
GENDER_OOD  = np.array(["X", "U", "?", "unknown"], dtype=object)
BOOL_OOD    = np.array([2, -1, 3, "yes", "no"], dtype=object)

# ============================================================
# RNG
# ============================================================

def _rng(seed: Optional[int]):
    return np.random.default_rng(None if seed is None else int(seed))

# ============================================================
# POLLUEUR PRINCIPAL
# ============================================================

def apply_validity_pollution_exact(
    df: pd.DataFrame,
    p: float,
    seed: int = 42,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Corrupt exactly p fraction of eligible cells.

    Returns:
        df_polluted, row_idx, col_idx
    """

    assert 0.0 <= p <= 1.0, "p must be in [0,1]"

    rng = _rng(seed)
    Xp = deepcopy(df)

    # Colonnes disponibles
    num_cols  = [c for c in NUM_COLS if c in Xp.columns and c not in PROTECTED]
    cat_cols  = [c for c in CAT_COLS if c in Xp.columns and c not in PROTECTED]
    bool_cols = [c for c in BOOL_COLS if c in Xp.columns and c not in PROTECTED]

    n_rows = len(Xp)
    if n_rows == 0 or p == 0.0:
        return Xp, np.array([], dtype=int), np.array([], dtype=int)

    # ------------------------------------------------------------
    # Cellules polluables (GLOBAL)
    # ------------------------------------------------------------
    polluable = []
    for c in num_cols:
        polluable.extend([(r, c) for r in range(n_rows)])
    for c in cat_cols:
        polluable.extend([(r, c) for r in range(n_rows)])
    for c in bool_cols:
        polluable.extend([(r, c) for r in range(n_rows)])

    if not polluable:
        return Xp, np.array([], dtype=int), np.array([], dtype=int)

    polluable = np.array(polluable, dtype=object)

    # ------------------------------------------------------------
    # Budget exact
    # ------------------------------------------------------------
    total_cells = len(polluable)
    budget = int(round(p * total_cells))
    budget = max(1, min(budget, total_cells))

    chosen_idx = rng.choice(total_cells, size=budget, replace=False)
    chosen_cells = polluable[chosen_idx]

    # ------------------------------------------------------------
    # Types d'erreurs
    # ------------------------------------------------------------
    if weights is None:
        weights = {
            "typing": 0.55,
            "cat_ood": 0.15,
            "bool_ood": 0.30,
        }

    types = np.array(list(weights.keys()))
    probs = np.array([weights[t] for t in types], dtype=float)
    probs /= probs.sum()

    assigned = rng.choice(types, size=len(chosen_cells), p=probs)

    # ------------------------------------------------------------
    # Application
    # ------------------------------------------------------------
    row_idx = []
    col_idx = []

    for (r, c), t in zip(chosen_cells, assigned):

        # TYPING
        if t == "typing" and c in num_cols:
            Xp.at[r, c] = TYPING_REPLS[rng.integers(0, len(TYPING_REPLS))]

        # CAT OOD
        elif t == "cat_ood" and c in cat_cols:
            Xp.at[r, c] = GENDER_OOD[rng.integers(0, len(GENDER_OOD))]

        # BOOL OOD
        elif t == "bool_ood" and c in bool_cols:
            Xp.at[r, c] = BOOL_OOD[rng.integers(0, len(BOOL_OOD))]

        else:
            # Type incompatible → fallback sûr
            if c in num_cols:
                Xp.at[r, c] = TYPING_REPLS[rng.integers(0, len(TYPING_REPLS))]
            elif c in cat_cols:
                Xp.at[r, c] = GENDER_OOD[rng.integers(0, len(GENDER_OOD))]
            elif c in bool_cols:
                Xp.at[r, c] = BOOL_OOD[rng.integers(0, len(BOOL_OOD))]
            else:
                continue

        row_idx.append(int(r))
        col_idx.append(Xp.columns.get_loc(c))

    return Xp, np.array(row_idx, dtype=int), np.array(col_idx, dtype=int)

# ============================================================
# WRAPPER COMMON_RUNNER
# ============================================================

def pollution_validity(df: pd.DataFrame, density: float, seed: int = 42):
    return apply_validity_pollution_exact(df, p=float(density), seed=int(seed))

def validity_pollution(df: pd.DataFrame, density: float, seed: int = 42):
    return pollution_validity(df, density=density, seed=seed)

pollution_validity.INTRODUCES_INVALID = True
pollution_validity.APPROXIMATE_RATE = True