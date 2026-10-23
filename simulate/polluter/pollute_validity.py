# -*- coding: utf-8 -*-
"""
VALIDITY POLLUTER — NUMERIC-CELL ALIGNED BUDGET
Compatible common_runner

- Budget calculé STRICTEMENT sur NUM_COLS
- Atteint p_real ≈ p_target même pour p=0.5
- Pas de double comptage
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List
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
    "wbc_min", "aniongap_min", "aniongap_max",
    "bun_min", "inr_min", "inr_max", "ptt_min", "urine_output",
    "dobutamine", "dopamine", "norepinephrine", "phenylephrine",
]

BOOL_COLS = ["dobutamine", "dopamine", "norepinephrine", "phenylephrine"]

PROTECTED = set(ID_COLS + [TARGET_COL])

# Valeurs invalides
TYPING_REPLS = np.array(["??", "N/A", "--", "abc", "1O.O"], dtype=object)
BOOL_OOD    = np.array([2, -1, 3], dtype=object)
GENDER_OOD  = np.array(["X", "U", "?"], dtype=object)

# ============================================================
# RNG
# ============================================================

def _rng(seed: Optional[int]):
    return np.random.default_rng(None if seed is None else int(seed))

# ============================================================
# POLLUEUR PRINCIPAL
# ============================================================

def apply_validity_pollution(
    df: pd.DataFrame,
    p: float,
    seed: int = 42,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Corrompt une fraction p EXACTE des cellules prédictrices : NUM_COLS, plus
    la colonne catégorielle gender lorsqu'elle est présente (binaire, survie).

    - typing    : valeurs non numériques
    - bool_ood  : valeurs invalides pour 0/1
    - cat_ood   : catégorie invalide pour gender, dans le budget

    Retour:
        df_polluted, row_idx, col_idx
    """

    assert 0.0 <= p <= 1.0

    rng = _rng(seed)
    Xp = deepcopy(df)

    num_cols = [c for c in NUM_COLS if c in Xp.columns and c not in PROTECTED]
    bool_cols = [c for c in BOOL_COLS if c in num_cols]
    cat_cols  = [c for c in CAT_COLS if c in Xp.columns]

    n_rows = len(Xp)
    if n_rows == 0 or p == 0.0 or not num_cols:
        return Xp, np.array([], int), np.array([], int)

    # ========================================================
    # ESPACE DE COMPTAGE = EXACTEMENT CELUI DU RUNNER
    # ========================================================

    polluable_cells: List[Tuple[int, str]] = [
        (r, c) for r in range(n_rows) for c in num_cols + cat_cols
    ]

    total_cells = len(polluable_cells)
    budget = int(round(p * total_cells))
    budget = max(1, min(budget, total_cells))

    rng.shuffle(polluable_cells)
    chosen_cells = polluable_cells[:budget]

    # ========================================================
    # TYPES D'ERREURS
    # ========================================================

    if weights is None:
        weights = {
            "typing": 0.6,
            "bool_ood": 0.4,
        }

    types = np.array(list(weights.keys()))
    probs = np.array([weights[t] for t in types], dtype=float)
    probs /= probs.sum()

    assigned = rng.choice(types, size=len(chosen_cells), p=probs)

    # ========================================================
    # APPLICATION
    # ========================================================

    row_idx, col_idx = [], []

    for (r, c), t in zip(chosen_cells, assigned):

        # ---------- CAT OOD (gender, dans le budget) ----------
        if c in cat_cols:
            Xp.at[r, c] = GENDER_OOD[rng.integers(0, len(GENDER_OOD))]

        # ---------- TYPING ----------
        elif t == "typing":
            Xp.at[r, c] = TYPING_REPLS[rng.integers(0, len(TYPING_REPLS))]

        # ---------- BOOL OOD ----------
        elif t == "bool_ood" and c in bool_cols:
            Xp.at[r, c] = BOOL_OOD[rng.integers(0, len(BOOL_OOD))]

        # ---------- FALLBACK SÛR ----------
        else:
            Xp.at[r, c] = TYPING_REPLS[rng.integers(0, len(TYPING_REPLS))]

        row_idx.append(r)
        col_idx.append(Xp.columns.get_loc(c))

    return Xp, np.array(row_idx), np.array(col_idx)

# ============================================================
# WRAPPER COMMON_RUNNER
# ============================================================

def pollution_validity(df: pd.DataFrame, density: float, seed: int = 42):
    return apply_validity_pollution(df, p=float(density), seed=int(seed))

pollution_validity.INTRODUCES_INVALID = True
pollution_validity.APPROXIMATE_RATE = True
