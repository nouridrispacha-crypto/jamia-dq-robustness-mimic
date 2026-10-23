# ============================================================
# POLLUEUR DE PRÉCISION — DATASET 18 VARIABLES (FIXED)
# Compatible common_runner
# ============================================================

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional, Tuple, Set


# ============================================================
# RNG
# ============================================================

def _rng(seed: Optional[int]):
    return np.random.default_rng(None if seed is None else int(seed))


# ============================================================
# GROUPES DE VARIABLES
# ============================================================

VITALS = {
    "hr_mean", "sbp_mean", "dbp_min", "rr_mean", "spo2_min",
}

LABS = {
    "wbc_min", "aniongap_min", "aniongap_max",
    "bun_min", "inr_min", "inr_max", "ptt_min",
}

DRUGS = {
    "dobutamine", "dopamine", "norepinephrine", "phenylephrine",
}

OTHER = {
    "age", "urine_output",
}


# ============================================================
# BIAIS SYSTÉMATIQUES
# ============================================================

ADDITIVE_BIAS = {
    "sbp_mean": 3.0,
    "dbp_min": 2.0,
    "rr_mean": 1.0,
    "spo2_min": -1.0,
}

MULTIPLICATIVE_BIAS = {
    "hr_mean": 0.03,
    "wbc_min": 0.05,
    "bun_min": 0.05,
    "inr_max": 0.05,
    "ptt_min": 0.05,
    "urine_output": 0.10,
}


# ============================================================
# POLLUEUR PRINCIPAL (FIXED)
# ============================================================

def pollution_precision(
    df: pd.DataFrame,
    density: float,
    seed: int = 42,
):
    """
    Pollution de précision (cell-wise, contrôlée)

    - d% des cellules numériques sont réellement modifiées
    - aucune cellule modifiée deux fois
    - aucune modification nulle
    """

    assert 0.0 <= density <= 1.0

    rng = _rng(seed)
    df = deepcopy(df)

    PROTECTED = {"subject_id", "hadm_id", "stay_id", "in_hospital_mortality"}

    num_cols = [
        c for c in df.columns
        if c not in PROTECTED and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not num_cols or density == 0.0:
        return df, np.array([], dtype=int), np.array([], dtype=int)

    X = df[num_cols].to_numpy(dtype=float)
    finite_mask = np.isfinite(X)

    total_valid = int(finite_mask.sum())
    budget = int(round(density * total_valid))
    budget = max(1, min(budget, total_valid))

    touched: Set[Tuple[int, int]] = set()
    row_idx = []
    col_idx = []

    changed = 0
    attempts = 0
    max_attempts = budget * 20

    n_rows, n_cols = X.shape

    while changed < budget and attempts < max_attempts:
        attempts += 1

        r = int(rng.integers(0, n_rows))
        c = int(rng.integers(0, n_cols))

        if not finite_mask[r, c]:
            continue
        if (r, c) in touched:
            continue

        col = num_cols[c]
        x = X[r, c]
        new_x = x

        # -------- Bruit gaussien --------
        if rng.random() < 0.6:
            if col in VITALS:
                cv = 0.05
            elif col in LABS:
                cv = 0.03
            elif col in DRUGS:
                cv = 0.10
            else:
                cv = 0.05
            sigma = cv * abs(x) if abs(x) > 0 else cv
            new_x = x + rng.normal(0, sigma)

        # -------- Biais systématique --------
        elif rng.random() < 0.75:
            if col in ADDITIVE_BIAS:
                new_x = x + ADDITIVE_BIAS[col]
            elif col in MULTIPLICATIVE_BIAS:
                new_x = x * (1.0 + MULTIPLICATIVE_BIAS[col])

        # -------- Bruit impulsionnel --------
        else:
            if col in DRUGS:
                new_x = 0.0 if x != 0 else 1.0
            else:
                new_x = x * rng.choice([0.1, 10.0])

        # 🔒 garde-fou critique
        if not np.isfinite(new_x):
            continue
        if new_x == x:
            continue

        X[r, c] = new_x
        touched.add((r, c))
        row_idx.append(r)
        col_idx.append(df.columns.get_loc(col))
        changed += 1

    df[num_cols] = X

    return (
        df,
        np.array(row_idx, dtype=int),
        np.array(col_idx, dtype=int),
    )


pollution_precision.APPROXIMATE_RATE = False
pollution_precision.INTRODUCES_INVALID = False
