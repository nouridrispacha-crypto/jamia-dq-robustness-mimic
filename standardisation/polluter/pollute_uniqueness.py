# ============================================================
# POLLUEUR D’UNICITÉ — DATASET 18 VARIABLES (FIXED)
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
# NOISE (léger mais effectif)
# ============================================================

def _noisify_block(
    X: pd.DataFrame,
    rng: np.random.Generator,
    num_noise_std: float = 0.05,
) -> pd.DataFrame:
    X = X.copy()

    num_cols = X.select_dtypes(include="number").columns
    for c in num_cols:
        sd = np.nanstd(X[c].values)
        if np.isfinite(sd) and sd > 0:
            X[c] = X[c] + rng.normal(0, num_noise_std * sd, size=len(X))
    return X


# ============================================================
# POLLUEUR PRINCIPAL
# ============================================================

def pollution_uniqueness(
    df: pd.DataFrame,
    density: float,
    seed: int = 42,
    age_col: str = "age",
    age_quantile: float = 0.80,
):
    """
    Pollution d’unicité (row-wise, contrôlée)

    - Atteint réellement p% de cellules modifiées
    - Une ligne est polluée une seule fois
    - Duplication exacte / noisy / ciblée âge
    """

    assert 0.0 <= density <= 1.0

    rng = _rng(seed)
    df = deepcopy(df).reset_index(drop=True)

    N = len(df)
    if N == 0 or density == 0.0:
        return df, np.array([], int), np.array([], int)

    PROTECTED = {"subject_id", "hadm_id", "stay_id", "in_hospital_mortality"}
    pollutable_cols = [c for c in df.columns if c not in PROTECTED]

    if not pollutable_cols:
        return df, np.array([], int), np.array([], int)

    total_cells = N * len(pollutable_cols)
    budget = int(round(density * total_cells))
    budget = max(1, min(budget, total_cells))

    touched_rows: Set[int] = set()
    row_idx, col_idx = [], []

    changed_cells = 0
    attempts = 0
    max_attempts = budget * 20

    # Pré-calc âge
    if age_col in df.columns:
        age_thr = df[age_col].quantile(age_quantile)
        age_pool = df.index[df[age_col] >= age_thr].to_numpy()
    else:
        age_pool = np.array([], int)

    while changed_cells < budget and attempts < max_attempts:
        attempts += 1

        t = int(rng.integers(0, N))
        if t in touched_rows:
            continue

        # Choix du scénario
        mode = rng.choice(["exact", "noisy", "age"])

        if mode == "age" and len(age_pool) > 0:
            s = int(rng.choice(age_pool))
        else:
            s = int(rng.integers(0, N))

        if s == t:
            continue

        before = df.loc[t, pollutable_cols].values
        src = df.loc[s, pollutable_cols]

        if mode == "noisy":
            src = _noisify_block(src.to_frame().T, rng).iloc[0]

        after = src.values

        # 🔒 garde-fou critique : changement réel
        diff_mask = before != after
        n_changed = int(diff_mask.sum())

        if n_changed == 0:
            continue

        df.loc[t, pollutable_cols] = after
        touched_rows.add(t)

        for c in np.array(pollutable_cols)[diff_mask]:
            row_idx.append(t)
            col_idx.append(df.columns.get_loc(c))

        changed_cells += n_changed

    return df, np.array(row_idx, int), np.array(col_idx, int)


pollution_uniqueness.INTRODUCES_INVALID = False
pollution_uniqueness.APPROXIMATE_RATE = True
