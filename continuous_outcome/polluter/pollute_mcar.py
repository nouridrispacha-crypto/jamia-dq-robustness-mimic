# ============================================
# POLLUEUR — MCAR STRICT (CELLWISE, SANS REMISE)
# ============================================

import numpy as np
import pandas as pd
from copy import deepcopy


def pollution_mcar(
    df: pd.DataFrame,
    density: float,
    seed=None,
    protected={"in_hospital_mortality"},
):
    """
    MCAR STRICT (cellwise, sans remise)

    - Une proportion `density` des cellules est rendue manquante
    - Tirage uniforme sur toutes les cellules (lignes × colonnes)
    - Indépendant :
        * des valeurs
        * des patients
        * des variables
        * du target
    - Sans remise → taux réel ≈ density
    - Génère uniquement des np.nan

    Compatible avec :
    - imputation
    - missing-as-category
    - sklearn / xgboost
    """

    assert 0.0 <= density <= 1.0

    df = deepcopy(df)
    rng = np.random.default_rng(seed)

    # Colonnes polluables
    cols = [c for c in df.columns if c not in protected]

    n_rows = df.shape[0]
    n_cols = len(cols)

    if n_cols == 0 or n_rows == 0 or density == 0.0:
        return df, np.array([], dtype=int), np.array([], dtype=int)

    # Nombre total de cellules polluables
    n_total = n_rows * n_cols
    n_to_corrupt = int(density * n_total)

    if n_to_corrupt == 0:
        return df, np.array([], dtype=int), np.array([], dtype=int)

    # Tirage uniforme SANS remise sur les cellules
    flat_idx = rng.choice(n_total, size=n_to_corrupt, replace=False)

    row_idx = flat_idx // n_cols
    col_idx = flat_idx % n_cols

    # Corruption
    for r, c in zip(row_idx, col_idx):
        df.iat[r, df.columns.get_loc(cols[c])] = np.nan

    return df, row_idx, col_idx


# ============================================================
# MÉTADONNÉE POUR LE COMMON RUNNER
# ============================================================
pollution_mcar.INTRODUCES_INVALID = False