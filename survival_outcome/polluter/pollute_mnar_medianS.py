# ============================================
# POLLUEUR — MNAR MEDIAN
# (Cellwise actif / Rowwise commenté)
# ============================================

import numpy as np
import pandas as pd
from copy import deepcopy


def pollution_mnar_median(
    df: pd.DataFrame,
    density: float,
    score_col: str = "sofa",
    center_frac: float = 0.35,   # ← FIXÉ (35 % autour de la médiane)
    seed: int | None = None,
):
    """
    MNAR MEDIAN — CELLWISE (FINAL, PROPRE) :

    - center_frac = fraction FIXE de patients autour de la médiane de sévérité
    - density     = proportion globale de cellules corrompues
    - corruption au niveau cellule (cellwise), PAS patient entier
    - missingness dépend de la sévérité (zone médiane)
    - aucune suppression systématique de patients
    - compatible avec :
        * missing-as-category
        * imputation
        * listwise deletion
    """

    rng = np.random.default_rng(seed)
    df = deepcopy(df)

    # Colonnes protégées (structure + cible)
    protected = {"subject_id", "hadm_id", "stay_id", "y_mort_hosp"}
    cols = [c for c in df.columns if c not in protected]

    n_rows = df.shape[0]
    n_cols = len(cols)

    # ---- budget global de cellules à corrompre ----
    n_to_corrupt = int(density * n_rows * n_cols)
    if n_to_corrupt == 0:
        return df, np.array([], dtype=int), np.array([], dtype=int)

    # ---- sélection FIXE des patients de sévérité médiane ----
    order = df[score_col].sort_values(ascending=True).index

    center_size = int(center_frac * n_rows)
    start = int(0.5 * n_rows - center_size / 2)
    stop  = start + center_size

    # sécurité bornes
    start = max(start, 0)
    stop = min(stop, n_rows)

    median_rows = order[start:stop]

    if len(median_rows) == 0:
        return df, np.array([], dtype=int), np.array([], dtype=int)

    # ---- tirage cellwise dans la zone médiane ----
    row_labels = rng.choice(
        median_rows.to_numpy(),
        size=n_to_corrupt,
        replace=True
    )

    col_idx = rng.integers(0, n_cols, size=n_to_corrupt)

    # ---- corruption cellule par cellule ----
    for r_lab, c in zip(row_labels, col_idx):
        df.loc[r_lab, cols[c]] = ""

    return df, row_labels, col_idx


# ------------------------------------------------------------------
# MNAR MEDIAN — ROWWISE (COMMENTÉ — POUR RÉFÉRENCE UNIQUEMENT)
# ------------------------------------------------------------------
"""
def pollution_mnar_median(
    df: pd.DataFrame,
    density: float,
    score_col: str = "sofa",
    center_frac: float = 0.35,
    seed: int | None = None,
):
    '''
    MNAR MEDIAN — ROWWISE :
    - center_frac des patients autour de la médiane sont éligibles
    - p% des patients sont ENTIEREMENT corrompus
    - équivalent à supprimer des patients
    - ❌ trop agressif, peu réaliste cliniquement
    '''

    rng = np.random.default_rng(seed)
    df = deepcopy(df)

    protected = {"subject_id", "hadm_id", "stay_id", "y_mort_hosp"}
    cols = [c for c in df.columns if c not in protected]

    n_rows = len(df)
    n_center = int(center_frac * n_rows)
    n_to_corrupt = int(density * n_rows)

    order = df[score_col].sort_values().index

    start = int(0.5 * n_rows - n_center / 2)
    stop  = start + n_center
    center_rows = order[start:stop]

    row_idx = rng.choice(center_rows, size=n_to_corrupt, replace=False)
    df.loc[row_idx, cols] = ""

    return df, row_idx.to_numpy(), cols
"""
