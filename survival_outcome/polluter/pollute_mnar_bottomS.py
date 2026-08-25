# ============================================
# POLLUEUR — MNAR BOTTOM
# (Cellwise actif / Rowwise commenté)
# ============================================

import numpy as np
import pandas as pd
from copy import deepcopy


def pollution_mnar_bottom(
    df: pd.DataFrame,
    density: float,
    score_col: str = "sofa",
    bottom_frac: float = 0.35,   # ← FIXÉ À 35 %
    seed: int | None = None,
):
    """
    MNAR BOTTOM — CELLWISE (FINAL, PROPRE) :

    - bottom_frac = 35% des patients les moins graves (FIXE)
    - density     = proportion globale de cellules corrompues
    - corruption au niveau cellule (cellwise), PAS patient entier
    - missingness dépend de la sévérité (MNAR)
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

    # ---- sélection FIXE des patients les moins graves ----
    order = df[score_col].sort_values(ascending=True).index
    n_bottom = int(bottom_frac * n_rows)
    bottom_rows = order[:n_bottom]

    if n_bottom == 0:
        return df, np.array([], dtype=int), np.array([], dtype=int)

    # ---- tirage cellwise dans le sous-groupe bottom ----
    # lignes tirées avec remise parmi les patients bottom
    row_labels = rng.choice(
        bottom_rows.to_numpy(),
        size=n_to_corrupt,
        replace=True
    )

    # colonnes tirées uniformément
    col_idx = rng.integers(0, n_cols, size=n_to_corrupt)

    # ---- corruption cellule par cellule ----
    for r_lab, c in zip(row_labels, col_idx):
        df.loc[r_lab, cols[c]] = ""

    return df, row_labels, col_idx


# ------------------------------------------------------------------
# MNAR BOTTOM — ROWWISE (COMMENTÉ — POUR RÉFÉRENCE UNIQUEMENT)
# ------------------------------------------------------------------
"""
def pollution_mnar_bottom(
    df: pd.DataFrame,
    density: float,
    score_col: str = "sofa",
    bottom_frac: float = 0.35,
    seed: int | None = None,
):
    '''
    MNAR BOTTOM — ROWWISE :
    - 35% des patients les moins graves sont sélectionnés
    - p% de ces patients sont ENTIEREMENT corrompus
    - équivalent à supprimer des patients
    - ❌ trop agressif, peu réaliste cliniquement
    '''

    rng = np.random.default_rng(seed)
    df = deepcopy(df)

    protected = {"subject_id", "hadm_id", "stay_id", "y_mort_hosp"}
    cols = [c for c in df.columns if c not in protected]

    n_rows = df.shape[0]
    n_bottom = int(bottom_frac * n_rows)
    n_to_corrupt = int(density * n_rows)

    order = df[score_col].sort_values(ascending=True).index
    bottom_rows = order[:n_bottom]

    row_idx = rng.choice(bottom_rows, size=n_to_corrupt, replace=False)
    df.loc[row_idx, cols] = ""

    return df, row_idx.to_numpy(), cols
"""
