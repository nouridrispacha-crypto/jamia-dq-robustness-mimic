# -*- coding: utf-8 -*-
"""
Pollueur MNAR MEDIAN-S (FULL ROWS) :

- Sélectionne les patients dont le SOFA est le plus proche de la médiane
- Pollue FULL-ROW : toutes les colonnes non protégées

Notes :
- Aucun remplacement par NaN (utilisation de "")
- Taille du dataset inchangée
- Compatible avec le train/test split
"""

import numpy as np
import pandas as pd


# ============================================================
#   FONCTION PRINCIPALE
# ============================================================

def apply_completeness_pollution_mnar_medianS_fullrows(
    df_orig: pd.DataFrame,
    target_missing_frac: float,
    *,
    sofa_col: str = None,
    protect_cols=("subject_id", "hadm_id", "stay_id"),
    rng: np.random.Generator | None = None,
):
    """
    MNAR-medianS FULL ROWS :
    - Sélectionne les lignes dont le SOFA est le plus proche de la médiane
    - Pollue ENTIEREMENT k lignes (aucune cellule partielle)
    - Colonnes protégées exclues
    """

    df = df_orig.copy().reset_index(drop=True)

    # RNG
    if rng is None:
        rng = np.random.default_rng()

    # Détection de la colonne SOFA si non spécifiée
    if sofa_col is None:
        for c in df.columns:
            if c.lower() in ("sofa", "sofa_score", "sofa24h"):
                sofa_col = c
                break

    if sofa_col is None:
        raise ValueError("Impossible de trouver une colonne SOFA (sofa / sofa_score / sofa24h).")

    # Colonnes à polluer (toutes sauf identifiants)
    cols_to_pollute = [c for c in df.columns if c not in protect_cols]

    if not cols_to_pollute:
        return df, 0.0

    n_rows = len(df)
    n_cols = len(cols_to_pollute)

    if n_rows == 0:
        return df, 0.0

    df = df.astype(object)

    # ============================================================
    # 1) Calcul distance à la médiane SOFA
    # ============================================================
    sofa_vals = pd.to_numeric(df[sofa_col], errors="coerce")

    median_sofa = sofa_vals.median()
    dist = (sofa_vals - median_sofa).abs()

    # Patients les plus proches de la médiane
    idx_sorted = dist.sort_values().index.to_numpy()

    # ============================================================
    # 2) Nombre de lignes à polluer
    # ============================================================
    target_rows = int(round(target_missing_frac * n_rows))
    target_rows = max(0, min(target_rows, n_rows))

    if target_rows == 0:
        achieved = 0.0
        return df, achieved

    chosen_rows = idx_sorted[:target_rows]

    # ============================================================
    # 3) Pollution FULL ROW
    # ============================================================
    for r in chosen_rows:
        for col in cols_to_pollute:
            df.loc[r, col] = ""   # pollution volontaire, pas NaN

    # ============================================================
    # 4) Taux obtenu
    # ============================================================
    total_cells = n_rows * n_cols
    missing_cells = len(chosen_rows) * n_cols
    achieved = missing_cells / total_cells if total_cells else 0.0

    return df, achieved


# ============================================================
#   WRAPPER compatible common_runner
# ============================================================

def pollution_mnar_median(df, density, rng=None, enable=None):
    """
    Signature attendue :
        df_pollué, mask, report
    """

    df_pollue, achieved = apply_completeness_pollution_mnar_medianS_fullrows(
        df_orig=df,
        target_missing_frac=float(density),
        rng=rng,
    )

    mask = None  # pas de mask cellulaire pour MNAR-FULLROWS

    report = {
        "type": "MNAR_medianS_fullrows",
        "target_frac": float(density),
        "achieved_frac": float(achieved),
        "n_rows": int(len(df)),
        "n_rows_modified": int(round(density * len(df))),
    }

    return df_pollue, mask, report
