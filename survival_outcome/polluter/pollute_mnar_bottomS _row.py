import numpy as np
import pandas as pd


def apply_completeness_pollution_mnar_bottomS_fullrows(
    df_orig: pd.DataFrame,
    target_missing_frac: float,
    *,
    sofa_col: str = "sofa",
    protect_cols=("subject_id", "hadm_id", "stay_id"),
    rng: np.random.Generator | None = None,
):
    """
    MNAR-bottomS FULL ROWS :
    - Cible les patients les moins graves (SOFA le plus bas).
    - On calcule k = round(target_missing_frac * n_rows).
    - On choisit k patients parmi les moins graves (bottom S).
    - Pour chaque patient sélectionné : toutes les colonnes NON protégées sont mises à "".
    - Aucune ligne partiellement polluée.
    - Le taux de pollution obtenu ≈ k / n_rows.
    """

    # Copie de sécurité
    df = df_orig.copy().reset_index(drop=True)

    # RNG
    if rng is None:
        rng = np.random.default_rng()

    # Colonnes à polluer
    cols_to_pollute = [c for c in df.columns if c not in protect_cols]
    if not cols_to_pollute:
        # Rien à polluer -> on renvoie DF + taux 0
        return df, 0.0

    n_rows = len(df)
    n_cols = len(cols_to_pollute)

    if n_rows == 0:
        return df, 0.0

    # ============================
    # 1) Patients les moins graves (bottom S)
    # ============================
    # SOFA numérique, on met les NaN très graves ou très peu graves selon la logique choisie
    sofa = pd.to_numeric(df[sofa_col], errors="coerce").fillna(9999)
    ranked = sofa.rank(method="first", ascending=True)  # moins grave = rang bas

    # Nombre de lignes à polluer
    target_rows = int(round(target_missing_frac * n_rows))
    target_rows = max(0, min(target_rows, n_rows))

    if target_rows == 0:
        return df, 0.0

    # On prend au moins target_rows candidats dans le bas de distribution
    bottom_idx = ranked.nsmallest(max(target_rows, 1)).index.to_numpy()

    # Si on a plus de candidats que nécessaire : tirage aléatoire
    if len(bottom_idx) > target_rows:
        chosen_rows = rng.choice(bottom_idx, size=target_rows, replace=False)
    else:
        # sinon on prend tout ce qu'on a (<= target_rows)
        chosen_rows = bottom_idx

    df = df.astype(object)

    # ============================
    # 2) Pollution LIGNES ENTIERES
    # ============================
    for r_idx in chosen_rows:
        for col in cols_to_pollute:
            df.loc[r_idx, col] = ""

    # ============================
    # 3) Taux obtenu
    # ============================
    total_cells = n_rows * n_cols
    missing_cells = len(chosen_rows) * n_cols
    achieved = missing_cells / total_cells if total_cells > 0 else 0.0

    return df, achieved


def pollution_mnar_bottom(df, density, rng=None, enable=None):
    """
    Wrapper appelé par common_runner.run_scenario.

    Signature attendue :
        pollution_fn(df, density=..., rng=None, enable=None)
    Doit retourner :
        df_pollue, mask, report
    """
    df_pollue, achieved = apply_completeness_pollution_mnar_bottomS_fullrows(
        df_orig=df,
        target_missing_frac=float(density),
        rng=rng,
    )

    # Pour l’instant on ne gère pas de mask cellulaire -> None
    mask = None

    report = {
        "type": "MNAR_bottomS_fullrows",
        "target_frac": float(density),
        "achieved_frac": float(achieved),
        "n_rows": int(len(df)),
    }

    return df_pollue, mask, report
