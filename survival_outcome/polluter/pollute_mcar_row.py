import numpy as np
import pandas as pd


def apply_completeness_pollution_mcar_fullrows(
    df_orig: pd.DataFrame,
    target_missing_frac: float,
    *,
    protect_cols=("subject_id", "hadm_id", "stay_id"),
    rng: np.random.Generator | None = None,
):
    """
    MCAR FULL ROWS :
    - On tire k lignes uniformément au hasard.
    - Sur ces lignes, toutes les colonnes non protégées sont mises à "".
    - Aucune ligne partiellement polluée.
    - Taux obtenu ≈ target_missing_frac (exactement k / n_rows).
    """
    df = df_orig.copy().reset_index(drop=True)

    if rng is None:
        rng = np.random.default_rng()

    cols_to_pollute = [c for c in df.columns if c not in protect_cols]
    if not cols_to_pollute:
        return df, 0.0

    n_rows = len(df)
    n_cols = len(cols_to_pollute)

    if n_rows == 0:
        return df, 0.0

    # Nombre de lignes à polluer
    target_rows = int(round(target_missing_frac * n_rows))
    target_rows = max(0, min(target_rows, n_rows))

    if target_rows == 0:
        return df, 0.0

    df = df.astype(object)

    # Tirage aléatoire des lignes
    chosen_rows = rng.choice(n_rows, size=target_rows, replace=False)

    # Pollution des lignes complètes
    for r_idx in chosen_rows:
        for col in cols_to_pollute:
            df.loc[r_idx, col] = ""

    total_cells = n_rows * n_cols
    missing_cells = len(chosen_rows) * n_cols
    achieved = missing_cells / total_cells if total_cells > 0 else 0.0

    return df, achieved


def pollution_mcar(df, density, rng=None, enable=None):
    """
    Wrapper appelé par common_runner.run_scenario
    Signature imposée : (df, density, rng=None, enable=None)
    Retour impératif : df_pollue, mask, report
    """
    df_pollue, achieved = apply_completeness_pollution_mcar_fullrows(
        df_orig=df,
        target_missing_frac=float(density),
        rng=rng,
    )

    mask = None
    report = {
        "type": "MCAR_fullrows",
        "target_frac": float(density),
        "achieved_frac": float(achieved),
        "n_rows": int(len(df)),
    }
    return df_pollue, mask, report
