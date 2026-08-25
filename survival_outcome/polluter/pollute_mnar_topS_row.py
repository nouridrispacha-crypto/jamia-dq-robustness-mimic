import numpy as np
import pandas as pd

def apply_completeness_pollution_mnar_topS_fullrows(
    df_orig: pd.DataFrame,
    target_missing_frac: float,
    *,
    sofa_col: str = "sofa",
    protect_cols=("subject_id", "hadm_id", "stay_id"),
    rng: np.random.Generator = None
):
    """
    MNAR-topS STRICT (version lignes complètes) :

    - On sélectionne les patients les plus graves (Top S selon SOFA).
    - On détermine k = round(target_missing_frac * n_rows).
    - On tire au hasard k patients parmi le Top S.
    - Pour chaque patient sélectionné : toutes les colonnes NON protégées sont vidées ("").
    - Aucune ligne partiellement polluée.
    - Le taux de pollution obtenu = (k / n_rows), soit l'approximation mathématique la plus précise.
    """

    # Copie sécurisée
    df = df_orig.copy().reset_index(drop=True)

    # RNG
    if rng is None:
        rng = np.random.default_rng()

    # Colonnes à polluer
    cols_to_pollute = [c for c in df.columns if c not in protect_cols]
    if not cols_to_pollute:
        return df, 0.0

    n_rows = len(df)
    n_cols = len(cols_to_pollute)

    if n_rows == 0:
        return df, 0.0

    # ============================
    # 1) Sélection TOP S patients
    # ============================
    sofa = pd.to_numeric(df[sofa_col], errors="coerce").fillna(-999)
    ranked = sofa.rank(method="first", ascending=False)  # plus grave = rang bas

    # Nombre de lignes à polluer (approximation du taux)
    target_rows = int(round(target_missing_frac * n_rows))
    target_rows = max(0, min(target_rows, n_rows))

    # Sélection du Top S (au moins autant que target_rows)
    top_idx = ranked.nsmallest(max(target_rows, 1)).index.to_numpy()

    # Tirage de target_rows parmi le Top S
    if len(top_idx) > target_rows:
        chosen_rows = rng.choice(top_idx, size=target_rows, replace=False)
    else:
        # Cas où top_idx contient moins ou égal au nombre souhaité
        chosen_rows = top_idx

    df = df.astype(object)

    # ============================
    # 2) Pollution par LIGNES ENTIERES
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
def pollution_mnar_top(df, density, rng=None, enable=None):
    df_pollue, achieved = apply_completeness_pollution_mnar_topS_fullrows(
        df_orig=df,
        target_missing_frac=density,
        rng=rng
    )
    mask = None
    report = {"missing_rows": int(round(density * len(df)))}
    return df_pollue, mask, report
