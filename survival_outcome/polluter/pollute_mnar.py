import numpy as np
import pandas as pd


def pollution_mnar_top(df: pd.DataFrame, p: float, seed: int = 0):
    """
    MNAR severity-driven (TOP)

    - La probabilité d'être manquant dépend de la gravité du patient
    - La gravité est définie ici par l'utilisation de la norépinéphrine
    - Les valeurs manquantes sont introduites préférentiellement
      chez les patients les plus graves
    - p correspond EXACTEMENT à p% de NaN réels (cellules)
    """

    rng = np.random.default_rng(seed)

    # Copie locale + reset index (CRITIQUE)
    df = df.copy().reset_index(drop=True)

    # Variables cliniquement critiques (MNAR)
    MNAR_VARS = [
        "urine_output",
        "age",
        "wbc_min",
        "spo2_min",
        "rr_mean",
        "bun_min",
        "sbp_mean",
        "ptt_min",
        "hr_mean",
    ]

    # Vérification minimale
    for v in MNAR_VARS + ["norepinephrine"]:
        if v not in df.columns:
            raise ValueError(f"Column '{v}' missing for MNAR pollution")

    # -----------------------------
    # 1) Définition de la gravité
    # -----------------------------
    # Plus norepinephrine est élevée, plus le patient est grave
    severity = df["norepinephrine"].fillna(0).values

    # Patients classés du plus grave au moins grave
    patient_order = np.argsort(-severity)

    # -----------------------------
    # 2) Calcul du nombre exact de NaN
    # -----------------------------
    n_cells_total = df.shape[0] * len(MNAR_VARS)
    n_to_nan = int(p * n_cells_total)

    # -----------------------------
    # 3) Sélection MNAR des cellules
    # -----------------------------
    cells = []
    for i in patient_order:
        for var in MNAR_VARS:
            cells.append((i, df.columns.get_loc(var)))
            if len(cells) >= n_to_nan:
                break
        if len(cells) >= n_to_nan:
            break

    # -----------------------------
    # 4) Application des NaN
    # -----------------------------
    for r, c in cells:
        df.iat[r, c] = np.nan

    return df
