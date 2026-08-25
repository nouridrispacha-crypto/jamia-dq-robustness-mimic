import numpy as np
import pandas as pd
from copy import deepcopy

def pollution_mnar_severity(
    df: pd.DataFrame,
    density: float,
    severity_col: str = "y_mort_hosp",
    target_vars=None,
    seed=None,
):
    """
    MNAR severity-based, cellwise
    - p% de NaN globaux EXACTS
    - Concentrés chez les patients graves
    - Concentrés sur des variables critiques
    """

    df = deepcopy(df)
    rng = np.random.default_rng(seed)

    if target_vars is None:
        raise ValueError("target_vars must be provided")

    protected = {"subject_id", "hadm_id", "stay_id", severity_col}
    cols = [c for c in df.columns if c not in protected]

    n_rows = df.shape[0]
    n_cols = len(cols)
    n_total = n_rows * n_cols
    n_to_corrupt = int(density * n_total)

    # ---------- Sévérité patient (0–1) ----------
    severity = df[severity_col].values.astype(float)
    severity = (severity - severity.min()) / (severity.max() - severity.min() + 1e-8)

    # ---------- Poids des colonnes ----------
    col_weights = np.ones(len(cols))
    for i, c in enumerate(cols):
        if c in target_vars:
            col_weights[i] = 3.0   # biais MNAR fort

    col_weights = col_weights / col_weights.sum()

    # ---------- Tirage biaisé ----------
    corrupted = set()

    while len(corrupted) < n_to_corrupt:
        r = rng.integers(0, n_rows)

        # proba ↑ si patient grave
        if rng.random() < severity[r]:
            c = rng.choice(len(cols), p=col_weights)
            corrupted.add((r, c))

    # ---------- Application ----------
    for r, c in corrupted:
        df.iat[r, df.columns.get_loc(cols[c])] = np.nan

    return df
