import numpy as np
import pandas as pd
from copy import deepcopy


def pollution_mnar_bottom(
    df: pd.DataFrame,
    density: float,
    seed: int = 0,
):
    """
    MNAR diffus severity-weighted (binaire, comparable MCAR) — BOTTOM

    - d % = d % EXACT de cellules manquantes (comme MCAR)
    - Sévérité définie par un indicateur binaire :
        norepinephrine ∈ {0,1}
    - Patients NON sous vasopresseur (sev=0) → probabilité plus élevée de valeurs manquantes
      (inverse de MNAR-TOP)
    - Diffus : tous les patients peuvent être touchés
    - Sans remise, seed contrôlée
    - Aucune falsification : uniquement des NaN
    - API compatible common_runner
    """

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------
    assert 0.0 <= density <= 1.0

    rng = np.random.default_rng(seed)
    df = deepcopy(df).reset_index(drop=True)

    TARGET = "in_hospital_mortality"
    PROTECTED = {"subject_id", "hadm_id", "stay_id", TARGET}

    # ------------------------------------------------------------
    # Colonnes polluables
    # ------------------------------------------------------------
    pollutable_cols = [c for c in df.columns if c not in PROTECTED]
    n_rows = len(df)
    n_cols = len(pollutable_cols)

    if density == 0.0 or n_rows == 0 or n_cols == 0:
        return df, np.array([], dtype=int), np.array([], dtype=int)

    # ------------------------------------------------------------
    # 1️⃣ Sévérité binaire (vasopresseur oui / non)
    # ------------------------------------------------------------
    # Hypothèse pipeline : pas de NaN avant pollution
    sev = df["norepinephrine"].to_numpy(dtype=int)  # 0 ou 1

    # ------------------------------------------------------------
    # 2️⃣ Probabilité MNAR par patient (2 groupes) — INVERSE de TOP
    # ------------------------------------------------------------
    alpha = 5.0  # force de séparation (↑ = plus de contraste)

    # mêmes valeurs que TOP, mais affectées à l'autre groupe
    p_high = 1.0 / (1.0 + np.exp(-alpha / 2))  # probabilité haute
    p_low  = 1.0 / (1.0 + np.exp(+alpha / 2))  # probabilité basse

    # BOTTOM : sev==0 (non sévère) -> p_high ; sev==1 (sévère) -> p_low
    p_row = np.where(sev == 0, p_high, p_low)

    # Normalisation → même budget global que MCAR
    p_row = p_row / p_row.mean()

    # ------------------------------------------------------------
    # 3️⃣ Poids par cellule (diffus)
    # ------------------------------------------------------------
    sub = df[pollutable_cols]
    not_nan = ~sub.isna().to_numpy()  # sécurité

    W = p_row[:, None] * not_nan
    W_sum = W.sum()

    if W_sum == 0:
        return df, np.array([], dtype=int), np.array([], dtype=int)

    # ------------------------------------------------------------
    # 4️⃣ Tirage SANS remise — budget exact
    # ------------------------------------------------------------
    n_total = n_rows * n_cols
    n_to_nan = int(density * n_total)

    p_flat = (W / W_sum).ravel()

    flat_idx = rng.choice(
        n_total,
        size=min(n_to_nan, int((p_flat > 0).sum())),
        replace=False,
        p=p_flat,
    )

    row_idx = flat_idx // n_cols
    col_local = flat_idx % n_cols

    # ------------------------------------------------------------
    # Application des NaN
    # ------------------------------------------------------------
    for r, c in zip(row_idx, col_local):
        df.iat[r, df.columns.get_loc(pollutable_cols[c])] = np.nan

    col_idx = np.array(
        [df.columns.get_loc(pollutable_cols[c]) for c in col_local],
        dtype=int,
    )

    return df, np.array(row_idx, dtype=int), col_idx

pollution_mnar_bottom.INTRODUCES_INVALID = False