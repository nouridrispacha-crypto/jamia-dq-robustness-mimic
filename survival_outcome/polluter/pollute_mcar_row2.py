import numpy as np
import pandas as pd

def apply_completeness_pollution_mcar_fullrows(
    df_orig,
    density,
    *,
    protect_cols=("subject_id", "hadm_id", "stay_id"),
    rng=None
):
    # --- NO OP ---
    if density <= 0:
        return df_orig, 0.0

    if rng is None:
        rng = np.random.default_rng()

    df = df_orig.copy()
    cols = [c for c in df.columns if c not in protect_cols]

    n = len(df)
    k = int(round(density * n))
    k = min(max(k, 0), n)

    if k == 0:
        return df_orig, 0.0

    df = df.astype(object)
    chosen = rng.choice(n, size=k, replace=False)

    for r in chosen:
        for c in cols:
            df.loc[r, c] = ""

    achieved = (k * len(cols)) / (n * len(cols))

    return df, achieved


def pollution_mcar(df, density, rng=None, enable=None):
    df_pol, achieved = apply_completeness_pollution_mcar_fullrows(
        df, density, rng=rng
    )
    return df_pol, None, {"type": "MCAR_fullrows", "achieved": achieved}
