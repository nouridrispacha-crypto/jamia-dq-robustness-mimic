
import numpy as np
import pandas as pd

def apply_completeness_pollution_mnar_bottomS_fullrows(
    df_orig,
    density,
    *,
    sofa_col="sofa",
    protect_cols=("subject_id","hadm_id","stay_id"),
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

    sofa = pd.to_numeric(df[sofa_col], errors="coerce").fillna(9999)
    ranked = sofa.rank(method="first", ascending=True)

    candidates = ranked.nsmallest(k).index.to_numpy()
    chosen = candidates if len(candidates) <= k else rng.choice(candidates, k, replace=False)

    df = df.astype(object)
    for r in chosen:
        for c in cols:
            df.loc[r, c] = ""

    achieved = (k * len(cols)) / (n * len(cols))
    return df, achieved


def pollution_mnar_bottom(df, density, rng=None, enable=None):
    df_pol, achieved = apply_completeness_pollution_mnar_bottomS_fullrows(
        df, density, rng=rng
    )
    return df_pol, None, {"type": "MNAR_bottom_fullrows", "achieved": achieved}
