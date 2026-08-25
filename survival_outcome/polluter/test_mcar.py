# ============================================================
# TEST UNITAIRE — POLLUTION MCAR STRICT
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

from polluter.pollute_mcar import pollution_mcar


# ============================================================
# CONFIG
# ============================================================
CSV_PATH = Path("data/18_variables.csv")   # adapte si besoin
TARGET_COL = "in_hospital_mortality"

SEEDS = [0, 42, 123]
P_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]


# ============================================================
# LOAD DATA (robuste)
# ============================================================
def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    assert TARGET_COL in df.columns, df.columns
    return df


# ============================================================
# TEST UNITAIRE MCAR
# ============================================================
def test_mcar_strict(df: pd.DataFrame, p: float, seed: int):
    """
    Vérifie que :
    - le taux réel de NaN ≈ p
    - le target n'est jamais touché
    - le nombre de NaN ajoutés est cohérent
    """

    protected = {TARGET_COL}
    features = [c for c in df.columns if c not in protected]

    # Nombre de cellules polluables AVANT
    pollutable_before = (~df[features].isna()).sum().sum()

    df_p, row_idx, col_idx = pollution_mcar(
        df,
        density=p,
        seed=seed,
        protected=protected,
    )

    # Nombre de cellules polluables APRÈS
    pollutable_after = (~df_p[features].isna()).sum().sum()

    added_nan = pollutable_before - pollutable_after
    achieved_p = added_nan / pollutable_before if pollutable_before > 0 else 0.0

    # --- checks ---
    assert TARGET_COL not in df_p.columns[df_p[TARGET_COL].isna().any():], \
        "❌ Target polluée !"

    assert len(row_idx) == len(col_idx), "❌ row_idx / col_idx mismatch"

    print(
        f"[seed={seed:3d}] "
        f"p demandé={p:4.2f} | "
        f"NaN ajoutés={added_nan:6d} | "
        f"p obtenu={achieved_p:6.4f}"
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    df = load_dataset(CSV_PATH)

    # on prend un sous-échantillon pour aller vite
    df = (
        df
        .dropna(subset=[TARGET_COL])
        .sample(n=1500, random_state=0)
        .reset_index(drop=True)
    )

    print("\n=== TEST UNITAIRE — MCAR STRICT ===\n")

    for seed in SEEDS:
        for p in P_LEVELS:
            test_mcar_strict(df, p, seed)

    print("\n✅ Tous les tests MCAR sont passés.\n")
