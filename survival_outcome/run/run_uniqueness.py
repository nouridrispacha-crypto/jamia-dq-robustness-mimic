# -*- coding: utf-8 -*-
"""
Runner UNICITÉ — harmonisé (bootstrap + Torch MLPs)

Structure attendue :

TEST2/
 ├── common_runner.py
 ├── polluter/
 │      └── pollute_uniqueness.py   (apply_uniqueness_pollution)
 ├── data/
 │      └── mqds_with_flags_24h.csv
 └── run/
        └── run_uniqueness.py   <-- ce fichier
"""

from pathlib import Path
import numpy as np
import sys

# -----------------------------------------------------------
# Ajouter TEST2 et TEST2/polluter au PYTHONPATH
# -----------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent      # .../TEST2/run
ROOT_DIR = THIS_DIR.parent                      # .../TEST2
POLLUTER_DIR = ROOT_DIR / "polluter"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(POLLUTER_DIR))

# -----------------------------------------------------------
# Import du runner générique
# -----------------------------------------------------------
from common_runner import (
    run_scenario,
    POLLUTION_LEVELS,
    SAMPLE_SIZES,
    TEST_SIZE,
    N_BOOTSTRAPS,
    MODEL_SEEDS,
)

# -----------------------------------------------------------
# Import du pollueur d’unicité
# -----------------------------------------------------------
from pollute_uniqueness import apply_uniqueness_pollution   # type: ignore
from pollute_uniqueness import uniqueness_pollution

# -----------------------------------------------------------
# Chemin du CSV et dossier de sortie
# -----------------------------------------------------------
CSV = str(ROOT_DIR / "data" / "mqds_with_flags_24h.csv")
OUT_ROOT = ROOT_DIR / "outputs_uniqueness_S1234_harmonized"


def apply_uniqueness(
    Xtr,
    Xte,
    p,
    fixed_train_pct,
    pollute_train,
    pollute_test,
    seed: int,
):
    """
    Pollution UNICITÉ appliquée comme wrapper pour run_scenario :

    - p = fraction de lignes à dupliquer/remplacer
    - fixed_train_pct = 0.20 pour S4
    - pollute_train / pollute_test = bool
    """

    # On reconstruit y_train et y_test plus proprement,
    # en détectant la colonne cible comme common_runner le fait.
    target_col = None
    for c in Xtr.columns:
        if c.lower() in ["sofa", "sofa_score", "y", "mortality",
                         "death", "y_mort_hosp", "target"]:
            target_col = c
            break

    if target_col is None:
        # fallback : dernière colonne du DF original
        target_col = Xtr.columns[-1]

    # Séparer X et y
    ytr = Xtr[target_col].astype(int).copy()
    Xtr = Xtr.drop(columns=[target_col]).copy()

    yte = Xte[target_col].astype(int).copy()
    Xte = Xte.drop(columns=[target_col]).copy()

    Xtr_pol = Xtr.copy()
    ytr_pol = ytr.copy()
    Xte_pol = Xte.copy()
    yte_pol = yte.copy()

    # -----------------------------
    # TRAIN
    # -----------------------------
    if pollute_train:
        d_train = float(p if fixed_train_pct is None else fixed_train_pct)
        Xtr_pol, ytr_pol, _ = apply_uniqueness_pollution(
            Xtr_pol,
            ytr_pol,
            d_frac=d_train,
            mode="noisy",          # <--- tu peux changer : 'exact' | 'noisy' | 'targeted'
            seed=seed,
        )

    # -----------------------------
    # TEST
    # -----------------------------
    if pollute_test:
        d_test = float(p)
        rs = seed + 1000
        Xte_pol, yte_pol, _ = apply_uniqueness_pollution(
            Xte_pol,
            yte_pol,
            d_frac=d_test,
            mode="noisy",
            seed=rs,
        )

    # On recompose les DataFrames X+y pour common_runner
    Xtr_pol[target_col] = ytr_pol
    Xte_pol[target_col] = yte_pol

    return Xtr_pol, Xte_pol


# =====================================================================
# 🚀 Lancement des 4 scénarios
# =====================================================================
if __name__ == "__main__":
    OUT_ROOT.mkdir(exist_ok=True)

    # S1 — TRAIN pollué uniquement
    run_scenario(
        "S1_train_only",
        CSV,
        OUT_ROOT / "S1_train_only",
        apply_uniqueness,
        pollute_train=True,
        pollute_test=False,
    )

    # S2 — TEST pollué uniquement
    run_scenario(
        "S2_test_only",
        CSV,
        OUT_ROOT / "S2_test_only",
        apply_uniqueness,
        pollute_train=False,
        pollute_test=True,
    )

    # S3 — TRAIN + TEST pollués indépendamment
    run_scenario(
        "S3_both_independent",
        CSV,
        OUT_ROOT / "S3_both_independent",
        apply_uniqueness,
        pollute_train=True,
        pollute_test=True,
    )

    # S4 — TRAIN fixe à 20 %, TEST variable
    run_scenario(
        "S4_train20_test_variable",
        CSV,
        OUT_ROOT / "S4_train20_test_variable",
        apply_uniqueness,
        pollute_train=True,
        pollute_test=True,
        fixed_train_pct=0.20,
    )
