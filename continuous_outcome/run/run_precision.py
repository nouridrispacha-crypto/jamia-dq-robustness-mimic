# -*- coding: utf-8 -*-
"""
Runner PRECISION — harmonisé (bootstrap + Torch MLPs)

Compatible avec la structure :

TEST2/
 ├── common_runner.py
 ├── polluter/
 │      └── pollute_precision_2.py
 ├── data/
 │      └── mqds_with_flags_24h.csv
 └── run/
        └── run_precision.py  <-- ce fichier
"""

from pathlib import Path
import numpy as np
import sys

# -----------------------------------------------------------
# Préparation des chemins / PYTHONPATH
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
# Import du pollueur de précision
# -----------------------------------------------------------
from pollute_precision import mix_precision_pollution_density   # type: ignore
from pollute_precision import precision_pollution


# -----------------------------------------------------------
# Chemin du CSV et dossier de sortie
# -----------------------------------------------------------
CSV = str(ROOT_DIR / "data" / "mqds_with_flags_24h.csv")
OUT_ROOT = ROOT_DIR / "outputs_precision_S1234_harmonized"


def apply_precision_pollution(
    Xtr,
    Xte,
    p,
    fixed_train_pct,
    pollute_train,
    pollute_test,
    seed: int,
):
    """
    Fonction conforme au schéma common_runner.run_scenario.

    p = pourcentage demandé de cellules à corrompre
    fixed_train_pct = pour S4 (train fixe à 20%)
    """

    Xtr_pol = Xtr.copy()
    Xte_pol = Xte.copy()

    # Colonnes numériques pour les deux datasets
    num_cols = [
        c for c in Xtr.columns
        if np.issubdtype(Xtr[c].dtype, np.number)
    ]

    # -----------------------------
    # TRAIN
    # -----------------------------
    if pollute_train:
        density_train = float(p if fixed_train_pct is None else fixed_train_pct)

        Xtr_pol = mix_precision_pollution_density(
            df=Xtr_pol,
            numeric_cols=num_cols,
            d=density_train,
            seed=seed,
        )

    # -----------------------------
    # TEST
    # -----------------------------
    if pollute_test:
        density_test = float(p)
        rs = seed + 10000     # seed indépendante comme tous les pollueurs

        Xte_pol = mix_precision_pollution_density(
            df=Xte_pol,
            numeric_cols=num_cols,
            d=density_test,
            seed=rs,
        )

    return Xtr_pol, Xte_pol


# =====================================================================
# 🚀 Lancement des 4 scénarios
# =====================================================================
if __name__ == "__main__":
    OUT_ROOT.mkdir(exist_ok=True)

    # S1 : TRAIN pollué uniquement
    run_scenario(
        "S1_train_only",
        CSV,
        OUT_ROOT / "S1_train_only",
        apply_precision_pollution,
        pollute_train=True,
        pollute_test=False,
    )

    # S2 : TEST pollué uniquement
    run_scenario(
        "S2_test_only",
        CSV,
        OUT_ROOT / "S2_test_only",
        apply_precision_pollution,
        pollute_train=False,
        pollute_test=True,
    )

    # S3 : TRAIN + TEST indépendants
    run_scenario(
        "S3_both_independent",
        CSV,
        OUT_ROOT / "S3_both_independent",
        apply_precision_pollution,
        pollute_train=True,
        pollute_test=True,
    )

    # S4 : TRAIN = 20% constant, TEST variable
    run_scenario(
        "S4_train20_test_variable",
        CSV,
        OUT_ROOT / "S4_train20_test_variable",
        apply_precision_pollution,
        pollute_train=True,
        pollute_test=True,
        fixed_train_pct=0.20,
    )
