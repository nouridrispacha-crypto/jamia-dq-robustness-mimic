# -*- coding: utf-8 -*-
"""
Runner VALIDITÉ — harmonisé (bootstrap + Torch MLPs)

Structure attendue :

TEST2/
 ├── common_runner.py
 ├── polluter/
 │      └── pollute_validity.py     (apply_validity_pollution)
 ├── data/
 │      └── mqds_with_flags_24h.csv
 └── run/
        └── run_validity.py         <-- ce fichier
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
# Import du pollueur de validité/format
# -----------------------------------------------------------
from pollute_validity import apply_validity_pollution   # type: ignore
from pollute_validity import validity_pollution

# -----------------------------------------------------------
# Chemin du CSV et dossier de sortie
# -----------------------------------------------------------
CSV = str(ROOT_DIR / "data" / "mqds_with_flags_24h.csv")
OUT_ROOT = ROOT_DIR / "outputs_validity_S1234_harmonized"


def apply_validity_wrapper(
    Xtr,
    Xte,
    p,
    fixed_train_pct,
    pollute_train,
    pollute_test,
    seed: int,
):
    """
    Wrapper de pollution pour la VALIDITÉ, compatible avec common_runner.run_scenario.

    - p : fraction globale de cellules à "rendre invalides" (0–1)
    - fixed_train_pct : si non None, remplace p pour le train (S4)
    - pollute_train / pollute_test : flags scénario S1–S4
    """

    Xtr_pol = Xtr.copy()
    Xte_pol = Xte.copy()

    # -----------------------------
    # TRAIN
    # -----------------------------
    if pollute_train:
        p_train = float(p if fixed_train_pct is None else fixed_train_pct)

        Xtr_pol = apply_validity_pollution(
            X=Xtr_pol,
            p=p_train,
            seed=seed,
            # scenario=None  # par défaut : mix typing/datefmt/cat_ood/unit
            # Tu peux forcer un scénario :
            # scenario="UNIT_US" ou "NUM_US_FORMAT"
        )

    # -----------------------------
    # TEST
    # -----------------------------
    if pollute_test:
        p_test = float(p)
        rs = seed + 1000

        Xte_pol = apply_validity_pollution(
            X=Xte_pol,
            p=p_test,
            seed=rs,
            # scenario=None
        )

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
        apply_validity_wrapper,
        pollute_train=True,
        pollute_test=False,
    )

    # S2 — TEST pollué uniquement
    run_scenario(
        "S2_test_only",
        CSV,
        OUT_ROOT / "S2_test_only",
        apply_validity_wrapper,
        pollute_train=False,
        pollute_test=True,
    )

    # S3 — TRAIN + TEST pollués indépendamment
    run_scenario(
        "S3_both_independent",
        CSV,
        OUT_ROOT / "S3_both_independent",
        apply_validity_wrapper,
        pollute_train=True,
        pollute_test=True,
    )

    # S4 — TRAIN fixe à 20 %, TEST variable
    run_scenario(
        "S4_train20_test_variable",
        CSV,
        OUT_ROOT / "S4_train20_test_variable",
        apply_validity_wrapper,
        pollute_train=True,
        pollute_test=True,
        fixed_train_pct=0.20,
    )
