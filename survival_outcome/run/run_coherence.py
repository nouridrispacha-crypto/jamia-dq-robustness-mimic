# -*- coding: utf-8 -*-
"""
Runner COHERENCE — harmonisé (bootstrap + Torch MLPs).

Structure attendue :

TEST2/
 ├── common_runner.py
 ├── polluter/
 │      └── coherence_polluter.py
 ├── data/
 │      └── mqds_with_flags_24h.csv
 └── run/
        └── run_coherence.py
"""

from pathlib import Path
import numpy as np
import sys

# -----------------------------------------------------------
# Ajouter TEST2 et TEST2/polluter dans le PYTHONPATH
# -----------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent        # .../TEST2/run
ROOT_DIR = THIS_DIR.parent                        # .../TEST2
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
# Import du pollueur
# -----------------------------------------------------------
from coherence_polluter import apply_coherence_pollution_vectorized as pollute_fn
# ou ta version exacte :
# from coherence_polluter import apply_coherence_pollution as pollute_fn


# -----------------------------------------------------------
# Chemin du CSV (dans TEST2/data/)
# -----------------------------------------------------------
CSV = str(ROOT_DIR / "data" / "mqds_with_flags_24h.csv")

# Dossier de sortie
OUT_ROOT = ROOT_DIR / "outputs_coherence_S1234_harmonized"


def apply_pollution_coherence(
    Xtr,
    Xte,
    p,
    fixed_train_pct,
    pollute_train,
    pollute_test,
    seed: int,
):
    """
    Pollution cohérence appliquée selon S1/S2/S3/S4.
    """

    Xtr_pol = Xtr.copy()
    Xte_pol = Xte.copy()

    # -----------------------------
    # 🎯 TRAIN
    # -----------------------------
    if pollute_train:
        density_train = float(p if fixed_train_pct is None else fixed_train_pct)
        Xtr_pol, _, _ = pollute_fn(
            Xtr_pol,
            density=density_train,
            rng=np.random.default_rng(seed),
        )

    # -----------------------------
    # 🎯 TEST
    # -----------------------------
    if pollute_test:
        density_test = float(p)
        rs = seed + 1000   # seed indépendante

        Xte_pol, _, _ = pollute_fn(
            Xte_pol,
            density=density_test,
            rng=np.random.default_rng(rs),
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
        apply_pollution_coherence,
        pollute_train=True,
        pollute_test=False,
    )

    # S2 — TEST pollué uniquement
    run_scenario(
        "S2_test_only",
        CSV,
        OUT_ROOT / "S2_test_only",
        apply_pollution_coherence,
        pollute_train=False,
        pollute_test=True,
    )

    # S3 — TRAIN + TEST pollués indépendamment
    run_scenario(
        "S3_both_independent",
        CSV,
        OUT_ROOT / "S3_both_independent",
        apply_pollution_coherence,
        pollute_train=True,
        pollute_test=True,
    )

    # S4 — TRAIN fixe à 20%, TEST variable
    run_scenario(
        "S4_train20_test_variable",
        CSV,
        OUT_ROOT / "S4_train20_test_variable",
        apply_pollution_coherence,
        pollute_train=True,
        pollute_test=True,
        fixed_train_pct=0.20,
    )
