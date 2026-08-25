from pathlib import Path
import numpy as np
import sys

# Dossiers
THIS_DIR = Path(__file__).resolve().parent      # .../TEST2/run
ROOT_DIR = THIS_DIR.parent                      # .../TEST2
POLLUTER_DIR = ROOT_DIR / "polluter"

# Pour que Python trouve common_runner et les polluters
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(POLLUTER_DIR))

# Import du runner générique
from common_runner import (
    run_scenario,
    POLLUTION_LEVELS,
    SAMPLE_SIZES,
    TEST_SIZE,
    N_BOOTSTRAPS,
    MODEL_SEEDS,
)

# Import de notre pollueur "autour de la médiane SOFA"
from pollute_mnar_medianS import apply_completeness_pollution_mnar_medianS_patients_and_cells
from pollute_mnar_medianS import pollution_mnar_medianS

# CSV et sortie
CSV = str(ROOT_DIR / "data" / "mqds_with_flags_24h.csv")
OUT_ROOT = ROOT_DIR / "outputs_mnar_medianS_S1234_harmonized"


def apply_pollution_mnar_medianS(
    Xtr,
    Xte,
    p,
    fixed_train_pct,
    pollute_train,
    pollute_test,
    seed: int,
):
    """
    Pollution MNAR basée sur la médiane de SOFA :
    on pollue les patients dont le SOFA est le plus proche de la médiane.
    """

    Xtr_pol = Xtr.copy()
    Xte_pol = Xte.copy()

    # TRAIN
    if pollute_train:
        density_train = float(p if fixed_train_pct is None else fixed_train_pct)
        Xtr_pol, _ = apply_completeness_pollution_mnar_medianS_patients_and_cells(
            Xtr_pol,
            target_missing_frac=density_train,
            rng=np.random.default_rng(seed),
        )

    # TEST
    if pollute_test:
        density_test = float(p)
        rs = seed + 1000
        Xte_pol, _ = apply_completeness_pollution_mnar_medianS_patients_and_cells(
            Xte_pol,
            target_missing_frac=density_test,
            rng=np.random.default_rng(rs),
        )

    return Xtr_pol, Xte_pol


if __name__ == "__main__":
    OUT_ROOT.mkdir(exist_ok=True)

    # S1 : train uniquement
    run_scenario(
        "S1_train_only",
        CSV,
        OUT_ROOT / "S1_train_only",
        apply_pollution_mnar_medianS,
        pollute_train=True,
        pollute_test=False,
    )

    # S2 : test uniquement
    run_scenario(
        "S2_test_only",
        CSV,
        OUT_ROOT / "S2_test_only",
        apply_pollution_mnar_medianS,
        pollute_train=False,
        pollute_test=True,
    )

    # S3 : train + test pollués indépendamment
    run_scenario(
        "S3_both_independent",
        CSV,
        OUT_ROOT / "S3_both_independent",
        apply_pollution_mnar_medianS,
        pollute_train=True,
        pollute_test=True,
    )

    # S4 : train fixé à 20 %, test varie
    run_scenario(
        "S4_train20_test_variable",
        CSV,
        OUT_ROOT / "S4_train20_test_variable",
        apply_pollution_mnar_medianS,
        pollute_train=True,
        pollute_test=True,
        fixed_train_pct=0.20,
    )
