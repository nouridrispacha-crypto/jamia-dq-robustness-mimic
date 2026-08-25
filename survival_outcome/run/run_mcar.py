from pathlib import Path
import sys

# --------------------------------------------------------------------
# Ajouter TEST2 et TEST2/polluter au PYTHONPATH
# --------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
POLLUTER_DIR = ROOT_DIR / "polluter"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(POLLUTER_DIR))

# --------------------------------------------------------------------
# Import du runner générique
# --------------------------------------------------------------------
from common_runner import run_scenario

# --------------------------------------------------------------------
# Import du pollueur MCAR (wrapper compatible)
# --------------------------------------------------------------------
from pollute_mcar import pollution_mcar

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
CSV = str(ROOT_DIR / "data" / "mqds_with_flags_24h.csv")
OUT_ROOT = ROOT_DIR / "outputs_mcar_S1234_harmonized"

if __name__ == "__main__":
    OUT_ROOT.mkdir(exist_ok=True)

    run_scenario(
        scenario_name="S1_train_only",
        csv_path=CSV,
        out_dir=OUT_ROOT / "S1_train_only",
        pollution_fn=pollution_mcar,
        pollute_train=True,
        pollute_test=False,
        fixed_train_pct=None,
    )

    run_scenario(
        scenario_name="S2_test_only",
        csv_path=CSV,
        out_dir=OUT_ROOT / "S2_test_only",
        pollution_fn=pollution_mcar,
        pollute_train=False,
        pollute_test=True,
        fixed_train_pct=None,
    )

    run_scenario(
        scenario_name="S3_both_independent",
        csv_path=CSV,
        out_dir=OUT_ROOT / "S3_both_independent",
        pollution_fn=pollution_mcar,
        pollute_train=True,
        pollute_test=True,
        fixed_train_pct=None,
    )

    run_scenario(
        scenario_name="S4_train20_test_variable",
        csv_path=CSV,
        out_dir=OUT_ROOT / "S4_train20_test_variable",
        pollution_fn=pollution_mcar,
        pollute_train=True,
        pollute_test=True,
        fixed_train_pct=0.20,
    )
