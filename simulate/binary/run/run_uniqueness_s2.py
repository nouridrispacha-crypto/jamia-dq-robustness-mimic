from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common_runner import run_scenario
from polluter.pollute_uniqueness import pollution_uniqueness

CSV = str(ROOT_DIR / "data" / "18_variables.csv")
OUT = ROOT_DIR /"Results" / "uniqueness"/ "outputs_uniqueness_S2"

if __name__ == "__main__":
    run_scenario(
        scenario_name="S2_test_only",
        csv_path=CSV,
        out_dir=OUT,
        pollution_fn=pollution_uniqueness,
        pollute_train=False,
        pollute_test=True,
    )
