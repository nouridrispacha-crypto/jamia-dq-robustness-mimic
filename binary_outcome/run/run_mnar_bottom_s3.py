from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common_runner import run_scenario
from polluter.pollute_mnar_bottomS import pollution_mnar_bottom


CSV = str(ROOT_DIR / "data" / "18_variables.csv")
OUT = ROOT_DIR / "Results" / "mnar_bottom" / "outputs_mnar_bottom_S3"

if __name__ == "__main__":
    run_scenario(
        scenario_name="S3_both_independent",
        csv_path=CSV,
        out_dir=OUT,
        pollution_fn=pollution_mnar_bottom,
        pollute_train=True,
        pollute_test=True,
    )
