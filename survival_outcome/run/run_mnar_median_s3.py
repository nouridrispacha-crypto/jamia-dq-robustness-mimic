from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))


from common_runner import run_scenario
from polluter.pollute_mnar_medianS import pollution_mnar_median



CSV = str(ROOT_DIR / "data" / "mqds_with_flags_24h.csv")
OUT = ROOT_DIR /"Results" / "mnar_median" / "outputs_mnar_median_S3"

if __name__ == "__main__":
    run_scenario(
        scenario_name="S3_both_independant",
        csv_path=CSV,
        out_dir=OUT,
        pollution_fn=pollution_mnar_median,
        pollute_train=False,
        pollute_test=True,
    )
