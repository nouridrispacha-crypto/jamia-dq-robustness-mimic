from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
POLLUTER_DIR = ROOT_DIR / "polluter"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(POLLUTER_DIR))


from common_runner import run_scenario
from polluter.coherence_polluter import pollution_coherence

CSV = str(ROOT_DIR / "data" / "18_variables.csv")
OUT = ROOT_DIR /"Results" / "Coherence" /"outputs_coherence_s2"


if __name__ == "__main__":
    run_scenario(
        scenario_name="S2_test_only",
        csv_path=CSV,
        out_dir=OUT,
        pollution_fn=pollution_coherence,
        pollute_train=False,
        pollute_test=True,
    )
