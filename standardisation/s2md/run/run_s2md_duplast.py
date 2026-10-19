"""S2-MD, cumulative design with duplication last (test-time, binary outcome, N = 3500).

Same runner and grid as run_s2md_cumulative.py; the only change is the
polluter: pollute_cumulative_duplast.py applies the cell-level defects first and
then copies records, so that a duplicate carries the defects of its source.

USAGE
    python run/run_s2md_duplast.py --smoke    # 3 replicates, c = 0 and 0.3
    python run/run_s2md_duplast.py            # 500 replicates, c = 0 ... 0.3
Resume is automatic.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Polluter"))

import common_runner                                          # noqa: E402
from common_runner import run_scenario                        # noqa: E402
from pollute_cumulative_duplast import make_cumulative_duplast_polluter      # noqa: E402
from pollute_mixed import W_OBSERVE                           # noqa: E402

CSV_PATH = ROOT / "data" / "18_variables.csv"
OUT_DIR = ROOT / "Results" / "S2MD_duplast"
SMOKE = "--smoke" in sys.argv

common_runner.SAMPLE_SIZES = [3500]
if SMOKE:
    common_runner.POLLUTION_LEVELS = [0.0, 0.3]
    common_runner.N_BOOTSTRAPS = 3
    SCENARIO = "S2MD_SMOKE_duplast_P4_observe"
else:
    common_runner.POLLUTION_LEVELS = [0.0, 0.1, 0.2, 0.3]
    common_runner.N_BOOTSTRAPS = 500
    SCENARIO = "S2MD_duplast_P4_observe_test"

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"data {CSV_PATH}\nout  {OUT_DIR / (SCENARIO + '.csv')}\n"
          f"levels {common_runner.POLLUTION_LEVELS}, replicates {common_runner.N_BOOTSTRAPS}")
    run_scenario(
        scenario_name=SCENARIO,
        csv_path=str(CSV_PATH),
        out_dir=OUT_DIR,
        pollution_fn=make_cumulative_duplast_polluter(W_OBSERVE),
        pollute_train=False,
        pollute_test=True,
    )
