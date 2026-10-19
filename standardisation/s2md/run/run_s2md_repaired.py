"""S2-MD with completeness repaired (test-time, binary outcome, N = 3500).

The test set is corrupted exactly as in run_s2md_duplast.py at c = 0.3 under
P_observed, and the true value is then written back into 86.7% of the missing
cells, so that the residual completeness defect rate corresponds to 2% instead
of the 15% measured in eHOP (2/15 of the defects are kept). Every other defect
-- precision, coherence, uniqueness, validity -- stays on the same cells, and
the repair is applied before the duplication step, so a duplicate copies an
already repaired record. About 12.2% of the cells remain altered.

Writing back the true value is a perfect repair, an upper bound on what
imputation could recover. Seeds are those of run_s2md_duplast.py, so replicate
b uses the same cohort, split, imputation and models: the two runs are paired.

USAGE
    python run/run_s2md_repaired.py --smoke    # 3 replicates
    python run/run_s2md_repaired.py            # 500 replicates, c = 0.3 only
Resume is automatic.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Polluter"))

import common_runner                                              # noqa: E402
from common_runner import run_scenario                            # noqa: E402
from pollute_cumulative_duplast import make_cumulative_duplast_polluter  # noqa: E402
from pollute_mixed import W_OBSERVE                               # noqa: E402

REPAIR = 1 - 2 / 15          # completeness defect rate taken from 15% down to 2%

CSV_PATH = ROOT / "data" / "18_variables.csv"
OUT_DIR = ROOT / "Results" / "S2MD_repaired"
SMOKE = "--smoke" in sys.argv

common_runner.SAMPLE_SIZES = [3500]
common_runner.POLLUTION_LEVELS = [0.3]      # clean reference: the p = 0 rows of S2MD_duplast
if SMOKE:
    common_runner.N_BOOTSTRAPS = 3
    SCENARIO = "S2MD_SMOKE_repaired"
else:
    common_runner.N_BOOTSTRAPS = 500
    SCENARIO = "S2MD_repaired_test"

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"data {CSV_PATH}\nout  {OUT_DIR / (SCENARIO + '.csv')}\n"
          f"levels {common_runner.POLLUTION_LEVELS}, replicates {common_runner.N_BOOTSTRAPS}, "
          f"repair {REPAIR:.3f}")
    run_scenario(
        scenario_name=SCENARIO,
        csv_path=str(CSV_PATH),
        out_dir=OUT_DIR,
        pollution_fn=make_cumulative_duplast_polluter(W_OBSERVE, "P_observed_repaired",
                                                      completeness_repair=REPAIR),
        pollute_train=False,
        pollute_test=True,
    )
