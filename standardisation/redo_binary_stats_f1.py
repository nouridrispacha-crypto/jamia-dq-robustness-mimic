"""The F1-macro half of the binary statistics, on the standardised arm.

redo_binary_stats.py skipped f1_macro because the October re-run had not recorded it. The
F1 queue has since produced it and fill_f1_scaled.py wrote it into the substituted inputs,
so the three missing rows of Supplementary Table S5 -- and the skewness that decides which
cases Supplementary Figure 5 should show -- can now be computed.

Writes into the same binary_stats_results_v2_scaled folder, beside the auc and brier files.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
IN = Path(__file__).resolve().parent / "scaled_inputs"
sys.path.insert(0, str(P))

import analyse_stats_v2 as V2                                         # noqa: E402

for dim in V2.BINARY.file_map:
    for scen in V2.BINARY.file_map[dim]:
        p = IN / dim / "{}.csv".format(scen)
        assert p.exists(), p
        V2.BINARY.file_map[dim][scen] = p
print("file_map binaire repointe sur le bras standardise")

OUT = P / "binary_stats_results_v2_scaled"
for scen in ("S1", "S2", "S3"):
    print("  {} f1_macro ...".format(scen), flush=True)
    V2.run_metric(V2.BINARY, scen, "f1_macro", OUT)
print("\ntermine")
