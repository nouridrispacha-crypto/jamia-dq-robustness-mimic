"""Re-run the binary statistical analysis on the standardised arm, into a separate folder.

analyse_stats_v2 writes three files per metric: the loss table, the residual diagnostics
that Supplementary Table S5 reports, and the contrasts behind Table 3. The binary ones on
disk date from 12 September, before the standardisation, so Supplementary Table S5 and
Figure 5 still describe superseded losses.

Nothing existing is overwritten: the output goes to binary_stats_results_v2_scaled, so the
new contrasts can be compared with the verdicts already in the manuscript before anything
is adopted.
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
print("  exemple :", V2.BINARY.file_map["validity"]["S2"])

OUT = P / "binary_stats_results_v2_scaled"
OUT.mkdir(exist_ok=True)
print("sortie :", OUT, "\n")

for scen in ("S1", "S2", "S3"):
    for metric in V2.BINARY.metrics:
        if metric == "f1_macro":
            print("  {} {} : saute (le re-run n'a pas enregistre le F1)".format(scen, metric))
            continue
        print("  {} {} ...".format(scen, metric), flush=True)
        V2.run_metric(V2.BINARY, scen, metric, OUT)
print("\ntermine")
