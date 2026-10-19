"""Supplementary Figure 3 (binary outcome at N = 7000) on the standardised arm.

Same mechanism as regen_f1_fig.py: BINARY.file_map is repointed at the substituted inputs,
which now carry the standardised N = 7000 rows (extend_scaled_n7000.py), and the paper's
own code draws the figure.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
HERE = Path(__file__).resolve().parent
IN = HERE / "scaled_inputs"
sys.path.insert(0, str(P))

import analyse_stats_v2 as V2

for dim in V2.BINARY.file_map:
    for scen in V2.BINARY.file_map[dim]:
        p = IN / dim / f"{scen}.csv"
        assert p.exists(), p
        V2.BINARY.file_map[dim][scen] = p
print("BINARY.file_map repointe sur le bras standardise")

import make_print_figures as M

M._cache.clear()
STAMP = f"{datetime.now():%Y%m%d_%H%M}"
for ext in ("pdf", "png"):
    f = P / "img" / f"figS5_binary_N7000.{ext}"
    if f.exists():
        shutil.copy(f, P / "img" / f"figS5_binary_N7000_backup_{STAMP}.{ext}")

SCEN3 = [("S1 (training-time)", None, "S1"), ("S2 (test-time)", None, "S2"),
         ("S3 (joint)", None, "S3")]
M.grid("binary",
       M.rows_for(V2.BINARY, M.DIMS),
       [(t, "auc", s) for t, _, s in SCEN3],
       "figS5_binary_N7000", figsize=(8.6, 8.8), n=7000, share=("auc",))

for ext in ("pdf", "png"):
    f = P / "img" / f"figS5_binary_N7000.{ext}"
    if f.exists():
        print("  figS5_binary_N7000.{:3s} {:7.0f} Ko  {}".format(
            ext, f.stat().st_size / 1024,
            datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")))
