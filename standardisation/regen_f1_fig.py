"""Supplementary Figure 1 (F1-macro) on the standardised arm.

Same mechanism as regen_figures.py: BINARY.file_map is repointed at the substituted inputs
before make_print_figures is used, so the figure is drawn by the paper's own code with no
change to it. What is new is that those inputs now carry f1_macro for the substituted rows
(fill_f1_scaled.py), which is why regen_figures.py had to skip this one.

make_print_figures is imported rather than run, so only this figure is rebuilt.
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
    f = P / "img" / f"figure_classif_f1.{ext}"
    if f.exists():
        shutil.copy(f, P / "img" / f"figure_classif_f1_backup_{STAMP}.{ext}")

SCEN3 = [("S1 (training-time)", None, "S1"), ("S2 (test-time)", None, "S2"),
         ("S3 (joint)", None, "S3")]
M.grid("binary",
       M.rows_for(V2.BINARY, M.DIMS),
       [(t, "f1_macro", s) for t, _, s in SCEN3],
       "figure_classif_f1", figsize=(8.6, 8.8), share=("f1_macro",))

for ext in ("pdf", "png"):
    f = P / "img" / f"figure_classif_f1.{ext}"
    if f.exists():
        print("  figure_classif_f1.{:3s} {:7.0f} Ko  {}".format(
            ext, f.stat().st_size / 1024,
            datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")))
