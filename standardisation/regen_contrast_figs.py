"""Redraw the three contrast figures after the symlog tick fix.

Same repointing as regen_figures.py: the binary config reads the substituted inputs, the
continuous and survival ones are left alone because those chains already standardised.
Only the x-axis ticks of the compressed panels change; no number moves.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
IN = Path(__file__).resolve().parent / "scaled_inputs"
sys.path.insert(0, str(P))

import analyse_stats_v2 as V2

for dim in V2.BINARY.file_map:
    for scen in V2.BINARY.file_map[dim]:
        p = IN / dim / f"{scen}.csv"
        assert p.exists(), p
        V2.BINARY.file_map[dim][scen] = p
print("BINARY.file_map repointe sur le bras standardise")

STAMP = f"{datetime.now():%Y%m%d_%H%M}"
TARGETS = ["figure6_contrasts_main", "suppfig10_contrasts_binary",
           "suppfig11_contrasts_continuous", "suppfig12_contrasts_survival"]
for name in TARGETS:
    for ext in ("pdf", "png"):
        f = P / "img" / f"{name}.{ext}"
        if f.exists():
            shutil.copy(f, P / "img" / f"{name}_backup_{STAMP}.{ext}")

import make_contrast_figure as C

C.figure(V2.BINARY, "S2", "figure6_contrasts_main", only="auc", scale=1.25)
for i, cfg in enumerate((V2.BINARY, V2.CONTINUOUS, V2.SURVIVAL)):
    C.figure(cfg, "S2", f"suppfig{10 + i}_contrasts_{cfg.name}")
C.csv()

for name in TARGETS:
    f = P / "img" / f"{name}.pdf"
    if f.exists():
        print("  {:34s} {:7.0f} Ko  {}".format(
            name, f.stat().st_size / 1024,
            datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")))
