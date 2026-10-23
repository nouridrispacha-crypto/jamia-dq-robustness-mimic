"""Redraw Figure 2 after repointing S2MD_CSV, and check it did not move.

make_print_figures.py pointed at the September S2-MD run while the published figure and
Table S8 report the standardised one; the figure must therefore come out identical to the
file already on disk. If it does not, the published figure was drawn from something else
again and that has to be understood before anything is archived.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PAPER
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(PAPER)
sys.path.insert(0, str(P))

OUT = P / "img" / "figure5_s2md_p4observed.png"
before = hashlib.md5(OUT.read_bytes()).hexdigest() if OUT.exists() else None
STAMP = f"{datetime.now():%Y%m%d_%H%M}"
for ext in ("pdf", "png"):
    f = P / "img" / f"figure5_s2md_p4observed.{ext}"
    if f.exists():
        shutil.copy(f, P / "img" / f"figure5_s2md_p4observed_backup_{STAMP}.{ext}")

import make_print_figures as M

print("source S2-MD :", M.S2MD_CSV)
assert M.S2MD_CSV.exists(), M.S2MD_CSV
M._cache.clear()
M.grid("binary", [("", {"S2": M.S2MD_CSV})],
       [("AUROC", "auc", "S2"), ("Brier score", "brier", "S2")],
       "figure5_s2md_p4observed", figsize=(7.6, 3.0))

after = hashlib.md5(OUT.read_bytes()).hexdigest()
print("md5 avant {}\nmd5 apres {}".format(before, after))
print("la figure publiee venait bien du run standardise"
      if before == after else "ATTENTION : la figure a change")
