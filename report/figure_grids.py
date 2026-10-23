"""Regenerate the three 30-panel grids with the settings the published figures used.

I previously called make_s2_grid.main() with its default arguments, which are portrait
(7.5 x 8.6 in) and shade=True, and so replaced a landscape figure without the "most
damaging" rectangles by an older design. The published PDFs are 13.15 x 6.31 in and carry
no rectangles, so figsize is taken from them and shade is off.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PAPER
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(PAPER)
sys.path.insert(0, str(P))
IN = Path(__file__).resolve().parent / "scaled_inputs"

import analyse_stats_v2 as V2
import make_s2_grid as G

# the size the published figures actually are, read from the backup rather than assumed
B = sorted(P.glob("img_backup_unstandardised_*"))[-1]
raw = (B / "figure_s2_grid.pdf").read_bytes()
v = [float(x) for x in re.findall(rb"/MediaBox\s*\[([^\]]*)\]", raw)[0].split()]
W, H = (v[2] - v[0]) / 72, (v[3] - v[1]) / 72
print(f"taille publiee : {W:.2f} x {H:.2f} pouces")
assert W > H, "la figure publiee n'est pas en paysage"

for dim in V2.BINARY.file_map:
    for sc in V2.BINARY.file_map[dim]:
        V2.BINARY.file_map[dim][sc] = IN / dim / f"{sc}.csv"
for c in (G._cache, G._lead_cache, G._model_cache):
    c.clear()
print("BINARY.file_map repointe sur le bras standardise")

for name, scen in (("figure_s2_grid", "S2"), ("suppfig_s1_grid", "S1"),
                   ("suppfig_s3_grid", "S3")):
    G.main(out_name=name, figsize=(W, H), scenario=scen, shade=False)
    out = P / "img" / f"{name}.pdf"
    r = out.read_bytes()
    q = [float(x) for x in re.findall(rb"/MediaBox\s*\[([^\]]*)\]", r)[0].split()]
    w, h = (q[2] - q[0]) / 72, (q[3] - q[1]) / 72
    assert w > h, f"{name} n'est pas en paysage"
    assert abs(w - W) < 0.3 and abs(h - H) < 0.3, (name, w, h)
    print(f"  {name:18s} {w:.2f} x {h:.2f} pouces, sans encadres")
print("\nles trois grilles regenerees au format publie")
