"""Regenerate the binary-outcome figures from the standardised arm.

BINARY.file_map is repointed at the substituted inputs before the figure scripts are
imported, so each runs unchanged. The continuous and survival configs are untouched: those
chains already standardised, so their panels are unaffected.

Supplementary Figure 1 (F1-macro) is deliberately NOT regenerated: the re-run recorded only
the two co-primary metrics, so f1_macro is empty on the replaced rows.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path("C:/Users/IDRINO/Desktop/JAMIA_PAPER")
sys.path.insert(0, str(P))
HERE = Path(__file__).resolve().parent
IN = HERE / "scaled_inputs"

import analyse_stats_v2 as V2

# ── keep the originals before anything overwrites them
BACKUP = P / f"img_backup_unstandardised_{datetime.now():%Y%m%d_%H%M}"
BACKUP.mkdir(exist_ok=True)
TARGETS = ["figure6_heatmap_s2", "figure_s2_grid", "suppfig9_heatmap_s1",
           "suppfig10_heatmap_s3", "suppfig_s1_grid", "suppfig_s3_grid",
           "figure6_contrasts_main", "suppfig10_contrasts_binary",
           "suppfig11_contrasts_continuous", "suppfig12_contrasts_survival"]
saved = 0
for name in TARGETS:
    for ext in ("pdf", "png"):
        f = P / "img" / f"{name}.{ext}"
        if f.exists():
            shutil.copy(f, BACKUP / f.name)
            saved += 1
print(f"{saved} fichiers sauvegardes dans {BACKUP.name}\n")

# ── repoint the binary config
old = {d: dict(s) for d, s in V2.BINARY.file_map.items()}
for dim in V2.BINARY.file_map:
    for scen in V2.BINARY.file_map[dim]:
        p = IN / dim / f"{scen}.csv"
        assert p.exists(), p
        V2.BINARY.file_map[dim][scen] = p
print("BINARY.file_map repointe sur le bras standardise")
print("  exemple :", V2.BINARY.file_map["validity"]["S2"])

print("  CONTINUOUS et SURVIVAL inchangees\n")

made = []

print("=== Figure 3 et Figures supp. 9 et 10 : cartes de chaleur ===")
import make_supp_heatmaps as H
H._cache.clear()
H.main() if hasattr(H, "main") else None
if not hasattr(H, "main"):
    # the script does its work under __main__; run it as a module instead
    import runpy
    runpy.run_path(str(P / "make_supp_heatmaps.py"), run_name="__main__")
made += ["figure6_heatmap_s2", "suppfig9_heatmap_s1", "suppfig10_heatmap_s3"]

print("\n=== Figure 4 et grilles supplementaires : courbes par niveau ===")
import make_s2_grid as G
for cache in (G._cache, G._lead_cache, G._model_cache):
    cache.clear()
G.main(out_name="figure_s2_grid", scenario="S2")
G.main(out_name="suppfig_s1_grid", scenario="S1")
G.main(out_name="suppfig_s3_grid", scenario="S3")
made += ["figure_s2_grid", "suppfig_s1_grid", "suppfig_s3_grid"]

print("\n=== Figures de contrastes ===")
import runpy
runpy.run_path(str(P / "make_contrast_figure.py"), run_name="__main__")
made += ["figure6_contrasts_main", "suppfig10_contrasts_binary"]

print("\n=== fichiers reecrits ===")
for name in sorted(set(made)):
    f = P / "img" / f"{name}.pdf"
    if f.exists():
        age = (datetime.now().timestamp() - f.stat().st_mtime)
        tag = "REECRIT" if age < 600 else "inchange"
        print(f"  {name:34s} {tag:9s} {f.stat().st_size / 1024:7.0f} Ko")
    else:
        print(f"  {name:34s} ABSENT")
print("\nSupplementary Figure 1 (F1-macro) NON regeneree : f1_macro absent du re-run")
