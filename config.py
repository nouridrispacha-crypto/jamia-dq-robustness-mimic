# -*- coding: utf-8 -*-
"""Every location the pipeline reads or writes, resolved in one place.

Each path is looked up in three places, in order: an environment variable, a `paths_local.py`
beside this file, and a default under the repository. The first two are how a reader points
the code at their own copies of the data; the third is only a resting place so an unset path
fails with a message naming the variable instead of a FileNotFoundError deep in a script.

`paths_local.py` is not version-controlled. It exists so a working machine keeps its own
absolute locations without those locations being published. A template is in
`paths_local.example.py`.

The result directories are the per-outcome simulation output, one directory per
(dimension, scenario) as the runners write it. The survival root is the 90-day relaunch:
horizon at day 90 from ICU admission with a 24-hour landmark.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Les modules se citent par leur nom (`import make_s2_grid`, `import analyse_stats_v2`),
# parce que cinq scripts s'importent ainsi et qu'un renommage pour la forme casserait le code
# qui produit les chiffres. Importer config suffit donc a rendre les trois dossiers visibles.
for _d in (ROOT, ROOT / "analyse", ROOT / "report"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

try:
    import paths_local as _local
except ImportError:                     # un depot neuf n'en a pas : les defauts s'appliquent
    _local = None


def _p(name, default):
    """Variable d'environnement, puis paths_local, puis le defaut sous le depot."""
    v = os.environ.get(name)
    if v:
        return Path(v)
    if _local is not None and hasattr(_local, name):
        return Path(getattr(_local, name))
    return ROOT / default


def need(path, name):
    """A appeler avant de lire : dit quelle variable fixer plutot que de planter plus loin."""
    if not Path(path).exists():
        raise SystemExit(
            "{} est introuvable.\n"
            "Fixez {} dans l'environnement, ou dans paths_local.py "
            "(voir paths_local.example.py).".format(path, name))
    return Path(path)


# ---- le manuscrit et ses images -------------------------------------------------
# Le dossier qui contient manuscript.tex et img/. Les scripts de report/ y ecrivent.
PAPER = _p("DQ_PAPER", "paper")
IMG = PAPER / "img"

# ---- resultats de simulation, une racine par issue ------------------------------
# Chaque racine contient <dimension>/<outputs_...>/<scenario>.csv
BINARY_RESULTS = _p("DQ_BINARY_RESULTS", "results/binary")
BINARY_RESULTS_RERUN = _p("DQ_BINARY_RESULTS_RERUN", "results/binary_rerun")
CONTINUOUS_RESULTS = _p("DQ_CONTINUOUS_RESULTS", "results/continuous")
CONTINUOUS_RESULTS_RERUN = _p("DQ_CONTINUOUS_RESULTS_RERUN", "results/continuous_rerun")
SURVIVAL_RESULTS = _p("DQ_SURVIVAL_RESULTS", "results/survival_j90")

# ---- bras standardise du binaire -----------------------------------------------
# Entrees substituees produites par simulate/standardised/build_scaled_inputs.py :
# un dossier par dimension, un CSV par scenario.
SCALED_INPUTS = _p("DQ_SCALED_INPUTS", "results/scaled_inputs")

# ---- donnees d'entree ----------------------------------------------------------
# Extractions MIMIC-IV. Non redistribuables : a reconstruire depuis MIMIC-IV.
DATA = _p("DQ_DATA", "data")

# ---- sorties d'analyse ---------------------------------------------------------
STATS_OUT = _p("DQ_STATS_OUT", "stats_results")

__all__ = ["ROOT", "PAPER", "IMG", "DATA", "STATS_OUT", "SCALED_INPUTS",
           "BINARY_RESULTS", "BINARY_RESULTS_RERUN", "CONTINUOUS_RESULTS",
           "CONTINUOUS_RESULTS_RERUN", "SURVIVAL_RESULTS", "need"]
