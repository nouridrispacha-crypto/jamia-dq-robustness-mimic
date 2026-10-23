# -*- coding: utf-8 -*-
"""Template for machine-specific paths. Copy to paths_local.py and edit.

paths_local.py is not version-controlled: it holds where the data and the simulation output
live on one machine, which is nobody else's business and would not be valid anywhere else.
Any name defined here overrides config.py's default; an environment variable of the same name
overrides this file in turn.
"""

# Dossier contenant manuscript.tex et img/
DQ_PAPER = "/chemin/vers/le/manuscrit"

# Extractions MIMIC-IV (non redistribuables)
DQ_DATA = "/chemin/vers/les/donnees"

# Sorties de simulation : <dimension>/<outputs_...>/<scenario>.csv
DQ_BINARY_RESULTS = "/chemin/vers/resultats/binary"
DQ_BINARY_RESULTS_RERUN = "/chemin/vers/resultats/binary_rerun"
DQ_CONTINUOUS_RESULTS = "/chemin/vers/resultats/continuous"
DQ_CONTINUOUS_RESULTS_RERUN = "/chemin/vers/resultats/continuous_rerun"
DQ_SURVIVAL_RESULTS = "/chemin/vers/resultats/survival_j90"

# Entrees substituees du bras standardise
DQ_SCALED_INPUTS = "/chemin/vers/resultats/scaled_inputs"
