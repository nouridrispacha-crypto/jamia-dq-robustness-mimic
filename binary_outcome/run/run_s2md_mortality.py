# -*- coding: utf-8 -*-
"""
LANCEMENT DU SCENARIO S2-MD — OUTCOME MORTALITE (binaire)
Arborescence MIE_2026 : ce script vit dans run/

S2-MD = degradation simultanee des cinq dimensions DQ a budget de corruption
egal, appliquee au test uniquement (pendant de S2, scenario de deploiement).

Le budget total T balaie la meme grille que les runs mono-dimension, ce qui
rend S2-MD directement comparable a S1/S2/S3 au meme niveau c.

PROFILS DE COMPOSITION
----------------------
Le poids w d'une dimension est la fraction de DOSSIERS qui presentent ce type
de defaut ; T est l'intensite du defaut a l'interieur de chaque dossier touche.

  P0_uniform         20 / 20 / 20 / 20 / 20   budget egal (corps de l'article)
  P1_documentation   45 / 10 / 15 / 20 / 10   unite surchargee
  P2_saisie          10 / 30 / 20 / 35 /  5   erreurs de saisie et capteurs
  P3_integration     15 / 10 / 30 /  5 / 40   fusion de systemes / migration
  (ordre : completeness / coherence / validity / precision / uniqueness)

Ce sont des profils STYLISES : ils encodent des situations operationnelles
plausibles, pas des prevalences mesurees. T est identique sur les quatre, de
sorte que seule la composition varie.

USAGE
-----
    cd chemin/vers/MIE_2026

    python run/run_s2md_mortality.py --smoke                    # 3 replicats, P0
    python run/run_s2md_mortality.py --quick --all              # 50 replicats, 4 profils
    python run/run_s2md_mortality.py --quick -p P3_integration  # 50 replicats, 1 profil
    python run/run_s2md_mortality.py --all                      # 500 replicats, 4 profils

Chaque profil ecrit dans son propre CSV. La reprise est automatique : un profil
deja calcule a 50 replicats repart a b=50 lorsqu'on relance en mode complet.
"""

import sys
from pathlib import Path

# ============================================================
# CHEMINS — resolus depuis l'emplacement du script
# ============================================================

ROOT = Path(__file__).resolve().parent.parent          # .../MIE_2026
sys.path.insert(0, str(ROOT))                          # common_runner.py
sys.path.insert(0, str(ROOT / "polluter"))             # pollueurs

import common_runner                                    # noqa: E402
from common_runner import run_scenario                   # noqa: E402
from pollute_mixed import make_mixed_polluter, PROFILES  # noqa: E402


# ============================================================
# A ADAPTER : nom du fichier de donnees mortalite
# ============================================================
# Le meme fichier que celui des runs mono-dimension S1/S2/S3, sinon la
# comparaison a iso-budget ne tient plus.

DATA_FILENAME = "18_variables.csv"

CSV_PATH = ROOT / "data" / DATA_FILENAME
OUT_DIR = ROOT / "Results" / "S2MD_mixed"


# ============================================================
# NOMS DE SORTIE PAR PROFIL
# ============================================================
# P0 conserve son nom historique pour que les 50 replicats deja calcules
# soient reconnus par la reprise et ne soient pas recalcules.

SCENARIO_NAMES = {
    "P0_uniform": "S2MD_mixed_uniform_test",
    "P1_documentation": "S2MD_mixed_P1_documentation_test",
    "P2_saisie": "S2MD_mixed_P2_saisie_test",
    "P3_integration": "S2MD_mixed_P3_integration_test",
    "P4_observe": "S2MD_mixed_P4_observe_test",
}

ORDER = ["completeness", "coherence", "validity", "precision", "uniqueness"]


# ============================================================
# PARAMETRES EXPERIMENTAUX
# ============================================================
#
#   --smoke  3 replicats, 2 niveaux   -> verification technique (CSV separe)
#   --quick  50 replicats, 6 niveaux  -> lecture provisoire des tendances
#   (defaut) 500 replicats, 6 niveaux -> run final de l'article
#
# --quick et le run final ecrivent dans LE MEME CSV, et c'est voulu : les
# graines du runner ne dependent que de l'indice de bootstrap b (10_000+b,
# 40_000+b, 50_000+b), jamais de N_BOOTSTRAPS. Les 50 replicats provisoires
# sont donc exactement les 50 premiers du run final.

SMOKE = "--smoke" in sys.argv
QUICK = "--quick" in sys.argv
ALL_PROFILES = "--all" in sys.argv

common_runner.SAMPLE_SIZES = [3500]

if SMOKE:
    common_runner.POLLUTION_LEVELS = [0.0, 0.5]
    common_runner.N_BOOTSTRAPS = 3
elif QUICK:
    common_runner.POLLUTION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    common_runner.N_BOOTSTRAPS = 50
else:
    common_runner.POLLUTION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    common_runner.N_BOOTSTRAPS = 500


def selected_profiles() -> list:
    """Profils a executer, dans l'ordre."""
    if ALL_PROFILES:
        return list(PROFILES.keys())

    for flag in ("-p", "--profile"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            if i + 1 < len(sys.argv):
                name = sys.argv[i + 1]
                if name not in PROFILES:
                    print(f"[ERREUR] profil inconnu : {name}")
                    print(f"         disponibles : {list(PROFILES)}")
                    sys.exit(1)
                return [name]

    return ["P0_uniform"]


# ============================================================
# VERIFICATION DES CHEMINS AVANT DE LANCER LE CALCUL
# ============================================================

def check_paths() -> bool:
    ok = True

    if not (ROOT / "common_runner.py").exists():
        print(f"[ERREUR] common_runner.py introuvable dans {ROOT}")
        ok = False

    if not (ROOT / "polluter").is_dir():
        print(f"[ERREUR] dossier Polluter introuvable dans {ROOT}")
        ok = False

    if not CSV_PATH.exists():
        print(f"[ERREUR] fichier de donnees introuvable :\n         {CSV_PATH}")
        data_dir = ROOT / "data"
        if data_dir.is_dir():
            candidates = sorted(
                p.name for p in data_dir.iterdir()
                if p.suffix.lower() in {".csv", ".parquet", ".txt"}
            )
            if candidates:
                print("\n         Fichiers disponibles dans data/ :")
                for c in candidates:
                    print(f"           - {c}")
                print("\n         Recopie le bon nom dans DATA_FILENAME, "
                      "en haut du script.")
        ok = False

    return ok


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not check_paths():
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = selected_profiles()

    print("=" * 70)
    print(f"Donnees       : {CSV_PATH}")
    print(f"Sorties       : {OUT_DIR}")
    print(f"N             : {common_runner.SAMPLE_SIZES}")
    print(f"Niveaux T     : {common_runner.POLLUTION_LEVELS}")
    print(f"Bootstraps    : {common_runner.N_BOOTSTRAPS}")
    print("Degradation   : test-time uniquement (pendant de S2)")
    print(f"Profils       : {', '.join(profiles)}")
    print("=" * 70)

    for pname in profiles:

        scenario = f"S2MD_SMOKE_{pname}" if SMOKE else SCENARIO_NAMES[pname]
        weights = PROFILES[pname]
        composition = " / ".join(f"{d[:5]} {weights[d]:.0%}" for d in ORDER)

        print()
        print("-" * 70)
        print(f"PROFIL   : {pname}")
        print(f"Scenario : {scenario}")
        print(f"Compo.   : {composition}")
        print("-" * 70)

        run_scenario(
            scenario_name=scenario,
            csv_path=str(CSV_PATH),
            out_dir=OUT_DIR,
            pollution_fn=make_mixed_polluter(weights, pname),
            pollute_train=False,   # S2-like
            pollute_test=True,
        )

    print()
    print("=" * 70)
    print("Termine. CSV produits :")
    for pname in profiles:
        name = f"S2MD_SMOKE_{pname}" if SMOKE else SCENARIO_NAMES[pname]
        print(f"  {OUT_DIR / (name + '.csv')}")
    print("=" * 70)
