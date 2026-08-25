# -*- coding: utf-8 -*-
"""
MIXED POLLUTER — SCENARIO S4 (budget-matched multi-dimension degradation)
Compatible common_runner (signature: fn(df, density, seed) -> (df, row_idx, col_idx))

PRINCIPE
--------
Les cinq dimensions DQ sont appliquees simultanement sur une meme cohorte,
a budget de corruption egal avec les runs mono-dimension.

Allocation disjointe par strates de lignes :
  - les lignes sont partitionnees aleatoirement en 5 strates de tailles w_k * N
  - la dimension k est appliquee a l'interieur de sa strate, a la densite T
  - aucune cellule ne peut recevoir deux mecanismes (disjonction par construction)
  - l'ordre d'application n'a aucun effet
  - aucune contamination de dtype entre pollueurs (validity ecrit des chaines
    uniquement dans sa propre strate, avant reassemblage)

BUDGET
------
Cellules corrompues = sum_k (w_k * N * P * T) = N * P * T
=> le taux global mesure vaut T, identique aux runs mono-dimension au meme c.
=> chaque dimension consomme exactement w_k du budget total.

Pour uniqueness (mecanisme au niveau ligne), la parite tombe automatiquement :
une strate de w_k*N lignes polluee a densite T remplace ~w_k*T*N lignes,
soit w_k*T de l'ensemble des cellules. A w=0.2 et T=0.5 : 10% des lignes.

LIMITE ASSUMEE
--------------
Un enregistrement donne recoit un seul type de defaut. Le melange est realise
au niveau de la cohorte, pas au niveau du dossier patient.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Le dossier Polluter/ est ajoute au chemin d'import : les cinq pollueurs
# sont ainsi trouves quel que soit le dossier depuis lequel le script est lance.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pollute_mnar_top import pollution_mnar_top
from coherence_polluter import pollution_coherence
from pollute_validity import pollution_validity
from pollute_precision import pollution_precision
from pollute_uniqueness import pollution_uniqueness


# ============================================================
# DIMENSIONS ET PROFILS
# ============================================================

DIMENSIONS = ["completeness", "coherence", "validity", "precision", "uniqueness"]

_POLLUTERS = {
    "completeness": pollution_mnar_top,
    "coherence": pollution_coherence,
    "validity": pollution_validity,
    "precision": pollution_precision,
    "uniqueness": pollution_uniqueness,
}

# Decalage de graine par dimension : garantit l'independance entre strates
# tout en restant deterministe pour un seed donne.
_SEED_OFFSET = {
    "completeness": 101,
    "coherence": 202,
    "validity": 303,
    "precision": 404,
    "uniqueness": 505,
}

# Profil principal (corps de l'article) : budget egal.
W_UNIFORM: Dict[str, float] = {d: 0.20 for d in DIMENSIONS}

# Profils stylises (materiel supplementaire).
W_DOCUMENTATION: Dict[str, float] = {
    "completeness": 0.45,
    "coherence": 0.10,
    "validity": 0.15,
    "precision": 0.20,
    "uniqueness": 0.10,
}

W_SAISIE: Dict[str, float] = {
    "completeness": 0.10,
    "coherence": 0.30,
    "validity": 0.20,
    "precision": 0.35,
    "uniqueness": 0.05,
}

W_INTEGRATION: Dict[str, float] = {
    "completeness": 0.15,
    "coherence": 0.10,
    "validity": 0.30,
    "precision": 0.05,
    "uniqueness": 0.40,
}

# Profil OBSERVE : pondere par les taux de defaut reellement mesures sur
# l entrepot eHOP du CHU d Angers, cohorte de reanimation adulte,
# janvier 2024 a mai 2026, 5 192 sejours.
#
#   dimension      taux mesure    poids normalise
#   completude        15 %            68,5 %
#   precision          2,8 %          12,8 %
#   unicite            2,0 %           9,1 %
#   coherence          1,9 %           8,7 %
#   validite           0,03 %          0,9 %
#
# La validite est residuelle mais non nulle : les donnees ont franchi la
# couche de controle a l integration, qui rejette les valeurs non
# parsables et hors domaine. Un ETL peut rejeter une valeur invalide, il
# ne peut pas inventer une valeur absente : d ou un profil reel domine
# par la completude.
W_OBSERVE: Dict[str, float] = {
    "completeness": 0.685,
    "coherence": 0.087,
    "validity": 0.009,
    "precision": 0.128,
    "uniqueness": 0.091,
}

PROFILES: Dict[str, Dict[str, float]] = {
    "P0_uniform": W_UNIFORM,
    "P1_documentation": W_DOCUMENTATION,
    "P2_saisie": W_SAISIE,
    "P3_integration": W_INTEGRATION,
    "P4_observe": W_OBSERVE,
}


# ============================================================
# PARTITION DES LIGNES (methode du plus fort reste)
# ============================================================

def _largest_remainder_sizes(n: int, weights: List[float]) -> List[int]:
    """Repartit n lignes selon weights en preservant exactement la somme n."""
    raw = [n * w for w in weights]
    base = [int(np.floor(x)) for x in raw]
    remainder = n - sum(base)

    if remainder > 0:
        frac = [(raw[i] - base[i], i) for i in range(len(raw))]
        frac.sort(key=lambda t: (-t[0], t[1]))
        for j in range(remainder):
            base[frac[j][1]] += 1

    return base


_PROTECTED = {"subject_id", "hadm_id", "stay_id", "in_hospital_mortality"}


def _prepare_sub(sub: pd.DataFrame, dim: str) -> pd.DataFrame:
    """
    Prepare les dtypes d'une strate avant pollution.

    - upcast en float64 des colonnes numeriques : evite les erreurs de
      setitem lossy (pandas >= 2.2 en mode strict, obligatoire en pandas 3)
      lorsqu'un swap de coherence ecrit une valeur continue dans une colonne
      entiere.
    - upcast en object pour la strate validity, qui ecrit des chaines.
    """
    sub = sub.copy()
    num_cols = [
        c for c in sub.columns
        if c not in _PROTECTED and pd.api.types.is_numeric_dtype(sub[c])
    ]

    for c in num_cols:
        sub[c] = sub[c].astype("float64")

    if dim == "validity":
        for c in num_cols:
            sub[c] = sub[c].astype(object)

    return sub


def _split_rows(
    n: int,
    weights: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Partition aleatoire disjointe des positions de lignes."""
    dims = [d for d in DIMENSIONS if weights.get(d, 0.0) > 0.0]
    w = np.array([weights[d] for d in dims], dtype=float)
    w = w / w.sum()

    sizes = _largest_remainder_sizes(n, list(w))
    perm = rng.permutation(n)

    out: Dict[str, np.ndarray] = {}
    start = 0
    for d, s in zip(dims, sizes):
        out[d] = perm[start:start + s]
        start += s

    return out


# ============================================================
# FABRIQUE DU POLLUEUR MIXTE
# ============================================================

def make_mixed_polluter(
    weights: Optional[Dict[str, float]] = None,
    profile_name: str = "P0_uniform",
    verbose: bool = False,
):
    """
    Retourne une fonction compatible common_runner.

    weights : poids par dimension, sommant a 1 (renormalise sinon).
              Par defaut : budget egal (0.20 chacun).
    """
    if weights is None:
        weights = W_UNIFORM

    missing = [d for d in weights if d not in DIMENSIONS]
    if missing:
        raise ValueError(f"Dimensions inconnues : {missing}")

    total_w = float(sum(weights.values()))
    if total_w <= 0:
        raise ValueError("La somme des poids doit etre strictement positive.")
    weights = {d: float(weights.get(d, 0.0)) / total_w for d in DIMENSIONS}

    def pollution_mixed(
        df: pd.DataFrame,
        density: float,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:

        assert 0.0 <= float(density) <= 1.0, "density doit etre dans [0,1]"

        n = len(df)
        if n == 0 or float(density) == 0.0:
            return df.copy(), np.array([], dtype=int), np.array([], dtype=int)

        # Index positionnel stable : le runner reaffecte la cible par alignement
        # d'index, donc l'ordre et les labels doivent etre preserves.
        original_index = df.index
        base = df.reset_index(drop=True)

        rng = np.random.default_rng(int(seed))
        strata = _split_rows(n, weights, rng)

        parts: List[pd.DataFrame] = []
        row_idx_all: List[np.ndarray] = []
        col_idx_all: List[np.ndarray] = []

        for dim, pos in strata.items():
            if len(pos) == 0:
                continue

            sub = base.iloc[pos].copy()
            sub_labels = sub.index.to_numpy()
            sub = sub.reset_index(drop=True)
            sub = _prepare_sub(sub, dim)

            fn = _POLLUTERS[dim]
            sub_seed = int(seed) + _SEED_OFFSET[dim]

            sub_polluted, r_loc, c_loc = fn(sub, float(density), seed=sub_seed)

            # Remise en place des labels de lignes d'origine
            sub_polluted = sub_polluted.reset_index(drop=True)
            sub_polluted.index = sub_labels
            parts.append(sub_polluted)

            if len(r_loc) > 0:
                row_idx_all.append(sub_labels[np.asarray(r_loc, dtype=int)])
                col_idx_all.append(np.asarray(c_loc, dtype=int))

            if verbose:
                print(
                    f"  [{profile_name}] {dim:13s} | "
                    f"lignes={len(pos):5d} ({len(pos)/n:5.1%}) | "
                    f"densite locale={density:.2f}"
                )

        out = pd.concat(parts, axis=0).sort_index()
        out = out.reindex(columns=df.columns)
        out.index = original_index

        row_idx = (
            np.concatenate(row_idx_all).astype(int)
            if row_idx_all else np.array([], dtype=int)
        )
        col_idx = (
            np.concatenate(col_idx_all).astype(int)
            if col_idx_all else np.array([], dtype=int)
        )

        return out, row_idx, col_idx

    # Coherence et uniqueness n'atteignent pas exactement la cible :
    # le taux global est approche, l'assert strict du runner est desactive.
    pollution_mixed.APPROXIMATE_RATE = True
    pollution_mixed.INTRODUCES_INVALID = True
    pollution_mixed.PROFILE_NAME = profile_name
    pollution_mixed.WEIGHTS = weights

    return pollution_mixed


# Pollueur principal du scenario S4 (corps de l'article)
pollution_mixed_uniform = make_mixed_polluter(W_UNIFORM, "P0_uniform")
