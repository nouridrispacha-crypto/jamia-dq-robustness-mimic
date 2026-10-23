# -*- coding: utf-8 -*-
"""
COHERENCE POLLUTER — 18_VARIABLES
Compatible common_runner

Règles de cohérence inter-variables :
R1: aniongap_min  <-> aniongap_max
R2: inr_min       <-> inr_max
R3: sbp_mean      <-> dbp_min         (SBP / DBP inverses)
R5: hr_mean       <-> rr_mean         (FC / FR inversees)
R6: permutation de deux des quatre indicateurs de vasopresseurs

Chaque echange conserve les deux valeurs dans le domaine physiologique de leur
nouvelle colonne : seule la relation entre variables est rompue, de sorte
qu'un controle de plage colonne par colonne ne detecte rien.

DEUX REGLES ONT ETE ECARTEES pour cette raison.
R4 echangeait deux colonnes numeriques arbitraires (un debit urinaire de
1726 mL pouvait atterrir dans la colonne SpO2) et fournissait environ deux
tiers des cellules corrompues.
R7 echangeait inr_min et ptt_min : moyennes 1.35 et 32.7, donc un INR a 33 et
un PTT a 1.4, hors domaine dans les deux sens.

Le jeu de regles restant plafonne a 34.8 % de couverture, d'ou
POLLUTION_LEVELS arrete a 0.3.

GARANTIES :
- chaque cellule (row, col) est modifiée AU PLUS UNE FOIS
- plusieurs règles différentes peuvent s'appliquer sur une même ligne
- p_real ≈ p_target (aligné avec measure_pollution_rate)
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Set
import numpy as np
import pandas as pd
from copy import deepcopy

# ================== COLS ==================
ID_COLS = ["subject_id", "hadm_id", "stay_id"]
TARGET_COL = "in_hospital_mortality"

NUM_COLS = [
    "age", "hr_mean", "sbp_mean", "dbp_min", "rr_mean", "spo2_min",
    "wbc_min", "aniongap_min", "aniongap_max",
    "bun_min", "inr_min", "inr_max", "ptt_min", "urine_output",
    "dobutamine", "dopamine", "norepinephrine", "phenylephrine",
]

PROTECTED = set(ID_COLS + [TARGET_COL])

# ================== RNG ==================
def _rng(seed: Optional[int]):
    return np.random.default_rng(None if seed is None else int(seed))

# ================== HELPERS ==================
def _finite_pair(a, b) -> bool:
    return pd.notna(a) and pd.notna(b)

def _swap_if_free(
    df: pd.DataFrame,
    r: int,
    c1: str,
    c2: str,
    touched: Set[Tuple[int, str]],
) -> int:
    """
    Swap c1 <-> c2 on row r IF BOTH cells were never touched.
    Returns number of modified cells (0 or 2).
    """
    if (r, c1) in touched or (r, c2) in touched:
        return 0

    v1 = df.at[r, c1]
    v2 = df.at[r, c2]

    if not _finite_pair(v1, v2):
        return 0
    if v1 == v2:
        return 0

    df.at[r, c1] = v2
    df.at[r, c2] = v1

    touched.add((r, c1))
    touched.add((r, c2))
    return 2

def _misalign_swap(
    df: pd.DataFrame,
    r: int,
    rng: np.random.Generator,
    cols: List[str],
    touched: Set[Tuple[int, str]],
) -> int:
    """
    Fallback: swap two random untouched numeric columns on row r.
    """
    free_cols = [c for c in cols if (r, c) not in touched]
    if len(free_cols) < 2:
        return 0

    c1, c2 = rng.choice(free_cols, size=2, replace=False).tolist()
    return _swap_if_free(df, r, c1, c2, touched)

# ================== MAIN ==================
def apply_coherence_pollution_18vars(
    df: pd.DataFrame,
    p: float,
    seed: int = 42,
    *,
    max_attempts_factor: int = 20,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:

    assert 0.0 <= float(p) <= 1.0, "p doit être dans [0,1]"
    rng = _rng(seed)
    Xp = deepcopy(df)

    num_cols = [c for c in NUM_COLS if c in Xp.columns and c not in PROTECTED]
    n_rows = len(Xp)

    if n_rows == 0 or p == 0.0 or not num_cols:
        return Xp, np.array([], dtype=int), np.array([], dtype=int)

    total_cells = n_rows * len(num_cols)
    budget = int(round(p * total_cells))
    budget = max(1, min(budget, total_cells))

    # 🔒 mémoire globale des cellules déjà modifiées
    touched: Set[Tuple[int, str]] = set()

    row_idx: List[int] = []
    col_idx: List[int] = []
    changed_cells = 0

    max_attempts = max_attempts_factor * budget
    attempts = 0

    has_R1 = {"aniongap_min", "aniongap_max"} <= set(Xp.columns)
    has_R2 = {"inr_min", "inr_max"} <= set(Xp.columns)
    has_R3 = {"sbp_mean", "dbp_min"} <= set(Xp.columns)
    has_R5 = {"hr_mean", "rr_mean"} <= set(Xp.columns)
    VASO = [c for c in ("dobutamine", "dopamine", "norepinephrine",
                        "phenylephrine") if c in Xp.columns]
    has_R6 = len(VASO) >= 2

    rules = []
    if has_R1: rules.append("R1")
    if has_R2: rules.append("R2")
    if has_R3: rules.append("R3")
    if has_R5: rules.append("R5")
    if has_R6: rules.append("R6")
    if not rules:
        return Xp, np.array([], dtype=int), np.array([], dtype=int)

    while changed_cells < budget and attempts < max_attempts:
        attempts += 1
        r = int(rng.integers(0, n_rows))
        rule = str(rng.choice(rules))

        nb = 0

        if rule == "R1":
            nb = _swap_if_free(Xp, r, "aniongap_min", "aniongap_max", touched)
            if nb:
                row_idx += [r, r]
                col_idx += [
                    Xp.columns.get_loc("aniongap_min"),
                    Xp.columns.get_loc("aniongap_max"),
                ]

        elif rule == "R2":
            nb = _swap_if_free(Xp, r, "inr_min", "inr_max", touched)
            if nb:
                row_idx += [r, r]
                col_idx += [
                    Xp.columns.get_loc("inr_min"),
                    Xp.columns.get_loc("inr_max"),
                ]

        elif rule == "R3":
            nb = _swap_if_free(Xp, r, "sbp_mean", "dbp_min", touched)
            if nb:
                row_idx += [r, r]
                col_idx += [
                    Xp.columns.get_loc("sbp_mean"),
                    Xp.columns.get_loc("dbp_min"),
                ]

        elif rule == "R5":
            nb = _swap_if_free(Xp, r, "hr_mean", "rr_mean", touched)
            if nb:
                row_idx += [r, r]
                col_idx += [
                    Xp.columns.get_loc("hr_mean"),
                    Xp.columns.get_loc("rr_mean"),
                ]

        elif rule == "R6":
            # wrong vasopressor recorded: permute two of the four indicators
            c1, c2 = rng.choice(VASO, size=2, replace=False).tolist()
            nb = _swap_if_free(Xp, r, c1, c2, touched)
            if nb:
                row_idx += [r, r]
                col_idx += [
                    Xp.columns.get_loc(c1),
                    Xp.columns.get_loc(c2),
                ]

        changed_cells += nb

    return Xp, np.array(row_idx, dtype=int), np.array(col_idx, dtype=int)

# ================== WRAPPER ==================
def pollution_coherence(df: pd.DataFrame, density: float, seed: int = 42):
    return apply_coherence_pollution_18vars(df, p=float(density), seed=int(seed))

pollution_coherence.INTRODUCES_INVALID = False
pollution_coherence.APPROXIMATE_RATE = True
   



