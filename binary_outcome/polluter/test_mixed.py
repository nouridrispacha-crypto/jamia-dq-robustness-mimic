# -*- coding: utf-8 -*-
"""
Test du pollueur mixte S4 sur une cohorte synthetique de type MIMIC-IV.

A placer dans Polluter/, a cote de pollute_mixed.py.

USAGE
-----
    cd chemin/vers/MIE_2026
    python Polluter/test_mixed.py

Aucune donnee reelle n'est necessaire : la cohorte est simulee.
Ce script ne verifie que la mecanique du pollueur (budget, disjonction,
reproductibilite), pas les performances des modeles.
"""

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pollute_mixed import (
    make_mixed_polluter,
    PROFILES,
    DIMENSIONS,
)

NUM_COLS = [
    "age", "hr_mean", "sbp_mean", "dbp_min", "rr_mean", "spo2_min",
    "wbc_min", "aniongap_min", "aniongap_max", "bun_min",
    "inr_min", "inr_max", "ptt_min", "urine_output",
    "dobutamine", "dopamine", "norepinephrine", "phenylephrine",
]
TARGET_COL = "in_hospital_mortality"
ID_COLS = ["subject_id", "hadm_id", "stay_id"]


def make_cohort(n=3500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "subject_id": np.arange(n),
        "hadm_id": np.arange(n) + 10 ** 6,
        "stay_id": np.arange(n) + 2 * 10 ** 6,
        "gender": rng.choice(["M", "F"], n),
        "age": rng.normal(66, 15, n).round(0),
        "hr_mean": rng.normal(83, 15, n).round(1),
        "sbp_mean": rng.normal(116, 18, n).round(1),
        "dbp_min": rng.normal(50, 12, n).round(1),
        "rr_mean": rng.normal(18.4, 3.5, n).round(1),
        "spo2_min": rng.normal(93, 4, n).round(0),
        "wbc_min": rng.lognormal(2.2, 0.4, n).round(1),
        "aniongap_min": rng.normal(13, 3, n).round(0),
        "aniongap_max": rng.normal(14, 3.5, n).round(0),
        "bun_min": rng.lognormal(2.9, 0.6, n).round(0),
        "inr_min": rng.normal(1.2, 0.2, n).round(2),
        "inr_max": rng.normal(1.3, 0.25, n).round(2),
        "ptt_min": rng.normal(29, 6, n).round(1),
        "urine_output": rng.normal(1500, 700, n).round(0),
        "dobutamine": rng.binomial(1, 0.013, n),
        "dopamine": rng.binomial(1, 0.028, n),
        "norepinephrine": rng.binomial(1, 0.174, n),
        "phenylephrine": rng.binomial(1, 0.191, n),
        TARGET_COL: rng.binomial(1, 0.11, n),
    })
    # Apres imputation hot-deck, toutes les colonnes numeriques sont en float
    for c in NUM_COLS:
        df[c] = df[c].astype(float)
    return df


def measure_pollution_rate(df_before, df_after, cols):
    """Copie exacte de la fonction du common_runner."""
    diff = 0
    total = 0
    for c in cols:
        if c not in df_before or c not in df_after:
            continue
        before = df_before[c].to_numpy()
        after = df_after[c].to_numpy()
        mask = ~(pd.isna(before) & pd.isna(after))
        diff += np.sum(before[mask] != after[mask])
        total += mask.sum()
    return diff / total if total > 0 else None


def per_row_changes(df_before, df_after, cols):
    """Nombre de cellules modifiees par ligne."""
    b = df_before[cols].to_numpy(dtype=object)
    a = df_after[cols].to_numpy(dtype=object)
    nan_b = pd.isna(b)
    nan_a = pd.isna(a)
    changed = (b != a) & ~(nan_b & nan_a)
    return changed.sum(axis=1)


def run_checks():
    df = make_cohort(3500, seed=0)
    fn = make_mixed_polluter(PROFILES["P0_uniform"], "P0_uniform")

    print("=" * 74)
    print("TEST 1 — Taux global mesure vs cible (profil uniforme, N=3500)")
    print("=" * 74)
    print(f"{'T cible':>9} | {'T mesure':>9} | {'ecart':>7} | {'lignes':>7} | {'ordre':>6}")
    print("-" * 74)

    for T in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        out, r, c = fn(df, T, seed=40_000)
        rate = measure_pollution_rate(df, out, NUM_COLS)
        rate = 0.0 if rate is None else rate
        same_order = bool((out.index == df.index).all())
        same_len = len(out) == len(df)
        print(
            f"{T:9.2f} | {rate:9.3f} | {rate - T:+7.3f} | "
            f"{'OK' if same_len else 'KO':>7} | {'OK' if same_order else 'KO':>6}"
        )

    print()
    print("=" * 74)
    print("TEST 2 — Parite du budget par dimension (T=0.5)")
    print("=" * 74)

    T = 0.5
    rng = np.random.default_rng(40_000)
    from pollute_mixed import _split_rows
    strata = _split_rows(len(df), {d: 0.2 for d in DIMENSIONS}, rng)

    out, _, _ = fn(df, T, seed=40_000)
    changed = per_row_changes(df, out, NUM_COLS)
    total_cells = len(df) * len(NUM_COLS)

    print(f"{'dimension':>14} | {'lignes':>7} | {'% lignes':>8} | "
          f"{'cellules':>9} | {'% budget':>8} | {'dens. locale':>12}")
    print("-" * 74)
    tot_changed = 0
    for d, pos in strata.items():
        nc = int(changed[pos].sum())
        tot_changed += nc
        local = nc / (len(pos) * len(NUM_COLS)) if len(pos) else 0
        print(
            f"{d:>14} | {len(pos):7d} | {len(pos)/len(df):8.1%} | "
            f"{nc:9d} | {nc/total_cells:8.1%} | {local:12.3f}"
        )
    print("-" * 74)
    print(f"{'TOTAL':>14} | {len(df):7d} | {1.0:8.1%} | "
          f"{tot_changed:9d} | {tot_changed/total_cells:8.1%} |")

    print()
    print("=" * 74)
    print("TEST 3 — Disjonction : aucune ligne touchee par deux mecanismes")
    print("=" * 74)
    all_pos = np.concatenate([p for p in strata.values()])
    print(f"lignes assignees      : {len(all_pos)}")
    print(f"lignes distinctes     : {len(np.unique(all_pos))}")
    print(f"couverture de N       : {len(np.unique(all_pos)) == len(df)}")

    print()
    print("=" * 74)
    print("TEST 4 — Effets de bord attendus apres coercition (T=0.5)")
    print("=" * 74)
    out, _, _ = fn(df, 0.5, seed=40_000)
    obj_cols = [c for c in NUM_COLS if out[c].dtype == object]
    n_str = 0
    for c in obj_cols:
        n_str += int(out[c].apply(lambda v: isinstance(v, str)).sum())
    coerced = out.copy()
    for c in NUM_COLS:
        coerced[c] = pd.to_numeric(coerced[c], errors="coerce")
    n_nan = int(coerced[NUM_COLS].isna().to_numpy().sum())
    n_nan_pre = int(out[NUM_COLS].isna().to_numpy().sum())
    print(f"colonnes passees en object    : {len(obj_cols)}/{len(NUM_COLS)}")
    print(f"cellules chaines (validity)   : {n_str}")
    print(f"NaN avant coercition          : {n_nan_pre}  (completeness)")
    print(f"NaN apres coercition          : {n_nan}  (completeness + validity)")
    print(f"cible sentinelle finale       : {n_nan/(len(df)*len(NUM_COLS)):.1%} des cellules")

    print()
    print("=" * 74)
    print("TEST 5 — Profils stylises (T=0.5) : % de lignes par dimension")
    print("=" * 74)
    print(f"{'profil':>18} | " + " | ".join(f"{d[:9]:>9}" for d in DIMENSIONS))
    print("-" * 74)
    for pname, w in PROFILES.items():
        rng2 = np.random.default_rng(40_000)
        st = _split_rows(len(df), w, rng2)
        cells = []
        for d in DIMENSIONS:
            frac = len(st.get(d, [])) / len(df) * 0.5
            cells.append(f"{frac:9.1%}")
        print(f"{pname:>18} | " + " | ".join(cells))
    print()
    print("Lecture : % de cellules du tableau total attribuees a chaque dimension a T=0.5.")

    print()
    print("=" * 74)
    print("TEST 6 — Reproductibilite et independance train/test")
    print("=" * 74)
    a1, _, _ = fn(df, 0.3, seed=40_000)
    a2, _, _ = fn(df, 0.3, seed=40_000)
    b1, _, _ = fn(df, 0.3, seed=50_000)
    same = a1[NUM_COLS].astype(str).equals(a2[NUM_COLS].astype(str))
    diff = not a1[NUM_COLS].astype(str).equals(b1[NUM_COLS].astype(str))
    print(f"meme seed  -> resultat identique  : {same}")
    print(f"seed train != seed test -> differe : {diff}")


if __name__ == "__main__":
    run_checks()
