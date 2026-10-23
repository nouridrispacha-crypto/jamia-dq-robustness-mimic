import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import PAPER
# -*- coding: utf-8 -*-
"""Table S8 recomputed with the extraction artefacts excluded from the scale.

The scale is the clean-column SD after winsorising at the 0.1st/99.9th
percentile. That removes the three impossible rr_mean values (max 269 267 for
a respiratory rate) without touching the ordinary right skew of the other
columns, and unlike a MAD-based scale it stays defined for the four binary
drug indicators, whose median absolute deviation is zero.

The corrupted values themselves are NOT winsorised: only the yardstick is.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = PAPER / "binary_outcome"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Polluter"))

from Polluter.pollute_mnar_top import pollution_mnar_top
from Polluter.coherence_polluter import pollution_coherence
from Polluter.pollute_validity import pollution_validity
from Polluter.pollute_precision import pollution_precision
from Polluter.pollute_uniqueness import pollution_uniqueness

MECH = {
    "Completeness": pollution_mnar_top,
    "Coherence": pollution_coherence,
    "Validity": pollution_validity,
    "Precision": pollution_precision,
    "Uniqueness": pollution_uniqueness,
}

df0 = pd.read_csv(ROOT / "data" / "18_variables.csv")
PROTECTED = {"subject_id", "hadm_id", "stay_id", "in_hospital_mortality"}
num_cols = [c for c in df0.select_dtypes(include="number").columns
            if c not in PROTECTED]

lo = df0[num_cols].quantile(0.001)
hi = df0[num_cols].quantile(0.999)
sd_w = df0[num_cols].clip(lower=lo, upper=hi, axis=1).std()
sd_raw = df0[num_cols].std()

print("Scale used, raw SD vs winsorised SD:")
print(f"  {'column':16s}{'raw SD':>12}{'winsorised':>12}{'factor':>9}")
for c in num_cols:
    print(f"  {c:16s}{sd_raw[c]:>12.2f}{sd_w[c]:>12.2f}{sd_raw[c]/sd_w[c]:>9.1f}")

rows = []
for name, fn in MECH.items():
    out = fn(df0.copy(), 0.5, seed=1)
    dfc = out[0] if isinstance(out, tuple) else out
    a = df0[num_cols].apply(pd.to_numeric, errors="coerce")
    b = dfc[num_cols].apply(pd.to_numeric, errors="coerce")
    changed = (a != b) & ~(a.isna() & b.isna())
    both = changed & a.notna() & b.notna()
    disp = (b - a).abs().div(sd_w, axis=1)[both].stack()
    n = int(changed.values.sum())
    n_void = n - len(disp)
    rows.append({
        "dimension": name,
        "cells_pct": 100 * n / (len(df0) * len(num_cols)),
        "void_pct": 100 * n_void / n if n else np.nan,
        "median": np.percentile(disp, 50) if len(disp) else np.nan,
        "q1": np.percentile(disp, 25) if len(disp) else np.nan,
        "q3": np.percentile(disp, 75) if len(disp) else np.nan,
        "gt5": 100 * float((disp > 5).mean()) if len(disp) else np.nan,
    })

t = pd.DataFrame(rows)
print("\n=== Table S8 recomputed on the winsorised scale ===")
print(f"  {'dimension':14s}{'cells%':>9}{'void%':>9}{'median':>9}"
      f"{'IQR':>20}{'>5 SD':>9}")
for r in rows:
    med = "--" if np.isnan(r["median"]) else f"{r['median']:.2f}"
    iqr = "--" if np.isnan(r["q1"]) else f"[{r['q1']:.2f}, {r['q3']:.2f}]"
    g5 = "--" if np.isnan(r["gt5"]) else f"{r['gt5']:.1f}%"
    print(f"  {r['dimension']:14s}{r['cells_pct']:>8.1f}%{r['void_pct']:>8.1f}%"
          f"{med:>9}{iqr:>20}{g5:>9}")

c = t.set_index("dimension")["median"]
print(f"\nprecision -> coherence ratio : {c['Coherence']/c['Precision']:.1f}x "
      f"({np.log10(c['Coherence']/c['Precision']):.2f} orders of magnitude)")
print(f"precision -> validity  ratio : {c['Validity']/c['Precision']:.1f}x "
      f"({np.log10(c['Validity']/c['Precision']):.2f} orders of magnitude)")
