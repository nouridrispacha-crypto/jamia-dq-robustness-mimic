# ============================================================
"""
COMMON RUNNER — DATA QUALITY EXPERIMENTS (FULL) — REGRESSION — OPTION A (.632)
Adapted for ICU LOS / log(ICU LOS)

Main changes vs classification version:
- Regression target
- Automatic creation of log_icu_los = log1p(icu_los_days)
- Non-stratified bootstrap
- Regression metrics (RMSE, MAE, R2)
- Regression models
- Regression diagnostics
"""
# ============================================================

# ================= CPU / RAM =================
import os
import csv
import time
import logging
import gc
import psutil

process = psutil.Process()
MAX_RAM_BYTES = 25 * (1024 ** 3)
try:
    process.rlimit(psutil.RLIMIT_AS, (MAX_RAM_BYTES, MAX_RAM_BYTES))
except Exception:
    pass

THREADS = "20"
for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = THREADS

# ================= IMPORTS =================
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ================= GLOBAL SETTINGS =================
ID_COLS = ["subject_id", "hadm_id", "stay_id"]

# Base LOS column expected in the dataset
BASE_TARGET_COL = "icu_los_days"

# Target actually used by models
TARGET_COL = "log_icu_los"

SEX_COL = "gender"

OOD_RANGES = {
    "age": [(-50, -1), (200, 500)],
    "hr_mean": [(-40, 0), (300, 800)],
    "sbp_mean": [(-50, 0), (300, 600)],
    "dbp_min": [(-30, 0), (200, 400)],
    "rr_mean": [(-20, 0), (100, 300)],
    "spo2_min": [(-20, 0), (120, 200)],
    "wbc_min": [(-10, 0), (200, 1000)],
    "aniongap_min": [(-50, 0), (200, 500)],
    "aniongap_max": [(-50, 0), (200, 500)],
    "bun_min": [(-50, 0), (500, 2000)],
    "inr_min": [(-10, 0), (50, 200)],
    "inr_max": [(-10, 0), (50, 200)],
    "ptt_min": [(-50, 0), (500, 2000)],
    "urine_output": [(-5000, -1), (1e6, 1e7)],
}

SENTINELS = {
    "age": -99.0,
    "hr_mean": -9.0,
    "sbp_mean": -1.0,
    "dbp_min": -1.0,
    "rr_mean": -1.0,
    "spo2_min": -5.0,
    "wbc_min": -1.0,
    "aniongap_min": -1.0,
    "aniongap_max": -1.0,
    "bun_min": -1.0,
    "inr_min": -1.0,
    "inr_max": -1.0,
    "ptt_min": -1.0,
    "urine_output": -9999.0,
    "dobutamine": -1.0,
    "dopamine": -1.0,
    "norepinephrine": -1.0,
    "phenylephrine": -1.0,
}

MISSING_TOKEN = "missing"

SAMPLE_SIZES = [3500]
POLLUTION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4 , 0.5]
N_BOOTSTRAPS = 500  # MODIFIÉ (temporaire) — 50 pour valider le fix StandardScaler
                    # avant de relancer les 500 replicats definitifs.

IMPUTATION_MODE = "HF_RANDOM"
MODELS_TO_RUN = None

CAT_COLS = ["gender"]
NUM_COLS = [
    "age", "hr_mean", "sbp_mean", "dbp_min", "rr_mean", "spo2_min",
    "wbc_min", "aniongap_min", "aniongap_max", "bun_min",
    "inr_min", "inr_max", "ptt_min", "urine_output",
    "dobutamine", "dopamine", "norepinephrine", "phenylephrine",
]

IMPUTE_COLS = NUM_COLS + ["gender"]

RF_NAME = "RandomForest"

# ================= LOGGER =================
def setup_logger(name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(out_dir / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_done_bootstraps(csv_file: Path):
    """
    Retourne un set de tuples (N, p, bootstrap) déjà calculés.
    Compatible avec les CSV sans header :
    [scenario, N, p, bootstrap, model, ...]
    """
    if not csv_file.exists():
        return set()

    done = set()
    try:
        with open(csv_file, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    N = int(row[1])
                    p = float(row[2])
                    b = int(row[3])
                    done.add((N, p, b))
                except Exception:
                    continue
    except Exception:
        return set()

    return done


# ================= DATA =================
def load_dataset(path):
    # MODIFIÉ (dépôt public) — utilise le chemin transmis par l'appelant
    # (relatif au script, voir run/*.py) au lieu d'un chemin absolu codé
    # en dur propre à la machine de développement.
    path = Path(path)
    pq = path.with_suffix(path.suffix + ".parquet")

    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
        df.to_parquet(pq, index=False)

    # Create log target automatically if needed
    if TARGET_COL not in df.columns:
        if BASE_TARGET_COL not in df.columns:
            raise ValueError(
                f"Dataset must contain either '{TARGET_COL}' or '{BASE_TARGET_COL}'."
            )
        df[TARGET_COL] = np.log1p(pd.to_numeric(df[BASE_TARGET_COL], errors="coerce"))

    return df


# ================= IMPUTATION =================
def build_sex_conditioned_pools(df, sex_col, cols):
    pools = {}
    if sex_col not in df.columns:
        return pools

    for sex, g in df.groupby(sex_col):
        pools[sex] = {
            c: g[c].dropna().to_numpy()
            for c in cols
            if c in g and g[c].notna().any()
        }
    return pools


def impute_hf_random_draw_sex(
    df_apply,
    sex_col,
    cols,
    pools,
    seed,
):
    rng = np.random.default_rng(seed)
    df = df_apply.copy()

    if sex_col not in df.columns or not pools:
        return df

    for sex, pool in pools.items():
        idx_sex = df[sex_col] == sex
        if not idx_sex.any():
            continue

        for c in cols:
            if c not in df or c not in pool or len(pool[c]) == 0:
                continue

            idx = idx_sex & df[c].isna()
            if idx.any():
                df.loc[idx, c] = rng.choice(pool[c], size=idx.sum())

    return df


def encode_nan_as_sentinel(df, cols):
    df = df.copy()
    for c in cols:
        if c in df and c in SENTINELS:
            df[c] = df[c].fillna(SENTINELS[c])
    return df


# ================= PREPROCESS =================
def enforce_column_types(df):
    df = df.copy()

    for c in NUM_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if BASE_TARGET_COL in df:
        df[BASE_TARGET_COL] = pd.to_numeric(df[BASE_TARGET_COL], errors="coerce")

    if TARGET_COL in df:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    for c in CAT_COLS:
        if c in df:
            df[c] = df[c].astype(object)

    return df


def prepare_data_for_models(df_train, df_eval, preprocessor=None):
    ytr = df_train[TARGET_COL].astype(float).to_numpy()
    yev = df_eval[TARGET_COL].astype(float).to_numpy()

    # Keep base LOS column out of X if present
    drop_cols = ID_COLS + [TARGET_COL]
    if BASE_TARGET_COL in df_train.columns:
        drop_cols.append(BASE_TARGET_COL)

    Xtr_df = df_train.drop(columns=drop_cols, errors="ignore").copy()
    Xev_df = df_eval.drop(columns=drop_cols, errors="ignore").copy()

    for c in CAT_COLS:
        if c in Xtr_df:
            Xtr_df[c] = Xtr_df[c].fillna(MISSING_TOKEN).astype(str)
        if c in Xev_df:
            Xev_df[c] = Xev_df[c].fillna(MISSING_TOKEN).astype(str)

    num_cols = [c for c in NUM_COLS if c in Xtr_df]
    cat_cols = [c for c in CAT_COLS if c in Xtr_df]

    if preprocessor is None:
        # MODIFIÉ — "passthrough" remplacé par StandardScaler. Les variables
        # numériques (ex. urine_output ~10^3 vs age ~10^1-2) étaient envoyées
        # brutes à LASSO et au MLP : la pénalité L1 de LASSO pénalise les
        # coefficients de façon asymétrique selon l'échelle native de chaque
        # variable, et l'initialisation par défaut de PyTorch suppose des
        # entrées d'échelle ~1. Sans standardisation, le MLP n'atteint pas un
        # R² positif même à c=0 (R²=-7.225), et le classement de variables de
        # LASSO est biaisé par l'échelle plutôt que par l'effet réel. DT, RF
        # et XGBoost sont inchangés : les splits d'arbre sont invariants à une
        # transformation monotone comme la standardisation.
        preprocessor = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop",
        )
        Xtr = preprocessor.fit_transform(Xtr_df)
    else:
        Xtr = preprocessor.transform(Xtr_df)

    Xev = preprocessor.transform(Xev_df)

    Xtr = np.nan_to_num(Xtr, nan=0.0).astype(np.float32, copy=False)
    Xev = np.nan_to_num(Xev, nan=0.0).astype(np.float32, copy=False)

    return preprocessor, Xtr, Xev, ytr, yev


# ================= BOOTSTRAP =================
def bootstrap_inbag_oob(df, seed):
    rng = np.random.RandomState(seed)
    n = len(df)

    inbag_idx = rng.choice(df.index.to_numpy(), size=n, replace=True)
    inbag = df.loc[inbag_idx].reset_index(drop=True)

    unique_inbag = np.unique(inbag_idx)
    oob_idx = np.setdiff1d(df.index.to_numpy(), unique_inbag)
    oob = df.loc[oob_idx].reset_index(drop=True)

    return inbag, oob


# ================= METRICS =================
def compute_metrics(y, yp):
    out = {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    if len(y) == 0:
        return out

    out["rmse"] = float(np.sqrt(mean_squared_error(y, yp)))
    out["mae"] = float(mean_absolute_error(y, yp))

    try:
        out["r2"] = float(r2_score(y, yp))
    except Exception:
        out["r2"] = np.nan

    return out


def combine_point632(a, b):
    return {k: 0.368 * a[k] + 0.632 * b[k] for k in a}


# ================= REGRESSION DIAGNOSTICS =================
def regression_diagnostics(y_true, y_pred):
    if len(y_pred) == 0:
        return {
            "yhat_mean": np.nan,
            "yhat_var": np.nan,
            "y_mean": np.nan,
            "y_var": np.nan,
            "bias": np.nan,
        }

    return {
        "yhat_mean": float(np.mean(y_pred)),
        "yhat_var": float(np.var(y_pred)),
        "y_mean": float(np.mean(y_true)),
        "y_var": float(np.var(y_true)),
        "bias": float(np.mean(y_pred - y_true)),
    }


# ================= MLP =================
class MLPRegressorTorch(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(Xtr, ytr, hidden, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPRegressorTorch(Xtr.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.float32),
        ),
        batch_size=128,
        shuffle=True,
    )

    model.train()
    for _ in range(10):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model


def predict_mlp(model, X):
    device = next(model.parameters()).device
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).to(device))
    return preds.cpu().numpy()


# ================= RUN MODELS =================
def run_all_models_point632_batched(df_inbag, df_oob, preprocessor):
    preprocessor, Xtr, Xoob, ytr, yoob = prepare_data_for_models(
        df_inbag, df_oob, preprocessor
    )

    res = {}
    diag = {}

    # ================= Linear Regression =================
    lr = LinearRegression()
    lr.fit(Xtr, ytr)

    p_tr = lr.predict(Xtr)
    p_oob = lr.predict(Xoob)

    res["LinearRegression"] = combine_point632(
        compute_metrics(ytr, p_tr),
        compute_metrics(yoob, p_oob),
    )
    diag["LinearRegression"] = regression_diagnostics(yoob, p_oob)

    # ================= LASSO =================
    lasso = Lasso(alpha=0.001, max_iter=5000)
    lasso.fit(Xtr, ytr)

    p_tr = lasso.predict(Xtr)
    p_oob = lasso.predict(Xoob)

    res["LASSO"] = combine_point632(
        compute_metrics(ytr, p_tr),
        compute_metrics(yoob, p_oob),
    )
    diag["LASSO"] = regression_diagnostics(yoob, p_oob)

    # ================= PLS =================
    n_comp = max(1, min(10, Xtr.shape[1] - 1)) if Xtr.shape[1] > 1 else 1
    pls = PLSRegression(n_components=n_comp)
    pls.fit(Xtr, ytr)

    p_tr = pls.predict(Xtr).ravel()
    p_oob = pls.predict(Xoob).ravel()

    res["PLS"] = combine_point632(
        compute_metrics(ytr, p_tr),
        compute_metrics(yoob, p_oob),
    )
    diag["PLS"] = regression_diagnostics(yoob, p_oob)

    # ================= KNN =================
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(Xtr, ytr)

    p_tr = knn.predict(Xtr)
    p_oob = knn.predict(Xoob)

    res["KNN"] = combine_point632(
        compute_metrics(ytr, p_tr),
        compute_metrics(yoob, p_oob),
    )
    diag["KNN"] = regression_diagnostics(yoob, p_oob)

    # ================= SVR =================
    svr = SVR(kernel="rbf", cache_size=500)
    svr.fit(Xtr, ytr)

    p_tr = svr.predict(Xtr)
    p_oob = svr.predict(Xoob)

    res["SVR"] = combine_point632(
        compute_metrics(ytr, p_tr),
        compute_metrics(yoob, p_oob),
    )
    diag["SVR"] = regression_diagnostics(yoob, p_oob)

    # ================= Decision Tree =================
    dt = DecisionTreeRegressor(
        max_depth=None,
        min_samples_leaf=10,
        random_state=0,
    )
    dt.fit(Xtr, ytr)

    p_tr = dt.predict(Xtr)
    p_oob = dt.predict(Xoob)

    res["DecisionTree"] = combine_point632(
        compute_metrics(ytr, p_tr),
        compute_metrics(yoob, p_oob),
    )
    diag["DecisionTree"] = regression_diagnostics(yoob, p_oob)

    # ================= Random Forest =================
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=0,
    )
    rf.fit(Xtr, ytr)

    p_tr = rf.predict(Xtr)
    p_oob = rf.predict(Xoob)

    res[RF_NAME] = combine_point632(
        compute_metrics(ytr, p_tr),
        compute_metrics(yoob, p_oob),
    )
    diag[RF_NAME] = regression_diagnostics(yoob, p_oob)

    # ================= XGBoost =================
    dtr = xgb.DMatrix(Xtr, label=ytr)
    doob = xgb.DMatrix(Xoob, label=yoob)

    xgb_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
    }
    if torch.cuda.is_available():
        xgb_params["device"] = "cuda"

    booster = xgb.train(
        xgb_params,
        dtr,
        num_boost_round=100,
    )

    p_tr = booster.predict(dtr)
    p_oob = booster.predict(doob)

    res["XGBoost"] = combine_point632(
        compute_metrics(ytr, p_tr),
        compute_metrics(yoob, p_oob),
    )
    diag["XGBoost"] = regression_diagnostics(yoob, p_oob)

    # ================= MLPs =================
    for name, hidden in {
        "MLP1": (64,),
        "MLP5": (64,) * 5,
        "MLP10": (64,) * 10,
    }.items():
        per_seed_metrics = []
        per_seed_preds = []

        for seed in [0, 1, 2]:
            model = train_mlp(Xtr, ytr, hidden, seed)

            p_tr = predict_mlp(model, Xtr)
            p_oob = predict_mlp(model, Xoob)

            per_seed_metrics.append(
                combine_point632(
                    compute_metrics(ytr, p_tr),
                    compute_metrics(yoob, p_oob),
                )
            )
            per_seed_preds.append(p_oob)

        res[name] = {
            k: float(np.mean([m[k] for m in per_seed_metrics]))
            for k in per_seed_metrics[0]
        }

        diag[name] = regression_diagnostics(
            yoob,
            np.mean(per_seed_preds, axis=0),
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return res, diag


def measure_pollution_rate(df_before, df_after, cols):
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


def count_missing_cells(df, cols):
    n_missing = 0
    n_total = 0
    for c in cols:
        if c not in df:
            continue
        n_missing += df[c].isna().sum()
        n_total += len(df)
    return n_missing, n_total, (n_missing / n_total if n_total > 0 else 0.0)


def ensure_schema(df):
    df = df.copy()

    for c in NUM_COLS:
        if c not in df:
            df[c] = np.nan

    for c in CAT_COLS:
        if c not in df:
            df[c] = np.nan

    if BASE_TARGET_COL not in df:
        df[BASE_TARGET_COL] = np.nan

    if TARGET_COL not in df and BASE_TARGET_COL in df:
        df[TARGET_COL] = np.log1p(pd.to_numeric(df[BASE_TARGET_COL], errors="coerce"))

    return df


# ================= RUN SCENARIO =================
def run_scenario(
    scenario_name,
    csv_path,
    out_dir,
    pollution_fn,
    pollute_train,
    pollute_test,
):
    out_dir = Path(out_dir)
    logger = setup_logger(scenario_name, out_dir)

    df = load_dataset(csv_path)
    csv_file = out_dir / f"{scenario_name}.csv"

    # Keep only rows with target present
    df = ensure_schema(df)
    df = enforce_column_types(df)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)

    # ================= INIT CSV =================
    if not csv_file.exists():
        with open(csv_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "scenario", "N", "p", "bootstrap", "model",
                "rmse", "mae", "r2",
                "yhat_mean", "yhat_var", "y_mean", "y_var", "bias",
            ])

    # ================= REPRISE =================
    done_bootstraps = load_done_bootstraps(csv_file)
    if done_bootstraps:
        logger.info(f"RESUME | {len(done_bootstraps)} bootstraps deja calcules")
    else:
        logger.info("AUCUNE REPRISE | scenario neuf")

    # ================= BOUCLE EXPERIMENTALE =================
    for N in SAMPLE_SIZES:
        logger.info(f"N = {N}")

        if N > len(df):
            logger.warning(f"SKIP N={N} | dataset trop petit ({len(df)} lignes)")
            continue

        df_sample = df.sample(n=N, random_state=42).reset_index(drop=True)

        # ======================================================
        # PREFIT PREPROCESSOR — UNE FOIS PAR N
        # ======================================================
        df_prefit = df_sample.copy()
        df_prefit = ensure_schema(df_prefit)
        df_prefit = enforce_column_types(df_prefit)

        SEX_POOLS = build_sex_conditioned_pools(
            df_prefit,
            SEX_COL,
            IMPUTE_COLS,
        )

        PREPROCESSOR, *_ = prepare_data_for_models(df_prefit, df_prefit)

        # ======================================================
        # POLLUTION / BOOTSTRAP
        # ======================================================
        for p in POLLUTION_LEVELS:
            logger.info(f"p = {p:.2f}")

            for b in range(N_BOOTSTRAPS):

                # -------- reprise --------
                if (N, p, b) in done_bootstraps:
                    if b % 10 == 0:
                        logger.info(f"SKIP | N={N} | p={p:.2f} | b={b:03d}")
                    continue

                t0 = time.time()

                # ---------------- Bootstrap .632 ----------------
                df_inbag, df_oob = bootstrap_inbag_oob(
                    df_sample,
                    seed=10_000 + b,
                )

                if len(df_oob) == 0:
                    logger.warning(f"SKIP | N={N} | p={p:.2f} | b={b:03d} | OOB vide")
                    continue

                # ---------------- Imputation HF (AVANT pollution) ----------------
                if IMPUTATION_MODE == "HF_RANDOM":
                    df_inbag = impute_hf_random_draw_sex(
                        df_inbag,
                        SEX_COL,
                        IMPUTE_COLS,
                        SEX_POOLS,
                        seed=20_000 + b,
                    )
                    df_oob = impute_hf_random_draw_sex(
                        df_oob,
                        SEX_COL,
                        IMPUTE_COLS,
                        SEX_POOLS,
                        seed=30_000 + b,
                    )

                # ================= POLLUTION TRAIN =================
                if pollute_train and len(df_inbag) > 0:
                    y_log = df_inbag[TARGET_COL].copy()
                    y_base = df_inbag[BASE_TARGET_COL].copy() if BASE_TARGET_COL in df_inbag else None
                    df_before = df_inbag.copy(deep=True)

                    out_tr = pollution_fn(df_inbag, p, seed=40_000 + b)
                    df_inbag = out_tr[0] if isinstance(out_tr, tuple) else out_tr

                    # restore targets
                    df_inbag[TARGET_COL] = y_log
                    if y_base is not None:
                        df_inbag[BASE_TARGET_COL] = y_base

                    pollution_rate_tr = measure_pollution_rate(
                        df_before, df_inbag, NUM_COLS
                    )

                    logger.info(
                        f"POLLUTION TRAIN | "
                        f"p_target={p:.3f} | p_real={pollution_rate_tr:.3f}"
                    )

                    

                # ================= POLLUTION TEST =================
                if pollute_test and len(df_oob) > 0:
                    y_log = df_oob[TARGET_COL].copy()
                    y_base = df_oob[BASE_TARGET_COL].copy() if BASE_TARGET_COL in df_oob else None
                    df_before = df_oob.copy(deep=True)

                    out_te = pollution_fn(df_oob, p, seed=50_000 + b)
                    df_oob = out_te[0] if isinstance(out_te, tuple) else out_te

                    # restore targets
                    df_oob[TARGET_COL] = y_log
                    if y_base is not None:
                        df_oob[BASE_TARGET_COL] = y_base

                    pollution_rate_te = measure_pollution_rate(
                        df_before, df_oob, NUM_COLS
                    )

                    logger.info(
                        f"POLLUTION TEST  | "
                        f"p_target={p:.3f} | p_real={pollution_rate_te:.3f}"
                    )

                    

                # ================= NORMALISATION =================
                df_inbag = ensure_schema(df_inbag)
                df_oob = ensure_schema(df_oob)

                df_inbag = enforce_column_types(df_inbag)
                df_oob = enforce_column_types(df_oob)

                # ================= NaN → OOD aléatoire =================
                df_inbag = encode_nan_as_sentinel(df_inbag, NUM_COLS)
                df_oob = encode_nan_as_sentinel(df_oob, NUM_COLS)

                # Ensure target is still present after pollution / coercion
                df_inbag = df_inbag[df_inbag[TARGET_COL].notna()].reset_index(drop=True)
                df_oob = df_oob[df_oob[TARGET_COL].notna()].reset_index(drop=True)

                if len(df_inbag) == 0 or len(df_oob) == 0:
                    logger.warning(f"SKIP | N={N} | p={p:.2f} | b={b:03d} | split vide apres nettoyage")
                    continue

                # ================= MODÈLES + SCORE .632 =================
                results, diagnostics = run_all_models_point632_batched(
                    df_inbag,
                    df_oob,
                    PREPROCESSOR,
                )

                # ================= CSV =================
                with open(csv_file, "a", newline="") as f:
                    w = csv.writer(f)
                    for model, m in results.items():
                        d = diagnostics.get(model, {})
                        w.writerow([
                            scenario_name, N, p, b, model,
                            m["rmse"],
                            m["mae"],
                            m["r2"],
                            d.get("yhat_mean", np.nan),
                            d.get("yhat_var", np.nan),
                            d.get("y_mean", np.nan),
                            d.get("y_var", np.nan),
                            d.get("bias", np.nan),
                        ])

                dt = time.time() - t0
                logger.info(
                    f"OK | N={N} | p={p:.2f} | b={b:03d}/{N_BOOTSTRAPS} | time={dt:5.1f}s"
                )

    logger.info("END SCENARIO")