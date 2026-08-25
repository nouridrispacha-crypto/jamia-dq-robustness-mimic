# ============================================================
"""
COMMON RUNNER — DATA QUALITY EXPERIMENTS (FULL) — OPTION A (.632)
⚡ PERFORMANCE-OPTIMIZED VERSION (NO METHODOLOGICAL CHANGE)

Changes:
- Pre-fit preprocessing once per N
- Single MLP fit per seed
- Reuse XGBoost DMatrix
- Reduced GC / CUDA pressure
"""
# ============================================================

# ================= CPU / RAM =================
import os
import csv
import time
import logging
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ================= GLOBAL SETTINGS =================
ID_COLS = ["subject_id", "hadm_id", "stay_id"]
TARGET_COL = "in_hospital_mortality"
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

SAMPLE_SIZES = [1500,3500,7000]
POLLUTION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_BOOTSTRAPS = 500

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
    path = Path(path)
    pq = path.with_suffix(path.suffix + ".parquet")
    if pq.exists():
        return pd.read_parquet(pq)

    df = pd.read_csv(path, sep=None, engine="python")
    df.to_parquet(pq, index=False)
    return df

# ================= IMPUTATION =================
def build_sex_conditioned_pools(df, sex_col, cols):
    pools = {}
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

    for c in CAT_COLS:
        if c in df:
            df[c] = df[c].astype(object)

    return df

def prepare_data_for_models(df_train, df_eval, preprocessor=None):
    ytr = df_train[TARGET_COL].astype(int).to_numpy()
    yev = df_eval[TARGET_COL].astype(int).to_numpy()

    Xtr_df = df_train.drop(columns=ID_COLS + [TARGET_COL], errors="ignore").copy()
    Xev_df = df_eval.drop(columns=ID_COLS + [TARGET_COL], errors="ignore").copy()

    # 🔒 GARANTIE : pas de NaN catégoriels
    for c in CAT_COLS:
        if c in Xtr_df:
            Xtr_df[c] = Xtr_df[c].astype(str).fillna(MISSING_TOKEN)
            Xev_df[c] = Xev_df[c].astype(str).fillna(MISSING_TOKEN)

    num_cols = [c for c in NUM_COLS if c in Xtr_df]
    cat_cols = [c for c in CAT_COLS if c in Xtr_df]

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", "passthrough", num_cols),
            ],
            remainder="drop",
        )
        Xtr = preprocessor.fit_transform(Xtr_df)
    else:
        Xtr = preprocessor.transform(Xtr_df)

    Xev = preprocessor.transform(Xev_df)

    # 🔒 GARANTIE FINALE : aucun NaN ne passe
    Xtr = np.nan_to_num(Xtr, nan=0.0).astype(np.float32, copy=False)
    Xev = np.nan_to_num(Xev, nan=0.0).astype(np.float32, copy=False)

    return preprocessor, Xtr, Xev, ytr, yev




# ================= BOOTSTRAP =================
def stratified_inbag_oob(df, ycol, seed):
    rng = np.random.RandomState(seed)
    inbag = []

    for _, g in df.groupby(ycol):
        inbag.append(
            g.sample(len(g), replace=True, random_state=rng.randint(1e6)).index
        )

    inbag_idx = np.concatenate(inbag)
    oob_idx = np.setdiff1d(df.index, inbag_idx)

    return (
        df.loc[inbag_idx].reset_index(drop=True),
        df.loc[oob_idx].reset_index(drop=True),
    )

# ================= METRICS =================
def compute_metrics(y, yp, pp):
    out = dict.fromkeys(["accuracy", "f1_macro", "auc", "brier", "auprc"], np.nan)
    if len(y) == 0:
        return out

    out["accuracy"] = accuracy_score(y, yp)
    out["f1_macro"] = f1_score(y, yp, average="macro")

    if len(np.unique(y)) > 1:
        out["auc"] = roc_auc_score(y, pp)
        out["auprc"] = average_precision_score(y, pp)

    out["brier"] = brier_score_loss(y, pp)
    return out

def combine_point632(a, b):
    return {k: 0.368 * a[k] + 0.632 * b[k] for k in a}

# ================= MLP =================
class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_mlp(Xtr, ytr, hidden, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(Xtr.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=128,
        shuffle=True,
    )

    model.train()
    for _ in range(10):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss(model(xb), yb).backward()
            opt.step()

    model.eval()
    return model

def predict_mlp(model, X):
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(X).to(device))
    return torch.softmax(logits, 1)[:, 1].cpu().numpy()

# ================= PREDICTION DIAGNOSTICS =================
def entropy_binary(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.mean(-p * np.log(p) - (1 - p) * np.log(1 - p))

def prediction_entropy(p):
    """
    Entropie binaire moyenne des probabilités prédites.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(
        np.mean(-p * np.log(p) - (1 - p) * np.log(1 - p))
    )


def prediction_diagnostics(p, prior):
    """
    Statistiques simples pour détecter le collapse vers la classe majoritaire.
    """
    return {
        "p_mean": float(np.mean(p)),
        "p_var": float(np.var(p)),
        "p_entropy": prediction_entropy(p),
        "p_abs_dist_prior": float(np.mean(np.abs(p - prior))),
    }

# ================= RUN MODELS =================

def run_all_models_point632_batched(df_inbag, df_oob, preprocessor):
    preprocessor, Xtr, Xoob, ytr, yoob = prepare_data_for_models(
        df_inbag, df_oob, preprocessor
    )

    res = {}
    diag = {}

    prior = ytr.mean()

    # ================= Logistic Regression =================
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr, ytr)

    p_tr = lr.predict_proba(Xtr)[:, 1]
    p_oob = lr.predict_proba(Xoob)[:, 1]

    res["LogisticRegression"] = combine_point632(
        compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob),
    )
    diag["LogisticRegression"] = prediction_diagnostics(p_oob, prior)

    # ================= LASSO =================
    lasso = Lasso(alpha=0.001, max_iter=5000)
    lasso.fit(Xtr, ytr)

    p_tr = np.clip(lasso.predict(Xtr), 0, 1)
    p_oob = np.clip(lasso.predict(Xoob), 0, 1)

    res["LASSO"] = combine_point632(
        compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob),
    )
    diag["LASSO"] = prediction_diagnostics(p_oob, prior)

    # ================= PLS =================
    pls = PLSRegression(n_components=min(10, Xtr.shape[1] - 1))
    pls.fit(Xtr, ytr)

    p_tr = np.clip(pls.predict(Xtr).ravel(), 0, 1)
    p_oob = np.clip(pls.predict(Xoob).ravel(), 0, 1)

    res["PLS"] = combine_point632(
        compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob),
    )
    diag["PLS"] = prediction_diagnostics(p_oob, prior)

    # ================= KNN =================
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr, ytr)

    p_tr = knn.predict_proba(Xtr)[:, 1]
    p_oob = knn.predict_proba(Xoob)[:, 1]

    res["KNN"] = combine_point632(
        compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob),
    )
    diag["KNN"] = prediction_diagnostics(p_oob, prior)

    # ================= SVM =================
    svm = SVC(kernel="rbf", probability=True, cache_size=500)
    svm.fit(Xtr, ytr)

    p_tr = svm.predict_proba(Xtr)[:, 1]
    p_oob = svm.predict_proba(Xoob)[:, 1]

    res["SVM"] = combine_point632(
        compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob),
    )
    diag["SVM"] = prediction_diagnostics(p_oob, prior)

    # ================= Decision Tree =================
    dt = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=10,
        random_state=0,
    )
    dt.fit(Xtr, ytr)

    p_tr = dt.predict_proba(Xtr)[:, 1]
    p_oob = dt.predict_proba(Xoob)[:, 1]

    res["DecisionTree"] = combine_point632(
        compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob),
    )
    diag["DecisionTree"] = prediction_diagnostics(p_oob, prior)


     # ================= Random Forest =================
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=0,
    )
    rf.fit(Xtr, ytr)

    p_tr = rf.predict_proba(Xtr)[:, 1]
    p_oob = rf.predict_proba(Xoob)[:, 1]

    res[RF_NAME] = combine_point632(
        compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob),
    )
    diag[RF_NAME] = prediction_diagnostics(p_oob, prior)


    # ================= XGBoost =================
    dtr = xgb.DMatrix(Xtr, label=ytr)
    doob = xgb.DMatrix(Xoob, label=yoob)

    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cuda",
        },
        dtr,
        num_boost_round=100,
    )

    p_tr = booster.predict(dtr)
    p_oob = booster.predict(doob)

    res["XGBoost"] = combine_point632(
        compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
        compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob),
    )
    diag["XGBoost"] = prediction_diagnostics(p_oob, prior)

    # ================= MLPs =================
    for name, hidden in {
        "MLP1": (64,),
        "MLP5": (64,) * 5,
        "MLP10": (64,) * 10,
    }.items():

        per_seed_metrics = []
        per_seed_probs = []

        for seed in [0, 1, 2]:
            model = train_mlp(Xtr, ytr, hidden, seed)

            p_tr = predict_mlp(model, Xtr)
            p_oob = predict_mlp(model, Xoob)

            per_seed_metrics.append(
                combine_point632(
                    compute_metrics(ytr, (p_tr > 0.5).astype(int), p_tr),
                    compute_metrics(yoob, (p_oob > 0.5).astype(int), p_oob),
                )
            )
            per_seed_probs.append(p_oob)

        res[name] = {
            k: np.mean([m[k] for m in per_seed_metrics])
            for k in per_seed_metrics[0]
        }

        diag[name] = prediction_diagnostics(
            np.mean(per_seed_probs, axis=0),
            prior,
        )

    torch.cuda.empty_cache()
    return res, diag



def measure_pollution_rate(df_before, df_after, cols):
    diff = 0
    total = 0
    for c in cols:
        if c not in df_before or c not in df_after:
            continue
        before = df_before[c].to_numpy()
        after  = df_after[c].to_numpy()
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

    # colonnes numériques attendues
    for c in NUM_COLS:
        if c not in df:
            df[c] = np.nan

    # colonnes catégorielles attendues
    for c in CAT_COLS:
        if c not in df:
            df[c] = np.nan

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

    # ================= INIT CSV =================
    if not csv_file.exists():
        with open(csv_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "scenario", "N", "p", "bootstrap", "model",
                "accuracy", "f1_macro", "auc", "brier", "auprc",
                "p_mean", "p_var", "p_entropy", "p_abs_dist_prior",
            ])

    # ================= REPRISE =================
    done_bootstraps = load_done_bootstraps(csv_file)
    if done_bootstraps:
        logger.info(f"RESUME | {len(done_bootstraps)} bootstraps déjà calculés")
    else:
        logger.info("AUCUNE REPRISE | scénario neuf")

    # ================= BOUCLE EXPÉRIMENTALE =================
    for N in SAMPLE_SIZES:
        logger.info(f"N = {N}")
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
                df_inbag, df_oob = stratified_inbag_oob(
                    df_sample,
                    TARGET_COL,
                    seed=10_000 + b,
                )

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
                    y = df_inbag[TARGET_COL].copy()
                    df_before = df_inbag.copy(deep=True)

                    out_tr = pollution_fn(df_inbag, p, seed=40_000 + b)
                    df_inbag = out_tr[0]
                    df_inbag[TARGET_COL] = y

                    pollution_rate_tr = measure_pollution_rate(
                        df_before, df_inbag, NUM_COLS
                    )

                    logger.info(
                        f"POLLUTION TRAIN | "
                        f"p_target={p:.3f} | p_real={pollution_rate_tr:.3f}"
                    )

                    # Assert UNIQUEMENT pour les pollueurs à densité exacte
                    if not getattr(pollution_fn, "APPROXIMATE_RATE", False):
                        assert abs(pollution_rate_tr - p) <= 0.01


                # ================= POLLUTION TEST =================
                if pollute_test and len(df_oob) > 0:
                    y = df_oob[TARGET_COL].copy()
                    df_before = df_oob.copy(deep=True)

                    out_te = pollution_fn(df_oob, p, seed=50_000 + b)
                    df_oob = out_te[0]
                    df_oob[TARGET_COL] = y

                    pollution_rate_te = measure_pollution_rate(
                        df_before, df_oob, NUM_COLS
                    )

                    logger.info(
                        f"POLLUTION TEST  | "
                        f"p_target={p:.3f} | p_real={pollution_rate_te:.3f}"
                    )

                    if not getattr(pollution_fn, "APPROXIMATE_RATE", False):
                        assert abs(pollution_rate_te - p) <= 0.01


                # ================= NORMALISATION =================
                df_inbag = ensure_schema(df_inbag)
                df_oob   = ensure_schema(df_oob)

                df_inbag = enforce_column_types(df_inbag)
                df_oob   = enforce_column_types(df_oob)

                # ================= NaN → OOD aléatoire =================
                df_inbag = encode_nan_as_sentinel(df_inbag, NUM_COLS)
                df_oob   = encode_nan_as_sentinel(df_oob,   NUM_COLS)

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
                            m["accuracy"],
                            m["f1_macro"],
                            m["auc"],
                            m["brier"],
                            m["auprc"],
                            d.get("p_mean", np.nan),
                            d.get("p_var", np.nan),
                            d.get("p_entropy", np.nan),
                            d.get("p_abs_dist_prior", np.nan),
                        ])

                dt = time.time() - t0
                logger.info(
                    f"OK | N={N} | p={p:.2f} | b={b:03d}/{N_BOOTSTRAPS} | time={dt:5.1f}s"
                )

    logger.info("END SCENARIO")

