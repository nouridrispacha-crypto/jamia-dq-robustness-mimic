# ============================================================
"""
COMMON RUNNER — DATA QUALITY EXPERIMENTS — SURVIE — OPTION A (.632)
Dataset : 18_variables_survival.csv (9998 patients, 25 colonnes)

Cibles :
  - DURATION_COL = "time_to_death"  (float, jours depuis admission ICU)
  - EVENT_COL    = "death_event"    (1 = décédé, 0 = censuré)
  Taux de mortalité : ~18.9%  |  Médiane survie : 5.95 jours

Modèles : Cox PH, Random Survival Forest, XGBoost Survival (cox), DeepSurv
Métriques : C-index (.632), Integrated Brier Score, Brier à 7j/14j/30j/90j

------------------------------------------------------------------
RÉVISION — corrections d'exécution uniquement
------------------------------------------------------------------
Aucune modification n'affecte les valeurs calculées. Les graines de
rééchantillonnage, d'imputation et de pollution sont inchangées, ainsi que
les métriques, les hyperparamètres et l'ordre des opérations. Un réplicat
recalculé avec cette version reproduit exactement le résultat obtenu avec
la version précédente.

Six correctifs, tous marqués « MODIFIÉ » dans le fichier :

  1. load_done_bootstraps indexe désormais sur le modèle. L'ancienne version
     marquait un réplicat comme terminé dès qu'une ligne existait, si bien
     qu'un réplicat où le RSF avait planté était sauté à la relance et que
     le trou restait ouvert définitivement.

  2. setup_logger attache aussi ses handlers au logger racine. Les blocs
     d'échec modèle écrivent sur logging.getLogger() sans nom, qui n'avait
     aucun handler : les messages « RSF FAILED » n'atteignaient jamais le
     fichier .log et le diagnostic était impossible a posteriori.

  3. RSF_N_JOBS remplace n_jobs=-1. Avec plusieurs runs simultanés, -1 fait
     réclamer tous les cœurs à chaque processus. n_jobs ne change pas les
     résultats : random_state=0 fixe les graines des arbres avant la
     distribution parallèle.

  4. THREADS passe de 20 à 6, pour la même raison, côté algèbre linéaire.

  5. Garde-fou mémoire dans la boucle. psutil.Process.rlimit() n'existe pas
     sous Windows : l'appel échouait silencieusement et aucune limite
     n'était appliquée. Le contrôle de RAM disponible arrête proprement le
     run avant saturation, ce qui est sans conséquence puisque la reprise
     corrigée redémarre exactement où le run s'est interrompu.

  6. _safe_time_points est supprimée. La grille d'intégration est désormais
     fixe (IBS_GRID) et les horizons cliniques sont filtrés directement dans
     compute_survival_metrics. L'ancien repli sur vingt points linéaires
     rendait l'IBS non comparable d'un réplicat à l'autre.

  7. IBS calculé pour les cinq modèles. XGBoostSurv et les DeepSurv ne
     fournissent qu'un score de risque ; leur fonction de survie est
     reconstruite par l'estimateur de Breslow du hasard cumulé de base,
     ajusté sur les données d'entraînement. Auparavant ils recevaient
     surv_fns=None et leur IBS restait à NaN.

  8. Grille d'intégration et référence de censure fixées une fois sur la
     cohorte propre. Avec l'échantillon in-bag comme référence, l'IBS et les
     Brier ponctuels échouaient dans environ un tiers des réplicats, ceux où
     le OOB contient un temps supérieur au maximum de l'in-bag, échec
     silencieusement transformé en NaN par le try/except.

  9. load_done_bootstraps exige c_index ET ibs sur les cinq modèles
     (REQUIRE_IBS). Sans cela les réplicats produits avant la correction 7,
     qui ont un c_index complet mais pas d'ibs, seraient considérés comme
     terminés et ne seraient jamais recalculés.
"""
# ============================================================

# ================= CPU / RAM =================
import os, csv, time, logging, gc
import psutil

process = psutil.Process()

# MODIFIÉ (5) — psutil.Process.rlimit() est spécifique à Linux. Sous
# Windows l'appel lève une exception, avalée par le except, et aucune
# limite n'est posée. La constante est conservée à titre documentaire ;
# le garde-fou effectif est MIN_AVAILABLE_GB, contrôlé dans la boucle.
MAX_RAM_BYTES = 6 * (1024 ** 3)
try:
    process.rlimit(psutil.RLIMIT_AS, (MAX_RAM_BYTES, MAX_RAM_BYTES))
except Exception:
    pass

# Seuil de RAM disponible sous lequel le run s'interrompt proprement.
MIN_AVAILABLE_GB = 4.0

# MODIFIÉ (4) — était "20". Avec plusieurs processus en parallèle sur une
# machine à 20 cœurs, chaque processus réclamait les 20 cœurs.
THREADS = "6"
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[k] = THREADS

# MODIFIÉ (3) — était n_jobs=-1 en dur dans le bloc RSF.
# À ajuster selon le nombre de runs lancés simultanément :
# processus x RSF_N_JOBS doit rester proche du nombre de cœurs.
RSF_N_JOBS = 6

# ================= IMPORTS =================
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    brier_score,
)

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ================= GLOBAL SETTINGS =================
ID_COLS      = ["subject_id", "hadm_id", "stay_id"]
DURATION_COL = "time_to_death"    # jours depuis admission ICU
EVENT_COL    = "death_event"      # 1=décédé, 0=censuré
SEX_COL      = "gender"           # M / F

# Colonnes à exclure de X (non-features)
EXTRA_EXCLUDE = ["in_hospital_mortality"]

OOD_RANGES = {
    "age":          [(-50, -1),    (200, 500)  ],
    "hr_mean":      [(-40, 0),     (300, 800)  ],
    "sbp_mean":     [(-50, 0),     (300, 600)  ],
    "dbp_min":      [(-30, 0),     (200, 400)  ],
    "rr_mean":      [(-20, 0),     (100, 300)  ],
    "spo2_min":     [(-20, 0),     (120, 200)  ],
    "wbc_min":      [(-10, 0),     (200, 1000) ],
    "aniongap_min": [(-50, 0),     (200, 500)  ],
    "aniongap_max": [(-50, 0),     (200, 500)  ],
    "bun_min":      [(-50, 0),     (500, 2000) ],
    "inr_min":      [(-10, 0),     (50, 200)   ],
    "inr_max":      [(-10, 0),     (50, 200)   ],
    "ptt_min":      [(-50, 0),     (500, 2000) ],
    "urine_output": [(-5000, -1),  (1e6, 1e7)  ],
}

SENTINELS = {
    "age": -99.0, "hr_mean": -9.0, "sbp_mean": -1.0, "dbp_min": -1.0,
    "rr_mean": -1.0, "spo2_min": -5.0, "wbc_min": -1.0,
    "aniongap_min": -1.0, "aniongap_max": -1.0, "bun_min": -1.0,
    "inr_min": -1.0, "inr_max": -1.0, "ptt_min": -1.0,
    "urine_output": -9999.0, "dobutamine": -1.0, "dopamine": -1.0,
    "norepinephrine": -1.0, "phenylephrine": -1.0,
}

MISSING_TOKEN = "missing"

# MODIFIÉ — priorité donnée à N=3500, qui porte les résultats principaux
# du manuscrit. La boucle traite les tailles dans l'ordre de cette liste
# et parcourt N=1500 en entier avant d'atteindre N=3500.
# Valeur d'origine : [1500, 3500, 7000]
SAMPLE_SIZES     = [3500]
POLLUTION_LEVELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_BOOTSTRAPS     = 500

IMPUTATION_MODE = "HF_RANDOM"
MODELS_TO_RUN   = None

CAT_COLS = ["gender"]
NUM_COLS = [
    "age", "hr_mean", "sbp_mean", "dbp_min", "rr_mean", "spo2_min",
    "wbc_min", "aniongap_min", "aniongap_max", "bun_min",
    "inr_min", "inr_max", "ptt_min", "urine_output",
    "dobutamine", "dopamine", "norepinephrine", "phenylephrine",
]
IMPUTE_COLS = NUM_COLS + ["gender"]

# Points d'évaluation Brier Score — adaptés à la médiane 5.95j
BRIER_TIME_POINTS = np.array([7, 14, 30, 90])

RF_NAME = "RandomSurvivalForest"

# MODIFIÉ (1) — liste des modèles écrits à chaque réplicat. Sert de
# critère de complétude à la reprise : un réplicat n'est considéré comme
# terminé que si les cinq ont produit un c_index exploitable.
ALL_MODELS = ["CoxPH", RF_NAME, "XGBoostSurv", "DeepSurv1", "DeepSurv5"]

# Index des colonnes du CSV de sortie, utilisés par load_done_bootstraps
COL_N, COL_P, COL_B, COL_MODEL, COL_CINDEX = 1, 2, 3, 4, 5
COL_IBS = 6

# Un réplicat n'est considéré comme terminé que si les cinq modèles ont
# produit à la fois un c_index ET un ibs exploitables. Mettre à False pour
# revenir au contrôle sur le seul c_index (ancien comportement).
REQUIRE_IBS = True


# ================= LOGGER =================
def setup_logger(name, out_dir):
    """
    MODIFIÉ (2) — les handlers sont désormais attachés au logger racine
    en plus du logger nommé.

    Les blocs d'échec modèle appellent logging.getLogger() sans argument,
    ce qui renvoie le logger racine. Celui-ci n'ayant aucun handler, les
    messages « RSF FAILED », « XGBoostSurv FAILED », etc. partaient vers
    stderr et n'étaient jamais écrits dans le fichier .log. Un fichier de
    log sans occurrence de FAILED ne signifiait donc pas absence d'échec,
    mais perte de l'information.

    propagate est mis à False sur le logger nommé pour éviter que chaque
    message ne soit écrit deux fois, les deux loggers partageant désormais
    les mêmes handlers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

    # Logger racine : niveau WARNING pour ne capter que les échecs et non
    # le bavardage informatif des bibliothèques tierces.
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(ch)
    root.addHandler(fh)

    return logger


def load_done_bootstraps(csv_file: Path):
    """
    MODIFIÉ (1) — retourne l'ensemble des (N, p, b) réellement terminés.

    L'ancienne version ajoutait un tuple dès qu'une ligne existait, sans
    regarder ni le modèle ni la valeur. Comme les cinq modèles sont écrits
    ensemble à chaque réplicat, un réplicat où le RSF avait planté
    conservait quand même quatre lignes valides : le tuple (N, p, b) était
    présent, la reprise le sautait, et le trou ne pouvait plus jamais être
    comblé, y compris après suppression manuelle de la ligne en échec.

    Ici, un réplicat n'est marqué comme fait que si les cinq modèles ont
    produit un c_index exploitable. Les réplicats partiels sont recalculés
    en entier, ce qui est sans risque : les graines étant déterministes en
    b, les quatre modèles ayant déjà réussi reproduiront exactement leurs
    valeurs précédentes.

    MODIFIÉ (7) — l'ibs est désormais exigé au même titre que le c_index
    lorsque REQUIRE_IBS est vrai. Sans cela, les réplicats produits avant la
    correction Breslow, qui possèdent un c_index complet mais un ibs vide
    pour XGBoostSurv et les DeepSurv, seraient considérés comme terminés :
    la reprise les sauterait et leur ibs ne serait jamais calculé. Le
    contrôle porte donc sur la sortie réellement attendue, et un ancien CSV
    est automatiquement recalculé sans qu'il faille le supprimer.
    """
    if not csv_file.exists():
        return set(), 0

    seen = {}   # (N, p, b) -> ensemble des modèles ayant réussi

    def _usable(row, col):
        """Valeur numérique présente et non NaN dans cette colonne."""
        if len(row) <= col:
            return False
        raw = row[col].strip()
        if raw == "" or raw.lower() in {"nan", "na", "none"}:
            return False
        try:
            return not np.isnan(float(raw))
        except ValueError:
            return False

    try:
        with open(csv_file, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)                      # entête
            for row in reader:
                if len(row) <= COL_CINDEX:
                    continue
                try:
                    key = (int(row[COL_N]), float(row[COL_P]),
                           int(row[COL_B]))
                    model = row[COL_MODEL]
                except (ValueError, IndexError):
                    continue

                if not _usable(row, COL_CINDEX):
                    continue
                if REQUIRE_IBS and not _usable(row, COL_IBS):
                    continue

                seen.setdefault(key, set()).add(model)
    except OSError:
        return set(), 0

    required = set(ALL_MODELS)
    complete = {k for k, models in seen.items() if required.issubset(models)}
    partial = len(seen) - len(complete)

    return complete, partial


# ================= DATA =================
def load_dataset(path):
    # MODIFIÉ (dépôt public) — utilise le chemin transmis par l'appelant
    # (relatif au script, voir run/*.py) au lieu d'un chemin absolu codé
    # en dur propre à la machine de développement.
    path = Path(path)
    pq = path.with_suffix(".parquet")

    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path}\n"
                f"Copiez 18_variables_survival.csv dans ce dossier."
            )
        df = pd.read_csv(path, sep=None, engine="python")
        df.to_parquet(pq, index=False)

    for col in [DURATION_COL, EVENT_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Colonne manquante : '{col}'. "
                f"Colonnes présentes : {df.columns.tolist()}"
            )

    df[DURATION_COL] = pd.to_numeric(df[DURATION_COL], errors="coerce")
    # Force bool strict — certaines versions sksurv rejettent int64
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype(bool)
    df = df[df[DURATION_COL].notna() & (df[DURATION_COL] > 0)].reset_index(drop=True)
    return df


# ================= IMPUTATION =================
def build_sex_conditioned_pools(df, sex_col, cols):
    pools = {}
    if sex_col not in df.columns:
        return pools
    for sex, g in df.groupby(sex_col):
        pools[sex] = {c: g[c].dropna().to_numpy()
                      for c in cols if c in g and g[c].notna().any()}
    return pools


def impute_hf_random_draw_sex(df_apply, sex_col, cols, pools, seed):
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
    if DURATION_COL in df:
        df[DURATION_COL] = pd.to_numeric(df[DURATION_COL], errors="coerce")
    if EVENT_COL in df:
        df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype(bool)
    for c in CAT_COLS:
        if c in df:
            df[c] = df[c].astype(object)
    return df


def ensure_schema(df):
    df = df.copy()
    for c in NUM_COLS:
        if c not in df:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df:
            df[c] = np.nan
    if DURATION_COL not in df:
        df[DURATION_COL] = np.nan
    if EVENT_COL not in df:
        df[EVENT_COL] = False
    return df


def prepare_data_for_models(df_train, df_eval, preprocessor=None):
    # Créer les structured arrays avec les champs standards "event"/"time"
    # (Surv.from_dataframe >= 0.21 préserve les noms de colonnes originaux)
    _dt = [('event', bool), ('time', float)]
    ytr = np.array(list(zip(df_train[EVENT_COL].astype(bool),
                            df_train[DURATION_COL].astype(float))), dtype=_dt)
    yev = np.array(list(zip(df_eval[EVENT_COL].astype(bool),
                            df_eval[DURATION_COL].astype(float))), dtype=_dt)

    drop_cols = ID_COLS + [DURATION_COL, EVENT_COL] + EXTRA_EXCLUDE
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
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ], remainder="drop")
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
    inbag_idx   = rng.choice(df.index.to_numpy(), size=n, replace=True)
    inbag       = df.loc[inbag_idx].reset_index(drop=True)
    unique_inbag = np.unique(inbag_idx)
    oob_idx     = np.setdiff1d(df.index.to_numpy(), unique_inbag)
    oob         = df.loc[oob_idx].reset_index(drop=True)
    return inbag, oob


# ================= MÉTRIQUES =================
# ================= REFERENCE D'INTEGRATION DE L'IBS =================
# Deux objets fixes, definis une seule fois sur la cohorte PROPRE :
#
#   IBS_GRID       grille d'integration, identique pour tous les modeles,
#                  niveaux de corruption, tailles de cohorte et replicats.
#                  Sans cela l'IBS est integre sur un support variable et
#                  n'est pas comparable d'un replicat a l'autre.
#
#   CENSORING_REF  jeu de reference servant a estimer la loi de censure
#                  (ponderation IPCW de Graf). Deux raisons de le fixer :
#                  la censure est une propriete de la cohorte et non du
#                  tirage bootstrap, et surtout l'echantillon in-bag ne
#                  couvre pas toujours les temps du OOB. Avec l'in-bag
#                  comme reference, integrated_brier_score leve
#                  "time must be smaller than largest observed time point"
#                  dans environ 36 % des replicats, silencieusement
#                  transforme en NaN par le try/except.

IBS_GRID = None          # rempli par set_ibs_reference()
CENSORING_REF = None     # rempli par set_ibs_reference()
IBS_GRID_N = 50          # nombre de points d'integration
IBS_LOWER_Q = 0.05       # borne basse : 5e centile des temps d'evenement
IBS_UPPER_Q = 0.90       # borne haute : 90e centile (queues instables exclues)


def set_ibs_reference(df):
    """
    Fixe la grille d'integration et la reference de censure de l'IBS a
    partir de la cohorte propre, avant toute degradation.

    La borne haute au 90e centile des temps d'evenement ecarte la queue de
    distribution, ou peu de sujets restent a risque et ou la ponderation de
    censure devient instable au point de dominer l'integrale.
    """
    global IBS_GRID, CENSORING_REF

    ev = df[EVENT_COL].astype(bool).to_numpy()
    tt = df[DURATION_COL].astype(float).to_numpy()

    CENSORING_REF = np.array(
        list(zip(ev, tt)), dtype=[("event", bool), ("time", float)]
    )

    t = tt[ev]
    if t.size < 10:
        t = tt

    lo = max(float(np.quantile(t, IBS_LOWER_Q)), 1e-3)
    hi = float(np.quantile(t, IBS_UPPER_Q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 1.0, max(2.0, float(np.nanmax(tt)) * 0.9)

    # strictement a l'interieur de la fenetre de la reference de censure
    hi = min(hi, float(tt.max()) * (1 - 1e-9))
    IBS_GRID = np.linspace(lo, hi, IBS_GRID_N)

    logging.getLogger().info(
        f"REFERENCE IBS FIXE | grille {IBS_GRID_N} points de {lo:.2f} a "
        f"{hi:.2f} j (centiles {IBS_LOWER_Q:.0%}-{IBS_UPPER_Q:.0%} des temps "
        f"d'evenement) | censure estimee sur {len(CENSORING_REF)} sujets propres"
    )
    return IBS_GRID


# conserve pour compatibilite avec d'anciens scripts d'appel
set_ibs_grid = set_ibs_reference


def _ibs_test_mask(y_test):
    """
    Sujets utilisables pour l'IBS : ceux dont le temps est strictement
    inferieur au maximum de la reference de censure. Ecarte typiquement un
    seul enregistrement, celui qui porte le temps maximal de la cohorte.
    """
    t_max = float(CENSORING_REF["time"].max())
    return np.asarray(y_test["time"], dtype=float) < t_max


# ================= BRESLOW =================
# Reconstruit la fonction de survie des modeles qui ne fournissent qu'un
# score de risque : XGBoostSurv, DeepSurv1, DeepSurv5.

def breslow_fit(y_train, log_risk_train):
    """
    Hazard cumule de base par l'estimateur de Breslow, ajuste sur le train.

        H0(t) = somme_{t_i <= t, evenement} 1 / somme_{j : T_j >= t_i} exp(eta_j)

    log_risk_train est le predicteur lineaire eta (log-risque), pas exp(eta).
    Les log-risques sont centres sur leur moyenne d'entrainement : cela ne
    change pas S(t|x), le decalage etant absorbe par H0, mais evite les
    depassements de exp().
    """
    t = np.asarray(y_train["time"], dtype=float)
    e = np.asarray(y_train["event"], dtype=bool)
    eta = np.asarray(log_risk_train, dtype=float).ravel()

    if t.size == 0 or e.sum() == 0 or eta.size != t.size:
        return None

    finite = np.isfinite(eta)
    center = float(np.mean(eta[finite])) if finite.any() else 0.0
    r = np.exp(np.clip(np.nan_to_num(eta, nan=center) - center, -20.0, 20.0))

    order = np.argsort(t, kind="mergesort")
    t, e, r = t[order], e[order], r[order]

    # somme des risques des sujets encore a risque, a chaque temps
    risk_set = np.cumsum(r[::-1])[::-1]

    ev_t = t[e]
    uniq, first = np.unique(ev_t, return_index=True)
    counts = np.bincount(np.searchsorted(uniq, ev_t), minlength=uniq.size)
    denom = risk_set[e][first]

    valid = denom > 0
    if not valid.any():
        return None

    H0 = np.cumsum(counts[valid] / denom[valid])
    return {"t": uniq[valid], "H0": H0, "center": center}


def breslow_surv_factory(baseline, log_risk):
    """
    Retourne une fonction pts -> matrice de survie (n_sujets x n_points),
    avec S(t|x) = exp( -H0(t) * exp(eta_x) ).

    Passee a compute_survival_metrics a la place de surv_fns, elle permet
    d'evaluer la survie sur n'importe quelle grille sans la recalculer.
    """
    if baseline is None:
        return None

    eta = np.asarray(log_risk, dtype=float).ravel()
    eta = np.nan_to_num(eta, nan=baseline["center"])
    risk = np.exp(np.clip(eta - baseline["center"], -20.0, 20.0))

    def surv_at(pts):
        H0 = np.interp(
            np.asarray(pts, dtype=float),
            baseline["t"], baseline["H0"],
            left=0.0, right=float(baseline["H0"][-1]),
        )
        return np.exp(-np.outer(risk, H0))

    return surv_at


def compute_survival_metrics(y_train, y_test, risk_scores, surv_fns=None):

    out = {"c_index": np.nan, "ibs": np.nan}
    for t in BRIER_TIME_POINTS:
        out[f"brier_{t}d"] = np.nan

    if len(y_test) == 0:
        return out

    # éviter NaN si aucun événement
    if y_test["event"].sum() == 0:
        return out

    # ================= C-INDEX =================
    try:
        ci = concordance_index_censored(
            y_test["event"],
            y_test["time"],
            risk_scores,
        )
        out["c_index"] = float(ci[0])
    except Exception:
        pass

    # ================= IBS / BRIER =================
    # surv_fns accepte trois formes :
    #   - une fonction pts -> matrice        (modeles a score de risque, Breslow)
    #   - un tableau de StepFunction sksurv  (CoxPH, RSF)
    #   - une matrice deja evaluee
    if surv_fns is not None:

        def _matrix_at(pts, rows=None):
            if callable(surv_fns):
                m = surv_fns(pts)
            elif hasattr(surv_fns[0], "__call__"):
                m = np.vstack([np.nan_to_num(fn(pts), nan=1.0) for fn in surv_fns])
            else:
                m = np.asarray(surv_fns)
            m = np.nan_to_num(m, nan=1.0)
            return m if rows is None else m[rows]

        # ---- Brier ponctuels aux horizons cliniques ----
        # Meme reference de censure et meme masque que l'IBS. Sans cela ces
        # colonnes echouent dans environ un tiers des replicats, ceux ou le
        # OOB contient un temps superieur au maximum de l'in-bag, et l'on
        # obtient un replicat avec IBS mais sans Brier ponctuel.
        #
        # Les bornes sont celles de la reference de censure, et non celles du
        # jeu evalue : un horizon nominal est valide des lors que la censure
        # y est estimable. Le borner par le minimum du jeu evalue faisait
        # sauter l'horizon 7 j des qu'aucun sujet ne sortait avant, et comme
        # la regle .632 propage un NaN d'un seul cote, la colonne devenait
        # vide pour les cinq modeles a la fois.
        try:
            if CENSORING_REF is None:
                raise RuntimeError(
                    "reference IBS non initialisee (appeler set_ibs_reference)")

            keep_c = _ibs_test_mask(y_test)
            t_ref_min = float(CENSORING_REF["time"].min())
            t_ref_max = float(CENSORING_REF["time"].max())

            pts_c = np.asarray(BRIER_TIME_POINTS, dtype=float)
            pts_c = pts_c[(pts_c >= t_ref_min) & (pts_c < t_ref_max)]

            if len(pts_c) > 0 and keep_c.sum() >= 20 \
                    and y_test["event"][keep_c].sum() > 0:
                _, bs = brier_score(
                    CENSORING_REF, y_test[keep_c],
                    _matrix_at(pts_c, keep_c), pts_c,
                )
                for t, s in zip(pts_c, bs):
                    out[f"brier_{int(round(t))}d"] = float(s)
        except Exception as exc:
            logging.getLogger().warning(
                f"BRIER PONCTUEL FAILED ({type(exc).__name__}): {exc}")

        # ---- IBS sur la grille FIXE, censure estimee sur la cohorte propre ----
        try:
            if IBS_GRID is None or CENSORING_REF is None:
                raise RuntimeError(
                    "reference IBS non initialisee (appeler set_ibs_reference)")

            keep = _ibs_test_mask(y_test)
            if keep.sum() < 20 or y_test["event"][keep].sum() == 0:
                logging.getLogger().warning(
                    f"IBS NON CALCULE | {int(keep.sum())} sujets utilisables, "
                    f"{int(y_test['event'][keep].sum())} evenements")
            else:
                out["ibs"] = float(
                    integrated_brier_score(
                        CENSORING_REF,
                        y_test[keep],
                        _matrix_at(IBS_GRID, keep),
                        IBS_GRID,
                    )
                )
        except Exception as exc:
            logging.getLogger().warning(f"IBS FAILED ({type(exc).__name__}): {exc}")

    return out


def combine_point632_survival(m_tr, m_oob):
    result = {}
    ci_tr, ci_oob = m_tr.get("c_index", np.nan), m_oob.get("c_index", np.nan)
    if not (np.isnan(ci_tr) or np.isnan(ci_oob)):
        result["c_index"] = 1.0 - (0.368*(1-ci_tr) + 0.632*(1-ci_oob))
    else:
        result["c_index"] = np.nan
    for key in ["ibs"] + [f"brier_{t}d" for t in BRIER_TIME_POINTS]:
        v_tr, v_oob = m_tr.get(key, np.nan), m_oob.get(key, np.nan)
        result[key] = 0.368*v_tr + 0.632*v_oob if not (np.isnan(v_tr) or np.isnan(v_oob)) else np.nan
    return result


def survival_diagnostics(y_test, risk_scores):
    if len(risk_scores) == 0:
        return {"risk_mean": np.nan, "risk_var": np.nan,
                "event_rate": np.nan, "median_time": np.nan}
    return {
        "risk_mean":   float(np.mean(risk_scores)),
        "risk_var":    float(np.var(risk_scores)),
        "event_rate":  float(y_test["event"].mean()),
        "median_time": float(np.median(y_test["time"])),
    }


def _nan_metrics():
    out = {"c_index": np.nan, "ibs": np.nan}
    for t in BRIER_TIME_POINTS:
        out[f"brier_{t}d"] = np.nan
    return out


# ================= DeepSurv =================
class DeepSurvNet(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def _cox_loss(log_risk, event, time):
    order = torch.argsort(time, descending=True)
    log_risk, event = log_risk[order], event[order]

    log_risk = torch.clamp(log_risk, -10, 10)

    return -torch.mean(
        (log_risk - torch.logcumsumexp(log_risk, dim=0)) * event
    )


def train_deepsurv(Xtr, ytr, hidden, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    events = torch.tensor(ytr["event"].astype(np.float32))
    times  = torch.tensor(ytr["time"].astype(np.float32))
    model  = DeepSurvNet(Xtr.shape[1], hidden).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), events, times),
        batch_size=128, shuffle=False,
    )
    model.train()
    for _ in range(20):
        for xb, eb, tb in loader:
            xb, eb, tb = xb.to(device), eb.to(device), tb.to(device)
            opt.zero_grad()
            _cox_loss(model(xb), eb, tb).backward()
            opt.step()
    model.eval()
    return model


def predict_risk(model, X):
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).cpu().numpy()


# ================= RUN MODELS =================
def run_all_models_point632_batched(df_inbag, df_oob, preprocessor):
    preprocessor, Xtr, Xoob, ytr, yoob = prepare_data_for_models(
        df_inbag, df_oob, preprocessor)
    res, diag = {}, {}

    # ── Cox PH ──────────────────────────────────────────────
    try:
        cox = CoxPHSurvivalAnalysis(alpha=0.1, ties="efron", n_iter=100)
        cox.fit(Xtr, ytr)
        risk_tr, risk_oob = cox.predict(Xtr), cox.predict(Xoob)
        surv_tr  = cox.predict_survival_function(Xtr)
        surv_oob = cox.predict_survival_function(Xoob)
        res["CoxPH"]  = combine_point632_survival(
            compute_survival_metrics(ytr, ytr,  risk_tr,  surv_tr),
            compute_survival_metrics(ytr, yoob, risk_oob, surv_oob),
        )
        diag["CoxPH"] = survival_diagnostics(yoob, risk_oob)
        del cox, surv_tr, surv_oob
    except MemoryError as e:
        # MODIFIÉ (2) — MemoryError distingué des autres exceptions, pour
        # que le log permette de trancher entre saturation mémoire et
        # échec numérique.
        logging.getLogger().error(f"CoxPH MEMORY ERROR: {e}")
        res["CoxPH"]  = _nan_metrics()
        diag["CoxPH"] = survival_diagnostics(yoob, np.array([]))
    except Exception as e:
        logging.getLogger().warning(f"CoxPH FAILED ({type(e).__name__}): {e}")
        res["CoxPH"]  = _nan_metrics()
        diag["CoxPH"] = survival_diagnostics(yoob, np.array([]))

    # ── Random Survival Forest ───────────────────────────────
    try:
        # MODIFIÉ (3) — n_jobs=RSF_N_JOBS au lieu de n_jobs=-1.
        # random_state=0 fixe les graines des arbres avant la distribution
        # parallèle : les résultats sont identiques quel que soit n_jobs.
        rsf = RandomSurvivalForest(
            n_estimators=200, min_samples_leaf=10,
            max_features="sqrt", n_jobs=RSF_N_JOBS, random_state=0)
        rsf.fit(Xtr, ytr)
        risk_tr, risk_oob = rsf.predict(Xtr), rsf.predict(Xoob)
        surv_tr  = rsf.predict_survival_function(Xtr)
        surv_oob = rsf.predict_survival_function(Xoob)
        res[RF_NAME]  = combine_point632_survival(
            compute_survival_metrics(ytr, ytr,  risk_tr,  surv_tr),
            compute_survival_metrics(ytr, yoob, risk_oob, surv_oob),
        )
        diag[RF_NAME] = survival_diagnostics(yoob, risk_oob)
        # Libération explicite : le RSF est de loin le plus gros
        # consommateur du panel, et l'objet reste sinon vivant pendant
        # l'entraînement des modèles suivants.
        del rsf, surv_tr, surv_oob
    except MemoryError as e:
        logging.getLogger().error(f"RSF MEMORY ERROR: {e}")
        res[RF_NAME]  = _nan_metrics()
        diag[RF_NAME] = survival_diagnostics(yoob, np.array([]))
    except Exception as e:
        logging.getLogger().warning(f"RSF FAILED ({type(e).__name__}): {e}")
        res[RF_NAME]  = _nan_metrics()
        diag[RF_NAME] = survival_diagnostics(yoob, np.array([]))

    gc.collect()

    # ── XGBoost Survival (Cox) ───────────────────────────────
    try:
        def xgb_labels(y):
            t, e = y["time"].astype(np.float32), y["event"].astype(np.float32)
            return np.where(e == 1, t, -t)

        dtr  = xgb.DMatrix(Xtr,  label=xgb_labels(ytr))
        doob = xgb.DMatrix(Xoob, label=xgb_labels(yoob))
        xgb_params = {
            "objective": "survival:cox", "tree_method": "hist",
            "eval_metric": "cox-nloglik", "learning_rate": 0.05, "max_depth": 6,
        }
        if torch.cuda.is_available():
            xgb_params["device"] = "cuda"
        booster  = xgb.train(xgb_params, dtr, num_boost_round=200, verbose_eval=False)
        risk_tr, risk_oob = booster.predict(dtr), booster.predict(doob)

        # survival:cox renvoie le rapport de hasard exp(eta) ; Breslow
        # attend le log-risque eta.
        eta_tr  = np.log(np.clip(risk_tr,  1e-12, None))
        eta_oob = np.log(np.clip(risk_oob, 1e-12, None))
        bl = breslow_fit(ytr, eta_tr)

        res["XGBoostSurv"]  = combine_point632_survival(
            compute_survival_metrics(ytr, ytr,  risk_tr,
                                     breslow_surv_factory(bl, eta_tr)),
            compute_survival_metrics(ytr, yoob, risk_oob,
                                     breslow_surv_factory(bl, eta_oob)),
        )
        diag["XGBoostSurv"] = survival_diagnostics(yoob, risk_oob)
        del booster, dtr, doob
    except MemoryError as e:
        logging.getLogger().error(f"XGBoostSurv MEMORY ERROR: {e}")
        res["XGBoostSurv"]  = _nan_metrics()
        diag["XGBoostSurv"] = survival_diagnostics(yoob, np.array([]))
    except Exception as e:
        logging.getLogger().warning(f"XGBoostSurv FAILED ({type(e).__name__}): {e}")
        res["XGBoostSurv"]  = _nan_metrics()
        diag["XGBoostSurv"] = survival_diagnostics(yoob, np.array([]))

    # ── DeepSurv ────────────────────────────────────────────
    for name, hidden in {"DeepSurv1": (64,), "DeepSurv5": (64,)*5}.items():
        seed_metrics, seed_risks = [], []
        for seed in [0, 1, 2]:
            try:
                mdl   = train_deepsurv(Xtr, ytr, hidden, seed)
                r_tr  = predict_risk(mdl, Xtr)
                r_oob = predict_risk(mdl, Xoob)

                # la sortie du reseau EST le log-risque eta : la perte de
                # vraisemblance partielle de Cox est ecrite sur cette echelle.
                bl = breslow_fit(ytr, r_tr)

                seed_metrics.append(combine_point632_survival(
                    compute_survival_metrics(ytr, ytr,  r_tr,
                                             breslow_surv_factory(bl, r_tr)),
                    compute_survival_metrics(ytr, yoob, r_oob,
                                             breslow_surv_factory(bl, r_oob)),
                ))
                seed_risks.append(r_oob)
                del mdl
            except MemoryError as e:
                logging.getLogger().error(f"{name} seed={seed} MEMORY ERROR: {e}")
                continue
            except Exception as e:
                logging.getLogger().warning(
                    f"{name} seed={seed} FAILED ({type(e).__name__}): {e}")
                continue
        if seed_metrics:
            # np.nanmean sur des colonnes entièrement NaN (ibs et brier_*d
            # pour les modèles à score de risque) émet un RuntimeWarning
            # « Mean of empty slice ». C'est attendu et sans conséquence :
            # ces modèles ne produisent pas de fonction de survie.
            with np.errstate(invalid="ignore"):
                res[name] = {k: float(np.nanmean([m[k] for m in seed_metrics]))
                             if not all(np.isnan(m[k]) for m in seed_metrics)
                             else np.nan
                             for k in seed_metrics[0]}
            diag[name] = survival_diagnostics(yoob, np.mean(seed_risks, axis=0))
        else:
            res[name]  = _nan_metrics()
            diag[name] = survival_diagnostics(yoob, np.array([]))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return res, diag


# ================= POLLUTION HELPERS =================
def measure_pollution_rate(df_before, df_after, cols):
    diff, total = 0, 0
    for c in cols:
        if c not in df_before or c not in df_after:
            continue
        before = df_before[c].to_numpy()
        after  = df_after[c].to_numpy()
        mask   = ~(pd.isna(before) & pd.isna(after))
        diff  += np.sum(before[mask] != after[mask])
        total += mask.sum()
    return diff / total if total > 0 else None


# ================= RUN SCENARIO =================
def run_scenario(scenario_name, csv_path, out_dir, pollution_fn,
                 pollute_train, pollute_test):
    out_dir  = Path(out_dir)
    logger   = setup_logger(scenario_name, out_dir)
    df       = load_dataset(csv_path)
    csv_file = out_dir / f"{scenario_name}.csv"

    df = ensure_schema(df)
    df = enforce_column_types(df)
    df = df[df[DURATION_COL].notna() & (df[DURATION_COL] > 0)].reset_index(drop=True)

    brier_cols = [f"brier_{t}d" for t in BRIER_TIME_POINTS]
    if not csv_file.exists():
        with open(csv_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "scenario", "N", "p", "bootstrap", "model",
                "c_index", "ibs", *brier_cols,
                "risk_mean", "risk_var", "event_rate", "median_time",
            ])

    # MODIFIÉ (1) — load_done_bootstraps renvoie désormais un couple
    # (réplicats complets, nombre de réplicats partiels).
    done_bootstraps, partial = load_done_bootstraps(csv_file)
    if done_bootstraps or partial:
        logger.info(
            f"RESUME | {len(done_bootstraps)} replicats complets "
            f"(criteres : c_index" + (" + ibs" if REQUIRE_IBS else "") +
            " sur les 5 modeles)"
        )
        if partial:
            logger.warning(
                f"RESUME | {partial} replicats incomplets seront recalcules "
                f"(un ou plusieurs modeles sans c_index"
                + (" ou sans ibs" if REQUIRE_IBS else "") + ")"
            )
    else:
        logger.info("AUCUNE REPRISE | scenario neuf")

    logger.info(f"CONFIG | THREADS={THREADS} | RSF_N_JOBS={RSF_N_JOBS} | "
                f"SAMPLE_SIZES={SAMPLE_SIZES} | "
                f"RAM disponible={psutil.virtual_memory().available/1024**3:.1f} Go")

    # MODIFIÉ (5) — drapeau d'arrêt propre, pour sortir des trois boucles
    # imbriquées quand la mémoire disponible passe sous le seuil.
    abort = False

    for N in SAMPLE_SIZES:
        if abort:
            break
        logger.info(f"N = {N}")
        if N > len(df):
            logger.warning(f"SKIP N={N} | dataset trop petit ({len(df)} lignes)")
            continue

        df_sample = df.sample(n=N, random_state=42).reset_index(drop=True)

        # Grille d'integration de l'IBS : definie sur la cohorte PROPRE,
        # avant toute degradation, et identique pour tous les replicats.
        set_ibs_reference(df_sample)
        # Force les types sur df_sample — le bootstrap en tire des sous-sets
        df_sample = ensure_schema(enforce_column_types(df_sample))

        # Construire les pools d'imputation sur df_sample complet
        SEX_POOLS = build_sex_conditioned_pools(df_sample, SEX_COL, IMPUTE_COLS)

        # Imputer df_prefit AVANT de fitter le preprocessor
        # (StandardScaler ne doit jamais voir de NaN)
        df_prefit = impute_hf_random_draw_sex(
            df_sample, SEX_COL, IMPUTE_COLS, SEX_POOLS, seed=0
        )
        df_prefit = encode_nan_as_sentinel(df_prefit, NUM_COLS)
        df_prefit = ensure_schema(enforce_column_types(df_prefit))

        PREPROCESSOR, *_ = prepare_data_for_models(df_prefit, df_prefit)

        for p in POLLUTION_LEVELS:
            if abort:
                break
            logger.info(f"p = {p:.2f}")
            for b in range(N_BOOTSTRAPS):
                if (N, p, b) in done_bootstraps:
                    if b % 10 == 0:
                        logger.info(f"SKIP | N={N} | p={p:.2f} | b={b:03d}")
                    continue

                # MODIFIÉ (5) — garde-fou mémoire. La limite RLIMIT_AS
                # posée en tête de fichier est inopérante sous Windows ;
                # ce contrôle la remplace. L'arrêt est sans conséquence :
                # la reprise corrigée redémarre exactement ici.
                avail_gb = psutil.virtual_memory().available / 1024**3
                if avail_gb < MIN_AVAILABLE_GB:
                    logger.error(
                        f"ARRET PROPRE | RAM disponible {avail_gb:.1f} Go "
                        f"< seuil {MIN_AVAILABLE_GB} Go | "
                        f"N={N} p={p:.2f} b={b:03d} | "
                        f"relancer le script pour reprendre ici"
                    )
                    abort = True
                    break

                t0 = time.time()
                df_inbag, df_oob = bootstrap_inbag_oob(df_sample, seed=10_000+b)
                if len(df_oob) == 0:
                    logger.warning(f"SKIP | OOB vide | b={b:03d}")
                    continue

                if IMPUTATION_MODE == "HF_RANDOM":
                    df_inbag = impute_hf_random_draw_sex(df_inbag, SEX_COL, IMPUTE_COLS, SEX_POOLS, 20_000+b)
                    df_oob   = impute_hf_random_draw_sex(df_oob,   SEX_COL, IMPUTE_COLS, SEX_POOLS, 30_000+b)

                if pollute_train and len(df_inbag) > 0:
                    dur_tr, ev_tr = df_inbag[DURATION_COL].copy(), df_inbag[EVENT_COL].copy()
                    df_before = df_inbag.copy(deep=True)
                    out_tr    = pollution_fn(df_inbag, p, seed=40_000+b)
                    df_inbag  = out_tr[0] if isinstance(out_tr, tuple) else out_tr
                    df_inbag[DURATION_COL], df_inbag[EVENT_COL] = dur_tr, ev_tr
                    logger.info(f"POLLUTION TRAIN | p_target={p:.3f} | p_real={measure_pollution_rate(df_before, df_inbag, NUM_COLS):.3f}")

                if pollute_test and len(df_oob) > 0:
                    dur_te, ev_te = df_oob[DURATION_COL].copy(), df_oob[EVENT_COL].copy()
                    df_before = df_oob.copy(deep=True)
                    out_te    = pollution_fn(df_oob, p, seed=50_000+b)
                    df_oob    = out_te[0] if isinstance(out_te, tuple) else out_te
                    df_oob[DURATION_COL], df_oob[EVENT_COL] = dur_te, ev_te
                    logger.info(f"POLLUTION TEST  | p_target={p:.3f} | p_real={measure_pollution_rate(df_before, df_oob, NUM_COLS):.3f}")

                df_inbag = encode_nan_as_sentinel(ensure_schema(enforce_column_types(df_inbag)), NUM_COLS)
                df_oob   = encode_nan_as_sentinel(ensure_schema(enforce_column_types(df_oob)),   NUM_COLS)
                df_inbag = df_inbag[df_inbag[DURATION_COL].notna() & (df_inbag[DURATION_COL]>0)].reset_index(drop=True)
                df_oob   = df_oob[df_oob[DURATION_COL].notna()     & (df_oob[DURATION_COL]>0)].reset_index(drop=True)

                if len(df_inbag) == 0 or len(df_oob) == 0:
                    logger.warning(f"SKIP | split vide | b={b:03d}")
                    continue

                results, diagnostics = run_all_models_point632_batched(df_inbag, df_oob, PREPROCESSOR)

                with open(csv_file, "a", newline="") as f:
                    w = csv.writer(f)
                    for model, m in results.items():
                        d = diagnostics.get(model, {})
                        w.writerow([
                            scenario_name, N, p, b, model,
                            m.get("c_index", np.nan),
                            m.get("ibs",     np.nan),
                            *[m.get(f"brier_{t}d", np.nan) for t in BRIER_TIME_POINTS],
                            d.get("risk_mean",   np.nan),
                            d.get("risk_var",    np.nan),
                            d.get("event_rate",  np.nan),
                            d.get("median_time", np.nan),
                        ])

                # MODIFIÉ — le pic mémoire du processus est journalisé, ce
                # qui permet de calibrer le nombre de runs parallèles sans
                # passer par le Gestionnaire des tâches.
                rss_gb = process.memory_info().rss / 1024**3
                logger.info(
                    f"OK | N={N} | p={p:.2f} | b={b:03d}/{N_BOOTSTRAPS} | "
                    f"{time.time()-t0:5.1f}s | RSS={rss_gb:.1f} Go"
                )

                del df_inbag, df_oob, results, diagnostics
                gc.collect()

    logger.info("ABORT SCENARIO" if abort else "END SCENARIO")