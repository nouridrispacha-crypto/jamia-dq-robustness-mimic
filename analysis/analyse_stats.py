# -*- coding: utf-8 -*-
"""
Analyse statistique revisee -- reconstruction de analyse_stats.py (introuvable
sur disque, jamais commite). Specification reconstituee et verifiee par le
compte exact d'observations/effets fixes donne par l'utilisateur (outcome
binaire) :

  perte ~ dimension*model + level + dimension:level + (1|replicate)

- Ajuste sur la PERTE intra-replicat (degradation appariee par rapport a
  c=0 pour le meme replicat et le meme modele), pas sur la valeur brute.
  A c=0 les 5 dimensions partagent exactement la meme valeur pour un
  replicat donne (aucune corruption n'est appliquee) : les inclure comme
  5 lignes distinctes gonflerait artificiellement l'effectif et, ajuste sur
  la valeur brute, l'intercept aleatoire s'estime a zero (le modele degenere
  en OLS). Le modele est donc ajuste UNIQUEMENT sur les niveaux 0.1-0.5.
- dimension : reference = precision (dimension la moins dommageable dans
  toutes les analyses descriptives -> reference naturelle).
- Groupes (effet aleatoire) : replicat bootstrap (b) SEUL.
- Contrastes : differences appariees a c=0.5 entre les 10 paires de
  dimensions (C(5,2)), test de Wilcoxon signe, correction de Holm sur les
  10 paires de chaque combinaison outcome x scenario x metrique. La perte
  est d'abord MOYENNEE SUR LES MODELES pour chaque replicat avant
  appariement : les modeles partagent le meme split in-bag/OOB pour un
  replicat donne (bootstrap_inbag_oob est seede par b seul, independamment
  du modele), donc traiter (modele, replicat) comme des paires
  independantes reintroduirait la pseudo-replication que la construction
  de la perte visait justement a eliminer. Les 500 replicats restent
  l'unique unite reellement independante, cohente avec (1|replicate).

Verification de la reconstruction (outcome binaire, exemple donne par
l'utilisateur, 5 modeles) : 5 dimensions x 5 modeles x 5 niveaux (0.1-0.5)
x 500 replicats = 62 500 observations ; 4 (dimension) + 4 (model) + 16
(dimension:model) + 4 (level) + 16 (dimension:level) = 45 effets fixes.
Les deux correspondent exactement aux chiffres donnes -> specification
consideree comme correctement reconstituee.

USAGE
-----
Modifier OUTCOME et SCENARIO en bas du fichier, ou appeler run_metric()
directement avec une config personnalisee. Trois configs pretes a l'emploi
sont definies ci-dessous : BINARY, CONTINUOUS, SURVIVAL.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, shapiro, skew
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as spstats

try:
    import statsmodels.formula.api as smf
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False

warnings.filterwarnings("ignore")

REFERENCE_DIMENSION = "precision"
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
REFERENCE_LEVEL = "0.1"

# MODIFIÉ (dépôt public) — les CSV de résultats (Results/) ne sont pas
# redistribués dans ce dépôt (données dérivées de MIMIC-IV, volumineuses).
# Reproduire localement : lancer les pipelines binary_outcome/,
# continuous_outcome/, survival_outcome/ (voir README), puis pointer
# DQ_RESULTS_ROOT vers le dossier contenant les trois sous-dossiers de
# resultats (par defaut : ./results, a cote de ce script).
RESULTS_ROOT = Path(os.environ.get("DQ_RESULTS_ROOT", "./results"))


# ============================================================
# CONFIGS PAR OUTCOME
# ============================================================

@dataclass
class OutcomeConfig:
    name: str                      # "binary" | "continuous" | "survival"
    n_fixed: int
    file_map: dict                 # dimension -> {scenario: Path}
    models: list                   # modeles mentionnes dans le manuscrit
    metrics: dict                  # metric_name -> (colonne CSV, direction)
    level_col: str = "p"           # nom de la colonne de niveau dans le CSV


def _survival_file_map():
    base = RESULTS_ROOT / "survival_outcome"
    return {
        "completeness": {
            "S1": base / "mnar_top" / "outputs_mnar_top_s1" / "S1_train_only.csv",
            "S2": base / "mnar_top" / "outputs_mnar_top_s2" / "S2_test_only.csv",
            "S3": base / "mnar_top" / "outputs_mnar_top_s3" / "S3_both_independent.csv",
        },
        "coherence": {
            "S1": base / "Coherence" / "outputs_coherence_s1" / "S1_train_only.csv",
            "S2": base / "Coherence" / "outputs_coherence_s2" / "S2_test_only.csv",
            "S3": base / "Coherence" / "outputs_coherence_s3" / "S3_train_test.csv",
        },
        "validity": {
            "S1": base / "validity" / "outputs_validity_S1" / "S1_train_only.csv",
            "S2": base / "validity" / "outputs_validity_S2" / "S2_test_only.csv",
            "S3": base / "validity" / "outputs_validity_S3" / "S3_both_independent.csv",
        },
        "precision": {
            "S1": base / "precision" / "outputs_precision_S1" / "S1_train_only.csv",
            "S2": base / "precision" / "outputs_precision_S2" / "S2_test_only.csv",
            "S3": base / "precision" / "outputs_precision_S3" / "S3_both_independent.csv",
        },
        "uniqueness": {
            "S1": base / "uniqueness" / "outputs_uniqueness_S1" / "S1_train_only.csv",
            "S2": base / "uniqueness" / "outputs_uniqueness_S2" / "S2_test_only.csv",
            "S3": base / "uniqueness" / "outputs_uniqueness_S3" / "S3_both_independent.csv",
        },
    }


def _binary_file_map():
    base = RESULTS_ROOT / "binary_outcome"
    return {
        "completeness": {
            "S1": base / "mnar_top" / "outputs_mnar_top_s1" / "S1_train_only.csv",
            "S2": base / "mnar_top" / "outputs_mnar_top_s2" / "S2_test_only.csv",
            "S3": base / "mnar_top" / "outputs_mnar_top_s3" / "S3_both_independent.csv",
        },
        "coherence": {
            "S1": base / "Coherence" / "outputs_coherence_s1" / "S1_train_only.csv",
            "S2": base / "Coherence" / "outputs_coherence_s2" / "S2_test_only.csv",
            "S3": base / "Coherence" / "outputs_coherence_s3" / "S3_train_test.csv",
        },
        "validity": {
            "S1": base / "validity" / "outputs_validity_S1" / "S1_train_only.csv",
            "S2": base / "validity" / "outputs_validity_S2" / "S2_test_only.csv",
            "S3": base / "validity" / "outputs_validity_S3" / "S3_both_independent.csv",
        },
        "precision": {
            "S1": base / "precision" / "outputs_precision_S1" / "S1_train_only.csv",
            "S2": base / "precision" / "outputs_precision_S2" / "S2_test_only.csv",
            "S3": base / "precision" / "outputs_precision_S3" / "S3_both_independent.csv",
        },
        "uniqueness": {
            "S1": base / "uniqueness" / "outputs_uniqueness_S1" / "S1_train_only.csv",
            "S2": base / "uniqueness" / "outputs_uniqueness_S2" / "S2_test_only.csv",
            "S3": base / "uniqueness" / "outputs_uniqueness_S3" / "S3_both_independent.csv",
        },
    }


def _continuous_file_map():
    base = RESULTS_ROOT / "continuous_outcome"
    return {
        "completeness": {
            "S1": base / "mnar_top" / "outputs_mnar_top_s1" / "S1_train_only.csv",
            "S2": base / "mnar_top" / "outputs_mnar_top_s2" / "S2_test_only.csv",
            "S3": base / "mnar_top" / "outputs_mnar_top_s3" / "S3_both_independent.csv",
        },
        "coherence": {
            "S1": base / "Coherence" / "outputs_coherence_s1" / "S1_train_only.csv",
            "S2": base / "Coherence" / "outputs_coherence_s2" / "S2_test_only.csv",
            "S3": base / "Coherence" / "outputs_coherence_s3" / "S3_train_test.csv",
        },
        "validity": {
            "S1": base / "validity" / "outputs_validity_S1" / "S1_train_only.csv",
            "S2": base / "validity" / "outputs_validity_S2" / "S2_test_only.csv",
            "S3": base / "validity" / "outputs_validity_S3" / "S3_both_independent.csv",
        },
        "precision": {
            "S1": base / "precision" / "outputs_precision_S1" / "S1_train_only.csv",
            "S2": base / "precision" / "outputs_precision_S2" / "S2_test_only.csv",
            "S3": base / "precision" / "outputs_precision_S3" / "S3_both_independent.csv",
        },
        "uniqueness": {
            "S1": base / "uniqueness" / "outputs_uniqueness_S1" / "S1_train_only.csv",
            "S2": base / "uniqueness" / "outputs_uniqueness_S2" / "S2_test_only.csv",
            "S3": base / "uniqueness" / "outputs_uniqueness_S3" / "S3_both_independent.csv",
        },
    }


SURVIVAL = OutcomeConfig(
    name="survival",
    n_fixed=3500,
    file_map=_survival_file_map(),
    models=["CoxPH", "RandomSurvivalForest", "XGBoostSurv", "DeepSurv1"],
    metrics={
        "c_index": ("c_index", "higher_better"),
        "ibs": ("ibs", "lower_better"),
    },
    level_col="p",
)

BINARY = OutcomeConfig(
    name="binary",
    n_fixed=3500,
    file_map=_binary_file_map(),
    models=["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"],
    metrics={
        "auc": ("auc", "higher_better"),
        "brier": ("brier", "lower_better"),
        "f1_macro": ("f1_macro", "higher_better"),
    },
    level_col="pollution",  # colonne de niveau nommee differemment dans ce runner
)

# NOTE : n_fixed=3500 seulement -- a n'utiliser que lorsque les reruns
# StandardScaler seront a 500/500 sur les 15 combinaisons (dimension x
# scenario). Statut au moment de l'ecriture de ce script : incomplet
# (profondeurs heterogenes, voir conversation). NE PAS LANCER sur ICIMTH
# tant que ce n'est pas confirme complet.
CONTINUOUS = OutcomeConfig(
    name="continuous",
    n_fixed=3500,
    file_map=_continuous_file_map(),
    models=["LASSO", "DecisionTree", "RandomForest", "XGBoost", "MLP1"],
    metrics={
        "rmse": ("rmse", "lower_better"),
        "r2": ("r2", "higher_better"),
    },
    level_col="p",
)


# ============================================================
# CHARGEMENT ET CONSTRUCTION DE LA PERTE
# ============================================================

def load_dimension(cfg: OutcomeConfig, dim: str, scenario: str) -> pd.DataFrame:
    df = pd.read_csv(cfg.file_map[dim][scenario])
    df = df[(df["N"] == cfg.n_fixed) & (df["model"].isin(cfg.models))].copy()
    df["dimension"] = dim
    return df


def build_loss_table(cfg: OutcomeConfig, scenario: str, metric_col: str, direction: str) -> pd.DataFrame:
    """
    Table longue [dimension, model, level, replicate, loss], restreinte a
    p in {0.1,...,0.5}. loss est positif quand la performance se degrade,
    quel que soit le sens naturel de la metrique.
    """
    rows = []
    for dim in cfg.file_map:
        df = load_dimension(cfg, dim, scenario)
        # Nom de colonne du niveau de corruption incoherent entre scenarios
        # pour l'outcome binaire (S1/S2: "pollution", S3: "p") -- detecte
        # au lieu de fixer une seule fois par config.
        lvl = "pollution" if "pollution" in df.columns else "p"

        base = df[df[lvl] == 0.0][["model", "bootstrap", metric_col]].rename(
            columns={metric_col: "baseline"}
        )
        sub = df[df[lvl].round(2).isin(LEVELS)][["model", "bootstrap", lvl, metric_col]]

        merged = sub.merge(base, on=["model", "bootstrap"], how="inner")
        if direction == "higher_better":
            merged["loss"] = merged["baseline"] - merged[metric_col]
        else:
            merged["loss"] = merged[metric_col] - merged["baseline"]

        merged["dimension"] = dim
        merged["level"] = merged[lvl].round(1).astype(str)
        merged["replicate"] = merged["bootstrap"]
        rows.append(merged[["dimension", "model", "level", "replicate", "loss"]])

    out = pd.concat(rows, ignore_index=True)
    out["dimension"] = pd.Categorical(
        out["dimension"],
        categories=[REFERENCE_DIMENSION] + [d for d in cfg.file_map if d != REFERENCE_DIMENSION],
    )
    out["level"] = pd.Categorical(
        out["level"],
        categories=[REFERENCE_LEVEL] + [f"{l:.1f}" for l in LEVELS if f"{l:.1f}" != REFERENCE_LEVEL],
    )
    return out


# ============================================================
# MODELE MIXTE
# ============================================================

def fit_mixed_model(loss_df: pd.DataFrame):
    if not HAVE_STATSMODELS:
        return None

    formula = (
        "loss ~ C(dimension, Treatment(reference='%s')) * C(model) "
        "+ C(level, Treatment(reference='%s')) "
        "+ C(dimension, Treatment(reference='%s')):C(level, Treatment(reference='%s'))"
    ) % (REFERENCE_DIMENSION, REFERENCE_LEVEL, REFERENCE_DIMENSION, REFERENCE_LEVEL)

    model = smf.mixedlm(formula, data=loss_df, groups=loss_df["replicate"])
    result = model.fit(reml=True)
    return result


def report_icc(result) -> float:
    var_re = float(result.cov_re.iloc[0, 0])
    var_resid = float(result.scale)
    return var_re / (var_re + var_resid)


def residual_diagnostics(result, cfg_name: str, scenario: str, metric_name: str, out_dir: Path):
    """
    Diagnostic des residus du modele mixte (demande co-auteur 3.2) :
    QQ-plot + skewness + Shapiro-Wilk (sur un sous-echantillon, le test
    etant sur-puissant au-dela de ~5000 observations).

    Les residus d'un modele mixte lineaire n'ont pas besoin d'etre
    gaussiens pour que les effets fixes restent des estimateurs sans biais
    (theoreme de Gauss-Markov generalise) ; la normalite conditionne
    l'exactitude des intervalles de confiance et des p-values sous
    l'hypothese de vraisemblance gaussienne. Un ecart notable justifie de
    lire les p-values du modele mixte comme indicatives plutot
    qu'exactes -- raison de plus pour que les contrastes Wilcoxon (non
    parametriques) restent le support principal des comparaisons par
    paires rapportees dans le texte.
    """
    resid = np.asarray(result.resid)
    resid = resid[np.isfinite(resid)]

    rng = np.random.default_rng(0)
    sample = resid if len(resid) <= 5000 else rng.choice(resid, size=5000, replace=False)
    try:
        sw_stat, sw_p = shapiro(sample)
    except Exception:
        sw_stat, sw_p = np.nan, np.nan
    sk = float(skew(resid))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    spstats.probplot(resid, dist="norm", plot=axes[0])
    axes[0].set_title("QQ-plot (residus)")
    axes[1].hist(resid, bins=60, color="#33506b", alpha=0.85)
    axes[1].set_title("Histogramme des residus")
    axes[1].set_xlabel("Residu")
    fig.suptitle(f"{cfg_name} | {scenario} | {metric_name} -- skew={sk:.2f}, "
                 f"Shapiro-Wilk p={sw_p:.2e} (n={len(sample)})", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / f"resid_{cfg_name}_{scenario}_{metric_name}.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {"skew": sk, "shapiro_stat": float(sw_stat), "shapiro_p": float(sw_p),
            "n_resid": len(resid), "png": str(out_path)}


# ============================================================
# CONTRASTES -- Wilcoxon signe + Holm, a c=0.5, sur les 10 paires
# ============================================================

def holm_correction(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def pairwise_contrasts(cfg: OutcomeConfig, loss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Paire = replicat SEUL (n=500) au sein d'UN MEME MODELE -- pas de
    moyenne sur les modeles.

    MODIFIE -- version precedente moyennait la perte sur les modeles avant
    d'apparier, dans l'idee d'obtenir 500 unites independantes (les
    modeles partageant le meme split in-bag/OOB pour un replicat donne).
    Mais moyenner ecrase les differences reelles entre architectures : la
    verification croisee contre le modele mixte a montre que le classement
    des dimensions differe selon le modele (ex. completeness domine pour
    CoxPH seul, alors que la version moyennee donnait coherence/validity/
    completeness quasi ex-aequo). Le modele mixte lui-meme ne suppose
    jamais un effet dimension independant du modele (interaction
    dimension:model complete) -- moyenner etait donc moins fidele a la
    structure du modele mixte que de garder les modeles separes.

    Les 10 paires sont donc testees SEPAREMENT pour chaque modele, chacune
    sur ses 500 replicats (independants entre eux, puisqu'un seul modele
    est implique -- plus de partage de split a corriger). Holm est
    applique sur les 10 paires DE CHAQUE MODELE separement (la famille de
    comparaisons reste "10 paires pour une analyse donnee", desormais
    outcome x scenario x metrique x modele plutot que sans le modele).
    """
    c5 = loss_df[loss_df["level"] == "0.5"].copy()
    dims = list(cfg.file_map.keys())

    all_rows = []
    for model_name in cfg.models:
        sub = c5[c5["model"] == model_name]
        pivot = sub.pivot_table(index="replicate", columns="dimension", values="loss")

        rows = []
        for d1, d2 in combinations(dims, 2):
            # PROVISOIRE (donnees incompletes, ex. ICIMTH en cours) : une
            # dimension peut n'avoir encore aucune ligne a c=0.5 pour ce
            # modele -- on saute la paire plutot que de planter, et on le
            # signale via n_pairs=0 au lieu de laisser une KeyError.
            if d1 not in pivot.columns or d2 not in pivot.columns:
                rows.append({
                    "model": model_name, "dim_A": d1, "dim_B": d2, "n_pairs": 0,
                    "median_diff": np.nan, "pct2_5": np.nan, "pct97_5": np.nan,
                    "wilcoxon_stat": np.nan, "p_raw": np.nan,
                })
                continue

            paired = pivot[[d1, d2]].dropna()
            if len(paired) < 2:
                rows.append({
                    "model": model_name, "dim_A": d1, "dim_B": d2, "n_pairs": len(paired),
                    "median_diff": np.nan, "pct2_5": np.nan, "pct97_5": np.nan,
                    "wilcoxon_stat": np.nan, "p_raw": np.nan,
                })
                continue

            diff = paired[d1] - paired[d2]
            try:
                stat, p = wilcoxon(diff)
            except ValueError:
                stat, p = np.nan, np.nan
            rows.append({
                "model": model_name, "dim_A": d1, "dim_B": d2, "n_pairs": len(diff),
                "median_diff": float(diff.median()),
                "pct2_5": float(diff.quantile(0.025)),
                "pct97_5": float(diff.quantile(0.975)),
                "wilcoxon_stat": stat, "p_raw": p,
            })

        block = pd.DataFrame(rows)
        valid = block["p_raw"].notna()
        block.loc[valid, "p_holm"] = holm_correction(block.loc[valid, "p_raw"].tolist())
        block["p_holm"] = block.get("p_holm", np.nan)
        block["ci_excludes_zero"] = (block["pct2_5"] > 0) | (block["pct97_5"] < 0)
        all_rows.append(block)

    return pd.concat(all_rows, ignore_index=True).sort_values(["model", "p_holm"])


# ============================================================
# RUN
# ============================================================

def run_metric(cfg: OutcomeConfig, scenario: str, metric_name: str, out_dir: Path):
    metric_col, direction = cfg.metrics[metric_name]
    print(f"\n{'=' * 70}\n{cfg.name} | {scenario} | {metric_name} ({direction})\n{'=' * 70}")

    loss_df = build_loss_table(cfg, scenario, metric_col, direction)
    n_obs = len(loss_df)
    n_expected = len(cfg.file_map) * len(cfg.models) * len(LEVELS) * loss_df["replicate"].nunique()
    print(f"Observations : {n_obs} (attendu si complet : {n_expected})")
    if n_obs != n_expected:
        print("ATTENTION : effectif incomplet -- verifier la couverture bootstrap "
              "avant d'interpreter (un ou plusieurs modeles/dimensions manquants "
              "pour certains replicats).")

    out_dir.mkdir(parents=True, exist_ok=True)
    loss_df.to_csv(out_dir / f"loss_{cfg.name}_{scenario}_{metric_name}.csv", index=False)

    if HAVE_STATSMODELS:
        try:
            result = fit_mixed_model(loss_df)
            icc = report_icc(result)
            print(f"Modele mixte converge. ICC = {icc:.1%}")
            with open(out_dir / f"mixedlm_{cfg.name}_{scenario}_{metric_name}.txt", "w") as f:
                f.write(result.summary().as_text())
                f.write(f"\n\nICC = {icc:.4f}\n")

            diag = residual_diagnostics(result, cfg.name, scenario, metric_name, out_dir)
            print(f"Diagnostic residus : skew={diag['skew']:.2f} | "
                  f"Shapiro-Wilk p={diag['shapiro_p']:.2e} (n={diag['n_resid']})")
            pd.DataFrame([{**{"outcome": cfg.name, "scenario": scenario, "metric": metric_name}, **diag}]) \
                .to_csv(out_dir / f"residdiag_{cfg.name}_{scenario}_{metric_name}.csv", index=False)
        except Exception as exc:
            print(f"ECHEC du modele mixte : {type(exc).__name__}: {exc}")
    else:
        print("statsmodels non installe -- modele mixte SAUTE.")

    contrasts = pairwise_contrasts(cfg, loss_df)
    contrasts.to_csv(out_dir / f"contrasts_{cfg.name}_{scenario}_{metric_name}.csv", index=False)

    print(f"\n{contrasts.to_string(index=False)}\n")
    for model_name in cfg.models:
        block = contrasts[contrasts["model"] == model_name]
        n_sig = (block["p_holm"] < 0.05).sum()
        n_sig_ci0 = ((block["p_holm"] < 0.05) & ~block["ci_excludes_zero"]).sum()
        print(f"  {model_name:22s} : {n_sig}/{len(block)} paires significatives (Holm<0.05), "
              f"dont {n_sig_ci0} avec un intervalle [2.5%,97.5%] qui contient zero.")

    return loss_df, contrasts


if __name__ == "__main__":
    # ------------------------------------------------------------
    # BINARY et SURVIVAL : 500/500 replicats confirmes complets sur les 15
    # combinaisons dimension x scenario pour ces deux outcomes (verifie
    # avant execution, y compris apres dedoublonnage de deux fichiers
    # survie -- voir conversation).
    #
    # CONTINUOUS (ICIMTH) : PROVISOIRE -- rerun StandardScaler encore en
    # cours au moment de l'execution (25%-100% des replicats selon la
    # combinaison). Lance quand meme a la demande explicite de
    # l'utilisateur pour produire une version rapide a envoyer au
    # co-auteur ; le controle d'effectif (n_obs vs n_expected) declenche
    # automatiquement un avertissement ATTENTION pour chaque combinaison
    # incomplete. NE PAS CITER CES CHIFFRES COMME DEFINITIFS.
    # ------------------------------------------------------------
    for CFG in [BINARY, SURVIVAL, CONTINUOUS]:
        OUT_DIR = Path(f"./{CFG.name}_stats_results")
        for SCENARIO in ["S1", "S2", "S3"]:
            for metric_name in CFG.metrics:
                run_metric(CFG, SCENARIO, metric_name, OUT_DIR)
