# Data-Quality Defects Do Not Fail Clinical Machine Learning Equally

Companion code repository for the manuscript submitted to *JAMIA*:

> **Data-Quality Defects Do Not Fail Clinical Machine Learning Equally: A Controlled Dimension-Wise Analysis on MIMIC-IV**

This repository contains the corruption mechanisms, model training pipelines,
and statistical analysis code used to produce every result reported in the
manuscript. It does **not** contain MIMIC-IV data or model outputs — see
[Data availability](#data-availability) below.

## Overview

The study independently degrades five data-quality (DQ) dimensions —
**completeness, coherence, validity, precision, uniqueness** — at six
corruption levels ($c = 0$ to $0.5$), under three scenarios (training-time,
test-time, joint), across three prediction tasks built on MIMIC-IV ICU
admissions:

| Outcome | Folder | Target | Models |
|---|---|---|---|
| Binary (in-hospital mortality) | `binary_outcome/` | `in_hospital_mortality` | LASSO, Decision Tree, Random Forest, XGBoost, MLP |
| Continuous (ICU length of stay) | `continuous_outcome/` | `log_icu_los` | LASSO, Decision Tree, Random Forest, XGBoost, MLP |
| Time-to-event (time to death) | `survival_outcome/` | `time_to_death` / `death_event` | CoxPH, Random Survival Forest, XGBoost-Cox, DeepSurv |

Each outcome folder is self-contained and follows the same layout:

```
<outcome>_outcome/
├── common_runner.py   # shared pipeline: imputation, corruption, bootstrap .632, model training, metrics
├── polluter/           # one script per DQ dimension, plus pollute_mixed.py for the multi-dimension scenario (S4, binary only)
└── run/                # one entry-point script per (dimension, scenario) combination
```

`analysis/analyse_stats.py` implements the revised statistical framework
(Section 3.7 / Evaluation protocol of the manuscript): per-replicate paired
loss relative to the clean baseline, a linear mixed-effects model
(`loss ~ dimension * model + level + dimension:level + (1|replicate)`), and
Wilcoxon signed-rank contrasts with Holm correction, computed separately per
predictive model to avoid pooling non-independent (model, replicate) pairs.

## Reproducing the results

1. **Get the data.** MIMIC-IV requires credentialed PhysioNet access (see
   [Data availability](#data-availability)). Extract the first-24h feature
   set described in Section 3.1 of the manuscript and place it as
   `<outcome>_outcome/data/<expected_filename>.csv` (see the `load_dataset`
   function in each `common_runner.py` for the exact expected filename and
   columns).

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run one (dimension, scenario) combination**, e.g. coherence under
   test-time degradation for the binary outcome:
   ```bash
   cd binary_outcome
   python run/run_coherence_s2.py
   ```
   Each script writes `Results/<Dimension>/outputs_<dimension>_<scenario>/<scenario>.csv`,
   resumable if interrupted (already-completed bootstrap replicates are
   detected and skipped). Running the full study is 5 dimensions × 3
   scenarios × 3 outcomes × 500 bootstrap replicates and is computationally
   substantial (single dimension/scenario/outcome combinations took on the
   order of hours to a day on the development machine).

4. **Run the statistical analysis**, once results exist:
   ```bash
   cd analysis
   export DQ_RESULTS_ROOT=/path/to/where/the/three/Results/folders/live
   python analyse_stats.py
   ```
   Edit the `__main__` block to select which outcome(s)/scenario(s) to
   analyze. Outputs (loss tables, mixed-model summaries, ICC, residual
   diagnostics, Wilcoxon+Holm contrast tables) are written to
   `<outcome>_stats_results/`.

## Scenario S4 (binary outcome only)

`binary_outcome/run/run_s4_mortality.py` runs the multi-dimension,
budget-matched degradation scenario described in Section 3.6 of the
manuscript (five weight profiles, including one derived from defect rates
measured on a real clinical data warehouse). See the script's docstring for
usage (`--smoke`, `--quick`, `--all` flags).

## Data availability

MIMIC-IV is a credentialed-access database; this repository cannot
redistribute it or any data derived from it. Access requires completing
PhysioNet's required human-subjects training and signing the data use
agreement: <https://physionet.org/content/mimiciv/>.

Model outputs (bootstrap-replicate-level CSVs, ~GB scale across the full
study) are likewise not included in this repository.

## Repository notes

- Corruption mechanism parameters (noise scale factors, bias magnitudes,
  sentinel values, etc.) are documented exactly in Supplementary Table S3 of
  the manuscript, extracted directly from the `polluter/` scripts in this
  repository.
- Some `run/` scripts predate the final (dimension, scenario)-suffixed
  naming convention (e.g. `run_coherence.py` alongside
  `run_coherence_s1.py`/`_s2.py`/`_s3.py`) and were superseded during
  development; the manuscript's reported results come from the
  scenario-suffixed scripts. These legacy scripts are kept for transparency
  but are not needed to reproduce the manuscript.
- File paths in `common_runner.py` and `analyse_stats.py` were adapted for
  this public repository (relative paths / `DQ_RESULTS_ROOT` environment
  variable) from the absolute paths used on the development machine; no
  computational logic was changed.

## Citation

If you use this code, please cite the manuscript (full citation to be added
on acceptance) and, if applicable, the archived release of this repository
(DOI via Zenodo, badge to be added).

## License

MIT — see [LICENSE](LICENSE).
