# Changelog

## v2.0.0 — code behind the revised manuscript

v1.0.0 was archived on 25 August 2026. Everything below was written after that date, so
the v1.0.0 archive does not reproduce any figure or table of the revised manuscript.

### Standardisation of the binary pipeline

The binary arm fitted LASSO and the MLP on unscaled predictors: its preprocessor used
`("num", "passthrough", num_cols)` where the continuous and survival arms already used
`StandardScaler`. The two models were re-run with the scaler. Clean AUROC moves from
0.7444 to 0.8224 for the MLP and by −0.002 for LASSO; the three tree models are invariant
to a per-column affine map and were not re-run.

Every binary figure, table and statistic in the manuscript now comes from this arm.

- `standardisation/runner_full.py` — the binary grid on both arms
- `standardisation/runner_scale.py`, `build_scaled_inputs.py` — substitute the re-run
  LASSO and MLP rows into the published inputs
- `standardisation/extend_scaled_n1500.py`, `extend_scaled_n7000.py` — the same for the
  two other cohort sizes
- `standardisation/redo_binary_stats.py`, `redo_binary_stats_f1.py` — re-run the mixed
  models and contrasts on the standardised arm

### Scenarios and analyses added since v1.0.0

- **S2-MD, composite degradation** (`standardisation/s2md/`): all five dimensions on the
  same test set, weighted by the defect rates of the eHOP warehouse, uniqueness applied
  last so a duplicated record carries the defects already written into its source.
- **Oracle repair counterfactual** (`s2md/run/run_s2md_repaired.py`): the true value is
  written back into 86.7 % of the cells the completeness mechanism emptied, which takes
  the share of emptied cells from 21.4 % to 2.9 %.
- **Encoding sensitivity** (`standardisation/runner_sens_std.py`): the out-of-domain
  sentinel against leaving the cell missing, and against in-bag median imputation with a
  missingness indicator. Under test-time degradation that indicator is identically zero
  in the in-bag set, so the two arms differ in the substituted value alone.
- **F1-macro grid** (`standardisation/runner_f1.py`): `runner_full` with a wider CSV
  writer. The October re-run recorded only the two co-primary metrics, which left
  Supplementary Figure 1 and three rows of Supplementary Table S5 on superseded data.
- **Bootstrap .632 against .632+** (`standardisation/compare_632plus.py`).
- **Preprocessing fitted in-bag only** (`standardisation/runner_noleak.py`), kept for a
  reviewer response: the published pipeline fits `SEX_POOLS` and the preprocessor once
  per cohort size.

### Figure and table generators

None of these were in v1.0.0. Each produces a numbered figure or table of the manuscript:

| script | output |
| --- | --- |
| `figures/figure1_protocol.py` | Figure 1 |
| `figures/make_s2md_schema.py` | Figure 2 (schematic) |
| `figures/make_print_figures.py` | Figures 2 and 5, Supplementary Figures 1 to 4 |
| `figures/make_supp_heatmaps.py` | Figure 3, Supplementary Figures 6 and 7 |
| `figures/make_s2_grid.py` | Figure 4, Supplementary Figures 2 and 3 grids |
| `figures/make_contrast_figure.py` | Supplementary Figures 8 to 10 |
| `figures/make_contrast_table.py` | Table 3 |
| `figures/make_resid_figures.py` | Supplementary Figure 5 |
| `analysis/analyse_stats_v2.py` | mixed models, contrasts, Tables 3, S5, S6 |
| `analysis/compute_severity_table_S8.py` | Supplementary Table S8 |
| `analysis/make_perf_tables.py` | performance tables |

### Corrections

- `figures/make_print_figures.py` read the composite S2-MD results from the September run
  while Figure 2 and Supplementary Table S8 report the standardised one. The path now
  points at the standardised results; the figure regenerates byte-for-byte identical to
  the published one, which is what confirmed the mismatch was in the path and not in the
  figure.
- `figures/make_contrast_figure.py` left the symmetric-log panels with matplotlib's
  default locator, which labelled every decade and made the R² / MLP axis unreadable at
  print size. The axis now carries zero, the unit decade and the outermost decade only.
- The corruption grid is capped at 0.3: coherence, confined to clinically linked
  variables, reaches at most about 35 % of predictor cells.

### Withdrawn in v2.0.0

The composite scenario was explored with five weight profiles; the manuscript reports only
P4_observe, derived from the eHOP defect rates. The other four -- uniform, documentation,
saisie, integration -- and the sweep script that ran them (run_s2md_mortality.py, with its
--all switch) are withdrawn. PROFILES keeps its single published entry, and
make_mixed_polluter now falls back to W_OBSERVE as the two cumulative polluters already
did. W_OBSERVE is untouched and every published call passes it explicitly, so the
corruption is unchanged: on the same cohort and seeds the composite still empties 21.4% of
cells and alters 30.2%, as before the withdrawal.

Two further S2-MD designs are left out, having produced nothing the manuscript reports:
run_s2md_cumulative.py, the earlier design in which duplication came first, and
run_s2md_remediated.py, the counterfactual of a warehouse whose completeness defect rate
had already fallen to 2%. pollute_cumulative.py stays despite belonging to the first of
them, because the published pollute_cumulative_duplast.py imports its constants.

### Known limitation

Paths are absolute and assume the author's layout (`JAMIA_PAPER`, `JAMIA_V2`, and the
results directories beside them). `analysis/analyse_stats.py` is the exception: it reads
`DQ_RESULTS_ROOT` from the environment. Running the rest elsewhere means editing the
`PAPER = Path(...)` line at the top of each script. The scripts were not rewritten to take
a configurable root for this release, because the figures were verified against these
paths and a mechanical rewrite could not be re-verified end to end.
