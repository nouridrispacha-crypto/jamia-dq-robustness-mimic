# Data-quality degradation and clinical ML robustness in MIMIC-IV

Code for the paper. Five data-quality dimensions are degraded in a controlled way, under
three scenarios, and the loss is measured on three prediction tasks: in-hospital mortality
(binary), ICU length of stay (continuous) and time to death (censored).

The MIMIC-IV extractions are not redistributable. Rebuild them from MIMIC-IV, then point the
code at them as described below.

## Where things live

```
config.py            every path, resolved once
simulate/            the degradation and fitting pipeline, one directory per outcome
  polluter/            the five degradation mechanisms
  run/                 one entry point per (dimension, scenario)
  standardised/        the binary arm refitted with predictor standardisation
  s2md/                the composite scenario weighted by observed defect rates
analyse/             the mixed model, the paired contrasts and the separability rule
report/              one script per table and per figure of the paper
verify/              recomputes the paper's numbers from the results and checks the text
```

## Paths

Nothing is hard-coded. `config.py` resolves each location from, in order: an environment
variable, a `paths_local.py` beside it, then a default under the repository.

Copy `paths_local.example.py` to `paths_local.py` and fill it in. That file is not
version-controlled, because where the data sits on one machine is of no use on another.

| name | what it points at |
|---|---|
| `DQ_PAPER` | the directory holding `manuscript.tex` and `img/` |
| `DQ_DATA` | the MIMIC-IV extractions |
| `DQ_BINARY_RESULTS`, `DQ_BINARY_RESULTS_RERUN` | binary simulation output |
| `DQ_CONTINUOUS_RESULTS`, `DQ_CONTINUOUS_RESULTS_RERUN` | continuous simulation output |
| `DQ_SURVIVAL_RESULTS` | survival simulation output |
| `DQ_SCALED_INPUTS` | the standardised binary inputs |

Two roots per outcome is not an accident: some dimensions were refitted later, and the file
maps in `analyse/analyse_stats_v2.py` say which come from where.

## Reproducing, in order

1. **Cohorts.** Rebuild the extractions from MIMIC-IV. The survival outcome runs from ICU
   admission to a fixed horizon at day 90, with a 24-hour landmark: a death recorded within
   the horizon is an event, every other patient is censored at day 90, and patients who died
   within the first 24 hours are excluded. The horizon is supportable because the extraction
   ascertains deaths to 90 days after hospital discharge, and discharge never precedes
   admission.

2. **Simulate.** Run the entry points under `simulate/<outcome>/run/`, one per dimension and
   scenario. Each writes one CSV per cell and resumes where it stopped, so interrupting costs
   only the replicate in flight. Thread counts are set inside each runner; with several
   processes in parallel, keep processes x threads near the core count.

   Each runner writes under its own outcome directory, so point the matching
   `DQ_*_RESULTS` at where it wrote. The five degradation mechanisms live once, in
   `simulate/polluter/`, and are shared by the three outcomes: they are identical, and a
   copy per outcome invited the reader to think otherwise.

3. **Standardised binary arm.** `simulate/standardised/build_scaled_inputs.py` then
   `runner_scale.py`. The paper's binary results are this arm.

4. **Analyse.** `analyse/analyse_stats.py` fits
   `loss ~ dimension*model + level + dimension:level + (1|replicate)` on corruption levels
   0.1 to 0.3 and writes the paired contrasts at c=0.3, Holm-corrected within the ten pairs
   of each model.

5. **Report.** The scripts in `report/` write the tables and figures and splice them into the
   manuscript. Each one guards what it must not change: the table scripts refuse to write if
   a cell outside their remit moves.

6. **Verify.** `verify/verify_all.py` recomputes the survival column of Supplementary
   Table S6, the residual skewness of Supplementary Table S5 and every counting claim in the
   Results from the result files, checks that no text from a superseded design survives, and
   confirms each figure is newer than the data it is drawn from.

## The separability rule

A degradation is called most damaging, for a given model, when it is reliably separable from
at least one other (Holm-adjusted p < 0.05 within the ten pairs of that model, and a
2.5th-97.5th percentile interval of the paired differences excluding zero), never on the less
damaging side, and costs more than the mean of the other four. Least damaging is the reverse.
A cell names a degradation with "all" when this holds for every model, otherwise with the
family for which it holds throughout.

The rule is not a ranking by mean loss. A degradation with a larger mean loss can fail to
qualify when its effect varies more across replicates, and two degradations that are not
separable from each other can both qualify.

## Requirements

`pip install -r requirements.txt`. The versions there are the ones the paper ran on.

## Licence

See `LICENSE`.
