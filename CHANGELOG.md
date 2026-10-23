# Changelog

## v3.0.0 — 2026-10-23

Rebuilt from v2.0.0, keeping only what regenerates the paper and making it runnable
elsewhere.

### Survival outcome redesigned

The outcome ran from ICU admission with follow-up ending at hospital discharge. Patients
leave hospital when they improve, so the censoring time carried prognosis and the
independent-censoring assumption behind the Cox model, the C-index and the inverse
probability of censoring weights did not hold.

It now runs to a fixed horizon at day 90 from ICU admission, with a 24-hour landmark.
Censoring is administrative for everyone not recorded as dying, which removes the dependence
rather than bounding it. The cohort is 9,768 patients with 1,613 events (16.5%), against
9,998 and 1,894 before; 230 patients who died within the first day are excluded.

Two defects of the old pipeline surfaced and are fixed. The subject mask of the Integrated
Brier Score tested `time < max`, which under a fixed horizon discards every patient alive at
it — 8,155 of 9,768 — leaving the score computed on the deaths alone; it is now a non-strict
inequality. And day 90 left the Brier time points, since it now sits outside the follow-up.

Consequences for the paper: the C-index results hold, with the same dimensions named and the
same ordering below the top two. The Integrated Brier Score results do not, which is expected
since it passes through the censoring distribution and the integration grid while the C-index
depends only on the ordering of event times. Coherence, whose only lead was the survival
Integrated Brier Score among tree-based models, now leads nowhere.

### Duplicates removed

The five degradation mechanisms were byte-identical in all five places they appeared, so
twenty of the twenty-five files were copies. They now live once, in `simulate/polluter/`,
shared by the three outcomes. The duplication was not only waste: five polluter directories
invited the reader to believe the mechanisms differ per outcome, and they do not.

`simulate/standardised/common_runner.py` was byte-identical to the binary one. It is the raw
pipeline that `runner_scale` compares its scaled arm against, which is right, but it did not
need to be a second copy.

`make_perf_tables.py` wrote "Supplementary Tables S9–S17", a series the supplement no longer
has and the manuscript never references.

Each of the 48 scripts that import a mechanism was redirected and then imported for real, to
prove the import resolves rather than assume it. 115 scripts become 93.

### Repository pruned

Thirteen one-off repair scripts removed. They recorded how results were patched into shape
and made the pipeline look more complicated than it is.

Absolute paths removed from all 24 files that carried them. `config.py` resolves every
location from an environment variable, then a `paths_local.py` that is not
version-controlled, then a default under the repository. The repository could not be run by
anyone else before this.

No cohort builder is included: the extraction is rebuilt from MIMIC-IV by the reader.

Filenames are unchanged. Five scripts import each other by module name, and renaming for
appearance is not worth breaking the code that produces the paper's numbers.

## v2.0.0

Archived release accompanying the submitted manuscript.
