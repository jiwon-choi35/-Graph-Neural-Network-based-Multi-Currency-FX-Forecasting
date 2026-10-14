# Paper Baseline/Evaluation Documentation

This directory documents the evaluation settings, reported baseline results, and baseline result provenance for the FX paper. It describes existing experiments; it does not introduce new training, tuning, evaluation, or checkpoint inference.

## Documented Baselines

- Naive-last
- Drift
- LSTM-33var residual_delta
- TCN-33var causal residual_delta
- Transformer-33var residual_delta
- Deterministic MTGNN

The existing `official_baseline_table` records these six baseline rows. B-MTGNN/Clean B-MTGNN results are not added here. ARIMA/ETS/Ridge and other excluded experiments remain historical references, as described in [HISTORICAL_EXPERIMENTS.md](HISTORICAL_EXPERIMENTS.md).

## Evaluation Protocol

- Train: 2011-01 through 2023-12.
- Validation: 2024-01 through 2024-12.
- Test: 2025-01 through 2025-12.
- Monthly one-step actual-update: a prediction at month t uses observations through t-1, not earlier predictions.
- Residual/delta outputs are restored to target levels before reporting per-target RRSE and Max RRSE.
- Neural configuration selection used 2024 validation Max RRSE; deterministic MTGNN epoch selection used mean validation RRSE.
- Transformer results include the recorded validation-selected alpha=0.5 persistence blend.

The calendar split and one-step evaluation semantics are shared. Input scopes, prediction parameterizations, post-processing, and selection objectives differ by model. The documentation does not claim identical inputs, identical post-processing, or a newly untouched project-wide holdout.

## Reading Order

1. [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md): calendar split, information boundary, level metrics, and selection rules.
2. [MODEL_AUDIT.md](MODEL_AUDIT.md): model-specific settings, source locations, and artifact references.
3. [BASELINE_RESULTS.md](BASELINE_RESULTS.md) and the [reported results table](tables/official_baseline_table.md): saved scores and provenance.
4. [HISTORICAL_EXPERIMENTS.md](HISTORICAL_EXPERIMENTS.md): historical experiments and excluded rows.
5. [REPRODUCE.md](REPRODUCE.md): original runner references and reproducibility boundaries.
6. [Existing verification record](manifests/official_artifact_verification.json): the previous local artifact inspection.

## Scope and Provenance

Only documentation, saved tables, and existing verification/manifests are included. Runners, verification scripts, prediction/metric/config snapshots, raw data, checkpoints, and external repository copies are not added. Original runner paths are documented rather than relocated.

CSV/JSON values, reported scores, and manifest hashes are preserved. Historical artifact paths and labels remain verbatim provenance fields, not links to files supplied with this directory. The manifest's `included_in_git=yes` field describes an earlier inclusion plan, not the current Git file list. Recorded document hashes describe earlier snapshots and are not regenerated after wording or directory changes.

## Reproducibility Boundaries

**This is not a standalone reproducibility package.** Reproduction requires the original experimental workspace, frozen inputs, dependencies, and any available selected weights. Neural selected weights were not persisted. The existing verification report's PASS describes a previous local inspection; it does not certify a new experiment or standalone execution from these documents.

No saved test results are used to reselect models or settings during documentation maintenance. Code authorship is not inferred from filenames or filesystem metadata.

## Data and Code Permissions

The original prediction files contain `y_true`. Data redistribution and external-code license permissions must be checked before those files or dependencies are shared. They are not included here, and this documentation does not authorize public release.
