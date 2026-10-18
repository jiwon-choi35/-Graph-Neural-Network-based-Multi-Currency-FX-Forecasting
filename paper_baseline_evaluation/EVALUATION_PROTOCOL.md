# Evaluation Protocol Documentation

This document describes the evaluation conditions of the saved baseline results. Source locations and artifact paths refer to the original experimental workspace. No runners, scripts, data, artifacts, or checkpoints are added, and no training, evaluation, or metric recalculation is performed.

## Data, Calendar Split, and Targets

Original data path: `external_team_repo_20260922_newzip/Cyber-trend-forecasting-main 3/B-MTGNN/data/sm_data.csv`. The directory name identifies the input location; it does not add B-MTGNN results to this baseline table.

Frozen SHA-256: `5086946995c5e2da176c17a73b94ebdbdb6c2e75f2a59b5cc5c1d932031132dd`.

The deterministic runner uses the corresponding data under `external_team_repo/Cyber-trend-forecasting-main 3/B-MTGNN/`. Its recorded data hash is identical. Recorded graph SHA-256: `53eb9bfaee6e07e9142fc8aefb2b61b59a906a0365d5d3895b3192c1d4480ea5`. Input data and external code are not included.

| Split | Calendar | 0-based rows | Months |
|---|---|---|---:|
| Train | 2011-01~2023-12 | 0..155 | 156 |
| Validation | 2024-01~2024-12 | 156..167 | 12 |
| Test | 2025-01~2025-12 | 168..179 | 12 |

Targets: US=`us_Trade Weighted Dollar Index`, KR=`kr_fx`, JP=`jp_fx`. Their indices in the numeric matrix excluding Date are US=0, KR=1, JP=2.

## Monthly One-Step Actual-Update

A prediction for month t uses observed history through t-1. After that prediction, the observed value at t becomes available for the next monthly input. Earlier predictions are not fed back. This is a sequence of one-step predictions, not a fixed-origin forecast of all twelve test months.

Naive/Drift append the actual observation after predicting (`run_baseline_robustness_20260923.py:74-88`). Neural inputs use observed-array slices `t-window:t` (`run_neural_baseline_preregistered_strengthening_20260930.py:169-177`). Deterministic MTGNN uses `series[t-24:t]` (`run_mtg_nn_matched.py:608-622`). Predictions do not overwrite the observed input arrays.

## Level Metrics

`RRSE = sqrt(sum((true_level - predicted_level)^2) / sum((true_level - mean(true_level))^2))`.

The denominator uses the evaluated target's twelve-month 2025 mean, not a training mean or a mean across all nodes.

`RMSE = sqrt(mean((true_level - predicted_level)^2))`.

`Max RRSE = max(US RRSE, KR RRSE, JP RRSE)`. Lower is better.

Residual/delta predictions are restored before scoring: `predicted_level[t] = actual_level[t-1] + predicted_delta[t]`. Training delta loss and reported level RRSE are distinct. The previous verifier compared saved scores; it is not included, and no new score is calculated here.

## Input Scope and Final Prediction

| Model | Input scope | Prediction / final output |
|---|---|---|
| Naive-last | target-only | previous actual level |
| Drift | target-only | previous actual level + observed-history average monthly drift |
| LSTM | 33-variable sequence | restored residual level; selected alpha=1; no extra blending |
| causal TCN | 33-variable sequence | restored residual level; selected alpha=1; no extra blending |
| Transformer | 33-variable sequence | restored residual level then alpha=0.5 persistence blending |
| Deterministic MTGNN | graph-33-variable | direct level; single eval forward; no anchor/debias |

Transformer: `final = alpha * restored_neural_level + (1-alpha) * last_actual`. The recorded validation-selected alpha is 0.5. No new blending or optimization is introduced.

The results share monthly one-step evaluation semantics but are not described as identical-input or identical-post-processing comparisons.

## Selection Boundary

Naive/Drift are fixed definitions. Recorded neural selection used 2024 validation Max RRSE for configuration, seed, epoch, and alpha. Deterministic MTGNN selected its epoch using mean validation RRSE across the three targets. These different objectives are preserved.

The protocol excludes test scores from configuration selection. A saved once-only field refers to its specific arm/run, not proof that the project never previously inspected test results. This documentation does not reselect saved results or newly certify independence of the historical research process.
