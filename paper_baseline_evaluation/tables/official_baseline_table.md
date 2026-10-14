# Reported Baseline Results

Existing saved baseline scores for paper baseline/evaluation documentation; lower is better. The six model rows, values, notes, and recorded artifact identifiers are preserved.

Historical artifact paths below are provenance fields, not links to files supplied with this directory. Prediction/metric snapshots, runners, data, and checkpoints are not included. These records do not certify a new experiment or a previously untouched project-wide holdout.

| Model | US RRSE | KR RRSE | JP RRSE | Max RRSE | Prediction artifact | Metric artifact | Notes |
|---|---:|---:|---:|---:|---|---|---|
| Naive-last | 0.642338 | 0.446334 | 0.310120 | 0.642338 | handoff_baselines/artifacts/naive_last/predictions_test_{US,KR,JP}.csv | handoff_baselines/artifacts/naive_last/test_metrics.csv | target-only; fixed previous actual level; no additional blending |
| Drift | 0.651703 | 0.397678 | 0.411072 | 0.651703 | handoff_baselines/artifacts/drift/predictions_test_{US,KR,JP}.csv | handoff_baselines/artifacts/drift/test_metrics.csv | target-only; observed-history average drift; no additional blending |
| LSTM-33var residual_delta | 0.667391 | 0.281353 | 0.394887 | 0.667391 | handoff_baselines/artifacts/lstm_33var/predictions_test_{US,KR,JP}.csv | handoff_baselines/artifacts/lstm_33var/test_metrics.csv | 33-variable; actual-last plus predicted delta; alpha=1; seeds 5/2 mean |
| TCN-33var causal residual_delta | 0.706799 | 0.600214 | 0.323992 | 0.706799 | handoff_baselines/artifacts/tcn_33var_causal/predictions_test_{US,KR,JP}.csv | handoff_baselines/artifacts/tcn_33var_causal/test_metrics.csv | 33-variable causal TCN; actual-last plus delta; alpha=1; seed 5 |
| Transformer-33var residual_delta | 0.697478 | 0.252448 | 0.327576 | 0.697478 | handoff_baselines/artifacts/transformer_33var/predictions_test_{US,KR,JP}.csv | handoff_baselines/artifacts/transformer_33var/test_metrics.csv | 33-variable; restored level then validation-selected alpha=0.5 blending |
| Deterministic MTGNN | 0.517669 | 0.215178 | 0.224434 | 0.517669 | handoff_baselines/artifacts/mtgnn_deterministic/predictions_test_{US,KR,JP}.csv | handoff_baselines/artifacts/mtgnn_deterministic/test_metrics.csv | graph-33-variable; direct level; eval single forward; no anchor/debias; epoch 28 |

Source: `runs/baseline_level_check/final_baseline_comparison_table_20261002.csv`.

Excluded source rows remain unchanged. See [HISTORICAL_EXPERIMENTS.md](../HISTORICAL_EXPERIMENTS.md).
