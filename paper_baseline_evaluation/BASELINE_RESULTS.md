# Reported Baseline Results and Provenance

Original source table: `runs/baseline_level_check/final_baseline_comparison_table_20261002.csv`.

The reported baseline table is available as [Markdown](tables/official_baseline_table.md) and [CSV](tables/official_baseline_table.csv). The source comparison table is not modified.

Saved scores are preserved. Prediction/metric/config snapshots, source runners, and the source comparison table are not added. Artifact paths identify original local snapshots. The previous verification PASS describes an earlier local inspection, not standalone reproducibility from these documents.

| Model | US RRSE | KR RRSE | JP RRSE | Max RRSE | Prediction artifact | Metric artifact | Notes |
|---|---:|---:|---:|---:|---|---|---|
| Naive-last | 0.642338 | 0.446334 | 0.310120 | 0.642338 | artifacts/naive_last/predictions_test_{US,KR,JP}.csv | artifacts/naive_last/test_metrics.csv | target-only; fixed |
| Drift | 0.651703 | 0.397678 | 0.411072 | 0.651703 | artifacts/drift/predictions_test_{US,KR,JP}.csv | artifacts/drift/test_metrics.csv | target-only; observed drift |
| LSTM-33var residual_delta | 0.667391 | 0.281353 | 0.394887 | 0.667391 | artifacts/lstm_33var/predictions_test_{US,KR,JP}.csv | artifacts/lstm_33var/test_metrics.csv | seeds 5/2 mean; alpha=1 |
| TCN-33var causal residual_delta | 0.706799 | 0.600214 | 0.323992 | 0.706799 | artifacts/tcn_33var_causal/predictions_test_{US,KR,JP}.csv | artifacts/tcn_33var_causal/test_metrics.csv | causal TCN; alpha=1 |
| Transformer-33var residual_delta | 0.697478 | 0.252448 | 0.327576 | 0.697478 | artifacts/transformer_33var/predictions_test_{US,KR,JP}.csv | artifacts/transformer_33var/test_metrics.csv | selected alpha=0.5 |
| Deterministic MTGNN | 0.517669 | 0.215178 | 0.224434 | 0.517669 | artifacts/mtgnn_deterministic/predictions_test_{US,KR,JP}.csv | artifacts/mtgnn_deterministic/test_metrics.csv | raw direct level; epoch 28 |

Values display saved metric scores to six decimal places. They are not new experimental results.

The deterministic MTGNN row is supported by recorded references to its original runner, selected checkpoint, graph/data, and saved outputs. Weights and external dependencies are not supplied, so this does not establish immediate inference reproducibility in a new environment. See [REPRODUCE.md](REPRODUCE.md).

Source mapping and hashes remain in the existing manifests. Their inclusion labels and document hashes describe earlier snapshots, not the current file list or revised document bytes. ARIMA/ETS/Ridge and B-MTGNN/Clean B-MTGNN are outside this table's scope; see [HISTORICAL_EXPERIMENTS.md](HISTORICAL_EXPERIMENTS.md). No authorship attribution is made.
