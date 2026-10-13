# Baseline results

원천 표: runs/baseline_level_check/final_baseline_comparison_table_20261002.csv. 전달표는 [tables/official_baseline_table.md](tables/official_baseline_table.md)와 [CSV](tables/official_baseline_table.csv)이다. 원천 final table은 수정하지 않았다.

1차 PR에서는 아래 표의 숫자를 그대로 보존하며 source comparison table, prediction/metric/config snapshot 및 runner 소스는 추가하지 않는다. Artifact 경로는 원 로컬 패키지의 참조이다. 기존 verification report의 PASS는 당시 로컬 artifact 검사 결과이며 이번 PR만으로 재현 가능하다는 뜻이 아니다.

| Model | US RRSE | KR RRSE | JP RRSE | Max RRSE | Prediction artifact | Metric artifact | Notes |
|---|---:|---:|---:|---:|---|---|---|
| Naive-last | 0.642338 | 0.446334 | 0.310120 | 0.642338 | artifacts/naive_last/predictions_test_{US,KR,JP}.csv | artifacts/naive_last/test_metrics.csv | target-only; fixed |
| Drift | 0.651703 | 0.397678 | 0.411072 | 0.651703 | artifacts/drift/predictions_test_{US,KR,JP}.csv | artifacts/drift/test_metrics.csv | target-only; observed drift |
| LSTM-33var residual_delta | 0.667391 | 0.281353 | 0.394887 | 0.667391 | artifacts/lstm_33var/predictions_test_{US,KR,JP}.csv | artifacts/lstm_33var/test_metrics.csv | seeds 5/2 mean; alpha=1 |
| TCN-33var causal residual_delta | 0.706799 | 0.600214 | 0.323992 | 0.706799 | artifacts/tcn_33var_causal/predictions_test_{US,KR,JP}.csv | artifacts/tcn_33var_causal/test_metrics.csv | causal TCN; alpha=1 |
| Transformer-33var residual_delta | 0.697478 | 0.252448 | 0.327576 | 0.697478 | artifacts/transformer_33var/predictions_test_{US,KR,JP}.csv | artifacts/transformer_33var/test_metrics.csv | selected alpha=0.5 |
| Deterministic MTGNN | 0.517669 | 0.215178 | 0.224434 | 0.517669 | artifacts/mtgnn_deterministic/predictions_test_{US,KR,JP}.csv | artifacts/mtgnn_deterministic/test_metrics.csv | raw direct level; epoch 28 |

위 값은 원 metric artifact를 소수 여섯 자리로 표시한 값이다. 새로운 실험 결과로 간주하지 않는다.

Deterministic MTGNN은 원 runner, selected checkpoint, graph/data reference와 저장 결과가 로컬에 존재해 포함한다. 패키지에는 weight와 외부 repo가 없으므로 독립적인 새 환경에서 곧바로 모형 추론을 재현할 수 있다는 뜻은 아니다. 기존 artifact 검증과 full training 재현의 요구사항은 [REPRODUCE.md](REPRODUCE.md)를 참조한다.

원 source mapping 및 hash는 manifests/official_artifacts_manifest.csv와 input_source_hashes.csv의 과거 기록을 참조한다. Manifest의 included_in_git 표시와 문서 hash는 당시 snapshot 기준이며 이번 PR의 포함 목록이나 갱신된 문서 hash를 대신하지 않는다. ARIMA/ETS/Ridge 및 B-MTGNN/Clean B-MTGNN은 요청된 표의 포함 범위에서 제외하며 제외 사유는 [HISTORICAL_EXPERIMENTS.md](HISTORICAL_EXPERIMENTS.md)에 있다. 코드 작성자나 개인 기여를 판단하지 않는다.
