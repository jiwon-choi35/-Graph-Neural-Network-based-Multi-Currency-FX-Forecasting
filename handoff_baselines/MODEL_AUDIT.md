# Model audit

경로는 모두 원 프로젝트 루트 기준이다. artifacts/의 경로는 기존 로컬 handoff_baselines/ 기준이며 1차 PR에는 해당 파일을 포함하지 않는다. 표의 yes는 모델 행이 전달 표에 포함됨을 뜻하며, 실행 코드나 artifact가 포함됨을 뜻하지 않는다. 실행 명령은 기록용이며 이번 PR에서 실행하지 않는다. 코드 존재, 파일명, metadata만으로 작성자나 개인 기여를 단정하지 않는다.

| Model | Included in official baseline handoff? | Input scope | Prediction target | Level/direct or delta-to-level | Uses previous actual level? | Persistence/Naive-last blending? | Evaluation loop | Recursive feedback? | Seed/config/checkpoint | Prediction CSV path | Metric CSV path | Reproduce command | Risk/notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive-last | yes | target-only | US/KR/JP | level baseline | yes | no extra | append actual after prediction | no | fixed; no model checkpoint | artifacts/naive_last/predictions_test_*.csv | artifacts/naive_last/test_metrics.csv | shared robustness --run (reference only) | original suite also runs excluded models |
| Drift | yes | target-only | US/KR/JP | level + observed average drift | yes | no extra | append actual after prediction | no | fixed; no model checkpoint | artifacts/drift/predictions_test_*.csv | artifacts/drift/test_metrics.csv | shared robustness --run (reference only) | drift estimate updates with observed history |
| LSTM-33var residual_delta | yes | 33-variable | delta for US/KR/JP | delta-to-level | yes | no extra; alpha=1 | observed-array window slices | no | w6/h24/L1/do0.1/lr0.0005/Huber; seeds 5,2; epochs 10,9; top2 mean | artifacts/lstm_33var/predictions_test_*.csv | artifacts/lstm_33var/test_metrics.csv | residual stability --run | selected weights not persisted |
| TCN-33var causal residual_delta | yes | 33-variable | delta for US/KR/JP | delta-to-level | yes | no extra; alpha=1 | observed-array window slices | no | w3/k3/ch16/do0.1/lr0.0005/Huber; seed5/epoch36 | artifacts/tcn_33var_causal/predictions_test_*.csv | artifacts/tcn_33var_causal/test_metrics.csv | TCN stability --run | weights not persisted; replaces older non-stability row |
| Transformer-33var residual_delta | yes | 33-variable | delta for US/KR/JP | delta-to-level + final blend | yes | yes; alpha=0.5 | observed-array window slices | no | w6/h8/L1/do0/lr0.001/Huber; seed5/epoch42 | artifacts/transformer_33var/predictions_test_*.csv | artifacts/transformer_33var/test_metrics.csv | residual stability --run | final score includes blending; weights not persisted |
| Deterministic MTGNN | yes; dependency-qualified | graph-33-variable | 33-node level; report US/KR/JP | direct level | yes, via history window | no; anchor/debias none | observed series[t-24:t] | no | seed1/w24/30 epochs; selected28; checkpoint_best.pt available locally | artifacts/mtgnn_deterministic/predictions_test_*.csv | artifacts/mtgnn_deterministic/test_metrics.csv | --arm mtgnn_deterministic --train-full / --eval-test | mean validation RRSE selection; external backbone; no weight copied |

## Code evidence

- Naive/Drift: run_baseline_robustness_20260923.py:74-88. Prediction saved first, observed.append(actual) after.
- Neural input/delta: run_neural_baseline_preregistered_strengthening_20260930.py:169-185. Level restored using actual[t-1]; anchored() at 189-191.
- LSTM/Transformer selected test path: run_neural_residual_stability_20260930.py:269-286. Selected member predictions averaged; then saved alpha applied.
- Causal TCN: run_tcn_stability_check_20261001.py:75-112,262-275. Left causal padding; delta to actual last level; alpha from tcn_selected_config.csv.
- Deterministic: run_mtg_nn_matched.py:72-77 and 608-622. eval + no_grad + single forward; data slice excludes current month.
- Historical selected metadata: each local artifacts/<model>/selected_config.json points to the original config/selected_configs.csv. These snapshots are not included in the first PR; the selected settings above are documentary records only.

## Reproducibility boundaries

기존 감사에서 원 코드와 저장 artifact의 존재를 확인한 기록을 인수인계한다. 이번 PR에서는 script나 model forward를 실행하지 않는다.

원 로컬 패키지의 prediction CSV는 기존 저장 결과의 복사본이었다. 기존 verification report는 copy hash와 날짜/행 수/finite 값, 저장 metric과 표를 대조한 과거 기록이며 이번 PR에는 prediction을 추가하지 않는다. 이 보고서의 PASS를 현재 PR의 독립 실행 또는 재현 인증으로 해석하지 않는다.

Root shared runners는 다른 arm도 포함하고 개별 모델 CLI가 없는 경우가 있다. 존재하지 않는 --models/--targets 옵션을 재현 명령에 추가하지 않는다. 기존 run 폴더의 보호를 해제하지 않으며, 개별 재실행에는 별도 환경과 담당 범위 정리가 필요하다.
