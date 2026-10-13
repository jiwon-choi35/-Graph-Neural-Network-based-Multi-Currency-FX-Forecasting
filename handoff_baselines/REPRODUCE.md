# Reproduction and saved-artifact verification

모든 경로와 명령은 원 프로젝트 루트 기준 기록이다. 1차 PR은 문서, 표, 기존 검증 기록만 전달하며 어떤 script도 실행하지 않는다. Runner 소스, scripts/, artifacts/, requirements.txt, 데이터 및 checkpoint는 포함하지 않으므로 아래 명령을 1차 PR만으로 실행할 수 있다고 해석하면 안 된다.

## Original artifact-only commands / 1차 PR에 포함되지 않는 도구

원 검증 도구는 Python 3.11+ 표준 라이브러리를 사용하지만 이번 PR에는 도구나 prediction snapshot을 추가하지 않는다. 아래는 기존 명령의 참고 기록이며 새 검증을 수행하는 안내가 아니다. Snapshot-based builder/verifier 개선도 보류한다.

~~~bash
python -B handoff_baselines/scripts/make_official_baseline_table.py
python -B handoff_baselines/scripts/verify_official_artifacts.py --save-report
~~~

표가 이미 있으면 table builder는 timestamp suffix를 붙여 기존 파일을 보존한다. Verifier의 기본 --table은 최초 official_baseline_table.csv를 읽는다. 새 suffix 표를 검증하려면 --table handoff_baselines/tables/<new-name>.csv를 명시한다.

Verifier는 저장 RRSE/Max를 원천 final CSV와 비교하고, 2025 prediction의 schema·12행·날짜·finite 값 및 copy hash를 검사한다. Prediction으로 RRSE를 새로 계산하거나 checkpoint를 로드하지 않는다. Missing artifact는 WARNING, 실제 수치/구조 불일치는 ERROR이다.

prepare_handoff_artifacts.py는 이전 로컬 패키지의 복사/manifest/키워드 스캔 도구이며 이번 PR에서는 제외한다. 기존 verification report의 PASS는 당시 원 로컬 파일에 대한 검사 기록이고 이번 PR의 독립 재현 인증이 아니다.

## 환경 설치 / Future model environment

~~~bash
python -m venv .venv
python -m pip install -r handoff_baselines/requirements.txt
~~~

위 설치 명령과 requirements.txt도 원 프로젝트의 참고 정보이며 이번 PR에서 실행하거나 추가하지 않는다. 기존 기록의 artifact 검증 환경은 Python 3.11.9였으며 model rerun 환경의 정확한 package lock은 저장돼 있지 않다. 원 requirements는 최소 dependency 목록이지 완전한 environment lock이 아니다.

## Frozen data

external_team_repo_20260922_newzip/Cyber-trend-forecasting-main 3/B-MTGNN/data/sm_data.csv.

SHA-256: 5086946995c5e2da176c17a73b94ebdbdb6c2e75f2a59b5cc5c1d932031132dd.

Deterministic source는 external_team_repo/Cyber-trend-forecasting-main 3/B-MTGNN/의 net.py/layer.py/util.py 및 동일 hash data/graph를 참조한다. 이 외부 코드·데이터는 공유 권한과 dependency 배치가 확보돼야 사용할 수 있으며 패키지는 원본을 복사하지 않는다. Local data/sm_data.csv로 대체하지 않는다.

## Original runner commands: reference only

아래 runner 소스는 이번 PR에 포함하지 않는다. 원래의 루트 배치를 유지하며 파일이나 output 폴더를 옮기지 않는다. 필요한 파일은 다음과 같다.

| Required file | Role |
|---|---|
| run_baseline_robustness_20260923.py | Naive-last/Drift 진입점; 다른 모델도 포함한 shared suite |
| run_neural_residual_stability_20260930.py | LSTM/Transformer residual 및 seed ensemble 진입점 |
| run_neural_baseline_preregistered_strengthening_20260930.py | Neural 모델/배열/복원/metric 공통 import 의존성 |
| run_tcn_stability_check_20261001.py | Causal TCN 진입점 |
| run_mtg_nn_matched.py | Deterministic MTGNN 학습 선택 및 locked-test 진입점 |

| Models | Original command | Limitation |
|---|---|---|
| Naive-last / Drift | python run_baseline_robustness_20260923.py --run | shared suite는 제외 statistical/neural 모델도 실행; 두 baseline만 선택하는 CLI 없음 |
| LSTM / Transformer | python run_neural_residual_stability_20260930.py --run | shared grid는 다른 model/scope도 실행; 이 handoff의 두 행만 재현하는 CLI 없음 |
| causal TCN | python run_tcn_stability_check_20261001.py --run | 별도 TCN suite; 기존 output 폴더가 있으면 재실행 거부 |
| Deterministic training selection | python run_mtg_nn_matched.py --arm mtgnn_deterministic --train-full --epochs 30 --seed 1 | 원 frozen dependencies 필요; 별도 신규 실행 환경에서만 |
| Deterministic locked test | python run_mtg_nn_matched.py --arm mtgnn_deterministic --eval-test | 원 checkpoint/config는 local path를 기록; 새 위치로 옮긴 artifact는 그대로 실행 안 될 수 있음; 이번에는 실행 금지 |

공식 표 구성은 위 명령을 재실행하여 test가 더 좋은 설정을 찾는 절차가 아니다. Source snippets와 설정은 MODEL_AUDIT을 참고한다. 기존 출력 보호를 해제하거나 기존 runs/를 지우는 안내는 하지 않는다.

### Historical runtime dependencies / 이번 PR에서 제외

Residual runner는 보고서 작성 시 runs/reviewer_matched_20260930/neural_baseline_preregistered_strengthening/final_test_metrics.csv를 읽는다. Causal TCN runner는 runs/reviewer_matched_20260930/neural_residual_stability/final_test_metrics.csv와 predictions/tcn_33_variable_residual_delta_{US,KR,JP}.csv를 읽는다. 원 runner를 그대로 실행하려면 이 과거 비교 자료도 원 경로에 필요하지만 1차 PR에는 추가하지 않는다. Prediction의 y_true와 데이터 재공유 권한을 확인하기 전 전달하지 않는다.

## What can be reproduced from this handoff?

- 1차 PR만으로는 문서, 저장 표, 원 source mapping 및 과거 검증 기록을 읽을 수 있다. 새 artifact 검증이나 모델 실행을 지원하는 완결 배포본이 아니다.
- 이전 로컬 패키지에는 copied prediction/metric/config가 있었으나 이번 PR에서는 제외한다. 표에 적힌 artifact 경로는 원 로컬 참조이며 해당 파일이 함께 배포됐다는 뜻이 아니다.
- Naive/Drift의 정의와 observed-history boundary는 문서화된 source 위치에서 확인할 수 있으나 source code 자체는 이번 PR에 포함하지 않는다.
- Neural selected weights는 저장되지 않아 선택된 가중치 forward를 재현할 수 없음. Seed/config/grid 기록은 존재하나 fresh training의 exact numerical identity를 보장하지 않음.
- Deterministic selected checkpoint는 원 로컬 runs/reviewer_matched_20260919/full_valid_select/mtgnn_deterministic/checkpoint_best.pt에 존재. 이 패키지에는 weight를 넣지 않으므로 모형 inference를 독립 재현하는 완결 배포본은 아님.
- 이 한계를 숨기지 않고 code dependency, selected metadata 및 실제 사용 command를 구분해 전달한다.
