# Evaluation protocol / 평가 조건

이 문서는 기존 저장 결과의 평가 조건을 기록한다. 1차 PR에는 runner, scripts/, artifacts/, 데이터 또는 checkpoint를 포함하지 않으며 학습, 평가, metric 재계산을 수행하지 않는다. Source code 위치와 artifact 경로는 원 프로젝트 기준 외부 참조이다.

## 데이터와 split

원 데이터는 프로젝트 루트 기준 external_team_repo_20260922_newzip/Cyber-trend-forecasting-main 3/B-MTGNN/data/sm_data.csv이다. 이 경로의 디렉터리명은 데이터 위치일 뿐, 다른 담당 모형의 성능을 handoff에 포함한다는 뜻이 아니다.

Frozen SHA-256: 5086946995c5e2da176c17a73b94ebdbdb6c2e75f2a59b5cc5c1d932031132dd.

Deterministic runner는 external_team_repo/Cyber-trend-forecasting-main 3/B-MTGNN/data/sm_data.csv를 사용한다. 기록된 data hash는 같다. Graph hash: 53eb9bfaee6e07e9142fc8aefb2b61b59a906a0365d5d3895b3192c1d4480ea5. 데이터 원본과 외부 repo는 패키지에 복사하지 않았다.

| Split | Calendar | 0-based rows | Months |
|---|---|---|---:|
| Train | 2011-01~2023-12 | 0..155 | 156 |
| Validation | 2024-01~2024-12 | 156..167 | 12 |
| Test | 2025-01~2025-12 | 168..179 | 12 |

Targets: US=us_Trade Weighted Dollar Index, KR=kr_fx, JP=jp_fx. Date를 제외한 numeric matrix의 target index는 US=0, KR=1, JP=2이다.

## Monthly one-step actual-update

월 t의 한 달 앞 예측에는 t-1까지 관측된 history를 사용한다. 월 t의 실제값은 예측이 완료된 후 다음 월 history에 들어간다. 저장 prediction을 다음 입력에 넣는 recursive feedback은 적용하지 않는다.

For month t, each baseline uses observed history through t-1. The realized observation at t becomes available for the next monthly forecast. Earlier predictions are not fed back as the next input. This is a sequence of one-step forecasts, not a forecast of all twelve test months from one fixed origin.

Naive/Drift는 history.append(actual)을 수행한다(run_baseline_robustness_20260923.py:74-88). Neural 입력은 원본 observed array의 t-window:t로 구성된다(run_neural_baseline_preregistered_strengthening_20260930.py:169-177). Deterministic MTGNN은 series[t-24:t]를 사용한다(run_mtg_nn_matched.py:608-622). Prediction 결과를 원 입력 배열에 덮어쓰지 않는다.

## Level RRSE

RRSE = sqrt(sum((true_level - predicted_level)^2) / sum((true_level - mean(true_level))^2)).

분모의 평균은 해당 target의 2025 평가 12개월 평균이다. Train 평균이나 33개 node 전체의 평균이 아니다. Max RRSE = max(US RRSE, KR RRSE, JP RRSE). 낮을수록 좋다.

Residual/delta model의 level 복원: predicted_level[t] = actual_level[t-1] + predicted_delta[t]. 복원 후 level과 actual level로 RRSE를 산출한다. Training delta loss와 exported level RRSE는 구별한다.

RRSE is calculated per target on restored levels over the twelve test months. Max RRSE is the largest of the three target scores. RMSE = sqrt(mean((true_level - predicted_level)^2)). The original handoff verification scripts compare saved scores, but those scripts are excluded from the first PR. No new score is calculated in this PR.

## 입력과 후처리 차이

| Model | Input scope | Prediction / final output |
|---|---|---|
| Naive-last | target-only | previous actual level |
| Drift | target-only | previous actual level + observed-history average monthly drift |
| LSTM | 33-variable sequence | restored residual level; selected alpha=1; no extra blending |
| causal TCN | 33-variable sequence | restored residual level; selected alpha=1; no extra blending |
| Transformer | 33-variable sequence | restored residual level then alpha=0.5 persistence blending |
| Deterministic MTGNN | graph-33-variable | direct level; single eval forward; no anchor/debias |

Transformer: final = alpha * restored_neural_level + (1-alpha) * last_actual. 기존 validation-selected alpha=0.5라는 사실만 인수인계한다. 새 blending 설계나 최적화는 수행하지 않는다.

동일 월별 한 달 앞 평가 방식에서의 최종 baseline 성능 비교이며, 동일 입력/동일 후처리 비교라고 기술하지 않는다.

## Selection boundary

Naive/Drift는 고정 정의이다. LSTM/TCN/Transformer는 저장 runner에서 2024 validation Max RRSE로 config/seed/epoch/alpha를 선택한다. Deterministic MTGNN은 세 타깃 평균 validation RRSE로 epoch를 선택한다. 서로 다른 selection objective를 동일하다고 고치지 않는다.

2025 test는 설정 선택에 사용하지 않는 것이 protocol이다. 저장 config의 once-only는 해당 arm/run의 평가 횟수이며, 프로젝트 전체가 과거에 test를 전혀 보지 않았다는 보장은 아니다. 이번 handoff는 저장 결과를 다시 선택하거나 과거 연구 과정의 독립성을 새로 인증하지 않는다.
