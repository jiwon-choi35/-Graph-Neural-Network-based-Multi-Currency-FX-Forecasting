# Historical / excluded experiments

1차 PR에는 이 요약 문서만 포함한다. 아래 source 결과, 과거 prediction, historical/excluded manifest 및 외부 repo는 원 로컬 참조이며 복사하거나 추가하지 않는다. 제외 사유는 전달 범위와 평가 조건의 차이이며 작성자나 코드 소유자를 단정하지 않는다.

## 봄학기 이력

2026년 2월 ARIMA, LSTM-delta, TCN-delta, Transformer-delta는 historical local-data 실험이다. 초기 neural delta는 delta 자체를 채점한 것이 아니라 실제 직전 level에 예측 delta를 더해 복원한 level을 평가했다. 주로 KR/JP 대상이며 US score는 없다.

초기 일부 결과는 monthly actual-update이다. 2월 19일 이후 일부 rolling 결과는 이전 예측값을 다음 입력에 쓰는 recursive 방식이다. rolling 순위표에 teacher-forced CSV를 recursive로 오분류한 사례와 test RMSE 기반 후보 ranking이 있다.

현재 frozen-data 저장 결과와 실제값도 다르다. 원 data_raw/data.csv가 현재 없으므로 동일 데이터 재현을 주장할 수 없다. Spring 결과를 현재 official handoff 또는 표 3 숫자로 대신 넣지 않는다. 파일을 삭제/이동하지 않고 historical_artifacts_manifest.csv에 archive-only 참조로 기록했다.

근거: runs/baseline_level_check/spring_2026_baseline_history_audit_20261008.md. 이 원 감사 문서는 local absolute-path 링크가 있어 1차 PR에 복사하지 않는다. 본 문서는 상대경로로 요약한 팀 내부 인수인계본이다. Git commit 이력이 없어 당시 날짜는 filesystem metadata only이며 작성자 증거가 아니다.

## 공식 handoff에서 제외한 모델/행

| Model/category | Reason | Existing source kept |
|---|---|---|
| ARIMA / Expanded ARIMA | 이번 최종 필수 baseline 범위에서 제외; statistical/contextual/historical 참조만 | runs/run_arima/ 및 runs/reviewer_matched_20260923/baseline_robustness/expanded_arima/ |
| ETS | 이번 필수 표 범위에서 제외 | runs/reviewer_matched_20260923/baseline_robustness/ets/ |
| Ridge-33var | 이번 필수 표 범위에서 제외 | runs/reviewer_matched_20260923/baseline_robustness/ridge_33var/ |
| B-MTGNN / Clean B-MTGNN | 요청된 baseline 전달 범위 밖; 구현·MC Dropout·anchor·공식 성능 확정은 이번 PR에서 다루지 않음; 작성자 판단 없음 | source final comparison table와 원래 run 폴더에 그대로 보존; 수치는 이 문서에 복사하지 않음 |
| GRU / FX-only / selected-small | 요청한 공식 baseline handoff 범위 밖 | 기존 neural 결과에서 보존; snapshot은 지정 33var arm만 필터링 |
| Older TCN-33var row | causal stability 결과로 handoff row를 교체한 이력 | 이전 source와 저장 결과를 보존; handoff에 새 training/selection 없음 |
| Recursive/fixed-origin/2026 forecasts | monthly one-step 공식 표와 다른 평가 문제 | spring report 및 기존 forecast 폴더에 보존 |

원 로컬 excluded metadata 목록은 manifests/excluded_large_files_manifest.csv에 있으나 이번 PR에서는 제외한다. 폴더명이 excluded라고 해서 원본 파일을 삭제했다는 뜻이 아니다. Source final comparison table의 제외 모델 행도 수정하지 않는다. Public 공개용 정리와 데이터/외부 코드 재공유 권한 확인은 별도 후속 작업이다.
