# FX baseline handoff / 환율 baseline 인수인계

이 폴더의 1차 PR은 공동연구 팀 내부용 문서, 공식 전달 표, 기존 artifact 검증 기록만 포함한다. 새 학습, 튜닝, 평가, checkpoint forward 또는 script 실행은 수행하지 않는다. 독립 실행 가능한 모델 배포본이 아니며, 작성자나 개인 기여를 인증하지 않는다.

포함 파일은 README.md, REPRODUCE.md, EVALUATION_PROTOCOL.md, MODEL_AUDIT.md, BASELINE_RESULTS.md, HISTORICAL_EXPERIMENTS.md, tables/official_baseline_table.csv 및 .md, manifests/official_artifact_verification.json, official_artifacts_manifest.csv, input_source_hashes.csv의 11개이다. Runner 실제 소스, scripts/, artifacts/, raw data, checkpoint, external repo 및 public 공개용 정리는 보류한다.

## 포함 범위 / Included models

Naive-last, Drift, LSTM-33var residual_delta, TCN-33var causal residual_delta, Transformer-33var residual_delta, Deterministic MTGNN.

공식 handoff 표는 요청된 여섯 baseline 행만 담는다. 표의 official이라는 파일명은 전달 범위를 뜻하며 전체 논문 모형의 공식 성능 확정을 대신하지 않는다. B-MTGNN/Clean B-MTGNN 결과 행과 해당 모델의 구현, MC Dropout, anchor correction 및 공식 성능 판단은 제외한다. ARIMA/ETS/Ridge는 historical 또는 excluded 참조로만 남긴다. 제외 사유는 전달 범위이며 코드 소유자나 기여자를 단정하지 않는다.

## 평가 조건 / Evaluation

- Train: 2011-01~2023-12; validation: 2024-01~2024-12; test: 2025-01~2025-12.
- Monthly one-step actual-update; 예측 시점 t에는 t-1까지의 실제 관측값을 사용.
- Delta는 level로 복원하고 US/KR/JP별 RRSE와 Max RRSE를 보고.
- Neural 설정은 2024 validation Max RRSE, deterministic MTGNN epoch는 2024 mean RRSE로 선택. 고정 Naive/Drift는 tuning 없음.
- 저장 test 결과를 보고 이번 handoff에서 모델이나 설정을 다시 선택하지 않음.
- Transformer의 기존 alpha=0.5 blending과 모델별 input scope를 명시.

The package documents saved monthly one-step baseline results under a common calendar split. Inputs and final prediction parameterizations differ by model. It does not claim identical inputs, identical post-processing, or a newly untouched holdout. The February local-data experiments are historical evidence and are not substituted into the current table.

## 읽는 순서 / Reading order

1. [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md): 정보 경계, level RRSE, 입력 범위.
2. [MODEL_AUDIT.md](MODEL_AUDIT.md): source code, 설정, 경로, 재현 한계.
3. [BASELINE_RESULTS.md](BASELINE_RESULTS.md) / [표](tables/official_baseline_table.md): 전달 수치.
4. [HISTORICAL_EXPERIMENTS.md](HISTORICAL_EXPERIMENTS.md): 제외 및 봄학기 이력.
5. [REPRODUCE.md](REPRODUCE.md): runner 목록, 원 실행 명령, 누락된 의존성과 재현 한계.
6. [기존 검증 기록](manifests/official_artifact_verification.json): 원 로컬 artifact에 대한 과거 검사 결과.

## 절차 / Workflow

1차 PR에서는 문서와 저장 표를 읽고 source mapping 및 기존 검증 기록을 검토한다. Builder/verifier 개선과 재실행은 하지 않는다. 원 runner는 프로젝트 루트 기준 경로를 사용하므로 이동하지 않으며, 진입점과 필요한 파일 목록만 REPRODUCE.md에 기록한다.

표와 MODEL_AUDIT의 artifacts/ 경로는 원 로컬 패키지의 snapshot 위치이며 이번 PR에는 포함되지 않는다. 원 runs/, source comparison table, runner, 외부 backbone, 데이터와 checkpoint도 참조 경로만 기록한다. 기존 verification report의 PASS는 당시 로컬 artifact 검사 결과이지 1차 PR 단독 재현 가능성이나 이번 PR의 새 검증 결과가 아니다.

## Git 준비 / Git preparation

루트 .gitignore는 바꾸지 않는다. 위의 승인된 11개 파일만 개별 추가하고 handoff_baselines/ 또는 저장소 전체를 추가하지 않는다. 원 runs/와 외부 repo를 통째로 강제 추가하지 않는다.

manifests/official_artifacts_manifest.csv와 input_source_hashes.csv는 기존 로컬 패키지 및 source의 과거 provenance 기록으로 원본을 보존한다. Manifest의 included_in_git=yes는 당시 포함 계획 표시이며 이번 PR의 실제 tracking 또는 포함 여부를 뜻하지 않는다. 문서 hash는 이번 PR 문서 갱신 전 snapshot 기준이므로 현재 문서 hash와 다를 수 있다. 실제 포함 파일은 위 11개 목록과 PR diff를 기준으로 확인한다.

The first PR contains documentation, saved table values, and historical verification records only. It does not include executable verification tools, prediction snapshots, runners, or model dependencies. Neural selected weights were not persisted, so exact selected-checkpoint inference is not supported by this PR. Builder/verifier changes and snapshot-based portability are deferred.

## 내부 공유 및 권한 / Internal sharing and permissions

팀 내부 repository라는 이유만으로 데이터나 외부 코드의 재공유 권한이 확보되는 것은 아니다. 필요한 원 데이터와 외부 backbone의 license/출처/배포 허가를 별도로 확인한다. 원 prediction에는 y_true가 포함되므로 내부 전달 승인과 데이터 재공유 권한을 확인해야 하며 public 공개는 보류한다. 파일명이나 filesystem metadata만으로 작성자 및 개인 기여를 판단하지 않는다.
