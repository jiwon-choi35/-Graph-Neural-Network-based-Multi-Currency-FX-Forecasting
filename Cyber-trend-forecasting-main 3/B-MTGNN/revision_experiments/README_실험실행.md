# 추가 실험 실행 안내 (JIIS-2026-079 심사 대응)

## 왜 실행이 필요한가
원고 표 2·3의 B-MTGNN 값(2025년 RRSE 달러지수 0.3507, 엔/달러 0.2881, 원/달러 0.2591)을 내는 모형을 로컬에서 찾지 못했습니다.
저장소에 커밋된 epoch 57 모형은 0.3181 / 0.2903 / 0.2557을 냅니다. 추가 실험 4개는 이 epoch 57 모형으로 돌렸습니다.

**원고 표를 만들 때 쓴 체크포인트(`model.pt`)를 가지고 있다면 아래 1~2단계를 실행해서 결과를 보내 주세요.**
그 모형이 원고 값을 재현하면, 같은 모형으로 3단계 실험을 다시 돌려서 원고 기준을 하나로 맞추려 합니다.

## 준비
- 아래 명령은 모두 이 폴더(`Cyber-trend-forecasting-main 3/B-MTGNN/revision_experiments`)에서 실행합니다.
- 패키지: `torch`, `numpy`, `pandas`, `scipy`, `matplotlib` (`../requirements.txt` 참고)
  - 참고 결과를 낸 환경: Python 3.12.10, torch 2.10.0, numpy 2.3.5, pandas 2.3.3 (Windows)
- **Windows에서는 짧은 경로에 클론하세요** (예: `C:/work/fx`). 폴더 이름에 공백이 있고 4단계가 하위 폴더를 더 만들기 때문에, 경로가 길면 "path longer than allowed" 오류가 납니다.
- **평가는 자동으로 CPU에서 실행됩니다.** MC 드롭아웃 난수가 CPU/GPU에서 다르게 나오기 때문에, 정확한 재현은 CPU에서만 됩니다. GPU는 4단계 학습에만 씁니다.

## 1단계: 환경 확인 (약 10초)
```
python check_model.py --model reference/paper_ep57.pt
```
마지막 줄에 `max |RRSE diff| vs epoch 57: 0.0000  <- MATCH`가 나오면 환경이 정상입니다.
안 나오면 출력 전체를 보내 주세요.

## 2단계: 가지고 있는 체크포인트 확인 (모형당 약 10초)
```
python check_model.py --model <체크포인트 경로> --seed <학습 때 --seed 값> --name <구분용 이름>
```
- `train_test.py`가 저장한 `model.pt`(모형 전체)를 넣어 주세요. 원고 설정(triple_050)이면 `--seed`는 1입니다.
- 출력 표(2025년 RRSE·RAE, 식(10) 구간 상·하한 평균, 피복 개월 수, 2024년 검증값)와
  `max |RRSE diff| vs manuscript Table 3` 줄을 보내 주세요. `<- MATCH`가 뜨면 원고 모형을 찾은 것입니다.
- 결과 파일: `results/check_model/<이름>_summary.csv`, `<이름>_monthly.csv`

## 3단계: 원고 모형을 찾은 경우, 그 모형으로 추가 실험 (CPU)
```
python exp12_recursive.py --model <원고 모형 경로>                         # 실험 1·2, 약 15~20분
python exp4_release_lag.py --models manuscript=<원고 모형 경로>            # 실험 4, 약 1분
```
- 실험 1 (`exp12_recursive.py`): 2024년 12월 시점에서 2025년 12개월 재귀 예측 백테스트 (식(7) 적용/미적용), 13개 시점 재귀 예측, MC 시드 민감도
- 실험 2 (같은 스크립트): 2026년 전망과 예측구간(MC 분산 + 백테스트 잔차 분산)
- 실험 4 (`exp4_release_lag.py`): 거시지표 발표 시차를 반영한 2025년 재평가
- 결과는 `results/exp12/`, `results/exp4/`에 저장됩니다. **`results/` 폴더 전체를 압축해서 보내 주세요.**

## 4단계 (선택): 제거 실험 재학습 (GPU 권장)
```
python train_ablation.py                      # 3개 구성 x 시드 1~5, GPU에서 약 1시간
python exp3_ablation_eval.py --model_dir runs/ablation/models --paper <원고 모형 경로>
```
- 구성: 원 모형(33노드) / 거시변수 제외(15노드) / 그래프 전파 차단(인접행렬 = 단위행렬)
- 학습은 `runs/ablation/box/`에 코드와 데이터를 복사해서 진행하므로 저장소의 결과 파일을 덮어쓰지 않습니다.
- GPU 메모리가 부족하면 `--parallel 1`로 줄이세요. 동작만 확인하려면 `--seeds 1 --epochs 2`.
- 결과: `results/exp3/` (같이 보내 주세요)

## 비교용 참고 결과
- `reference_results/`: epoch 57 모형으로 같은 명령을 실행한 결과입니다 (실험 3은 시드 1~5 재학습 모형).
- `reference_results/exp4`의 `paper_sm_data` 값(달러지수 0.3197)이 1단계 값(0.3181)과 조금 다른 것은 정상입니다. 실험 4는 모든 입력 변형에 같은 MC 마스크를 쓰려고 평가 경로를 따로 두었기 때문이며, 실험 4 안에서의 비교에는 영향이 없습니다.
- `reference/paper_ep57.pt`: 저장소에 커밋된 epoch 57 체크포인트 (원본: `AXIS/model/Bayesian/model.pt`, 커밋 7819f90).

## 파일 구성
| 파일 | 역할 |
|---|---|
| `rev_common.py` | 공통 함수. `train_test.py`의 평가(one-step, 식(7) 보정, 식(10) 구간)와 `forecast.py`의 재귀 예측을 그대로 구현 |
| `check_model.py` | 체크포인트 하나로 원고 표 2·3 값 계산 |
| `exp12_recursive.py` | 실험 1·2 |
| `exp4_release_lag.py` | 실험 4 |
| `train_ablation.py`, `exp3_ablation_eval.py` | 실험 3 학습과 평가 |
| `../train_test.py` | `--adj_mode identity` 옵션 추가(기본값은 기존과 같음) |
| `../net.py`, `../layer.py`, `../util.py`, `../trainer.py` | 모형 코드 (기존과 같음) |
| `../data/sm_data.csv`, `../data/data.csv`, `../data/graph.csv` | 평활 자료, 원자료(실험 4 재평활용), 그래프 파일 |
