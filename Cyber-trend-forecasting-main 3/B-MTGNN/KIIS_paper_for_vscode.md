
## 1. 개요 (HY중고딕 11pt, 굵게)

최근 글로벌 금융 시장은 국가 간 경제적 밀착도가 심화됨에 따라 환율 변동의 복잡성과 상호의존성이 급격히 증대되고 있다 [1]. 환율은 단순한 개별 국가의 경제 지표를 넘어, 인접 국가 및 주요 교역국 간의 유기적인 관계 속에서 실시간으로 변화하는 동적 특성을 지닌다 [2]. 그러나 ARIMA와 같은 전통적인 통계 모델이나 단순 LSTM 계열의 딥러닝 모델들은 이러한 국가 간의 공간적 상관관계를 충분히 반영하지 못하고, 단일 국가의 과거 패턴 학습에 치중해 왔다는 한계가 있다 [3]. 이는 급변하는 매크로 경제 상황 속에서 국가 간 금융 전이 효과를 간과하게 만들며, 결과적으로 환율의 장기적 추세를 정확히 포착하고 변동 원인을 해석하는 데 걸림돌이 된다 [4].
이에 본 연구는 환율이 개별적으로 존재하는 것이 아니라 상호 동적인 관계로 연결되어 있다는 점에 주목하여, 국가 간 잠재적 관계를 스스로 학습하는 그래프 신경망(Graph Neural Networks, GNN) 기반의 MTGNN 모델을 제안한다 [5]. 특히 명시적인 인접 행렬이 존재하지 않는 외환 시장의 특성을 고려하여, 적응형 그래프 학습 모듈(Adaptive Graph Learning Layer)을 통해 주요 3개국(미국, 한국, 일본) 간의 숨겨진 시공간적 정보를 통합 모델링하였다. 또한, 금리 및 물가 등 국가별 거시경제 지표를 노드 피처(Node Feature)로 결합함으로써, 모델이 단순 시계열 패턴을 넘어 다차원적인 경제적 맥락을 학습할 수 있도록 확장성을 부여하였다 [6].
본 연구의 핵심적인 차별점은 단순한 수치 예측력 향상에 그치지 않고, SLM(Small Language Model) 기반의 다중 에이전트 시스템(Multi-Agent System) 을 결합하여 예측 결과에 대한 고도의 해석력을 확보했다는 점에 있다 [7]. 기존 연구들이 단일 수치 예측에 치중했던 것과 달리, 본 프레임워크는 전문화된 다중 에이전트 간의 순환형 체이닝(Chaining) 구조를 도입하여 예측 데이터의 시계열 일관성을 엄밀히 검증하고 심층적인 원인 분석을 수행한다 [8]. 이를 통해 환율 변동의 동인에 대한 설명 가능한(Explainable) 근거를 구축하며, 에이전트의 다각적 해석이 담긴 장기 예측 보고서를 제공함으로써 외환과 글로벌 경제에 대한 근본적인 분석을 제안한다.
단순한 기술적인 정확도 개선을 넘어, 데이터 기반의 정량적 예측과 경제적 추론 기반의 정성적 분석이 결합된 새로운 외환 분석 패러다임을 제시한다는 점에서 본 연구는 중요한 의의를 갖는다. 특히 인공지능이 도출한 수치에 전문가적 해석을 덧붙임으로써, 금융의 실질적인 활용 가능성을 입증하고자 한다. 이는 변동성이 큰 글로벌 경제 환경에서 보다 선제적이고 입체적인 리스크 대응 체계를 구축하는 데 기여할 것으로 기대된다.
본 논문의 구성은 다음과 같다. 2장에서는 환율 예측 및 거시경제 요인에 관한 선행 연구를 고찰하고, 3장에서는 제안하는 MTGNN 및 다중 에이전트 시스템의 상세 구조를 설명한다. 4장에서는 실험 과정과 결과를 통해 모델의 유효성을 검증하며, 마지막으로 결론 및 향후 연구 방향에 대해 논한다.
## 2. 관련 연구HY중고딕 11pt, 굵게)
## 2. 1 외환 환율 예측 및 딥러닝 모델
외환 환율 예측 분야는 2000년대 이전부터 계량경제학적 접근을 통해 활발히 연구되었다. 초기 연구들은 주로 벡터 자기회귀(VAR)나 GARCH와 같은 통계적 모델을 사용하여 환율의 변동성을 설명하고자 하였다. [9]. 그러나 이러한 전통적인 방식은 통계적 가정을 필요로 하며, 금융 시장의 복잡한 비선형 역학을 포착하는 데 한계가 존재한다 [10]. 이러한 한계를 해결하기 위해 최근에는 기계학습 및 딥러닝 기반의 방법론이 제안되었다.
딥러닝 모델인 LSTM, CNN, 그리고 최근 자연어 처리 분야에서 혁신을 일으킨 Transformer 등이 외환 예측에 적용되어 기존 통계 모델 대비 우수한 성능을 보였다. 특히 시계열 데이터의 장기 의존성(Long-term dependency)과 미시적 변동성을 포착하기 위해 Zhou et al.(2021)은 시간적 특징을 임베딩(Temporal Embeddings)하여 어텐션 메커니즘을 개선한 Informer 모델을 제안하였다 [11]. 이 연구는 단변량 데이터보다 거시경제 지표를 포함한 다변량(Multivariate) 데이터를 사용할 때 트랜스포머가 LSTM이나 ARIMA보다 월등한 예측 성능과 수익률을 달성함을 입증하였다.
## 2. 2 거시경제 요인과 기존 방법론의 한계
선행 연구들은 외환 시장의 예측 정확도를 높이기 위해 다양한 시도를 해왔다. 금리평가설(IRP)이나 구매력평가설과 같은 이론에 기반하여 금리, 인플레이션율, 통화 정책과 같은 거시경제 지표를 특징(feature)으로 포함하는 연구들이 진행되었다 [12]. Yao et al.(2018)의 연구 역시 기술적 지표 외에 펀더멘털(Fundamental) 데이터를 결합한 Dual-Stage Attention 구조를 통해 예측력을 강화하였다 [13].
그러나 이러한 최신 연구들조차 다음과 같은 구조적 한계를 지닌다. 첫째, 다중 통화 간의 구조적 상호 의존성(Interrelationships)을 간과하였다. 대부분의 딥러닝 기반 연구는 입력 데이터를 평면적인 벡터 형태로 처리할 뿐, 경제학적 이론에 근거한 통화 삼각 관계(triplet)나 통화 간의 네트워크 구조를 명시적으로 모델링하지 않는다. 둘째, 시장 마찰 요인에 따른 가치 평가의 상대성을 충분히 반영하지 못했다. 신용도나 은행 규제 등으로 인해 통화의 가치 평가는 상대 통화에 따라 달라질 수 있으나, 기존 연구는 이를 전체 통화 네트워크 관점에서 포괄적으로 활용하지 못하는 경향이 있다.
## 2. 3 그래프 신경망(GNN) 기반의 다변량 시계열 예측
앞서 언급한 외환 시장의 구조적 상호 의존성을 포착하기 위해, 최근 시계열 예측 분야에서는 그래프 신경망(Graph Neural Networks, GNN)을 활용한 연구가 주목받고 있다 [6], [14]. 다변량 시계열 데이터에서 각 변수를 그래프의 노드(Node)로 간주할 경우, GNN은 인접한 노드들의 정보를 집계하여 잠재된 공간적 의존성(Spatial Dependency)을 효과적으로 학습할 수 있다는 장점이 있다 [15] ,[16]. 특히 기존의 CNN이나 RNN 기반 방법론들이 변수 쌍(Pair-wise) 간의 관계를 명시적으로 모델링하지 못해 모델의 해석력이 떨어진다는 한계를 GNN을 통해 극복할 수 있다 [17].
그러나 외환 시장과 같은 일반적인 다변량 시계열 데이터는 변수 간의 연결 구조(Graph Structure)가 사전에 정의되어 있지 않다는 문제점이 존재한다. 이를 해결하기 위해 Wu et al.(2020)은 사전에 정의된 그래프 구조 없이도 데이터로부터 변수 간의 관계를 자동으로 추출할 수 있는 그래프 학습 모듈(Graph Learning Module)을 제안하였다 [5]. 이 연구에서 제안한 MTGNN(Multivariate Time Series Grpah Neural Network) 프레임워크는 그래프 학습 레이어를 통해 변수 간의 단방향(Uni-directed) 관계를 학습하여 인접 행렬(Adjacency Matrix)을 생성하고, 이를 바탕으로 그래프 컨볼루션(Graph Convolution)을 수행하여 공간적 관계를 포착한다.
또한, 시계열 데이터의 시간적 패턴을 학습하기 위해 팽창 컨볼루션(Dilated Convolution)과 인셉션 레이어(Inception Layer)를 결합한 시간 컨볼루션 모듈을 함께 사용하여, 시공간적(Spatial-Temporal) 의존성을 동시에 모델링하는 종단간(End-to-End) 학습 구조를 확립하였다[5]. 이러한 접근 방식은 환율과 같이 변수 간의 구조적 관계가 뚜렷하지 않은 데이터에서도 잠재된 상호 연관성을 발견하고 예측 성능을 향상시킬 수 있음을 시사한다.
## Chapter 3. Methodology
## 3. 1 문제 정의 및 개요
본 연구는 다수 국가(또는 통화)로 구성된 글로벌 외환(FX) 시스템에서, 국가 간 상호의존성(그래프 구조)과 시계열 동학(시간 의존성)을 함께 고려하여 다중 스텝(Multi-horizon) 환율 경로를 확률적으로 예측하는 문제를 다룬다.
국가(또는 통화) 집합을 라 하고, 각 국가는 FX 네트워크의 노드로 표현한다. 관측 시점  동안 국가별 환율 및 거시·금융 특징이 주어졌을 때, 목표는 다음의 조건부 예측분포를 추정하는 것이다.

여기서 는 예측 구간(horizon)이며, 는 미래 스텝의 환율(또는 환율 상태/수익률) 벡터 시퀀스이다. 이 정의는 한 점 예측이 아니라, 미래 경로 전체에 대한 분포를 추정한다.
예를 들어 학습 구간이 2011-01부터 T=2025-10까지라면, T+1은 2025-11이고, H=36이면 2025-11부터 36개월 구간을 예측한다. 이때 모델은 와 그래프 에 조건부로 의 확률분포를 추정한다.

## 3. 2 입력 변수 및 특징 구성
환율은 단일 요인으로 움직이지 않으며, (i) 환율 자체의 동학, (ii) 변동성·유동성(시장 활동), (iii) 거시·정책 환경(공통 컨텍스트)의 결합에 의해 비대칭적으로 공진화(co-evolve)한다. 이에 따라 본 연구는 각 국가 i에 대해 다음의 구성요소를 결합한다.

## 3. 2.1 환율 동학
국가 i의 환율 관측을 라 할 때, 로그수익률(또는 정규화된 환율 상태)을 로 정의한다.

실험/데이터 특성에 따라 를 정규화된 레벨로 두고 변환을 생략할 수도 있으나, 본문에서는 를 환율 상태(수익률 포함)로 정의한다.

## 3. 2.2 변동성·시장 활동
환율 리스크를 반영하기 위해 롤링 변동성 와, 유동성/거래강도를 대변하는 거래량(또는 대체 지표) 를 포함한다.

## 3. 2.3 거시·정책 컨텍스트
여러 국가에 동시에 영향을 주는 거시·금융 공통요인을 로 둔다(예: 글로벌 위험선호, 금리, 원자재 등)

## 3. 2.4 최종 특징 벡터
최종적으로 국가 i의 시점 t 입력 특징을 다음과 같이 정의한다.

시장 휴일/비동기 캘린더/거시지표 결측 등으로 발생하는 결측은 전진 채움(forward filling) 및 마스킹으로 처리한다.

## 3. 3 국가 상호작용 그래프 구성
국가 간 구조적 의존성을 명시적으로 반영하기 위해 가중 그래프를 정의한다.

여기서 는 국가(통화) 노드 집합, 는 국가 간 관계, 는 가중 인접행렬이다. 엣지 는 무역노출, 금융통합, 통화정책 스필오버, 지역 인접성, 환율 상관 기반 동조화 등으로 인해 i의 변화가 j에 영향을 줄 수 있음을 의미한다.
초기 인접행렬 는 상관, 경제/지역 근접성, 잠재 의존 성분을 혼합해 구성할 수 있으며,

처럼 가중 결합 형태로 표현된다. 학습 과정에서 그래프 구조는 메시지 패싱 등을 통해 동적으로 보정(refinement)될 수 있다.

## 3. 4 베이지안 MTGNN 예측
## 3. 4.1 모델 입력/출력 및 예측 목표
시점 t에서 국가별 특징 행렬을

로 두고, 각 행이 에 대응한다. 관측  및 그래프 가 주어졌을 때, 목적은 미래 환율 상태의 예측분포를 추정하는 것이다.

## 3. 4.2 베이지안 예측(불확실성 포함)과 구간(밴드) 해석
추론 시 드롭아웃을 활성화한 채 S회 확률적 전방추론을 수행한다.

$$\hat{y}_{t+h}^{(s)} \sim p(y_{t+h} | X_{1:t}, G; \theta^{(s)}), \quad s = 1, \ldots, S$$

이로부터 예측 평균과 분산을 다음과 같이 근사한다.

$$\mu_{t+h} = \frac{1}{S} \sum_{s=1}^{S} \hat{y}_{t+h}^{(s)}, \quad \sigma^2_{\text{MC},t+h} = \frac{1}{S-1} \sum_{s=1}^{S} (\hat{y}_{t+h}^{(s)} - \mu_{t+h})^2 \quad (11)$$

그러나 MC dropout은 모델의 인식론적 불확실성(Epistemic Uncertainty)만 포착하며, 데이터의 고유한 노이즈인 우연적 불확실성(Aleatoric Uncertainty)을 반영하지 못한다 [18]. 특히 dropout 비율이 낮을 경우($p_{\text{drop}} \approx 0.02$), MC 샘플 간 분산이 극히 작아 예측구간이 지나치게 좁아진다는 한계가 있다.

이를 해결하기 위해 본 연구는 **잔차 기반 예측구간(Residual-based Prediction Interval)** 방법을 제안한다. 검증 데이터에서 실제 예측 오차의 분산($\sigma^2_{\text{residual}}$)을 추정하고, 이를 MC dropout 분산과 결합하여 현실적인 예측구간을 산출한다.

$$\text{Residual: } r_{t} = \hat{y}_{t} - y_{t}, \quad \sigma^2_{\text{residual}} = \frac{1}{T_{\text{val}}} \sum_{t \in \text{val}} r_{t}^2 \quad (12)$$

$$\text{Combined Variance: } \sigma^2_{\text{total}} = \sigma^2_{\text{MC}} + \sigma^2_{\text{residual}} \quad (13)$$

$$\text{95% Prediction Interval: } \text{PI}_{95\%} = \mu_{t+h} \pm 1.96 \times \sqrt{\sigma^2_{\text{total}}} \times \beta \quad (14)$$

여기서 $\beta \in [0.5, 1.0]$는 구간 너비 조정 계수로, 사용자가 시각화 목적에 맞춰 조정할 수 있다. 본 연구의 실험에서는 $\beta = 0.5$를 적용하여 과도한 불확실성 팽창을 억제하였다.

이 방법은 단순히 모델 내부 분산만 사용하는 기존 접근과 달리, 실제 예측 오차 분포를 반영하여 금융 리스크 관리에 더 적합한 예측구간을 제공한다는 장점이 있다 [19]. 또한 구현 측면에서 분위수(quantile) 구간으로도 예측구간을 제공한다(예: 10%–90% 분위). forecast.py 기본 설정은 $\alpha=0.05$, $\alpha=0.95$를 사용한다.

## 3. 4.3 국가 간 환율 격차 정의
국가 i와 j의 상대적 환율 갭을 예측 평균 기반으로 정의한다.

$$\Delta_{i,j}(t+h) = \mu_{i,t+h} - \mu_{j,t+h} \quad (15)$$

이는 국가 간 상대적 강세/약세를 정량화하며, 분위수 구간을 함께 제시하면 갭의 불확실성까지 해석할 수 있다. 예측구간을 함께 고려하면 다음과 같이 표현된다.

$$\text{PI}_{\Delta_{i,j}} = [\mu_{i,t+h} - \mu_{j,t+h}] \pm 1.96 \times \sqrt{\sigma^2_{\text{total},i} + \sigma^2_{\text{total},j}} \quad (16)$$

## 4. 실험 결과 및 분석

## 4.1 데이터셋 및 실험 설정
본 연구는 미국(US Trade Weighted Dollar Index), 한국(kr_fx), 일본(jp_fx) 3개국의 월별 환율 데이터를 사용하였다. 학습 구간은 2011년 7월부터 2023년 12월까지(150개월), 검증 구간은 2024년 1월부터 12월까지(12개월), 테스트 구간은 2025년 1월부터 12월까지(12개월)로 설정하였다. 예측 horizon은 12개월(1년)로 설정하여 2025년 전체 기간에 대한 환율 예측을 수행하였다. 

주요 하이퍼파라미터는 다음과 같다: 입력 시퀀스 길이($T_{\text{in}}$) = 40, 출력 시퀀스 길이($T_{\text{out}}$) = 1 (슬라이딩 윈도우 방식), 학습률($\eta$) = 0.00015, Dropout 비율 = 0.02, 은닉층 차원 = 256, 에폭 수 = 180. MC dropout 샘플 수($S$) = 10으로 설정하였다.

## 4.2 예측 성능 평가
예측 성능은 RSE(Root Relative Squared Error)를 주요 지표로 사용하였으며, 목표 RSE < 0.5를 설정하였다. 실험 결과, 제안 모델은 3개 주요 타겟 국가(미국, 일본, 한국)에서 모두 목표를 달성하였다.

**표 2. 타겟 국가별 RSE 성능**
| 국가 | RSE | 목표 달성 여부 |
|------|-----|--------------|
| US (Trade Weighted Dollar Index) | 0.4884 | ✓ |
| JP (jp_fx) | 0.2721 | ✓ |
| KR (kr_fx) | 0.2970 | ✓ |

또한 래깅(Lagging) 탐지 진단을 통해 예측값이 실제값을 단순 추종하는 것이 아님을 검증하였다. Lag-0 상관계수가 0.96 이상, 방향 일치도가 72~100%로 나타났으며, 범위 비율(range ratio)이 0.97~1.18로 적절한 변동성을 유지하였다.

## 4.3 잔차 기반 예측구간의 효과
기존 MC dropout만 사용한 예측구간은 dropout=0.02의 낮은 값으로 인해 밴드 폭이 지나치게 좁았다. 제안된 잔차 기반 방법을 적용한 결과, 예측구간이 실제 예측 오차 분포를 반영하여 적절한 너비로 확장되었다. $\beta=0.5$ 조정 계수를 통해 시각적 명확성과 통계적 신뢰성 사이의 균형을 달성하였다.

그림 2는 일본 엔화(jp_fx)의 2025년 테스트 예측 결과를 보여준다. 분홍색 밴드가 실제값의 변동을 적절히 포괄하면서도 과도하게 넓지 않아 실용적 리스크 평가에 활용 가능함을 확인할 수 있다.

## 2. 본문 작성요령
본문은 필요에 따라 3~4개의 장으로 구분하여 방법론, 실험 설계 및 결과, 결론 등으로 나누어 서술할 수 있으며, 그 이하에는 절을 두어 내용을 세분화할 수 있다.
학술용어는 될 수 있는 한 국문으로 쓰는 것을 권장한다. 다만, 적당한 번역이 어렵거나 영문 작성이 필수적일 때 영어로 쓸 수 있다. 번역 용어의 경우 이해를 돕기 위해 괄호 안에 영문을 함께 적어야 한다. 본 양식을 통해 작성한 논문은 기본적으로 2쪽 이내이다.

## 2. 1 수식 삽입
수식의 경우, 위-아래 1줄을 띄워 분리하고, 아래와 같이 작성한다.

(1)

수식을 삽입한 경우, 수식에 대한 설명이 포함되어야 한다. 식의 번호는 괄호로 둘러싸서 나타내고 맨 오른쪽에 붙여서 표기한다.

## 2. 2 그림 삽입
그림의 경우, 위-아래 1줄을 띄워 분리하고, 아래와 같이 삽입한다. 캡션은 국문으로 먼저 작성하고, 그 다음 줄에 영문으로 작성한다.

그림 1. 역전사 연산자와 형질도입연산자.
Fig. 1. Reverse transcription operator. (신명조 9pt, 가운데 맞춤)

## 2. 3 표 삽입
표의 경우, 위-아래 1줄을 띄워 분리하고, 아래와 같이 작성한다. 캡션은 국문으로 먼저 작성하고, 그다음 줄에 영문으로 작성한다.
표에 들어가는 단어, 문장 등은 모두 영문 표기를 원칙으로 하되, 부득이하게 한글이 들어갈 경우에는 영문을 병기한다.

표 1. 모델링을 위한 초기 파라미터들.
Table 1. Parameters. (신명조 9pt, 가운데 맞춤)

표를 삽입한 경우, 표에 대한 설명이 포함되어야 한다.

## 3. 인용 및 참고문헌 작성요령(HY중고딕 11pt, 굵게, 가운데 맞춤)

## 3. 1 인용(신명조 9pt, 굵게)
본문에서 인용할 경우, 인용 문장 끝에 참고문헌의 번호를 대괄호 안에 넣어 기재한다. 예시는 다음과 같다.
퍼지 모델을 구하기 위하여 50개의 입출력 데이터가 이용되었다 [1].
본문 내에서 저자를 언급(In-text Citation)할 경우, 저자의 이름을 적고 참고문헌의 번호를 대괄호 안에 넣어 기재한다. 예시는 다음과 같다.
Smith et al. [2]은 신규 알고리즘을 제시하였다. (신명조 9pt, 줄간격 130%)

## 3. 2 참고문헌
참고문헌의 경우, 원칙적으로 한국지능시스템학회 국문지 참고문헌표기법을 따른다. 참고로 이 표기법은 IEEE (Institute of Electrical and Electronics Engineers) 스타일과 Elsevier의 표기법을 절충하여 작성되었다. 이 표기법은 숫자형 인용 방식을 사용하며, 참고문헌 리스트를 본문에 등장한 순서대로 번호를 매긴다 (알파벳 순이 아님). 저자 이름은 이니셜 + 성(last name) 형식을 따른다 (예: J. K. Author).
대략적인 참고문헌의 작성요령은 다음과 같다. 자세한 내용은 IEEE 스타일과 한국지능시스템학회 국문지 작성요령을 참고하는 것을 권장한다.

# 학술 논문
[1] J. K. Author, “Title of paper,” Title of Journal, vol. X, no. X, pp. xxx–xxx, Year.

# 학술대회 논문
[2] J. K. Author, “Title of paper,” Proc. Abbrev. Conf. Name, City, Country, pp. xxx–xxx, Year.

# 단행본
[3] J. K. Author, Title of Book, xth ed. Publisher, Year.

# 웹사이트
[4] Author (if available), “Document title,” Available: URL, [Accessed Date: Date].

이하 참고문헌 섹션에 들어있는 문헌들은 스타일을 참고하기 위한 용도이다.

## 참고문헌

[1] S. Miranda-Agrippino and H. Rey, “US Monetary Policy and the Global Financial Cycle,” The Review of Economic Studies, vol. 87, no. 6, pp. 2754-2776, 2020.

[2] J. Baruník and T. Křehlík, “Measuring the Frequency Dynamics of Financial Connectedness and Systemic Risk,” Journal of Financial Econometrics, vol. 16, no. 2, pp. 271-296, 2018.

[3] Z. Pan, Y. Liang, W. Wang, Y. Yu, M. Li, and J. Zhang, “Urban Traffic Prediction from Spatio-Temporal Data: A Deep Learning Approach,” IEEE Transactions on Knowledge and Data Engineering, vol. 32, no. 4, pp. 720-733, 2019.

[4] D. Diebold and K. Yilmaz, “Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers,” International Journal of Forecasting, vol. 28, no. 1, pp. 57-66, 2012.

[5]  Wu, Z., Pan, S., Chen, G., Long, G., Zhang, C., & Philip, S. Y. (2020). "Connecting the dots: Multivariate time series forecasting with graph neural networks." Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

[6]  Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting." International Conference on Learning Representations (ICLR).

[7] Yu, S., et al. (2024). "FinAgent: A Multi-modal Generalist Agent for Financial Analysis and Reasoning." arXiv preprint arXiv:2402.14411.

[8]  Wu, Q., et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." Proceedings of the 40th International Conference on Machine Learning (ICML).

[9] Pradeepkumar, D., & Ravi, V. (2018). Forecasting financial time series volatility using particle swarm optimization trained quantile regression neural network. Applied Soft Computing, 58, 35-52.

[10] Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions.

[11] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting.

[12] Amat, C., Michalski, T., & Stoltz, G. (2018). Fundamentals and exchange rate forecastability with machine learning methods. Journal of International Money and Finance, 88, 58-74.

[13] Yao, Q., Song, D., Chen, H., Wei, A., Cottrell, G. W., & Bian, G. W. (2018). A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction.

[14] Yu, B., et al. (2018). Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. IJCAI.

[15] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.

[16] Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for deep spatial-temporal graph modeling. Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI), 1907-1913.

[17] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph Attention Networks. International Conference on Learning Representations (ICLR).
[18] Kendall, A., & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 5574-5584.

[19] Gneiting, T., & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation," Journal of the American Statistical Association, vol. 102, no. 477, pp. 359-378.
[3] C. W. Xu, “Fuzzy Model Identification and Self-learning for Dynamic Systems,” IEEE Trans. Syst. Man. Cybern., vol. 17, no. 4, pp. 683-689, 1987.
[4] H. Takagi and M. Sugeno, “Fuzzy Identification of Systems and Its Application to Modeling and Control,” IEEE Trans. on Sys. Man and Cybern., vol. 15, pp. 116-132, 1985.
[5] J. H. Holland, Genetic Algorithms in Search, Optimization and Machine Learning, Addison- Wesley, 1989. (신명조 9pt)

### 표 1
| 그래프 신경망 기반 다중 환율 예측: MTGNN 변형과 앙상블 예측 |
| --- |
| Graph Neural Network-based Multi-Currency FX Forecasting: MTGNN Variants, Ensemble Prediction |
| 김수민 1․ 최지원 2․ 백승환 3․ 한용섭 4․ 박대승 5 Kim Soo-min, Choi Ji-won, Baek Seung-hwan, Han Yong-seop, Park Dae-seung |
| 1수원대학교 컴퓨터학부 SW학과 E-mail: [본인 이메일 입력]@suwon.ac.kr 2수원대학교 컴퓨터학부 SW학과 E-mail: 23017097@suwon.ac.kr 3수원대학교 컴퓨터학부 SW학과 E-mail: sqortmdghks1@suwon.ac.kr 4수원대학교 컴퓨터학부 SW학과 E-mail: [본인 이메일 입력]@suwon.ac.kr 5수원대학교 컴퓨터학부 SW학과 E-mail: [본인 이메일 입력]@suwon.ac.kr |
| 요 약 (HY중고딕 11pt 굵게)   본 연구는 환율 예측의 전통적인 시계열 모델이 간과해 온 국가 간 상호의존성을 포착하는 데 있어 한계를 극복하고, 환율 변동의 복잡한 메커니즘을 규명하는 데 목적이 있다. 이를 위해 그래프 신경망(GNN) 기반의 MTGNN 모델을 채택하여 주요 3개국(미국, 한국, 일본)의 월간 환율 데이터와 거시경제 지표를 결합한 시공간적 상관관계를 통합 분석하였다. 특히 데이터로부터 잠재적 관계를 스스로 학습하는 적응형 그래프 학습 모듈을 통해 예측의 정교함과 확장성을 검증하였다.  연구 결과, 제안 모델은 기존 LSTM 및 통계 모델 대비 우수한 예측 정확도를 기록하였다. 수치 예측의 한계를 넘어 SLM(Small Language Model) 기반의 다중 에이전트 시스템을 결합함으로써 분석의 신뢰성을 강화하였으며, 전문화된 에이전트 기법으로 순환형 체이닝 구조를 통해 예측값의 시계열 일관성을 검증하였다. 또한 특정 오차의 원인을 식별하여 모델의 개선 방향을 제시하고, 장기 예측 결과에 대한 국가별 상호작용 해석과 설명 가능한(Explainable) 인사이트 도출을 지원한다. 결론적으로 본 연구는 인공지능 기술을 활용하여 수치 예측과 경제적 맥락 해석이 통합된 차세대 외환 분석 프레임워크를 정립하였다는 데 학술적 의의가 있다.  (신명조 8.5pt)       키워드 - 환율 예측, 그래프 신경망(GNN), 다중 에이전트 시스템(MAS), 설명 가능한 AI(XAI), 시계열 일관성 검증   This study aims to overcome the limitations of traditional time-series models in capturing cross-country dependencies and to elucidate the complex mechanisms of exchange rate fluctuations. By adopting the MTGNN (Multivariate Time Series Forecasting with Graph Neural Networks) model, this research conducts an integrated analysis of spatio-temporal correlations by combining monthly exchange rate data from five major IMF countries with macroeconomic indicators. In particular, the precision and scalability of the forecasts are validated through an adaptive graph learning module that autonomously identifies latent relationships within the data.  Experimental results demonstrate that the proposed model achieves superior prediction accuracy compared to existing LSTM and statistical models. Beyond numerical forecasting, the reliability of the analysis is reinforced by integrating a Small Language Model (SLM)-based multi-agent system. This specialized agent technique utilizes a recursive chaining flow to verify the time-series consistency of forecasted values while identifying the causes of specific errors to suggest model improvements. Furthermore, this approach facilitates the interpretation of inter-country interactions and the derivation of explainable insights for long-term forecasts. Consequently, this research holds significant academic value by establishing a next-generation foreign exchange analysis framework that integrates numerical prediction with economic contextual interpretation using artificial intelligence. Keywords — Exchange Rate Forecasting, Graph Neural Networks (GNN), Multi-Agent System (MAS), Explainable AI (XAI), Time-series Consistency Verification |

### 표 2
|  | (1) |
| --- | --- |

### 표 3
|  | (2) |
| --- | --- |

### 표 4
|  | (3) |
| --- | --- |

### 표 5
|  | (4) |
| --- | --- |

### 표 6
|  | (5) |
| --- | --- |

### 표 7
|  | (6) |
| --- | --- |

### 표 8
|  | (7) |
| --- | --- |

### 표 9
|  | (8) |
| --- | --- |

### 표 10
|  | (9) |
| --- | --- |

### 표 11
|  | (10) |
| --- | --- |

### 표 12
| Parameter | Value |
| --- | --- |
| Population number | 100 |
| Mutation rate | 0.2 |
