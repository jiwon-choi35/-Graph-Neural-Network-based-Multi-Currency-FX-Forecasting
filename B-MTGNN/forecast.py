"""
Index Data Forecasting Script (달러 인덱스 또는 가변 지수 데이터 예측)
- 일반 환율과 달리, 인덱스는 1.0에 고정되지 않음
- 실제 인덱스 값 변화를 유지하며 예측
- [수정] 학습 코드와 동일한 Standardization (Mean/Std) 정규화 적용
"""

import numpy as np
import os
import torch
import sys
import pandas as pd
from matplotlib import pyplot
import matplotlib.dates as mdates
import csv
from collections import defaultdict

# Import from current directory
from net import gtnet

# 기본 설정
pyplot.rcParams['savefig.dpi'] = 300

# ==========================================
# Helper Functions
# ==========================================

def exponential_smoothing(series, alpha):
    """지수평활"""
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result

def consistent_name(name):
    """컬럼명 정리"""
    name = name.replace('-ALL', '').replace('Mentions-', '').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2: 
                result += word
            else: 
                result += word[0].upper() + word[1:]
            if i < len(words) - 1: 
                result += ' '
        return result
    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word: 
            result += word
        else: 
            result += word[0] + (word[1:].lower())
        if i < len(words) - 1: 
            result += ' '
    return result

def zero_negative_curves(data, forecast):
    """음수값 제거"""
    data = torch.clamp(data, min=0)
    forecast = torch.clamp(forecast, min=0)
    return data, forecast

def save_data(data, forecast, confidence, variance, col, output_dir):
    """예측 데이터 저장"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(data.shape[1]):
        d = data[:,i]
        f = forecast[:,i]
        c = confidence[:,i]
        v = variance[:,i]
        name = col[i]
        
        safe_name = name.replace('/', '_')
        with open(os.path.join(output_dir, safe_name + '.txt'), 'w') as ff:
            ff.write('Data: ' + str(d.tolist()) + '\n')
            ff.write('Forecast: ' + str(f.tolist()) + '\n')
            ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
            ff.write('Variance: ' + str(v.tolist()) + '\n')

# ==========================================
# Plotting Functions
# ==========================================
def plot_forecast(data, forecast, confidence, name, dates_hist, dates_future, output_dir, color="#1f77b4", linestyle='-', is_index=False):
    """개별 노드 플롯"""
    data, forecast = zero_negative_curves(data, forecast)
    if torch.is_tensor(data): 
        data = data.cpu()
    if torch.is_tensor(forecast): 
        forecast = forecast.cpu()
    if torch.is_tensor(confidence): 
        confidence = confidence.cpu()

    pyplot.style.use("default") 
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

    d = torch.cat((data, forecast[0:1]), dim=0).numpy()
    f = forecast.numpy()
    c = confidence.numpy()
    clean_name = consistent_name(name)
    all_dates = dates_hist + dates_future

    ax.plot(range(len(d)), d, '-', color=color, label=clean_name, linewidth=2)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, linestyle=linestyle, color=color, linewidth=2)
    
    # 모든 데이터에 신뢰도 영역 표시
    ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=color, alpha=0.3)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, color=color, linewidth=0.8, alpha=0.6)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f + c, color=color, linewidth=0.8, alpha=0.6)

    x_ticks_pos = [i for i, date in enumerate(all_dates) if date.month == 1]
    last_pos = len(all_dates) - 1
    if last_pos not in x_ticks_pos:
        x_ticks_pos.append(last_pos)

    ax.set_xticks(x_ticks_pos)
    ax.set_xticklabels(
        [all_dates[i].strftime('%Y') if all_dates[i].month == 1 else all_dates[i].strftime('%b-%y') for i in x_ticks_pos],
        rotation=90, fontsize=13
    )
    ax.set_ylabel(f"{consistent_name(name)}", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(clean_name, y=1.03, fontsize=18)
    fig.set_size_inches(10, 7)

    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    safe_name = clean_name.replace('/', '_')
    pyplot.savefig(os.path.join(output_dir, safe_name + '.png'), bbox_inches="tight")
    pyplot.close(fig)
    print(f"Individual Plot saved: {safe_name}.png (Color: {color})")


def plot_multi_node(dates_hist, dates_future, smoothed_hist, smoothed_fut, smoothed_conf_fut,
                    target_indices, col, index_idx, plot_colours, out_path,
                    x_start=None, x_end=None, add_last_month_tick=True):
    """다중 노드 플롯 - 하나의 그림에 5개 노드"""
    fig, ax = pyplot.subplots(figsize=(15, 10))

    connect_date = dates_future[0]
    x_past = dates_hist + [connect_date]

    for idx, i in enumerate(target_indices):
        # 각 노드의 첫 값 기준으로 정규화 (1.0부터 시작)
        base_value = smoothed_hist[0, i]
        
        y_past = torch.cat((smoothed_hist[:, i], smoothed_fut[0:1, i]), dim=0).numpy() / base_value
        y_fut = smoothed_fut[:, i].numpy() / base_value
        c_fut = smoothed_conf_fut[:, i].numpy() / base_value

        color = plot_colours[idx % len(plot_colours)]
        is_index = 'weighted' in col[i].lower() or 'trade' in col[i].lower()
        
        # 보라색은 무조건 점선, 그 외는 is_index에 따라 결정
        linestyle = '--' if color == '#9467bd' else ('-' if is_index else '--')

        ax.plot(x_past, y_past, '-', label=consistent_name(col[i]), color=color, linewidth=1.5)
        ax.plot(dates_future, y_fut, linestyle=linestyle, color=color, linewidth=2)

        # 신뢰도 영역 표시
        ax.fill_between(dates_future, y_fut - c_fut, y_fut + c_fut, color=color, alpha=0.25)

    # 예측 시작점에 세로선 추가
    ax.axvline(x=dates_future[0], color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    if x_start is None:
        x_start = dates_hist[0]
    if x_end is None:
        # 마지막 틱(2026/Dec)이 그래프 맨 끝에 오도록, 최소한의 여유만 둠
        x_end = dates_future[-1] + pd.Timedelta(days=15)
    ax.set_xlim(pd.Timestamp(x_start), pd.Timestamp(x_end))

    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if add_last_month_tick:
        end_tick = pd.Timestamp(dates_future[-1])
        year_ticks = pd.date_range(pd.Timestamp(x_start).normalize(), end_tick.normalize(), freq="YS")
        ticks = list(year_ticks)
        labels = [t.strftime('%Y') for t in year_ticks]

        # 마지막 틱 추가
        if end_tick not in ticks:
            ticks.append(end_tick)
            # 마지막 틱이 연말이고, 같은 연도의 연초 틱이 있으면 연초 틱 제거 (공백 방지)
            end_year = end_tick.year
            if len(year_ticks) > 0 and year_ticks[-1].year == end_year:
                # 같은 연도의 연초 틱과 라벨 제거
                ticks = [t for t in ticks[:-1] if t.year != end_year] + [end_tick]
                labels = [l for i, l in enumerate(labels) if year_ticks[i].year != end_year]
            labels.append(end_tick.strftime('%Y/%b'))

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1.03))
    ax.grid(True, linestyle=':', alpha=0.6)
    pyplot.title("US Index & FX Rate Forecast (Normalized to 1.0)", fontsize=18)
    pyplot.savefig(out_path, bbox_inches="tight")
    pyplot.close()


# ==========================================
# Main Execution Block
# ==========================================

# 경로 설정 (train.py와 동일: 스크립트 위치 = 프로젝트 루트)
script_dir = os.path.dirname(os.path.abspath(__file__))
# train_test/train과 동일: data/data.csv 우선
data_file = os.path.join(script_dir, 'data', 'data.csv')
if not os.path.exists(data_file):
    data_file = os.path.join(script_dir, 'ExchangeRate_dataset.csv')
# 원본 README: 운용 모델 o_model.pt로 미래 예측. 없으면 model.pt 사용
model_file = os.path.join(script_dir, 'model', 'Bayesian', 'o_model.pt')
if not os.path.exists(model_file):
    model_file = os.path.join(script_dir, 'model', 'Bayesian', 'model.pt')

# 출력 디렉토리
plot_dir = os.path.join(script_dir, 'model', 'Bayesian', 'forecast', 'plots')
pt_plots_dir = os.path.join(script_dir, 'model', 'Bayesian', 'forecast', 'pt_plots')
data_out_dir = os.path.join(script_dir, 'model', 'Bayesian', 'forecast', 'data')

for d in [plot_dir, pt_plots_dir, data_out_dir]:
    if not os.path.exists(d): 
        os.makedirs(d, exist_ok=True)

# Device 설정
device = torch.device('cpu')
print(f"Using device: {device}")

# 데이터 로드
try:
    print(f"Reading data from: {data_file}")
    df_raw = pd.read_csv(data_file)

    # 날짜 컬럼 찾기
    date_col = next((c for c in ["Date", "date", "DATA", "data"] if c in df_raw.columns), None)

    if date_col is not None:
        dates_all = pd.to_datetime(df_raw[date_col], errors="coerce")
        df = df_raw.drop(columns=[date_col])
        if dates_all.isna().all():
            dates_all = None
    else:
        dates_all = None
        df = df_raw

    # 수치 변환 및 결측치 처리
    df = df.apply(pd.to_numeric, errors="coerce").ffill().fillna(0)

    # 날짜 설정
    # 규칙 4: forecast 예측기간은 26년 1월~12월이므로, 마지막 관측은 25년 12월
    if dates_all is None:
        LAST_OBS = pd.Timestamp("2025-12-01")
        dates_all = pd.date_range(end=LAST_OBS, periods=len(df), freq="MS").tolist()
    else:
        dates_all = pd.Series(dates_all).ffill()
        dates_all = [pd.Timestamp(d).to_period("M").to_timestamp() for d in dates_all.tolist()]

    col = df.columns.tolist()
    rawdat = df.values
    n, m = rawdat.shape
    print(f"Data loaded: {n} months, {m} nodes | last={pd.Timestamp(dates_all[-1]).strftime('%Y-%m')}")

except FileNotFoundError:
    print(f"❌ Error: 파일을 찾을 수 없습니다: {data_file}")
    sys.exit()

# Index 컬럼 찾기
index_idx = next((i for i, name in enumerate(col) 
                   if 'us_' in name.lower() or 'dollar' in name.lower() or 'index' in name.lower()), -1)
if index_idx != -1: 
    print(f"Found Index column at index {index_idx}: {col[index_idx]}")

# ==============================================================================
# 5. Normalization: Z-score (train.py normalize=3과 동일)
# ==============================================================================
mean_vals = np.mean(rawdat, axis=0)
std_vals = np.std(rawdat, axis=0)

# 표준편차가 0인 경우(상수 데이터) 1로 대체하여 나눗셈 에러 방지
std_vals[std_vals == 0] = 1.0 

dat = (rawdat - mean_vals) / std_vals

print(f"Data normalized (Standardization). Mean/Std applied.")

# 모델 로드
print(f"Loading model from: {model_file}")
try:
    with open(model_file, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
        model.to(device)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

# Input sequence length 결정
try:
    seq_len = int(getattr(model, 'seq_length', None) or 
                  (getattr(model.module, 'seq_length', None) if hasattr(model, 'module') else None))
except Exception:
    seq_len = None

if seq_len is None:
    seq_len = 10

print(f"Using input sequence length (seq_len) = {seq_len}")

# 초기 입력 준비 (전역 Z-score 정규화된 데이터)
X_init = torch.from_numpy(dat[-seq_len:, :]).float().to(device)

# Bayesian Estimation (Dropout MC)
# 규칙 4: forecast 예측기간은 26년 1월~12월 (1년 = 12개월)
num_runs, horizon = 20, 12
outputs = []

print(f"Running Bayesian Forecast ({num_runs} MC runs, {horizon} month horizon)...")
print(f"[RevIN] Applying Reversible Instance Normalization per window (matching training)")

P = seq_len

model.train()
with torch.no_grad():
    # Test shape with RevIN
    tmp_in = X_init.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).contiguous()  # [1, 1, N, T]
    # RevIN: Per-Window Normalization
    w_mean = tmp_in.mean(dim=-1, keepdim=True)  # [1, 1, N, 1]
    w_std = tmp_in.std(dim=-1, keepdim=True)    # [1, 1, N, 1]
    w_std[w_std == 0] = 1
    tmp_in_norm = (tmp_in - w_mean) / w_std
    tmp_out = model(tmp_in_norm)
    pred_len = int(tmp_out.size(1))

for r in range(num_runs):
    curr_X = X_init.clone()  # 전역 Z-score 정규화된 데이터
    preds = []
    len_preds = 0

    model.train()
    with torch.no_grad():
        while len_preds < horizon:
            # 입력 준비: [1, 1, N, T] 형태로 변환
            curr_input = curr_X.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).contiguous()  # [1, 1, N, T]
            
            # ===== RevIN: Per-Window Normalization (학습 시와 동일) =====
            w_mean = curr_input.mean(dim=-1, keepdim=True)  # [1, 1, N, 1]
            w_std = curr_input.std(dim=-1, keepdim=True)    # [1, 1, N, 1]
            w_std[w_std == 0] = 1
            curr_input_norm = (curr_input - w_mean) / w_std
            
            # 나중에 역정규화를 위해 통계 저장
            wm = w_mean[0, 0, :, 0]  # [N]
            ws = w_std[0, 0, :, 0]   # [N]
            # ============================================================
            
            # 모델 예측 (RevIN 정규화된 입력 사용)
            out = model(curr_input_norm)
            
            # 출력 형태 변환
            pred_block = out.squeeze(3).squeeze(0)  # [T_out, N]
            
            # ===== RevIN 역정규화 =====
            # pred_block의 각 시간 단계에 대해 역정규화
            pred_level = pred_block * ws.unsqueeze(0) + wm.unsqueeze(0)  # [T_out, N]
            # ===========================

            need = horizon - len_preds
            take = min(pred_level.size(0), need)
            take_block = pred_level[:take, :].cpu().numpy()

            preds.append(take_block)
            len_preds += take

            # 다음 입력 업데이트 (RevIN 역정규화된 예측값 사용)
            new_X = np.concatenate([curr_X.cpu().numpy(), take_block], axis=0)
            curr_X = torch.from_numpy(new_X[-P:, :]).float().to(device).contiguous()

    outputs.append(torch.tensor(np.concatenate(preds, axis=0)))

print(f"✅ Forecast generation complete")

# 통계 계산
outputs = torch.stack(outputs)
Y = torch.mean(outputs, dim=0)
std_dev = torch.std(outputs, dim=0)
confidence = 1.96 * std_dev / torch.sqrt(torch.tensor(num_runs))
variance = torch.var(outputs, dim=0)

# ==============================================================================
# [수정] 9. Denormalization (Standardization 역변환)
# 원래 값 = (정규화된 값 * 표준편차) + 평균
# ==============================================================================
mean_torch = torch.from_numpy(mean_vals).float()
std_torch = torch.from_numpy(std_vals).float()

dat_denorm = torch.from_numpy(dat).float() * std_torch + mean_torch
Y_denorm = Y * std_torch + mean_torch
confidence_denorm = confidence * std_torch
variance_denorm = variance * (std_torch ** 2)

save_data(dat_denorm, Y_denorm, confidence_denorm, variance_denorm, col, data_out_dir)

# Smoothing
all_data = torch.cat((dat_denorm, Y_denorm), dim=0)
all_conf = torch.cat((torch.zeros_like(dat_denorm), confidence_denorm), dim=0)

smoothed_data_list, smoothed_conf_list = [], []
for i in range(m):
    smoothed_data_list.append(exponential_smoothing(all_data[:, i].cpu().numpy(), 0.3))
    smoothed_conf_list.append(exponential_smoothing(all_conf[:, i].cpu().numpy(), 0.1))

smoothed_dat = torch.tensor(np.array(smoothed_data_list)).T
smoothed_confidence = torch.tensor(np.array(smoothed_conf_list)).T
smoothed_hist = smoothed_dat[:-horizon, :]
smoothed_fut = smoothed_dat[-horizon:, :]
smoothed_conf_fut = smoothed_confidence[-horizon:, :]

# 날짜 설정
# 규칙 4: forecast 예측기간은 26년 1월~12월 (1년 = 12개월)
HIST_END = pd.Timestamp("2025-12-01")
dates_hist = pd.date_range(end=HIST_END, periods=len(df), freq="MS").tolist()

FORECAST_START = pd.Timestamp("2026-01-01")  # 26년 1월부터 시작
dates_future = pd.date_range(start=FORECAST_START, periods=horizon, freq="MS").tolist()

print(f"Forecast range: {dates_future[0].strftime('%Y-%m')} ~ {dates_future[-1].strftime('%Y-%m')}")

# 플롯 대상 선택: 달러 인덱스(us_Trade Weighted Dollar Index), kr_fx, jp_fx만 사용
desired_targets = ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx']
desired_lower = {t.lower() for t in desired_targets}
target_indices = [i for i, n in enumerate(col) if n.lower() in desired_lower]

# 만약 컬럼명이 조금 다를 경우(대소문자/공백 등) 대비용 fallback
if not target_indices:
    target_indices = []
    for i, n in enumerate(col):
        name_lower = n.lower()
        if 'us_trade weighted dollar index' in name_lower or name_lower == 'us_trade weighted dollar index':
            target_indices.append(i)
        elif 'kr_fx' in name_lower:
            target_indices.append(i)
        elif 'jp_fx' in name_lower:
            target_indices.append(i)

target_indices = sorted(list(set(target_indices)))
if not target_indices:
    target_indices = list(range(m))

plot_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# 개별 플롯 생성
print("Generating individual plots...")
for idx, i in enumerate(target_indices):
    plot_forecast(smoothed_hist[:, i], smoothed_fut[:, i], smoothed_conf_fut[:, i], col[i], 
                  dates_hist, dates_future, pt_plots_dir, 
                  color=plot_colours[idx % len(plot_colours)], 
                  linestyle='-',
                  is_index=False)

# Multi-Node Plot - FULL
print("Generating multi-node plots...")
# 마지막 예측 날짜(2026-12-01)가 그래프 맨 끝에 오도록 x_end 설정
forecast_end = dates_future[-1] + pd.Timedelta(days=15)  # 약간의 여유만 둠
plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=smoothed_hist,
    smoothed_fut=smoothed_fut,
    smoothed_conf_fut=smoothed_conf_fut,
    target_indices=target_indices,
    col=col,
    index_idx=index_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "Multi_Node_Index_FULL.png"),
    x_start=dates_hist[0],
    x_end=forecast_end,
)

# Multi-Node Plot - ZOOM
plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=smoothed_hist,
    smoothed_fut=smoothed_fut,
    smoothed_conf_fut=smoothed_conf_fut,
    target_indices=target_indices,
    col=col,
    index_idx=index_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "Multi_Node_Index_ZOOM.png"),
    x_start=pd.Timestamp("2022-08-01"),
    x_end=forecast_end,
)

print("=== 최종 완료 ===")
print(f"All outputs saved to: {plot_dir}")