"""
Index Data Forecasting Script (달러 인덱스 또는 가변 지수 데이터 예측)
- 일반 환율과 달리, 인덱스는 1.0에 고정되지 않음
- 실제 인덱스 값 변화를 유지하며 예측
"""

import numpy as np
import os
import torch
import sys
import re
import argparse
import pandas as pd
from matplotlib import pyplot
import matplotlib.dates as mdates

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


def should_clamp_nonnegative(name):
    lower = name.lower()
    if 'trade_balance' in lower or 'balanced_of_trade' in lower:
        return False
    return True

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


def resolve_latest_tuned_model(script_dir, fallback_model_file):
    tuning_root = os.path.join(script_dir, 'tuning_runs')
    if not os.path.isdir(tuning_root):
        return fallback_model_file

    run_dirs = [
        os.path.join(tuning_root, d)
        for d in os.listdir(tuning_root)
        if os.path.isdir(os.path.join(tuning_root, d))
    ]
    run_dirs.sort(reverse=True)

    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, 'best_summary.txt')
        if not os.path.exists(summary_path):
            continue
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                text = f.read()
            m = re.search(r'log_file=(.*run_(\d+)\.log)', text)
            if not m:
                continue
            run_id = int(m.group(2))
            ckpt_path = os.path.join(run_dir, 'checkpoints', f'model_{run_id:03d}.pt')
            if os.path.exists(ckpt_path):
                return ckpt_path
        except Exception:
            continue

    return fallback_model_file


def smooth_series(arr, alpha):
    if alpha >= 0.999:
        return arr
    return np.array(exponential_smoothing(arr, alpha), dtype=np.float32)

# ==========================================
# Plotting Functions
# ==========================================
def plot_forecast(data, forecast, confidence, name, dates_hist, dates_future, output_dir, color="#1f77b4", linestyle='--', is_index=False):
    """개별 노드 플롯"""
    if should_clamp_nonnegative(name):
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

        ax.plot(x_past, y_past, '-', label=consistent_name(col[i]), color=color, linewidth=1.5)
        ax.plot(dates_future, y_fut, linestyle='--', color=color, linewidth=2)

        # 신뢰도 영역 표시
        ax.fill_between(dates_future, y_fut - c_fut, y_fut + c_fut, color=color, alpha=0.25)

    if x_start is None:
        x_start = dates_hist[0]
    if x_end is None:
        x_end = dates_future[-1] + pd.Timedelta(days=30)
    ax.set_xlim(pd.Timestamp(x_start), pd.Timestamp(x_end))

    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if add_last_month_tick:
        end_tick = pd.Timestamp(dates_future[-1])
        year_ticks = pd.date_range(pd.Timestamp(x_start).normalize(), end_tick.normalize(), freq="YS")
        ticks = list(year_ticks)

        if end_tick not in ticks:
            ticks.append(end_tick)

        labels = [t.strftime('%Y') for t in year_ticks]
        if ticks[-1] == end_tick and (len(labels) == len(ticks) - 1):
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

parser = argparse.ArgumentParser(description='Forecast plotting and export')
parser.add_argument('--model', type=str, default='', help='optional model checkpoint path (.pt)')
parser.add_argument('--mc_runs', type=int, default=20)
parser.add_argument('--horizon', type=int, default=36)
parser.add_argument('--hist_alpha', type=float, default=0.3, help='history smoothing alpha')
parser.add_argument('--future_alpha', type=float, default=1.0, help='future smoothing alpha (1.0 means no smoothing)')
args = parser.parse_args()

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_file = os.path.join(script_dir, 'data', 'sm_data.csv')
model_file = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'model.pt')
if args.model.strip():
    model_file = args.model
else:
    model_file = resolve_latest_tuned_model(script_dir, model_file)

# 출력 디렉토리
plot_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'plots')
pt_plots_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'pt_plots')
data_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'data')

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
    if dates_all is None:
        LAST_OBS = pd.Timestamp("2025-07-01")
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

# Normalization (인덱스 값을 강제로 1.0으로 고정하지 않음)
scale = np.ones(m)
dat = np.zeros(rawdat.shape)

for i in range(m):
    scale[i] = np.max(np.abs(rawdat[:, i]))
    if scale[i] == 0: 
        scale[i] = 1.0
    dat[:, i] = rawdat[:, i] / scale[i]

print(f"Data normalized. Scale values: {scale[:5]}...")  # 처음 5개만 표시

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

# 초기 입력 준비
X_init = torch.from_numpy(dat[-seq_len:, :]).float().to(device)

# Bayesian Estimation (Dropout MC)
num_runs, horizon = args.mc_runs, args.horizon
outputs = []

print(f"Running Bayesian Forecast ({num_runs} MC runs, {horizon} month horizon)...")

P = seq_len

model.train()
with torch.no_grad():
    tmp_in = X_init.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).contiguous()
    tmp_out = model(tmp_in)
    pred_len = int(tmp_out.size(1))

for r in range(num_runs):
    curr_X = X_init.clone()
    preds = []
    len_preds = 0

    model.train()
    with torch.no_grad():
        while len_preds < horizon:
            curr_input = curr_X.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).contiguous()

            # train_test.py와 동일한 RevIN 전처리/복원
            w_mean = curr_input.mean(dim=-1, keepdim=True)
            w_std = curr_input.std(dim=-1, keepdim=True)
            w_std[w_std == 0] = 1
            curr_input_norm = (curr_input - w_mean) / w_std

            out = model(curr_input_norm)

            pred_block = out.squeeze(3).squeeze(0)  # [T_out, N]
            wm = w_mean[0, 0, :, 0]
            ws = w_std[0, 0, :, 0]
            pred_level = pred_block * ws.unsqueeze(0) + wm.unsqueeze(0)

            # 인덱스를 강제로 1.0으로 고정하지 않음
            # if index_idx != -1:
            #     pred_level[:, index_idx] = 1.0

            need = horizon - len_preds
            take = min(pred_level.size(0), need)
            take_block = pred_level[:take, :].cpu().numpy()

            preds.append(take_block)
            len_preds += take

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

# Denormalization
scale_torch = torch.from_numpy(scale).float()
dat_denorm = torch.from_numpy(dat).float() * scale_torch
Y_denorm = Y * scale_torch
confidence_denorm = confidence * scale_torch
variance_denorm = variance * scale_torch

save_data(dat_denorm, Y_denorm, confidence_denorm, variance_denorm, col, data_out_dir)

# Smoothing
all_data = torch.cat((dat_denorm, Y_denorm), dim=0)
all_conf = torch.cat((torch.zeros_like(dat_denorm), confidence_denorm), dim=0)

hist_plot_list, fut_plot_list, conf_plot_list = [], [], []
for i in range(m):
    hist_arr = dat_denorm[:, i].cpu().numpy()
    fut_arr = Y_denorm[:, i].cpu().numpy()
    conf_arr = confidence_denorm[:, i].cpu().numpy()

    hist_plot_list.append(smooth_series(hist_arr, args.hist_alpha))
    fut_plot_list.append(smooth_series(fut_arr, args.future_alpha))
    conf_plot_list.append(smooth_series(conf_arr, min(args.future_alpha, 0.5)))

hist_plot = torch.tensor(np.array(hist_plot_list)).T
fut_plot = torch.tensor(np.array(fut_plot_list)).T
conf_plot_fut = torch.tensor(np.array(conf_plot_list)).T

# 날짜 설정
HIST_END = pd.Timestamp("2025-07-01")
dates_hist = pd.date_range(end=HIST_END, periods=len(df), freq="MS").tolist()

FORECAST_START = HIST_END + pd.DateOffset(months=1)
dates_future = pd.date_range(start=FORECAST_START, periods=horizon, freq="MS").tolist()

print(f"Forecast range: {dates_future[0].strftime('%Y-%m')} ~ {dates_future[-1].strftime('%Y-%m')}")

# 플롯 대상 선택 (US Trade Weighted Dollar Index + 주요 FX)
preferred_names = [
    'us_Trade Weighted Dollar Index',
    'kr_fx',
    'jp_fx',
    'cn_fx',
    'uk_fx',
]
target_indices = [i for i, name in enumerate(col) if name in preferred_names]

# 대소문자/표기 차이 대비 fallback
if not target_indices:
    fallback_tokens = ['trade weighted dollar index', 'kr_fx', 'jp_fx', 'cn_fx', 'uk_fx']
    target_indices = sorted(list(set([
        i for token in fallback_tokens for i, n in enumerate(col)
        if token in n.lower() and 'trade_balance' not in n.lower() and 'balanced_of_trade' not in n.lower()
    ])))

if not target_indices:
    target_indices = list(range(m))

plot_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# 개별 플롯 생성
print("Generating individual plots...")
for idx, i in enumerate(target_indices):
    plot_forecast(hist_plot[:, i], fut_plot[:, i], conf_plot_fut[:, i], col[i], 
                  dates_hist, dates_future, pt_plots_dir, 
                  color=plot_colours[idx % len(plot_colours)], 
                  linestyle='--',
                  is_index=False)

# Multi-Node Plot - FULL
print("Generating multi-node plots...")
plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=hist_plot,
    smoothed_fut=fut_plot,
    smoothed_conf_fut=conf_plot_fut,
    target_indices=target_indices,
    col=col,
    index_idx=index_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "Multi_Node_Index_FULL.png"),
    x_start=dates_hist[0],
    x_end=pd.Timestamp("2028-07-31"),
)

# Multi-Node Plot - ZOOM
plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=hist_plot,
    smoothed_fut=fut_plot,
    smoothed_conf_fut=conf_plot_fut,
    target_indices=target_indices,
    col=col,
    index_idx=index_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "Multi_Node_Index_ZOOM.png"),
    x_start=pd.Timestamp("2022-08-01"),
    x_end=pd.Timestamp("2028-07-31"),
)

print("=== 최종 완료 ===")
print(f"All outputs saved to: {plot_dir}")
