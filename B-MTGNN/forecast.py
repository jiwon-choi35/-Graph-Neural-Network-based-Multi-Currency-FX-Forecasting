import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from collections import defaultdict
from matplotlib import pyplot
import matplotlib.dates as mdates
import random
import pandas as pd

# Import from current directory
from net import gtnet

# 기본 설정
pyplot.rcParams['savefig.dpi'] = 300 

# ==========================================
# 1. Rebase / Gap Param Functions
# ==========================================
def make_gap_params(
    hist: torch.Tensor,        # [T, N] 과거
    col_names,                 # list[str]
    base_mode="first",         # "first"
    base_index=1.0,            # 시작점 통일 값
    eps=1e-8
):
    if base_mode == "first":
        base = hist[0, :]      # [N]
    elif base_mode == "last":
        base = hist[-1, :]     # [N]
    else:
        raise ValueError("base_mode must be 'first' or 'last'")

    base = base.detach().cpu()
    base_np = base.numpy()

    # 곱셈 rebase 계수
    mul = (base_index / (torch.abs(base) + eps)).cpu()  # [N]
    mul_np = mul.numpy()

    # 덧셈 shift 오프셋
    add = (torch.tensor(base_index) - base).cpu()       # [N]
    add_np = add.numpy()

    df = pd.DataFrame({
        "node": col_names,
        "base_mode": [base_mode]*len(col_names),
        "base_value": base_np,
        "mul_factor": mul_np,
        "add_offset": add_np,
        "base_index": [base_index]*len(col_names),
    })
    return df, base, mul, add

def apply_rebase_mul(x: torch.Tensor, mul_factor: torch.Tensor):
    return x * mul_factor.to(x.device)

# ==========================================
# 2. Helper Functions
# ==========================================

def exponential_smoothing(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result

def consistent_name(name):
    name = name.replace('-ALL', '').replace('Mentions-', '').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2: result += word
            else: result += word[0].upper() + word[1:]
            if i < len(words) - 1: result += ' '
        return result
    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word: result += word
        else: result += word[0] + (word[1:].lower())
        if i < len(words) - 1: result += ' '
    return result

def zero_negative_curves(data, forecast):
    data = torch.clamp(data, min=0)
    forecast = torch.clamp(forecast, min=0)
    return data, forecast

def save_data(data, forecast, confidence, variance, col, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(data.shape[1]):
        d = data[:,i]
        f = forecast[:,i]
        c = confidence[:,i]
        v = variance[:,i]
        name = col[i]
        
        safe_name = name.replace('/','_')
        with open(os.path.join(output_dir, safe_name+'.txt'), 'w') as ff:
            ff.write('Data: '+str(d.tolist())+'\n')
            ff.write('Forecast: '+str(f.tolist())+'\n')
            ff.write('95% Confidence: '+str(c.tolist())+'\n')
            ff.write('Variance: '+str(v.tolist())+'\n')

def save_gap_fx_yearly(forecast, ref_name, comp_names, index, output_dir, dates_future):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    year_indices = defaultdict(list)
    for idx, date in enumerate(dates_future):
        year_indices[date.year].append(idx)
    
    sorted_years = sorted(year_indices.keys())
    header = ['Comparison', 'Target_Country'] + [str(y) for y in sorted_years]
    filename = 'US_FX_Gap_Analysis.csv'
    
    if ref_name not in index:
        print(f"Error: Reference {ref_name} not found in index.")
        return

    ref_idx = index[ref_name]
    us_data = forecast[:, ref_idx].detach().cpu().numpy() if torch.is_tensor(forecast) else forecast[:, ref_idx]

    table = []
    for comp in comp_names:
        if comp not in index: continue
        target_clean = consistent_name(comp)
        row = [f"US vs {target_clean}", target_clean]
        comp_idx = index[comp]
        comp_data = forecast[:, comp_idx].detach().cpu().numpy() if torch.is_tensor(forecast) else forecast[:, comp_idx]
            
        gap_values = []
        for year in sorted_years:
            indices = year_indices[year]
            us_mean = np.mean(us_data[indices])
            comp_mean = np.mean(comp_data[indices])
            gap = us_mean - comp_mean
            gap_values.append(gap)
            
        row.extend(gap_values)
        table.append(row)
    
    sorted_table = sorted(table, key=lambda r: abs(sum(r[2:])), reverse=True)
    with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in sorted_table:
            writer.writerow(row)
    print(f"✅ FX Gap CSV saved: {os.path.join(output_dir, filename)}")

# ==========================================
# 3. Plotting Functions
# ==========================================
def plot_forecast(data, forecast, confidence, name, dates_hist, dates_future, output_dir, color="#1f77b4", linestyle='-', is_us=False):
    data, forecast = zero_negative_curves(data, forecast)
    if torch.is_tensor(data): data = data.cpu()
    if torch.is_tensor(forecast): forecast = forecast.cpu()
    if torch.is_tensor(confidence): confidence = confidence.cpu()

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
    
    if not is_us:
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=color, alpha=0.3)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, color=color, linewidth=0.8, alpha=0.6)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f + c, color=color, linewidth=0.8, alpha=0.6)

    x_ticks_pos = [i for i, date in enumerate(all_dates) if date.month == 1]
    # 끝(예: Jul-28) 라벨도 포함
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

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    safe_name = clean_name.replace('/', '_')
    pyplot.savefig(os.path.join(output_dir, safe_name + '.png'), bbox_inches="tight")
    pyplot.close(fig)
    print(f"Individual Plot saved: {safe_name}.png (Color: {color})")


def build_year_ticks(all_dates):
    # all_dates may contain daily timestamps; pick first January occurrence per year
    pos = []
    lab = []
    seen = set()
    for i, d in enumerate(all_dates):
        y = int(d.year)
        if (y not in seen) and (int(d.month) == 1):
            pos.append(i)
            lab.append(str(y))
            seen.add(y)
    return pos, lab


# --- Main Execution Block ---

# 프로젝트 디렉토리 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 데이터 및 모델 파일
data_file = os.path.join(script_dir, 'data', 'ExchangeRate_dataset.csv')
model_file = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'model.pt')

# 출력 디렉토리
plot_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'plots')
pt_plots_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'pt_plots')
data_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'data')
rebased_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'data_rebased')
gap_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'gap')

# 모든 저장 디렉토리 자동 생성 (workspace 내)
for d in [plot_dir, pt_plots_dir, data_out_dir, rebased_out_dir, gap_out_dir]:
    if not os.path.exists(d): os.makedirs(d, exist_ok=True)

# 3. Device 설정 (CPU 전용)
device = torch.device('cpu')
print(f"Using device: {device}")

# 4. 데이터 로드

try:
    print(f"Reading data from: {data_file}")
    df_raw = pd.read_csv(data_file)

    # Date 컬럼이 있으면 사용하고(라벨용), 모델 입력에서는 제거
    date_col = next((c for c in ["Date", "date", "DATA", "data"] if c in df_raw.columns), None)

    if date_col is not None:
        dates_all = pd.to_datetime(df_raw[date_col], errors="coerce")
        df = df_raw.drop(columns=[date_col])
        if dates_all.isna().all():
            dates_all = None
    else:
        dates_all = None
        df = df_raw

    # 숫자화(값은 그대로 유지, Date 컬럼이 섞여있어도 안전)
    df = df.apply(pd.to_numeric, errors="coerce").ffill().fillna(0)

    # Date 컬럼이 없으면: 월별 데이터가 2025/Jul에 끝난다고 가정해 라벨 생성
    if dates_all is None:
        LAST_OBS = pd.Timestamp("2025-07-01")  # 최종 관측: 25/Jul
        dates_all = pd.date_range(end=LAST_OBS, periods=len(df), freq="MS").tolist()
    else:
        # 월 시작으로 정규화
        dates_all = pd.Series(dates_all).ffill()
        dates_all = [pd.Timestamp(d).to_period("M").to_timestamp() for d in dates_all.tolist()]

    col = df.columns.tolist()
    index = {name: i for i, name in enumerate(col)}
    rawdat = df.values
    n, m = rawdat.shape
    print(f"Data loaded: {n} months, {m} nodes | last={pd.Timestamp(dates_all[-1]).strftime('%Y-%m')}")
except FileNotFoundError:
    print(f"❌ Error: 파일을 찾을 수 없습니다: {data_file}")
    sys.exit()

# US FX 인덱스 찾기
us_idx = next((i for i, name in enumerate(col) if 'us_fx' in name.lower()), -1)
if us_idx != -1: print(f"Found US FX at index: {us_idx} ({col[us_idx]})")

# 5. Normalization
scale = np.ones(m)
dat = np.zeros(rawdat.shape)
if us_idx != -1: rawdat[:, us_idx] = 1.0

for i in range(m):
    scale[i] = np.max(np.abs(rawdat[:, i]))
    if scale[i] == 0: scale[i] = 1.0
    dat[:, i] = rawdat[:, i] / scale[i]

# 6. Load Model (map_location='cpu' 명시)
print(f"Loading Model from {model_file}...")
try:
    with open(model_file, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
        model.to(device)
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    sys.exit()

# 7. Input Preparation: use model's expected sequence length
try:
    seq_len = int(getattr(model, 'seq_length', None) or getattr(model, 'module', None) and getattr(model.module, 'seq_length', None))
except Exception:
    seq_len = None
if seq_len is None:
    # fallback to 10 if not found
    seq_len = 10
print(f"Using input sequence length (seq_len) = {seq_len}")
X_init = torch.from_numpy(dat[-seq_len:, :]).float().to(device)

# 8. Bayesian Estimation (정확히 horizon개월만 생성)
num_runs, horizon = 20, 36
outputs = []

print(f"Running Bayesian Forecast for {horizon} months...")

# Ensure rolling window size P equals model input length
P = seq_len

# Warm up: estimate model output block length (pred_len)
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
            out = model(curr_input)

            pred_block = out.squeeze(3).squeeze(0)  # [pred_len, N]
            pred_level = pred_block  # 누적합 없이 바로 사용

            if us_idx != -1:
                pred_level[:, us_idx] = 1.0

            need = horizon - len_preds
            take = min(pred_level.size(0), need)
            take_block = pred_level[:take, :].cpu().numpy()

            preds.append(take_block)
            len_preds += take

            # roll forward by 'take' steps
            new_X = np.concatenate([curr_X.cpu().numpy(), take_block], axis=0)
            curr_X = torch.from_numpy(new_X[-P:, :]).float().to(device).contiguous()

    outputs.append(torch.tensor(np.concatenate(preds, axis=0)))  # [horizon, N]

outputs = torch.stack(outputs)  # [num_runs, horizon, N]
Y = torch.mean(outputs, dim=0)              # [horizon, N]
std_dev = torch.std(outputs, dim=0)         # [horizon, N]
confidence = 1.96 * std_dev / torch.sqrt(torch.tensor(num_runs))
variance = torch.var(outputs, dim=0)

# 9. Denormalization & Rebase
scale_torch = torch.from_numpy(scale).float()
dat_denorm = torch.from_numpy(dat).float() * scale_torch
Y_denorm, confidence_denorm, variance_denorm = Y * scale_torch, confidence * scale_torch, variance * scale_torch

# Rebase 없이 실제 스케일로 저장
save_data(dat_denorm, Y_denorm, confidence_denorm, variance_denorm, col, data_out_dir)


# 10. Smoothing
# 실제 스케일로 smoothing
all_data = torch.cat((dat_denorm, Y_denorm), dim=0)
all_conf = torch.cat((torch.zeros_like(dat_denorm), confidence_denorm), dim=0)
smoothed_data_list, smoothed_conf_list = [], []
for i in range(m):
    smoothed_data_list.append(exponential_smoothing(all_data[:, i].cpu().numpy(), 0.3))
    smoothed_conf_list.append(exponential_smoothing(all_conf[:, i].cpu().numpy(), 0.1))

smoothed_dat = torch.tensor(np.array(smoothed_data_list)).T
smoothed_confidence = torch.tensor(np.array(smoothed_conf_list)).T
smoothed_hist, smoothed_fut = smoothed_dat[:-horizon, :], smoothed_dat[-horizon:, :]
smoothed_conf_fut = smoothed_confidence[-horizon:, :]

# === X축(월) 기준 고정: "마지막 관측월 = 2025/Jul" ===
# 고정된 마지막 관측월을 기준으로 과거 월렬과 미래 월렬을 생성합니다.
HIST_END = pd.Timestamp("2025-07-01")  # 최종 학습 데이터 마지막 월
dates_hist = pd.date_range(end=HIST_END, periods=len(df), freq="MS").tolist()

# 미래예측: 2025/Aug ~ 2028/Jul (36개월)
FORECAST_START = HIST_END + pd.DateOffset(months=1)  # 2025-08-01
dates_future = pd.date_range(start=FORECAST_START, periods=horizon, freq="MS").tolist()

print("Forecast range:", dates_future[0].strftime('%Y-%m'), "~", dates_future[-1].strftime('%Y-%m'))

# 11. Individual Plotting Loop
target_keywords = ['us_fx', 'kr_fx', 'uk_fx', 'jp_fx', 'cn_fx']
target_indices = [i for k in target_keywords for i, n in enumerate(col) if k.lower() in n.lower()]
if not target_indices: target_indices = range(m)
plot_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

print("Generating Plots...")
for idx, i in enumerate(target_indices):
    plot_forecast(smoothed_hist[:, i], smoothed_fut[:, i], smoothed_conf_fut[:, i], col[i], dates_hist, dates_future, pt_plots_dir, 
                  color=plot_colours[idx % len(plot_colours)], linestyle='-' if i == us_idx else '--', is_us=(i == us_idx))

# 12. Multi-Node Plot (DATE AXIS + FULL & ZOOM images)
import matplotlib.dates as mdates

def plot_multi_node(dates_hist, dates_future,
                    smoothed_hist, smoothed_fut, smoothed_conf_fut,
                    target_indices, col, us_idx, plot_colours,
                    out_path,
                    x_start=None, x_end=None,
                    add_last_month_tick=True):
    fig, ax = pyplot.subplots(figsize=(15, 10))

    # 연결점(예측 첫 달)을 과거 라인 끝에 붙여서 기존 모양 유지
    connect_date = dates_future[0]
    x_past = dates_hist + [connect_date]   # len = len(hist)+1

    for idx, i in enumerate(target_indices):
        # 각 노드의 시작값(첫 달)을 1.0으로 정규화 (Rebase)
        base_value = smoothed_hist[0, i]
        
        y_past = torch.cat((smoothed_hist[:, i], smoothed_fut[0:1, i]), dim=0).numpy() / base_value
        y_fut  = smoothed_fut[:, i].numpy() / base_value
        c_fut  = smoothed_conf_fut[:, i].numpy() / base_value

        color = plot_colours[idx % len(plot_colours)]
        is_us = (i == us_idx)

        ax.plot(x_past, y_past, '-', label=consistent_name(col[i]), color=color, linewidth=1.5)
        ax.plot(dates_future, y_fut, linestyle='-' if is_us else '--', color=color, linewidth=2)

        if not is_us:
            ax.fill_between(dates_future, y_fut - c_fut, y_fut + c_fut, color=color, alpha=0.25)

    # 예측 시작점에 세로선 추가
    ax.axvline(x=dates_future[0], color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # 축 범위
    if x_start is None:
        x_start = dates_hist[0]
    if x_end is None:
        x_end = dates_future[-1] + pd.Timedelta(days=30)  # 마지막 달이 잘리면 안 보여서 여유
    ax.set_xlim(pd.Timestamp(x_start), pd.Timestamp(x_end))

    # 연도 tick + 마지막(2028/Jul)만 강제 표기(요청사항)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if add_last_month_tick:
        end_tick = pd.Timestamp(dates_future[-1])  # 2028-07-01
        year_ticks = pd.date_range(pd.Timestamp(x_start).normalize(), end_tick.normalize(), freq="YS")
        ticks = list(year_ticks)

        if end_tick not in ticks:
            ticks.append(end_tick)

        labels = [t.strftime('%Y') for t in year_ticks]
        if ticks[-1] == end_tick and (len(labels) == len(ticks) - 1):
            labels.append(end_tick.strftime('%Y/%b'))  # "2028/Jul"

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1.03))
    ax.grid(True, linestyle=':', alpha=0.6)
    pyplot.title("Multi-Node Forecast (Actual Scale)", fontsize=18)
    pyplot.savefig(out_path, bbox_inches="tight")
    pyplot.close()


# === (A) 전체: 2011 ~ 2028/Jul ===
plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=smoothed_hist,
    smoothed_fut=smoothed_fut,
    smoothed_conf_fut=smoothed_conf_fut,
    target_indices=target_indices,
    col=col,
    us_idx=us_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "Multi_Node_Normalized_FULL.png"),
    x_start=dates_hist[0],
    x_end=pd.Timestamp("2028-07-31"),
)

# === (B) 확대: 2022/Aug ~ 2028/Jul ===
plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=smoothed_hist,
    smoothed_fut=smoothed_fut,
    smoothed_conf_fut=smoothed_conf_fut,
    target_indices=target_indices,
    col=col,
    us_idx=us_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "Multi_Node_Normalized_ZOOM.png"),
    x_start=pd.Timestamp("2022-08-01"),
    x_end=pd.Timestamp("2028-07-31"),
)

# --- 추가: 기존 스타일로 Zoom 이미지 생성 (Aug/22 ~ Jul/28) ---
zoom_start = pd.Timestamp("2022-08-01")
zoom_end = pd.Timestamp("2028-07-01")

all_dates = dates_hist + dates_future
left = next((i for i, d in enumerate(all_dates) if d >= zoom_start), 0)
right = next((i for i, d in reversed(list(enumerate(all_dates))) if d <= zoom_end), len(all_dates) - 1)

fig, ax = pyplot.subplots(figsize=(15, 10))
for idx, i in enumerate(target_indices):
    # 각 노드의 시작값 기준으로 정규화 (Rebase)
    base_value = smoothed_hist[0, i]
    d = torch.cat((smoothed_hist[:, i], smoothed_fut[0:1, i]), dim=0).numpy() / base_value
    f, c = smoothed_fut[:, i].numpy() / base_value, smoothed_conf_fut[:, i].numpy() / base_value
    color = plot_colours[idx % len(plot_colours)]
    
    # 보라색 점선 처리
    is_purple = color == '#9467bd'
    linestyle = '-' if i == us_idx else ('--' if is_purple else '--')
    
    ax.plot(range(len(d)), d, '-', label=consistent_name(col[i]), color=color, linewidth=1.5)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, linestyle=linestyle, color=color, linewidth=2)
    if i != us_idx:
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=color, alpha=0.25)

# 예측 시작점에 세로선 추가
forecast_boundary = len(d) - 1  # 과거 데이터의 마지막 인덱스
ax.axvline(x=forecast_boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

ax.set_xlim(left, right)

# 6개월 간격 + 시작/끝 포함
step = 6
tick_positions = list(range(left, right + 1, step))
if right not in tick_positions:
    tick_positions.append(right)
tick_positions = sorted(set(tick_positions))
ax.set_xticks(tick_positions)
ax.set_xticklabels([all_dates[p].strftime('%b-%y') for p in tick_positions], rotation=90, fontsize=13)

ax.legend(loc="upper left", bbox_to_anchor=(1, 1.03))
ax.grid(True, linestyle=':', alpha=0.6)
pyplot.title("Multi-Node Forecast (Normalized Trend) - Zoom", fontsize=18)
pyplot.savefig(os.path.join(plot_dir, 'Multi_Node_Normalized_Zoom.png'), bbox_inches="tight")
pyplot.close()

# 13. Gap Analysis
found_comps = [n for cand in ['kr_fx', 'uk_fx', 'jp_fx', 'cn_fx'] for n in col if cand in n.lower()]
if us_idx != -1 and found_comps:
    save_gap_fx_yearly(smoothed_fut, col[us_idx], found_comps, index, gap_out_dir, dates_future)

print("=== 최종 완료 ===")
