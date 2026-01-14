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
    x_ticks_labels = [all_dates[i].strftime('%Y') for i in x_ticks_pos]
    ax.set_xticks(x_ticks_pos)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_ylabel("Trend", fontsize=15)
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


# --- Main Execution Block ---

# 1. 모든 경로를 구글 드라이브 절대 경로로 직접 지정 (project_root 미사용)
data_file = '/content/drive/MyDrive/Cyber-trend-forecasting-main/Cyber-trend-forecasting-main/B-MTGNN/data/ExchangeRate_dataset.csv'
model_file = '/content/drive/MyDrive/Cyber-trend-forecasting-main/Cyber-trend-forecasting-main/model/model.pt'

plot_dir = '/content/drive/MyDrive/Cyber-trend-forecasting-main/Cyber-trend-forecasting-main/model/Bayesian/forecast/plots'
pt_plots_dir = '/content/drive/MyDrive/Cyber-trend-forecasting-main/Cyber-trend-forecasting-main/model/Bayesian/forecast/pt_plots'
data_out_dir = '/content/drive/MyDrive/Cyber-trend-forecasting-main/Cyber-trend-forecasting-main/model/Bayesian/forecast/data'
rebased_out_dir = '/content/drive/MyDrive/Cyber-trend-forecasting-main/Cyber-trend-forecasting-main/model/Bayesian/forecast/data_rebased'
gap_out_dir = '/content/drive/MyDrive/Cyber-trend-forecasting-main/Cyber-trend-forecasting-main/model/Bayesian/forecast/gap'

# 2. 모든 저장 디렉토리 자동 생성
for d in [plot_dir, pt_plots_dir, data_out_dir, rebased_out_dir, gap_out_dir]:
    if not os.path.exists(d): os.makedirs(d, exist_ok=True)

# 3. Device 설정 (CPU 전용)
device = torch.device('cpu')
print(f"Using device: {device}")

# 4. 데이터 로드
try:
    print(f"Reading data from: {data_file}")
    df = pd.read_csv(data_file)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.ffill().fillna(0)
    dates_hist = pd.date_range(start='2011-01-01', periods=len(df), freq='MS').tolist()
    col = df.columns.tolist()
    index = {name: i for i, name in enumerate(col)}
    rawdat = df.values
    n, m = rawdat.shape
    print(f"Data loaded: {n} months, {m} nodes")
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

# 6. Input Preparation
P = 36
X_init = torch.from_numpy(dat[-P:, :]).float().to(device)

# 7. Load Model (map_location='cpu' 명시)
print(f"Loading Model from {model_file}...")
try:
    with open(model_file, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
        model.to(device)
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    sys.exit()

# 8. Bayesian Estimation
num_runs, horizon = 20, 36
outputs = []
print(f"Running Bayesian Forecast for {horizon} months...")
steps_needed = horizon // 12

for r in range(num_runs):
    curr_X = X_init.clone()
    sample_preds = []
    model.train() 
    with torch.no_grad():
        for step in range(steps_needed):
            curr_input = curr_X.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).contiguous()
            output = model(curr_input)
            pred_block = output.squeeze(3).squeeze(0)
            last = curr_input[:, 0, :, -1]
            last_rep = last.unsqueeze(1).repeat(1, pred_block.size(0), 1)
            pred_level = pred_block.unsqueeze(0) + last_rep
            if us_idx != -1: pred_level[:, :, us_idx] = 1.0
            sample_preds.append(pred_level.squeeze(0).cpu().numpy())
            new_X = np.concatenate([curr_X.cpu().numpy(), pred_level.squeeze(0).cpu().numpy()], axis=0)
            curr_X = torch.from_numpy(new_X[-36:, :]).float().to(device).contiguous()
    outputs.append(torch.tensor(np.concatenate(sample_preds, axis=0)))

outputs = torch.stack(outputs) 
Y, std_dev = torch.mean(outputs, dim=0), torch.std(outputs, dim=0)
confidence = 1.96 * std_dev / torch.sqrt(torch.tensor(num_runs))
variance = torch.var(outputs, dim=0)

# 9. Denormalization & Rebase
scale_torch = torch.from_numpy(scale).float()
dat_denorm = torch.from_numpy(dat).float() * scale_torch
Y_denorm, confidence_denorm, variance_denorm = Y * scale_torch, confidence * scale_torch, variance * scale_torch

print("Applying Rebase (Start Points to 1.0)...")
gap_df, _, mul_factor, _ = make_gap_params(hist=dat_denorm, col_names=col, base_mode="first", base_index=1.0)
dat_rebased = apply_rebase_mul(dat_denorm, mul_factor)
Y_rebased = apply_rebase_mul(Y_denorm, mul_factor)
confidence_rebased = apply_rebase_mul(confidence_denorm, mul_factor)
variance_rebased = apply_rebase_mul(variance_denorm, mul_factor)
save_data(dat_rebased, Y_rebased, confidence_rebased, variance_rebased, col, rebased_out_dir)

# 10. Smoothing
all_data = torch.cat((dat_rebased, Y_rebased), dim=0)
all_conf = torch.cat((torch.zeros_like(dat_rebased), confidence_rebased), dim=0)
smoothed_data_list, smoothed_conf_list = [], []
for i in range(m):
    smoothed_data_list.append(exponential_smoothing(all_data[:, i].cpu().numpy(), 0.1))
    smoothed_conf_list.append(exponential_smoothing(all_conf[:, i].cpu().numpy(), 0.1))

smoothed_dat = torch.tensor(np.array(smoothed_data_list)).T
smoothed_confidence = torch.tensor(np.array(smoothed_conf_list)).T
smoothed_hist, smoothed_fut = smoothed_dat[:-horizon, :], smoothed_dat[-horizon:, :]
smoothed_conf_fut = smoothed_confidence[-horizon:, :]
dates_future = [dates_hist[-1] + pd.DateOffset(months=i + 1) for i in range(horizon)]

# 11. Individual Plotting Loop
target_keywords = ['us_fx', 'kr_fx', 'uk_fx', 'jp_fx', 'cn_fx']
target_indices = [i for k in target_keywords for i, n in enumerate(col) if k.lower() in n.lower()]
if not target_indices: target_indices = range(m)
plot_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

print("Generating Plots...")
for idx, i in enumerate(target_indices):
    plot_forecast(smoothed_hist[:, i], smoothed_fut[:, i], smoothed_conf_fut[:, i], col[i], dates_hist, dates_future, pt_plots_dir, 
                  color=plot_colours[idx % len(plot_colours)], linestyle='-' if i == us_idx else '--', is_us=(i == us_idx))

# 12. Multi-Node Plot
fig, ax = pyplot.subplots(figsize=(15, 10))
for idx, i in enumerate(target_indices):
    d = torch.cat((smoothed_hist[:, i], smoothed_fut[0:1, i]), dim=0).numpy()
    f, c = smoothed_fut[:, i].numpy(), smoothed_conf_fut[:, i].numpy()
    color = plot_colours[idx % len(plot_colours)]
    ax.plot(range(len(d)), d, '-', label=consistent_name(col[i]), color=color, linewidth=1.5)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, linestyle='-' if i == us_idx else '--', color=color, linewidth=2)
    if i != us_idx: ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=color, alpha=0.25)

all_dates = dates_hist + dates_future
x_ticks_pos = [i for i, date in enumerate(all_dates) if date.month == 1]
ax.set_xticks(x_ticks_pos)
ax.set_xticklabels([all_dates[i].strftime('%Y') for i in x_ticks_pos], rotation=90, fontsize=13)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1.03))
ax.grid(True, linestyle=':', alpha=0.6)
pyplot.title("Multi-Node Forecast (Normalized Trend)", fontsize=18)
pyplot.savefig(os.path.join(plot_dir, 'Multi_Node_Normalized.png'), bbox_inches="tight")
pyplot.close()

# 13. Gap Analysis
found_comps = [n for cand in ['kr_fx', 'uk_fx', 'jp_fx', 'cn_fx'] for n in col if cand in n.lower()]
if us_idx != -1 and found_comps:
    save_gap_fx_yearly(smoothed_fut, col[us_idx], found_comps, index, gap_out_dir, dates_future)

print("=== 최종 완료 ===")
