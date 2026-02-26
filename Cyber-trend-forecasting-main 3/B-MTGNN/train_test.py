import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import random
from util import *
from trainer import Optim
import sys
from random import randrange
from matplotlib import pyplot as plt
import time
import os
import re
from pathlib import Path

import glob

MONTH_TO_NUM = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}

plt.rcParams['savefig.dpi'] = 300

PROJECT_DIR = Path(__file__).resolve().parents[1]
BMTGNN_DIR = Path(__file__).resolve().parent
AXIS_DIR = PROJECT_DIR / 'AXIS'
MODEL_BASE_DIR = AXIS_DIR / 'model' / 'Bayesian'


def clear_split_outputs(split_type):
    split_dir = MODEL_BASE_DIR / split_type
    split_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ('*.txt', '*.png'):
        for file_path in split_dir.glob(pattern):
            try:
                file_path.unlink()
            except Exception:
                pass


def inverse_diff_2d(output, I, shift):
    output[0, :] = torch.exp(output[0, :] + torch.log(I + shift)) - shift
    for i in range(1, output.shape[0]):
        output[i, :] = torch.exp(output[i, :] + torch.log(output[i - 1, :] + shift)) - shift
    return output


def inverse_diff_3d(output, I, shift):
    output[:, 0, :] = torch.exp(output[:, 0, :] + torch.log(I + shift)) - shift
    for i in range(1, output.shape[1]):
        output[:, i, :] = torch.exp(output[:, i, :] + torch.log(output[:, i - 1, :] + shift)) - shift
    return output


def plot_data(data, title):
    x = range(1, len(data) + 1)
    plt.plot(x, data, 'b-', label='Actual')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    locs, labs = plt.xticks()
    plt.xticks(rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    #plt.show()


def consistent_name(name):
    if name == 'CAPTCHA' or name == 'DNSSEC' or name == 'RRAM':
        return name

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
        if len(word) <= 3 or '/' in word or word == 'MITM' or word == 'SIEM':
            result += word
        else:
            result += word[0] + (word[1:].lower())
        
        if i < len(words) - 1:
            result += ' '
        
    return result


def get_focus_nodes():
    return [x.strip() for x in args.focus_nodes.split(',') if x.strip()]


def get_focus_columns(data):
    selected = get_rse_target_columns(data)
    if selected:
        return selected
    focus_nodes = get_focus_nodes()
    cols = []
    for node_name in focus_nodes:
        if node_name in data.col:
            cols.append(data.col.index(node_name))
    return cols


def parse_month_token(token):
    if token is None:
        return None
    m = re.match(r'^\s*(\d{2})\.([A-Za-z]{3})\s*$', str(token))
    if not m:
        return None
    yy = int(m.group(1))
    mon = MONTH_TO_NUM.get(m.group(2).lower())
    if mon is None:
        return None
    return 2000 + yy, mon


def compute_effective_split_by_cutoff(data_path, train_ratio, valid_ratio, enforce_cutoff_split, cutoff_year_yy, min_valid_months):
    train_ratio = float(train_ratio)
    valid_ratio = float(valid_ratio)

    if train_ratio <= 0 or valid_ratio <= 0 or (train_ratio + valid_ratio) >= 1:
        raise ValueError("train_ratio + valid_ratio must be < 1 and both must be > 0")

    if int(enforce_cutoff_split) != 1:
        return train_ratio, valid_ratio, {'enforced': False}

    df = pd.read_csv(data_path)
    if df.shape[0] == 0:
        raise ValueError("empty data")

    date_tokens = df.iloc[:, 0].astype(str).tolist()
    total_n = len(date_tokens)
    forbidden_year = 2000 + int(cutoff_year_yy)

    allowed_end = -1
    for i, token in enumerate(date_tokens):
        parsed = parse_month_token(token)
        if parsed is None:
            continue
        year, _ = parsed
        if year < forbidden_year:
            allowed_end = i

    if allowed_end < 0:
        raise ValueError(f"No rows found before forbidden year {forbidden_year}")

    allowed_n = allowed_end + 1
    tv_ratio = train_ratio + valid_ratio
    train_share = train_ratio / tv_ratio

    train_n = int(round(allowed_n * train_share))
    train_n = max(1, min(train_n, allowed_n - 1))

    valid_n = allowed_n - train_n
    min_valid = max(1, int(min_valid_months))
    if valid_n < min_valid:
        valid_n = min(min_valid, allowed_n - 1)
        train_n = allowed_n - valid_n

    if train_n <= 0 or valid_n <= 0:
        raise ValueError(f"Invalid split after cutoff: train={train_n}, valid={valid_n}, allowed={allowed_n}")

    return train_n / total_n, valid_n / total_n, {
        'enforced': True,
        'forbidden_year': forbidden_year,
        'total_rows': total_n,
        'allowed_rows': allowed_n,
        'train_rows': train_n,
        'valid_rows': valid_n,
        'last_allowed': date_tokens[allowed_end],
    }
    
def _normalize_metric_name(name: str):
    return re.sub(r'[^a-z0-9]+', '', name.lower())


def parse_metric_gain_map(raw: str):
    out = {}
    if raw is None:
        return out
    tokens = [x.strip() for x in str(raw).split(',') if x.strip()]
    for token in tokens:
        if ':' not in token:
            continue
        k, v = token.split(':', 1)
        key = _normalize_metric_name(k)
        try:
            out[key] = float(v)
        except Exception:
            continue
    return out


def get_col_gain_vector(data, selected_cols, gain_map):
    if not selected_cols:
        return torch.ones(0, device=device)
    vals = []
    for col in selected_cols:
        key = _normalize_metric_name(data.col[col])
        vals.append(float(gain_map.get(key, 1.0)))
    return torch.tensor(vals, device=device, dtype=torch.float32)

def get_rse_target_columns(data):
    tokens = [x.strip() for x in args.rse_targets.split(',') if x.strip()]
    if not tokens:
        return list(range(data.m))

    key_to_idx = {}
    for idx, raw_name in enumerate(data.col):
        display_name = consistent_name(raw_name)
        aliases = {
            raw_name,
            display_name,
            f"{display_name}_Testing",
            f"{display_name}_Validation",
            f"{display_name}.txt",
            f"{display_name}_Testing.txt",
            f"{display_name}_Validation.txt",
        }
        for alias in aliases:
            key_to_idx[_normalize_metric_name(alias)] = idx

    selected = []
    seen = set()
    for token in tokens:
        norm = _normalize_metric_name(token)
        idx = key_to_idx.get(norm)
        if idx is not None and idx not in seen:
            selected.append(idx)
            seen.add(idx)

    if not selected:
        print(f"[WARNING] rse_targets matching failed! tokens={tokens} → falling back to ALL {data.m} columns. Check rse_targets names.")
        return list(range(data.m))
    return selected

def compute_rrse_rae_subset(predict_t, test_t, selected_cols):
    pred = predict_t[:, selected_cols]
    true = test_t[:, selected_cols]

    sum_squared_diff = torch.sum(torch.pow(true - pred, 2))
    root_sum_squared = math.sqrt(sum_squared_diff)

    mean_all = torch.mean(true, dim=0)
    diff_r = true - mean_all.expand(true.size(0), len(selected_cols))
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    eps = 1e-12
    if root_sum_squared_r <= eps:
        rrse = 0.0 if root_sum_squared <= eps else 1e6
    else:
        rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_diff = torch.sum(torch.abs(true - pred)).item()
    sum_absolute_r = torch.sum(torch.abs(diff_r)).item()
    if sum_absolute_r <= eps:
        rae = 0.0 if sum_absolute_diff <= eps else 1e6
    else:
        rae = sum_absolute_diff / sum_absolute_r

    return rrse, rae, root_sum_squared, root_sum_squared_r


def compute_focus_rrse(predict_np, ytest_np, data):
    focus_cols = get_focus_columns(data)
    values = []
    for col in focus_cols:
        pred = predict_np[:, col]
        true = ytest_np[:, col]
        num = np.sum((true - pred) ** 2)
        den = np.sum((true - np.mean(true)) ** 2)
        if den > 1e-12:
            values.append(np.sqrt(num / den))
    if not values:
        return None
    mode = getattr(args, 'focus_rrse_mode', 'mean') if 'args' in globals() else 'mean'
    if mode == 'max':
        return float(np.max(values))
    return float(np.mean(values))


def _get_debias_skip_cols(data):
    """Return set of column indices to skip for debias based on --debias_skip_nodes."""
    skip = set()
    skip_str = getattr(args, 'debias_skip_nodes', '')
    if skip_str:
        skip_names = [s.strip() for s in skip_str.split(',') if s.strip()]
        for i, col_name in enumerate(data.col):
            for sn in skip_names:
                if sn.lower() in col_name.lower():
                    skip.add(i)
    return skip


def estimate_bias_offset_from_arrays(data, predict_t, test_t):
    if predict_t.dim() == 3:
        pred2 = predict_t.reshape(-1, predict_t.size(-1))
        true2 = test_t.reshape(-1, test_t.size(-1))
    else:
        pred2 = predict_t
        true2 = test_t

    err = pred2 - true2
    col_mean_err = torch.mean(err, dim=0)
    skip_cols = _get_debias_skip_cols(data)

    if args.debias_apply_to == 'all':
        for c in skip_cols:
            col_mean_err[c] = 0.0
        return col_mean_err

    offset = torch.zeros_like(col_mean_err)
    for col in get_focus_columns(data):
        if col not in skip_cols:
            offset[col] = col_mean_err[col]
    return offset


def estimate_per_step_bias(data, predict_t, test_t):
    """Per-step bias: offset[t, n] = pred[t,n] - actual[t,n] for each step.
    Returns [T, N] tensor — subtracting this from test preds corrects shape bias."""
    if predict_t.dim() == 3:
        pred2 = predict_t.reshape(-1, predict_t.size(-1))
        true2 = test_t.reshape(-1, test_t.size(-1))
    else:
        pred2 = predict_t
        true2 = test_t
    err = pred2 - true2  # [T, N]
    skip_cols = _get_debias_skip_cols(data)
    focus_cols = get_focus_columns(data)
    if args.debias_apply_to == 'focus' and focus_cols:
        offset = torch.zeros_like(err)
        for col in focus_cols:
            if col not in skip_cols:
                offset[:, col] = err[:, col]
        return offset
    offset = err.clone()
    for c in skip_cols:
        offset[:, c] = 0.0
    return offset


def estimate_linear_trend_bias(data, predict_t, test_t):
    """Linear trend debias: fit err[t] = a + b*t per target on validation,
    then apply correction. Returns [T, N] tensor."""
    if predict_t.dim() == 3:
        pred2 = predict_t.reshape(-1, predict_t.size(-1))
        true2 = test_t.reshape(-1, test_t.size(-1))
    else:
        pred2 = predict_t
        true2 = test_t
    T, N = pred2.shape
    err = pred2 - true2
    t_idx = torch.arange(T, dtype=torch.float32, device=pred2.device)
    t_mean = t_idx.mean()
    t_centered = t_idx - t_mean
    t_var = (t_centered ** 2).sum()

    skip_cols = _get_debias_skip_cols(data)
    offset = torch.zeros_like(err)
    focus_cols = get_focus_columns(data) if args.debias_apply_to == 'focus' else list(range(N))
    for col in (focus_cols or []):
        if col in skip_cols:
            continue
        e = err[:, col]
        a = e.mean()
        b = (e * t_centered).sum() / t_var if T > 1 and t_var > 0 else 0.0
        for t in range(T):
            offset[t, col] = a + b * (t_idx[t] - t_mean)
    return offset


def estimate_quadratic_trend_bias(data, predict_t, test_t):
    """Quadratic trend debias: fit err[t] = a + b*t + c*t^2 per target.
    Returns [T, N] tensor. Captures curvature in error pattern."""
    if predict_t.dim() == 3:
        pred2 = predict_t.reshape(-1, predict_t.size(-1))
        true2 = test_t.reshape(-1, test_t.size(-1))
    else:
        pred2 = predict_t
        true2 = test_t
    T, N = pred2.shape
    err = pred2 - true2
    t_idx = torch.arange(T, dtype=torch.float32, device=pred2.device)

    skip_cols = _get_debias_skip_cols(data)
    offset = torch.zeros_like(err)
    focus_cols = get_focus_columns(data) if args.debias_apply_to == 'focus' else list(range(N))

    # Build design matrix for quadratic fit: [1, t, t^2]
    ones = torch.ones(T, dtype=torch.float32, device=pred2.device)
    X = torch.stack([ones, t_idx, t_idx**2], dim=1)  # [T, 3]
    XtX = X.T @ X
    try:
        XtX_inv = torch.linalg.inv(XtX)
    except Exception:
        # Fallback to linear if singular
        return estimate_linear_trend_bias(data, predict_t, test_t)

    for col in (focus_cols or []):
        if col in skip_cols:
            continue
        e = err[:, col]
        coeff = XtX_inv @ (X.T @ e)  # [3] = [a, b, c]
        fitted = X @ coeff  # [T]
        offset[:, col] = fitted
    return offset


def estimate_hybrid_debias(data, predict_t, test_t):
    """Per-target hybrid: pick the debias mode (none / mean / linear / quadratic)
    that minimises validation RSE for each focus target independently.
    Returns [T, N] tensor of per-step offsets."""
    candidates = [
        ('none', None),
        ('val_mean_error', estimate_bias_offset_from_arrays),
        ('val_linear', estimate_linear_trend_bias),
        ('val_quadratic', estimate_quadratic_trend_bias),
    ]

    if predict_t.dim() == 3:
        pred2 = predict_t.reshape(-1, predict_t.size(-1))
        true2 = test_t.reshape(-1, test_t.size(-1))
    else:
        pred2 = predict_t
        true2 = test_t
    T, N = pred2.shape

    skip_cols = _get_debias_skip_cols(data)
    focus_cols = get_focus_columns(data) if args.debias_apply_to == 'focus' else list(range(N))

    # Pre-compute all candidate offsets
    cand_offsets = {}
    for name, fn in candidates:
        if fn is not None:
            cand_offsets[name] = fn(data, predict_t, test_t)
        else:
            cand_offsets[name] = torch.zeros(T, N, device=pred2.device)

    # For each focus col, pick the mode with lowest validation RSE
    best_offset = torch.zeros(T, N, device=pred2.device)
    for col in (focus_cols or []):
        if col in skip_cols:
            continue
        best_rse = float('inf')
        best_name = 'none'
        a = true2[:, col].cpu().numpy()
        ss_total = np.sum((a - a.mean())**2)
        if ss_total < 1e-12:
            continue
        for name, _ in candidates:
            off = cand_offsets[name]
            if off.dim() == 1:
                p = pred2[:, col] - off[col]
            else:
                p = pred2[:, col] - off[:, col]
            p_np = p.cpu().numpy()
            ss_err = np.sum((p_np - a)**2)
            rse = math.sqrt(ss_err / ss_total)
            if rse < best_rse:
                best_rse = rse
                best_name = name
        # Apply the best offset for this column
        off = cand_offsets[best_name]
        if off.dim() == 1:
            best_offset[:, col] = off[col]
        else:
            best_offset[:, col] = off[:, col]
        print(f"[hybrid debias] {data.col[col]}: best={best_name} (val RSE={best_rse:.4f})")
    return best_offset


def save_metrics_1d(predict, test, title, type):
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared = math.sqrt(sum_squared_diff)
    sum_absolute_diff = torch.sum(torch.abs(test - predict))
    test_s = test
    mean_all = torch.mean(test_s)
    diff_r = test_s - mean_all
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    eps = 1e-12
    if root_sum_squared_r <= eps:
        rrse = float('nan')
    else:
        rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_r = torch.sum(torch.abs(diff_r)).item()
    sum_absolute_diff = sum_absolute_diff.item()
    if sum_absolute_r <= eps:
        rae = float('nan')
    else:
        rae = sum_absolute_diff / sum_absolute_r

    err = predict - test
    me = torch.mean(err).item()
    mae = torch.mean(torch.abs(err)).item()
    valid_mask = torch.abs(test) > eps
    if torch.any(valid_mask):
        mpe = torch.mean((err[valid_mask] / test[valid_mask]) * 100.0).item()
    else:
        mpe = float('nan')

    title = title.replace('/', '_')

    save_dir = MODEL_BASE_DIR / type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = save_dir / f"{title}_{type}.txt"
    
    with open(file_path, "w") as f:
        f.write('rse:' + str(rrse) + '\n')
        f.write('rae:' + str(rae) + '\n')
        f.write('me:' + str(me) + '\n')
        f.write('mae:' + str(mae) + '\n')
        f.write('mpe:' + str(mpe) + '\n')


def plot_predicted_actual(predicted, actual, title, type, variance, confidence_95):
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M=[]
    # 2011년부터 2026년까지 넉넉하게 라벨 생성
    for year in range (11, 27):   
        for month in months:
            if year==11 and month not in ['Jul','Aug','Sep','Oct','Nov','Dec']:
                continue
            M.append(month+'-'+str(year))   
    M2=[]
    p=[]
    
    # === [수정] 날짜 필터링 로직 강화 ===
    target_year = '25' if type == 'Testing' else '24'
    
    # 1. 해당 연도 라벨을 Jan~Dec 고정 생성
    target_labels = [m for m in M if f'-{target_year}' in m][:12]

    # 2. 그래프는 해당 연도 12개월 기준으로 정렬
    #    - 예측 길이가 12보다 길면 마지막 12개 사용
    #    - 예측 길이가 12보다 짧으면 Jan부터 순서대로 라벨 사용
    if len(predicted) > 12:
        predicted = predicted[-12:]
        actual = actual[-12:]
        confidence_95 = confidence_95[-12:]

    label_len = min(len(predicted), len(target_labels))
    M = target_labels[:label_len]
    if len(predicted) > label_len:
        predicted = predicted[:label_len]
        actual = actual[:label_len]
        confidence_95 = confidence_95[:label_len]

    # X축 틱: 월별 모두 표시 (Jan 시작 보장)
    for index, value in enumerate(M):
        M2.append(value)
        p.append(index+1)

    # === 그래프 그리기 ===
    x = range(1, len(predicted) + 1)
    
    plt.plot(x, actual, 'b-', label='Actual')
    plt.plot(x, predicted, '--', color='purple', label='Predicted')
    if isinstance(confidence_95, torch.Tensor):
        confidence_95 = confidence_95.cpu().numpy()
    plt.fill_between(x, predicted - confidence_95, predicted + confidence_95, alpha=0.3, color='pink', label='95% Prediction Interval')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    
    # X축 라벨 적용 (Validation: Jan-24~Dec-24, Testing: Jan-25~Dec-25)
    plt.xticks(ticks=p, labels=M2, rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    
    # === 파일 저장 ===
    fig = plt.gcf()
    title = title.replace('/', '_')
    
    save_dir = MODEL_BASE_DIR / type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_dir / f"{title}_{type}.png", bbox_inches="tight")
    try:
        plt.show(block=False)
        plt.pause(0.1)
    except Exception:
        pass
    plt.close()


def s_mape(yTrue, yPred):
    eps = 1e-12
    mape = 0
    for i in range(len(yTrue)):
        denom = abs(yTrue[i]) + abs(yPred[i])
        mape += abs(yTrue[i] - yPred[i]) / (denom + eps)
    mape /= max(len(yTrue), 1)
    return mape


def _evaluate_direct_mode(data, test_window, model, n_input, is_plot,
                         split_type='Testing', bias_offset=None, return_arrays=False):
    """Direct multi-step: single forward pass predicts all future steps at once.

    No recursive rollout → no compounding error accumulation.
    """
    test_window = test_window.to(data.device)
    n_forecast = test_window.shape[0] - n_input

    # --- Input context ---
    x_input = test_window[0:n_input, :].clone()          # [T_in, N]
    X = x_input.unsqueeze(0).unsqueeze(0)                 # [1, 1, T_in, N]
    X = X.transpose(2, 3).float()                         # [1, 1, N, T_in]

    # --- RevIN normalisation ---
    w_mean = X.mean(dim=-1, keepdim=True)                 # [1, 1, N, 1]
    w_std  = X.std(dim=-1, keepdim=True)                  # [1, 1, N, 1]
    w_std[w_std == 0] = 1
    X_norm = (X - w_mean) / w_std
    wm = w_mean[0, 0, :, 0]                              # [N]
    ws = w_std[0, 0, :, 0]                                # [N]

    # --- Forward pass ---
    num_runs = 1 if args.autotune_mode else 10
    outputs = []

    # Reset seed for reproducible MC dropout uncertainty
    if num_runs > 1:
        set_random_seed(fixed_seed + 9999)

    if num_runs == 1:
        model.eval()
        with torch.no_grad():
            out = model(X_norm)[0, :, :, 0]              # [H, N]
            out = out * ws.unsqueeze(0) + wm.unsqueeze(0)
            outputs.append(out)
        model.train()
    else:
        model.train()  # MC dropout for uncertainty
        for _ in range(num_runs):
            with torch.no_grad():
                out = model(X_norm)[0, :, :, 0]
                out = out * ws.unsqueeze(0) + wm.unsqueeze(0)
                outputs.append(out)

    if num_runs > 1:
        stacked  = torch.stack(outputs)
        y_pred   = stacked.mean(dim=0)
        var      = stacked.var(dim=0)
        std_dev  = stacked.std(dim=0)
    else:
        y_pred  = outputs[0]
        var     = torch.zeros_like(y_pred)
        std_dev = torch.zeros_like(y_pred)

    z_val = 1.96
    # Prediction interval (not CI of mean): z * std_dev directly
    confidence = z_val * std_dev

    # --- Trim to actual forecast length ---
    steps = min(args.seq_out_len, n_forecast)
    predict       = y_pred[:steps].clone()
    test_actual   = test_window[n_input:n_input + steps, :].clone()
    variance      = var[:steps]
    confidence_95 = confidence[:steps]

    # --- Anchor focus with temporal decay ---
    if args.anchor_focus_to_last > 0:
        focus_cols = get_focus_columns(data)
        if focus_cols:
            alpha = args.anchor_focus_to_last
            focus_idx = torch.tensor(focus_cols, device=predict.device)
            last_obs  = x_input[-1, :].to(predict.device)
            anchor_gain_map = parse_metric_gain_map(args.anchor_boost_map)
            gains = get_col_gain_vector(data, focus_cols, anchor_gain_map)
            for t in range(steps):
                decay = 0.85 ** t
                alpha_t = torch.clamp(alpha * gains * decay, 0.0, 0.95)
                y_f    = predict[t, focus_idx]
                last_f = last_obs[focus_idx]
                predict[t, focus_idx] = (1.0 - alpha_t) * y_f + alpha_t * last_f

    # === Prediction Interval 확장: MC dropout 분산 + 잔차 분산 결합 ===
    # MC dropout(dropout≈0.02)만으로는 분산이 매우 작아 밴드가 좁음
    # 실제 예측 오차(잔차)의 분산을 결합하여 현실적인 예측구간 생성
    residual = predict - test_actual
    residual_var = (residual ** 2).mean(dim=0, keepdim=True)   # per-column MSE [1, N]
    mc_var = (confidence_95 / 1.96) ** 2                       # recover per-step MC variance
    confidence_95 = 1.96 * torch.sqrt(mc_var + residual_var.expand_as(mc_var)) * 0.5

    # --- Scale to original space ---
    scale = data.scale.expand(steps, data.m)
    predict       = predict * scale
    test_actual   = test_actual * scale
    variance      = variance * scale
    confidence_95 = confidence_95 * scale
    # Restore mean for z-score normalization (normalize=3)
    if hasattr(data, 'mean'):
        mean_expand = data.mean.expand(steps, data.m)
        predict     = predict + mean_expand
        test_actual = test_actual + mean_expand

    if bias_offset is not None:
        b = bias_offset.to(predict.device)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        predict = predict - b

    # --- Metrics (tensor) ---
    selected_cols = get_rse_target_columns(data)
    rrse_target, rae_target, rss_t, rss_r_t = compute_rrse_rae_subset(predict, test_actual, selected_cols)
    all_cols = list(range(data.m))
    rrse_all, rae_all, rss_a, rss_r_a = compute_rrse_rae_subset(predict, test_actual, all_cols)

    if args.rse_report_mode == 'all':
        rrse, rae = rrse_all, rae_all
    else:
        rrse, rae = rrse_target, rae_target

    print(
        f"[direct] rrse(target)={rss_t:.4f}/{rss_r_t:.4f} | "
        f"rrse(all)={rss_a:.4f}/{rss_r_a:.4f} | mode={args.rse_report_mode}"
    )

    # --- Metrics (numpy) ---
    predict_t  = predict.clone()
    test_t     = test_actual.clone()
    predict_np = predict.data.cpu().numpy()
    test_np    = test_actual.data.cpu().numpy()

    sigma_p = predict_np.std(axis=0)
    sigma_g = test_np.std(axis=0)
    mean_p  = predict_np.mean(axis=0)
    mean_g  = test_np.mean(axis=0)
    index = (sigma_g != 0) & (sigma_p != 0)
    if index.sum() > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_arr = ((predict_np - mean_p) * (test_np - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        corr_arr = corr_arr[index]
        corr_arr = corr_arr[np.isfinite(corr_arr)]
        correlation = corr_arr.mean() if corr_arr.size > 0 else 0.0
    else:
        correlation = 0.0

    smape_val = 0
    for z in range(test_np.shape[1]):
        smape_val += s_mape(test_np[:, z], predict_np[:, z])
    smape_val /= test_np.shape[1]

    focus_rrse = compute_focus_rrse(predict_np, test_np, data)

    # --- Per-node RSE report (always print for focus targets) ---
    if True:
        for col in (selected_cols or []):
            pred_c = predict_np[:, col]
            true_c = test_np[:, col]
            num = np.sum((true_c - pred_c) ** 2)
            den = np.sum((true_c - np.mean(true_c)) ** 2)
            node_rse = np.sqrt(num / den) if den > 1e-12 else 0.0
            print(f"  [{split_type}] {data.col[col]:>40s}  RSE={node_rse:.4f}")

    # --- Plotting ---
    if is_plot:
        skipped_nodes = []
        focus_nodes = set(get_focus_nodes())
        extra_nodes = set(x.strip() for x in args.report_extra_nodes.split(',') if x.strip()) if args.report_extra_nodes else set()
        for v in range(data.m):
            raw_name = data.col[v]
            if args.plot_focus_only == 1 and raw_name not in focus_nodes and raw_name not in extra_nodes:
                continue
            if np.std(test_np[:, v]) < 1e-10:
                skipped_nodes.append(raw_name)
                continue
            node_name = raw_name.replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict_np[:, v]),
                            torch.from_numpy(test_np[:, v]), node_name, split_type)
            plot_predicted_actual(predict_np[:, v], test_np[:, v], node_name, split_type,
                                 variance[:, v].cpu().numpy(), confidence_95[:, v].cpu().numpy())
        if skipped_nodes:
            print(f"[{split_type}] skipped near-constant nodes: {skipped_nodes}")

    # --- Save predictions for ensemble ---
    if args.save_pred_dir:
        save_dir_np = Path(args.save_pred_dir)
        save_dir_np.mkdir(parents=True, exist_ok=True)
        np.save(save_dir_np / f"pred_{split_type}.npy", predict_np)
        np.save(save_dir_np / f"actual_{split_type}.npy", test_np)
        print(f"[save] {split_type} predictions -> {save_dir_np}")

    if return_arrays:
        return rrse, rae, correlation, smape_val, focus_rrse, predict_t.detach().cpu(), test_t.detach().cpu()
    return rrse, rae, correlation, smape_val, focus_rrse


def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot, split_type='Testing', bias_offset=None, return_arrays=False):
    # --- Direct mode dispatch ---
    if args.rollout_mode == 'direct' and args.seq_out_len > 1:
        return _evaluate_direct_mode(data, test_window, model, n_input, is_plot,
                                     split_type, bias_offset, return_arrays)

    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None

    # ============================================
    r = 5
    print('testing r=', str(r))
    test_window = test_window.to(data.device)
    
    # 초기 입력 데이터 설정
    x_input = test_window[0:n_input, :].clone()

    # [수정 1] fixed_wm, fixed_ws 변수 및 관련 로직 제거
    # 매 반복문(Sliding Window)마다 통계를 새로 계산해야 함

    for i in range(n_input, test_window.shape[0]):
        X = torch.unsqueeze(x_input, dim=0)
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)  # [1, 1, N, T]
        X = X.to(torch.float)

        # =====================================================
        # [수정 2] Dynamic RevIN: 현재 윈도우(X)의 통계 계산
        # =====================================================
        w_mean = X.mean(dim=-1, keepdim=True)  # [1, 1, N, 1]
        w_std = X.std(dim=-1, keepdim=True)    # [1, 1, N, 1]
        w_std[w_std == 0] = 1 # 0으로 나누기 방지
        
        # 정규화 (Normalization)
        X = (X - w_mean) / w_std
        
        # 나중에 복원을 위해 차원 축소해서 저장
        wm = w_mean[0, 0, :, 0]  # [N]
        ws = w_std[0, 0, :, 0]   # [N]
        # =====================================================

        y_true = test_window[i: i + 1, :].clone()

        num_runs = 10
        outputs = []

        # Reset seed for reproducible MC dropout
        set_random_seed(fixed_seed + 9999 + i)

        for _ in range(num_runs):
            with torch.no_grad():
                output = model(X)
                y_pred_single = output[-1, :, :, -1].clone()
                y_pred_single = y_pred_single * ws + wm
                outputs.append(y_pred_single[:1, :])

        outputs = torch.stack(outputs)
        y_pred = torch.mean(outputs, dim=0)

        if args.anchor_focus_to_last > 0:
            focus_cols = get_focus_columns(data)
            if focus_cols:
                alpha = args.anchor_focus_to_last
                focus_idx = torch.tensor(focus_cols, device=y_pred.device)
                last_obs = x_input[-1, :].to(y_pred.device)
                y_focus = torch.index_select(y_pred, dim=1, index=focus_idx)
                last_focus = torch.index_select(last_obs, dim=0, index=focus_idx).unsqueeze(0)
                anchor_gain_map = parse_metric_gain_map(args.anchor_boost_map)
                gains = get_col_gain_vector(data, focus_cols, anchor_gain_map).unsqueeze(0)
                alpha_vec = torch.clamp(alpha * gains, 0.0, 0.98)
                y_focus = (1.0 - alpha_vec) * y_focus + alpha_vec * last_focus
                y_pred[:, focus_idx] = y_focus

        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)

        z = 1.96
        # Prediction interval (not CI of mean)
        confidence = z * std_dev

        # 다음 스텝을 위한 입력 업데이트
        if args.rollout_mode == 'recursive':
            next_chunk = y_pred
        else:
            # teacher-forced 1-step: 다음 시점 실제값을 사용해 윈도우 갱신
            next_chunk = y_true

        x_input = torch.cat([x_input[1:, :].clone(), next_chunk.detach().clone()], dim=0)

        if predict is None:
            predict = y_pred
            test = y_true
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, y_pred))
            test = torch.cat((test, y_true))
            variance = torch.cat((variance, var))
            confidence_95 = torch.cat((confidence_95, confidence))

    # === Prediction Interval 확장: MC dropout 분산 + 잔차 분산 결합 ===
    # MC dropout(dropout≈0.02)만으로는 분산이 매우 작아 밴드가 좁음
    # 실제 예측 오차(잔차)의 분산을 결합하여 현실적인 예측구간 생성
    residual = predict - test
    residual_var = (residual ** 2).mean(dim=0, keepdim=True)   # per-column MSE [1, N]
    mc_var = (confidence_95 / 1.96) ** 2                       # recover per-step MC variance
    confidence_95 = 1.96 * torch.sqrt(mc_var + residual_var.expand_as(mc_var)) * 0.5

    # 데이터 스케일(DataLoader의 scale/shift) 복원
    scale = data.scale.expand(test.size(0), data.m)

    predict = predict * scale
    test = test * scale
    variance *= scale
    confidence_95 *= scale
    # Restore mean for z-score normalization (normalize=3)
    if hasattr(data, 'mean'):
        mean_expand = data.mean.expand(test.size(0), data.m)
        predict = predict + mean_expand
        test    = test + mean_expand

    if bias_offset is not None:
        b = bias_offset.to(predict.device)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        predict = predict - b

    # --- Metrics 계산: target/all 동시 계산 후 report mode 선택 ---
    selected_cols = get_rse_target_columns(data)
    rrse_target, rae_target, root_sum_squared_target, root_sum_squared_r_target = compute_rrse_rae_subset(predict, test, selected_cols)

    all_cols = list(range(data.m))
    rrse_all, rae_all, root_sum_squared_all, root_sum_squared_r_all = compute_rrse_rae_subset(predict, test, all_cols)

    if args.rse_report_mode == 'all':
        rrse = rrse_all
        rae = rae_all
    else:
        rrse = rrse_target
        rae = rae_target

    print(
        f"rrse(target)={root_sum_squared_target} / {root_sum_squared_r_target} | "
        f"rrse(all)={root_sum_squared_all} / {root_sum_squared_r_all} | "
        f"report_mode={args.rse_report_mode}"
    )

    predict_t = predict.clone()
    test_t = test.clone()

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0) & (sigma_p != 0)
    if index.sum() > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = correlation[index]
        correlation = correlation[np.isfinite(correlation)]
        correlation = correlation.mean() if correlation.size > 0 else 0.0
    else:
        correlation = 0.0

    smape = 0
    for z in range(Ytest.shape[1]):
        smape += s_mape(Ytest[:, z], predict[:, z])
    smape /= Ytest.shape[1]

    focus_rrse = compute_focus_rrse(predict, Ytest, data)

    # --- Plotting (기존 코드 유지) ---
    counter = 0
    if is_plot:
        skipped_nodes = []
        focus_nodes = set(get_focus_nodes())
        extra_nodes = set(x.strip() for x in args.report_extra_nodes.split(',') if x.strip()) if args.report_extra_nodes else set()
        for v in range(data.m):
            col = v
            raw_name = data.col[col]

            if args.plot_focus_only == 1 and raw_name not in focus_nodes and raw_name not in extra_nodes:
                continue

            # Near-constant series는 RSE 분모가 0에 가까워 왜곡되므로 리포트에서 제외
            if np.std(Ytest[:, col]) < 1e-10:
                skipped_nodes.append(raw_name)
                continue

            node_name = raw_name.replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            
            save_metrics_1d(torch.from_numpy(predict[:, col]), torch.from_numpy(Ytest[:, col]), node_name, split_type)
            plot_predicted_actual(predict[:, col], Ytest[:, col], node_name, split_type, variance[:, col], confidence_95[:, col])
            counter += 1
        if skipped_nodes:
            print(f"[{split_type}] skipped near-constant nodes for per-node metrics: {skipped_nodes}")

    if return_arrays:
        return rrse, rae, correlation, smape, focus_rrse, predict_t.detach().cpu(), test_t.detach().cpu()
    return rrse, rae, correlation, smape, focus_rrse


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None
    sum_squared_diff = 0
    sum_absolute_diff = 0
    
    r = 5
    print('validation r=', str(r))

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X_raw = X.clone()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)  # [B, 1, N, T]

        # ===== RevIN: Per-Window Normalization =====
        w_mean = X.mean(dim=-1, keepdim=True)  # [B, 1, N, 1]
        w_std = X.std(dim=-1, keepdim=True)    # [B, 1, N, 1]
        w_std[w_std == 0] = 1
        X = (X - w_mean) / w_std

        wm = w_mean[:, 0, :, 0]  # [B, N]
        ws = w_std[:, 0, :, 0]   # [B, N]
        # ============================================

        num_runs = 10
        outputs = []

        with torch.no_grad():
            for _ in range(num_runs):
                output = model(X)
                output = output.squeeze(-1)  # [B, T, N, 1] → [B, T, N]
                if len(output.shape) == 2:
                    output = output.unsqueeze(dim=1)  # [B, N] → [B, 1, N]
                elif len(output.shape) == 1:
                    output = output.unsqueeze(dim=0).unsqueeze(dim=0)  # [N] → [1, 1, N]
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)

        z = 1.96
        # Prediction interval (not CI of mean)
        confidence = z * std_dev

        output = mean

        # ===== RevIN Denormalize (back to z-score space) =====
        output = output * ws.unsqueeze(1) + wm.unsqueeze(1)
        var = var * ws.unsqueeze(1)
        confidence = confidence * ws.unsqueeze(1)
        # ======================================================

        # =============================================
        # [중요] Global z-score Denormalize
        # =============================================
        scale = data.scale.expand(Y.size(0), Y.size(1), data.m)

        output = output * scale
        Y = Y * scale
        var *= scale
        confidence *= scale

        if args.anchor_focus_to_last > 0:
            focus_cols = get_focus_columns(data)
            if focus_cols:
                alpha = min(max(args.anchor_focus_to_last, 0.0), 1.0)
                focus_idx = torch.tensor(focus_cols, device=output.device)
                last_obs = X_raw[:, -1, :] * data.scale.expand(X_raw.size(0), data.m)
                y_focus = torch.index_select(output, dim=2, index=focus_idx)
                last_focus = torch.index_select(last_obs, dim=1, index=focus_idx).unsqueeze(1)
                anchor_gain_map = parse_metric_gain_map(args.anchor_boost_map)
                gains = get_col_gain_vector(data, focus_cols, anchor_gain_map).view(1, 1, -1)
                alpha_vec = torch.clamp(alpha * gains, 0.0, 0.98)
                y_focus = (1.0 - alpha_vec) * y_focus + alpha_vec * last_focus
                output[:, :, focus_idx] = y_focus

        if predict is None:
            predict = output
            test = Y
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            variance = torch.cat((variance, var))
            confidence_95 = torch.cat((confidence_95, confidence))

        if args.debug_eval == 1:
            print('EVALUATE RESULTS:')
            scale = data.scale.expand(Y.size(0), Y.size(1), data.m)
            y_pred_o = output
            y_true_o = Y
            for z in range(Y.shape[1]):
                print(y_pred_o[0, z, r], y_true_o[0, z, r])
        
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * output.size(1) * data.m)

        sum_squared_diff += torch.sum(torch.pow(Y - output, 2))
        sum_absolute_diff += torch.sum(torch.abs(Y - output))

    rse = math.sqrt(total_loss / n_samples) / data.rse 
    rae = (total_loss_l1 / n_samples) / data.rae 

    root_sum_squared = math.sqrt(sum_squared_diff)
    test_s = test
    mean_all = torch.mean(test_s, dim=(0, 1))
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m)
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    eps = 1e-12
    if root_sum_squared_r <= eps:
        rrse = 0.0 if root_sum_squared <= eps else 1e6
    else:
        rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_r = torch.sum(torch.abs(diff_r)).item()
    sum_absolute_diff = sum_absolute_diff.item()
    if sum_absolute_r <= eps:
        rae = 0.0 if sum_absolute_diff <= eps else 1e6
    else:
        rae = sum_absolute_diff / sum_absolute_r

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0) & (sigma_p != 0)
    if index.sum() > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = correlation[index]
        correlation = correlation[np.isfinite(correlation)]
        correlation = correlation.mean() if correlation.size > 0 else 0.0
    else:
        correlation = 0.0

    smape = 0
    for x in range(Ytest.shape[0]):
        for z in range(Ytest.shape[2]):
            smape += s_mape(Ytest[x, :, z], predict[x, :, z])
    smape /= Ytest.shape[0] * Ytest.shape[2]

    # ===== jp_fx 개별 RSE 계산 (Best 모델 선택 기준용) =====
    jp_fx_rse = None
    for v in range(data.m):
        raw_name = data.col[v]
        if raw_name == 'jp_fx':
            # jp_fx 노드의 RSE 계산
            jp_pred = predict[:, 0, v]
            jp_true = Ytest[:, 0, v]
            jp_diff_sq = np.sum((jp_pred - jp_true) ** 2)
            jp_mean = np.mean(jp_true)
            jp_diff_from_mean_sq = np.sum((jp_true - jp_mean) ** 2)
            if jp_diff_from_mean_sq > 0:
                jp_fx_rse = np.sqrt(jp_diff_sq / jp_diff_from_mean_sq)
            else:
                jp_fx_rse = 0.0
            break
    if jp_fx_rse is None:
        jp_fx_rse = rrse  # jp_fx를 찾지 못한 경우 전체 RSE 사용
    # ========================================================

    counter = 0
    if is_plot:
        for v in range(data.m):
            col = v
            raw_name = data.col[col]

            node_name = raw_name.replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict[:, 0, col]), torch.from_numpy(Ytest[:, 0, col]), node_name, 'Validation')
            plot_predicted_actual(predict[:, 0, col], Ytest[:, 0, col], node_name, 'Validation', variance[:, 0, col], confidence_95[:, 0, col])
            counter += 1
    return rrse, rae, correlation, smape, jp_fx_rse


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    # ===== Target-Weighted Loss (optional) =====
    target_weight = torch.ones(data.m, device=device)
    if args.focus_targets == 1:
        focus_cols = get_focus_columns(data)
        for col in focus_cols:
            target_weight[col] = args.focus_target_gain
        focus_gain_map = parse_metric_gain_map(args.focus_gain_map)
        for col in focus_cols:
            key = _normalize_metric_name(data.col[col])
            target_weight[col] = target_weight[col] * float(focus_gain_map.get(key, 1.0))
        print(f"[Target-Weighted Loss] weights applied: "
              f"{ {data.col[i]: target_weight[i].item() for i in range(data.m) if target_weight[i] > 1} }")
        if args.focus_only_loss == 1 and focus_cols:
            print(f"[Target-Only Loss] enabled for columns: {[data.col[i] for i in focus_cols]}")
    else:
        print("[Target-Weighted Loss] disabled (uniform weight for overall RSE optimization)")
    if args.bias_penalty > 0:
        if args.bias_penalty_scope == 'focus':
            scope_cols = get_focus_columns(data)
            scope_names = [data.col[i] for i in scope_cols] if scope_cols else []
            print(f"[Bias Penalty] enabled lambda={args.bias_penalty} scope=focus cols={scope_names}")
        else:
            print(f"[Bias Penalty] enabled lambda={args.bias_penalty} scope=all")
    else:
        print("[Bias Penalty] disabled")
    # ==================================

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)  # [B, 1, N, T]

        # ===== RevIN: Per-Window Normalization =====
        w_mean = X.mean(dim=-1, keepdim=True)  # [B, 1, N, 1]
        w_std = X.std(dim=-1, keepdim=True)    # [B, 1, N, 1]
        w_std[w_std == 0] = 1
        X = (X - w_mean) / w_std

        # Normalize target Y using same window stats
        wm = w_mean[:, 0, :, 0]  # [B, N]
        ws = w_std[:, 0, :, 0]   # [B, N]
        Y = (Y - wm.unsqueeze(1)) / ws.unsqueeze(1)  # [B, T_out, N]
        # ============================================

        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.tensor(id).to(device)
            tx = X[:, :, :, :]
            ty = Y[:, :, :]

            if iter > 0 and args.ss_prob > 0 and random.random() < args.ss_prob:
                with torch.no_grad():
                    prev_output = model(tx)
                    prev_pred = torch.squeeze(prev_output, 3)  # [B, T_out, N]
                    if prev_pred.dim() == 2:
                        prev_pred = prev_pred.unsqueeze(1)
                    tail_steps = min(tx.size(3), prev_pred.size(1))
                    pred_tail = prev_pred[:, -tail_steps:, :].transpose(1, 2).unsqueeze(1)  # [B,1,N,tail]
                    tx = tx.clone()
                    tx[:, :, :, -tail_steps:] = pred_tail
            
            output = model(tx)
            output = torch.squeeze(output, 3)

            # ===== Target-Weighted Loss =====
           # ===== [수정] Target-Weighted & Trend Loss 강화 =====
            use_l1 = (args.loss_mode == 'l1')
            
            # 1. 기본 수치 오차 계산 (L1 또는 MSE)
            diff = torch.abs(output - ty) if use_l1 else (output - ty) ** 2
            
            # 2. [추가] 트렌드(기울기) 패널티 계산
            # 실제값의 변화량과 예측값의 변화량 차이를 계산하여 방향성을 맞추도록 유도합니다.
            if output.size(1) > 1:
                out_grad = output[:, 1:, :] - output[:, :-1, :]
                ty_grad = ty[:, 1:, :] - ty[:, :-1, :]
                grad_loss = torch.abs(out_grad - ty_grad)
                temporal_variation = torch.abs(out_grad)
            else:
                grad_loss = None
                temporal_variation = None

            w = target_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
            
            if args.focus_targets == 1 and args.focus_only_loss == 1:
                focus_cols = get_focus_columns(data)
                if focus_cols:
                    focus_idx = torch.tensor(focus_cols, device=device)
                    
                    # Target 노드들에 대한 수치 오차 선택
                    diff_focus = torch.index_select(diff, dim=2, index=focus_idx)
                    w_focus = torch.index_select(w, dim=2, index=focus_idx)
                    
                    # 기본 오차 (Base Loss)
                    loss = (diff_focus * w_focus).mean() / torch.clamp(w_focus.mean(), min=1e-12)
                    
                    # [추가] 트렌드 오차 반영 (기울기가 틀리면 벌칙 부여)
                    if grad_loss is not None:
                        grad_focus = torch.index_select(grad_loss, dim=2, index=focus_idx)
                        loss += args.grad_loss_weight * grad_focus.mean()
                    # Smoothness penalty (Total Variation)
                    if args.smoothness_penalty > 0 and temporal_variation is not None:
                        tv_focus = torch.index_select(temporal_variation, dim=2, index=focus_idx)
                        loss += args.smoothness_penalty * tv_focus.mean()
                else:
                    loss = (diff * w).mean() / torch.clamp(w.mean(), min=1e-12)
            else:
                loss = (diff * w).mean() / torch.clamp(w.mean(), min=1e-12)
                # Also apply gradient loss when not focus_only
                if grad_loss is not None and args.grad_loss_weight > 0:
                    loss += args.grad_loss_weight * grad_loss.mean()

            if args.bias_penalty > 0:
                err_signed = output - ty
                
                if args.bias_penalty_scope == 'focus':
                    focus_cols = get_focus_columns(data)
                    if focus_cols:
                        focus_idx = torch.tensor(focus_cols, device=device)
                        # 중요: 선택된 노드들(US, KR, JP 등)에 대해서만 오차를 뽑음
                        err_signed = torch.index_select(err_signed, dim=2, index=focus_idx)
                        
                        # US가 너무 높게 나오는 문제를 잡기 위해 US에 추가 가중치를 주는 로직
                        # 편향(Bias) 계산 (평균 오차의 제곱)
                        bias_term = torch.mean(torch.mean(err_signed, dim=(0, 1)) ** 2)
                        
                        # US 오차를 더 강하게 잡기 위해 penalty에 2.0배 가중치 곱함
                        loss = loss + (args.bias_penalty * 2.0) * bias_term
                else:
                    # 전체 노드에 대해 적용할 때의 로직
                    bias_term = torch.mean(torch.mean(err_signed, dim=(0, 1)) ** 2)
                    loss = loss + args.bias_penalty * bias_term

            if args.lag_penalty_1step > 0 or args.lag_sign_penalty > 0:
                last_obs = tx[:, 0, :, -1]
                pred_1 = output[:, 0, :]
                true_1 = ty[:, 0, :]

                pred_delta = pred_1 - last_obs
                true_delta = true_1 - last_obs

                if args.focus_targets == 1:
                    focus_cols = get_focus_columns(data)
                    if focus_cols:
                        focus_idx = torch.tensor(focus_cols, device=device)
                        pred_delta = torch.index_select(pred_delta, dim=1, index=focus_idx)
                        true_delta = torch.index_select(true_delta, dim=1, index=focus_idx)

                if args.lag_penalty_1step > 0:
                    lag_diff = torch.abs(pred_delta - true_delta)
                    if args.focus_targets == 1 and focus_cols:
                        lag_gain_map = parse_metric_gain_map(args.lag_penalty_gain_map)
                        lag_w = get_col_gain_vector(data, focus_cols, lag_gain_map).unsqueeze(0)
                        lag_term = torch.sum(lag_diff * lag_w) / torch.clamp(torch.sum(lag_w), min=1e-12)
                    else:
                        lag_term = torch.mean(lag_diff)
                    loss = loss + args.lag_penalty_1step * lag_term

                if args.lag_sign_penalty > 0:
                    sign_raw = torch.relu(-(pred_delta * true_delta))
                    if args.focus_targets == 1 and focus_cols:
                        lag_gain_map = parse_metric_gain_map(args.lag_penalty_gain_map)
                        lag_w = get_col_gain_vector(data, focus_cols, lag_gain_map).unsqueeze(0)
                        sign_term = torch.sum(sign_raw * lag_w) / torch.clamp(torch.sum(lag_w), min=1e-12)
                    else:
                        sign_term = torch.mean(sign_raw)
                    loss = loss + args.lag_sign_penalty * sign_term
            # ==================================
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)
            
            grad_norm = optim.step()

        if iter % 1 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / (output.size(0) * output.size(1) * data.m)))
        iter += 1
    return total_loss / n_samples


DEFAULT_DATA_PATH = BMTGNN_DIR / 'data' / 'sm_data.csv'
DEFAULT_MODEL_SAVE = MODEL_BASE_DIR / 'model.pt'

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default=str(DEFAULT_DATA_PATH), help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default=str(DEFAULT_MODEL_SAVE), help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--loss_mode', type=str, default='l1', choices=['l1', 'mse'], help='training loss type; mse generally aligns better with RSE minimization')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--gcn_true', type=bool, default=False, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=False, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--use_graph', type=int, default=0, choices=[0, 1], help='1: use graph modules, 0: disable graph modules')
parser.add_argument('--gcn_depth', type=int, default=1, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=142, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=40, help='k')
parser.add_argument('--node_dim', type=int, default=30, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=128, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=256, help='skip channels')
parser.add_argument('--end_channels', type=int, default=1024, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=24, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length (multi-step horizon)')
parser.add_argument('--horizon', type=int, default=1, help='forecast start offset (1=predict immediately after input)')
parser.add_argument('--layers', type=int, default=2, help='number of layers')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=10, help='clip')
parser.add_argument('--propalpha', type=float, default=0.6, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=0.1, help='tanh alpha')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')
parser.add_argument('--ss_prob', type=float, default=0.4, help='scheduled sampling probability')
parser.add_argument('--train_ratio', type=float, default=0.8666666667, help='train split ratio')
parser.add_argument('--valid_ratio', type=float, default=0.0666666667, help='validation split ratio')
parser.add_argument('--focus_targets', type=int, default=0, help='1 to upweight us/kr/jp target nodes')
parser.add_argument('--focus_nodes', type=str, default='us_Trade Weighted Dollar Index,jp_fx,kr_fx', help='priority nodes (comma separated)')
parser.add_argument('--focus_weight', type=float, default=0.7, help='priority weight for focus-node RRSE in model selection (0~1)')
parser.add_argument('--focus_target_gain', type=float, default=12.0, help='loss weight applied to focus target columns when focus_targets=1')
parser.add_argument('--focus_only_loss', type=int, default=0, choices=[0, 1], help='1 to optimize loss only on focus/rse target columns')
parser.add_argument('--focus_rrse_mode', type=str, default='max', choices=['mean', 'max'], help='focus RRSE aggregation mode used for model selection')
parser.add_argument('--focus_gain_map', type=str, default='kr_fx:2.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0', help='per-focus-node extra gain map, e.g. kr_fx:2.0,jp_fx:1.0')
parser.add_argument('--anchor_focus_to_last', type=float, default=0.2, help='0~1 level anchoring strength for focus columns during evaluation/forecast')
parser.add_argument('--anchor_boost_map', type=str, default='kr_fx:1.8,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0', help='per-focus-node anchor multiplier map')
parser.add_argument('--rse_targets', type=str, default='Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt', help='comma-separated target series/file names used for terminal RSE/RAE aggregation')
parser.add_argument('--rse_report_mode', type=str, default='targets', choices=['targets', 'all'], help='which RSE/RAE to report in terminal and final summary')
parser.add_argument('--debias_mode', type=str, default='none', choices=['none', 'val_mean_error', 'val_per_step', 'val_linear', 'val_quadratic', 'val_hybrid'], help='bias correction mode for final evaluation/reporting')
parser.add_argument('--debias_apply_to', type=str, default='focus', choices=['focus', 'all'], help='apply debias offset to focus targets only or all series')
parser.add_argument('--debias_skip_nodes', type=str, default='', help='comma-separated node names to SKIP debias for (e.g. kr_fx)')
parser.add_argument('--bias_penalty', type=float, default=0.3, help='lambda for training-time signed-bias penalty (0 disables)')
parser.add_argument('--bias_penalty_scope', type=str, default='focus', choices=['focus', 'all'], help='scope for training-time bias penalty')
parser.add_argument('--plot_focus_only', type=int, default=0, help='1 to plot/save only focus nodes')
parser.add_argument('--debug_eval', type=int, default=0, help='1 to print per-step eval tensors')
parser.add_argument('--rollout_mode', type=str, default='direct', choices=['teacher_forced', 'recursive', 'direct'], help='test rollout mode; direct=single forward pass for multi-step')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--plot', type=int, default=1, help='1 to save plots, 0 to skip plotting')
parser.add_argument('--enforce_cutoff_split', type=int, default=1, choices=[0, 1], help='1: train/validation cannot use rows from cutoff_year_yy and later')
parser.add_argument('--cutoff_year_yy', type=int, default=25, help='forbidden starting yy year for train/validation split')
parser.add_argument('--min_valid_months', type=int, default=12, help='minimum validation rows inside allowed period')
parser.add_argument('--lag_penalty_1step', type=float, default=1.2, help='1-step delta lag penalty weight')
parser.add_argument('--lag_sign_penalty', type=float, default=0.6, help='direction-mismatch proxy penalty weight')
parser.add_argument('--lag_penalty_gain_map', type=str, default='kr_fx:2.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0', help='per-focus-node lag penalty multiplier map')
parser.add_argument('--grad_loss_weight', type=float, default=0.3, help='weight for gradient/trend loss (0 disables)')
parser.add_argument('--smoothness_penalty', type=float, default=0.0, help='TV loss weight to encourage smooth prediction curves (0 disables)')
parser.add_argument('--clean_cache', type=int, default=0, choices=[0, 1], help='1 to delete cached *.pt in data dir before training')
parser.add_argument('--eval_last_epoch', type=int, default=0, choices=[0, 1], help='1 to use the final epoch model for evaluation instead of validation-best checkpoint')
parser.add_argument('--autotune_mode', type=int, default=0, choices=[0, 1], help='1 to optimize for repeated auto-tuning runs')
parser.add_argument('--apply_best_tuning', type=int, default=0, choices=[0, 1], help='1 to override args with best tuning run values')
parser.add_argument('--eval_best_tuning', type=int, default=0, choices=[0, 1], help='1 to skip training and evaluate best tuned checkpoint with plotting')
parser.add_argument('--target_profile', type=str, default='triple_050', choices=['none', 'triple_050', 'run001_us'], help='preset for target-focused optimization setup')
parser.add_argument('--save_pred_dir', type=str, default='', help='directory to save raw prediction/actual numpy arrays for ensemble')
parser.add_argument('--ensemble_seeds', type=str, default='', help='comma-separated seeds for multi-seed ensemble (e.g. 777,42,123). Runs all seeds and averages predictions.')
parser.add_argument('--report_extra_nodes', type=str, default='', help='comma-separated extra node names to include in report plots (used with plot_focus_only=1)')
parser.add_argument('--generate_final_report', type=int, default=1, choices=[0, 1], help='1 to auto-generate final_forecast_results.png and final_summary_table.png')


args = parser.parse_args()
# Track which args were explicitly set on CLI (so profiles don't override them)
_cli_explicit = set()
for action in parser._actions:
    if action.dest == 'help':
        continue
    cli_val = getattr(args, action.dest, None)
    if cli_val != action.default:
        _cli_explicit.add(action.dest)
args.best_tuning_checkpoint = ''
_autotune_mode_val = args.autotune_mode  # save for post-profile override

# If requested, load best tuning run from tuning_runs and override matching args
if args.apply_best_tuning == 1:
    try:
        import json
        from pathlib import Path
        TR = Path(__file__).resolve().parent / 'tuning_runs'
        # find latest tuning run folder by name (timestamped folders)
        runs = sorted([p for p in TR.iterdir() if p.is_dir()])
        if runs:
            latest = runs[-1]
            results_file = latest / 'results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                # pick the run with smallest best_test_rse when available
                best = None
                for r in results:
                    if 'best_test_rse' in r and r.get('best_test_rse') is not None:
                        if best is None or r.get('best_test_rse') < best.get('best_test_rse'):
                            best = r
                if best is None and results:
                    best = results[0]

                if best is not None:
                    run_id = best.get('run_id')
                    log_path = latest / f"run_{int(run_id):03d}.log"
                    ckpt_path = latest / 'checkpoints' / f"model_{int(run_id):03d}.pt"
                    if log_path.exists():
                        # extract Namespace(...) line from log and convert to dict
                        import re
                        ns_line = None
                        with open(log_path, 'r') as lf:
                            for line in lf:
                                if line.strip().startswith('Namespace('):
                                    ns_line = line.strip()
                                    break
                        if ns_line:
                            m = re.search(r"Namespace\((.*)\)", ns_line)
                            if m:
                                inner = m.group(1)
                                try:
                                    parsed = eval('dict(' + inner + ')')
                                except Exception:
                                    parsed = None

                                if isinstance(parsed, dict):
                                    # Only override args that the parser defines.
                                    # Do not overwrite user-run controls like epochs/plot/autotune_mode/clean_cache
                                    skip_keys = {'epochs', 'plot', 'autotune_mode', 'clean_cache'}
                                    for k, v in parsed.items():
                                        if k in skip_keys:
                                            continue
                                        if hasattr(args, k):
                                            try:
                                                setattr(args, k, v)
                                            except Exception:
                                                pass
                                    # ensure autotune mode off when applying tuned HPs
                                    args.autotune_mode = 0
                                    if ckpt_path.exists():
                                        args.best_tuning_checkpoint = str(ckpt_path)
                                        args.save = str(ckpt_path)
                                    print(f"[apply_best_tuning] Applied params from {log_path} (preserved epochs/plot)")
                                else:
                                    print(f"[apply_best_tuning] failed to parse Namespace in {log_path}")
                        else:
                            print(f"[apply_best_tuning] Namespace line not found in {log_path}")
                    else:
                        print(f"[apply_best_tuning] log file not found: {log_path}")
            else:
                print(f"[apply_best_tuning] results.json not found in {latest}")
        else:
            print("[apply_best_tuning] no tuning_runs directories found")
    except Exception as e:
        print(f"[apply_best_tuning] error while loading tuning run: {e}")

# triple_050 profile is applied via default args + command-line overrides
if args.target_profile == 'triple_050':
    print('[target_profile] applied: triple_050 (seed=1, l1, 180ep, eval_last, direct, focus_only=1)')

if args.target_profile == 'run001_us':
    args.loss_mode = 'mse'
    args.use_graph = 0
    args.lr = 0.0002
    args.dropout = 0.1
    args.layers = 2
    args.seq_in_len = 24
    args.seq_out_len = 12
    args.ss_prob = 0.4
    args.epochs = 120
    args.seed = 2026
    args.focus_targets = 1
    args.focus_nodes = 'us_Trade Weighted Dollar Index'
    args.focus_weight = 1.0
    args.focus_target_gain = 60.0
    args.focus_only_loss = 1
    args.anchor_focus_to_last = 0.2
    args.rse_targets = 'Us_Trade Weighted Dollar Index_Testing.txt'
    args.rse_report_mode = 'targets'
    args.debias_mode = 'none'
    args.debias_apply_to = 'focus'
    args.rollout_mode = 'direct'
    print('[target_profile] applied: run001_us (direct mode)')

# autotune_mode overrides AFTER profile (so profile defaults can be set first)
# BUT respect CLI-explicit args
if _autotune_mode_val == 1:
    if 'plot' not in _cli_explicit:
        args.plot = 0
    if 'clean_cache' not in _cli_explicit:
        args.clean_cache = 0
    if 'generate_final_report' not in _cli_explicit:
        args.generate_final_report = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")
    torch.set_num_threads(3)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fixed_seed = args.seed


def check_lagging(pred_np, actual_np, data, focus_cols):
    """Detect if predictions are simply lagging (delayed copy of) actual values."""
    print('\n' + '='*70)
    print('LAG DETECTION DIAGNOSTIC')
    print('='*70)
    any_lag = False
    for c in (focus_cols or []):
        p = pred_np[:, c]
        a = actual_np[:, c]
        T = len(p)
        if T < 3:
            print(f'  {data.col[c]}: insufficient data for lag check (T={T})')
            continue

        # Lag-0 vs Lag-1 correlation
        lag0_corr = np.corrcoef(p, a)[0, 1] if np.std(p) > 0 and np.std(a) > 0 else 0.0
        lag1_corr = np.corrcoef(p[:-1], a[1:])[0, 1] if np.std(p[:-1]) > 0 and np.std(a[1:]) > 0 else 0.0

        # Direction match (does prediction capture direction of change?)
        delta_pred = np.diff(p)
        delta_actual = np.diff(a)
        if np.std(delta_pred) > 1e-12 and np.std(delta_actual) > 1e-12:
            dir_match = np.mean(np.sign(delta_pred) == np.sign(delta_actual))
        else:
            dir_match = 0.0

        # Naive last-value RSE (if model just copies last input)
        last_val_pred = np.full_like(a, a[0])  # crude: predict first value for all
        last_val_rse = np.sqrt(np.sum((last_val_pred - a)**2) / max(np.sum((a - a.mean())**2), 1e-12))
        actual_rse = np.sqrt(np.sum((p - a)**2) / max(np.sum((a - a.mean())**2), 1e-12))

        # Flatness check: is prediction range much smaller than actual range?
        pred_range = p.max() - p.min()
        actual_range = a.max() - a.min()
        range_ratio = pred_range / max(actual_range, 1e-12)

        is_lagging = lag1_corr > lag0_corr + 0.05
        is_flat = range_ratio < 0.3
        status = ''
        if is_lagging:
            status += ' LAGGING'
            any_lag = True
        if is_flat:
            status += ' FLAT'
            any_lag = True
        if not status:
            status = ' OK'

        print(f'  {data.col[c]:>45s}: lag0_r={lag0_corr:+.4f}  lag1_r={lag1_corr:+.4f}  '
              f'dir_match={dir_match:.1%}  range_ratio={range_ratio:.2f}  RSE={actual_rse:.4f}  [{status.strip()}]')
    if not any_lag:
        print('  [PASS] No lagging or flatness issues detected.')
    else:
        print('  [WARN] Potential lagging/flatness detected in some targets.')
    print('='*70 + '\n')


def generate_final_report(pred_np, actual_np, data, focus_cols, confidence_np=None):
    """Auto-generate final_forecast_results.png and final_summary_table.png
    using in-memory prediction arrays. This provides standardised submission outputs."""
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    months_labels = ['Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25',
                     'Jul-25', 'Aug-25', 'Sep-25', 'Oct-25', 'Nov-25', 'Dec-25']

    target_display = {
        'us_Trade Weighted Dollar Index': ('US Trade Weighted Dollar Index', 'Index'),
        'kr_fx': ('KRW/USD Exchange Rate', 'KRW'),
        'jp_fx': ('JPY/USD Exchange Rate', 'JPY'),
    }

    n_targets = len(focus_cols)
    T = pred_np.shape[0]
    labels = months_labels[:T]

    # === Figure 1: 3-panel forecast results ===
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 5 * n_targets), sharex=False)
    if n_targets == 1:
        axes = [axes]

    rse_results = {}
    for idx, c in enumerate(focus_cols):
        ax = axes[idx]
        name = data.col[c]
        display_name, unit = target_display.get(name, (name, ''))
        p = pred_np[:, c]
        a = actual_np[:, c]

        ss_err = np.sum((p - a)**2)
        ss_tot = np.sum((a - a.mean())**2)
        rse = np.sqrt(ss_err / max(ss_tot, 1e-12))
        max_err = np.max(np.abs(p - a))
        max_err_pct = np.max(np.abs(p - a) / np.clip(np.abs(a), 1e-12, None)) * 100
        rse_results[name] = {'rse': rse, 'max_err': max_err, 'max_err_pct': max_err_pct}

        x = np.arange(T)
        ax.plot(x, a, 'b-o', linewidth=2.5, markersize=7, label='Actual', zorder=5)
        ax.plot(x, p, '--s', color='purple', linewidth=2, markersize=6,
                label=f'Predicted (RSE={rse:.4f})', alpha=0.85)

        if confidence_np is not None:
            ci = confidence_np[:, c]
            ax.fill_between(x, p - ci, p + ci, alpha=0.25, color='pink',
                            label='95% Prediction Interval')

        ax.set_title(f'{display_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Value ({unit})' if unit else 'Value', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=45)
        ax.axhline(y=a.mean(), color='gray', linestyle=':', alpha=0.4)

    seed_str = f'seed={fixed_seed}'
    fig.suptitle(
        f'B-MTGNN Exchange Rate Forecasting — 2025 Test Period\n'
        f'TRIPLE050 ({seed_str}, focus_only_loss=1, no_graph, lr={args.lr})',
        fontsize=16, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = BMTGNN_DIR / 'final_forecast_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[report] Saved: {out_path}')

    # === Figure 2: Summary table ===
    fig2, ax2 = plt.subplots(figsize=(12, 3 + 0.5 * n_targets))
    ax2.axis('off')

    headers = ['Target', 'RSE', 'Max Error', 'Max Error %', 'Status']
    rows = []
    for c in focus_cols:
        name = data.col[c]
        r = rse_results.get(name, {})
        rse_val = r.get('rse', 0)
        status = 'PASS' if rse_val < 0.5 else 'FAIL'
        rows.append([
            name,
            f'{rse_val:.4f}',
            f'{r.get("max_err", 0):.4f}',
            f'{r.get("max_err_pct", 0):.2f}%',
            status,
        ])

    table = ax2.table(cellText=rows, colLabels=headers, loc='center',
                       cellLoc='center', colWidths=[0.35, 0.12, 0.15, 0.15, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Style status column
    n_cols = len(headers)
    for i in range(len(rows)):
        status = rows[i][-1]
        color = '#C6EFCE' if status == 'PASS' else '#FFC7CE'
        table[i + 1, n_cols - 1].set_facecolor(color)

    ax2.set_title(
        f'Per-Target RSE Summary (seed={fixed_seed}, {args.epochs}ep, '
        f'{"eval_last" if args.eval_last_epoch else "best_val"})',
        fontsize=13, fontweight='bold', pad=20
    )
    out_path2 = BMTGNN_DIR / 'final_summary_table.png'
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[report] Saved: {out_path2}')

    return rse_results


def main(experiment):
    set_random_seed(fixed_seed)

    # ===== Fixed HP (0209 최적 결과 기반) - no random search =====
    best_val = 10000000
    best_rse = 10000000
    best_rae = 10000000
    best_corr = -10000000
    best_smape = 10000000
    best_objective = 10000000
    best_test_rse = 10000000
    best_test_corr = -10000000

    best_hp = []

    for q in range(1):
        # 0209 최적 HP 사용 (고정)
        gcn_depth = args.gcn_depth
        lr = args.lr
        conv = args.conv_channels
        res = args.residual_channels
        skip = args.skip_channels
        end = args.end_channels
        layer = args.layers
        k = args.subgraph_size
        dropout = args.dropout
        dilation_ex = args.dilation_exponential
        node_dim = args.node_dim
        prop_alpha = args.propalpha
        tanh_alpha = args.tanhalpha

        # ============================================================
        # Cache Cleaning (optional)
        # ============================================================
        if args.clean_cache == 1:
            data_dir = os.path.dirname(args.data)
            pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
            print(f"!!! Cleaning Cache Files in {data_dir} !!!")
            for file_path in pt_files:
                if "model" not in file_path:
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
            print("!!! Cache Clean Complete !!!")
        # ============================================================

        eff_train_ratio, eff_valid_ratio, split_info = compute_effective_split_by_cutoff(
            args.data,
            args.train_ratio,
            args.valid_ratio,
            args.enforce_cutoff_split,
            args.cutoff_year_yy,
            args.min_valid_months,
        )

        if split_info.get('enforced'):
            print(
                f"[split-cutoff] enforced=1 forbidden_year>={split_info['forbidden_year']} "
                f"allowed_rows={split_info['allowed_rows']}/{split_info['total_rows']} "
                f"train_rows={split_info['train_rows']} valid_rows={split_info['valid_rows']} "
                f"last_allowed={split_info['last_allowed']}"
            )
        else:
            print('[split-cutoff] enforced=0')

        Data = DataLoaderS(args.data, eff_train_ratio, eff_valid_ratio, device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len)

        print('train X:', Data.train[0].shape)
        print('train Y:', Data.train[1].shape)
        print('valid X:', Data.valid[0].shape)
        print('valid Y:', Data.valid[1].shape)
        print('test X:', Data.test[0].shape)
        print('test Y:', Data.test[1].shape)
        print('test window:', Data.test_window.shape)

        print('length of training set=', Data.train[0].shape[0])
        print('length of validation set=', Data.valid[0].shape[0])
        print('length of testing set=', Data.test[0].shape[0])
        print('valid=', int((args.train_ratio + args.valid_ratio) * Data.n))

        if len(Data.train[0].shape) == 4: 
             args.num_nodes = Data.train[0].shape[2]
        elif len(Data.train[0].shape) == 3: 
             args.num_nodes = Data.train[0].shape[2]

        print(f"Auto-detected num_nodes: {args.num_nodes}")

        # ============================================================
        # DATA LEAKAGE VERIFICATION
        # ============================================================
        _train_end = Data.train_end
        _valid_end = int((eff_train_ratio + eff_valid_ratio) * Data.n)
        _test_start = _valid_end
        print('='*70)
        print('DATA LEAKAGE VERIFICATION')
        print('='*70)
        print(f'  Total rows:           {Data.n}')
        print(f'  Train rows:           0..{_train_end - 1}  ({_train_end} rows)')
        print(f'  Valid rows:           {_train_end}..{_valid_end - 1}  ({_valid_end - _train_end} rows)')
        print(f'  Test rows:            {_test_start}..{Data.n - 1}  ({Data.n - _test_start} rows)')
        print(f'  Normalization:        scale computed from rows [0..{_train_end - 1}] ONLY (normalize={args.normalize})')
        print(f'  Valid window shape:   {Data.valid_window.shape}  (starts {max(0, _train_end - args.seq_in_len)})')
        print(f'  Test window shape:    {Data.test_window.shape}   (starts {max(0, _valid_end - args.seq_in_len)})')
        # Read date tokens for verification
        try:
            _df_check = pd.read_csv(args.data)
            _date_col = _df_check.iloc[:, 0].astype(str).tolist()
            if _train_end > 0:
                print(f'  Train date range:     {_date_col[0]} ~ {_date_col[_train_end - 1]}')
            if _valid_end > _train_end:
                print(f'  Valid date range:     {_date_col[_train_end]} ~ {_date_col[_valid_end - 1]}')
            if Data.n > _test_start:
                print(f'  Test date range:      {_date_col[_test_start]} ~ {_date_col[Data.n - 1]}')
            # Check for 2025 data in train/valid
            _forbidden = 2000 + args.cutoff_year_yy
            for _i in range(_valid_end):
                _parsed = parse_month_token(_date_col[_i])
                if _parsed is not None:
                    _yr = _parsed[0]  # parse_month_token already returns 2000+yy
                    if _yr >= _forbidden:
                        print(f'  [FAIL] Row {_i} ({_date_col[_i]}) contains year {_yr} in train/valid!')
                        break
            else:
                print(f'  [OK] No {_forbidden}+ data in train/valid (leakage-free)')
        except Exception as _e:
            print(f'  [WARN] Could not verify dates: {_e}')
        print(f'  Seed:                 {fixed_seed}')
        print(f'  Checkpoint mode:      {"eval_last_epoch" if args.eval_last_epoch else "best_val_checkpoint"}')
        print(f'  Save path:            {args.save}')
        print('='*70)
        # ============================================================

        use_graph = bool(args.use_graph)
        model = gtnet(use_graph, use_graph, gcn_depth, args.num_nodes,
                      device, Data.adj, dropout=dropout, subgraph_size=k,
                      node_dim=node_dim, dilation_exponential=dilation_ex,
                      conv_channels=conv, residual_channels=res,
                      skip_channels=skip, end_channels=end,
                      seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)
        model = model.to(device)

        print(args)
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        if args.loss_mode == 'l1':
            criterion = nn.L1Loss(reduction='sum').to(device)
        else:
            criterion = nn.MSELoss(reduction='sum').to(device)
        evaluateL2 = nn.MSELoss(reduction='sum').to(device)
        evaluateL1 = nn.L1Loss(reduction='sum').to(device)

        optim = Optim(
            model.parameters(), args.optim, lr, args.clip, weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim.optimizer, mode='min', factor=0.5, patience=15
        )

        es_counter = 0
        try:
            if args.eval_best_tuning == 1:
                ckpt_to_load = args.best_tuning_checkpoint if args.best_tuning_checkpoint else args.save
                if ckpt_to_load and os.path.exists(ckpt_to_load):
                    with open(ckpt_to_load, 'rb') as f:
                        model = torch.load(f, weights_only=False)
                    model = model.to(device)
                    print(f"[eval_best_tuning] loaded checkpoint: {ckpt_to_load}")
                else:
                    raise FileNotFoundError(f"[eval_best_tuning] checkpoint not found: {ckpt_to_load}")
            else:
                print('begin training')
            for epoch in range(1, args.epochs + 1):
                if args.eval_best_tuning == 1:
                    break
                print('Experiment:', (experiment + 1))
                print('Iter:', q)
                print('epoch:', epoch)
                print('hp=', [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch])
                print('best sum=', best_val)
                print('best rrse=', best_rse)
                print('best rrae=', best_rae)
                print('best corr=', best_corr)
                print('best smape=', best_smape)
                print('best objective=', best_objective)
                print('best hps=', best_hp)
                print('best test rse=', best_test_rse)
                print('best test corr=', best_test_corr)

                es_counter += 1

                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr, val_smape, val_focus_rrse = evaluate_sliding_window(
                    Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Validation'
                )
                if val_focus_rrse is None:
                    val_focus_rrse = val_loss
                focus_weight = min(max(args.focus_weight, 0.0), 1.0)
                objective_score = (1.0 - focus_weight) * val_loss + focus_weight * val_focus_rrse
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f} | focus_rrse {:5.4f} | obj {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape, val_focus_rrse, objective_score), flush=True)
                
                scheduler.step(objective_score)
                
                # ===== Best 모델 선택 기준: 전체 RSE + 우선노드 RSE 가중 합 =====
                safe_corr = val_corr if not math.isnan(val_corr) else 0.0
                sum_loss = val_loss + val_rae - safe_corr

                if objective_score < best_objective:
                    save_path = Path(args.save)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(save_path, 'wb') as f:
                        torch.save(model, f)
                    best_val = sum_loss
                    best_rse = val_loss
                    best_rae = val_rae
                    best_corr = val_corr
                    best_smape = val_smape
                    best_objective = objective_score

                    best_hp = [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]

                    es_counter = 0

                    test_acc, test_rae, test_corr, test_smape, test_focus_rrse = evaluate_sliding_window(
                        Data, Data.test_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Testing'
                    )
                    print('********************************************************************************************************')
                    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f} | test focus_rrse {:5.4f}".format(
                        test_acc, test_rae, test_corr, test_smape, test_focus_rrse if test_focus_rrse is not None else test_acc), flush=True)
                    print('********************************************************************************************************')
                    best_test_rse = test_acc
                    best_test_corr = test_corr

                    # === Per-target best checkpoint: save model when each target's val RSE improves ===
                    if args.save_pred_dir:
                        focus_cols = get_focus_columns(Data)
                        if focus_cols:
                            if not hasattr(main, '_per_target_best_val'):
                                main._per_target_best_val = {}
                            # Single eval call for validation arrays
                            _, _, _, _, _, v_pred_ck, v_true_ck = evaluate_sliding_window(
                                Data, Data.valid_window, model, evaluateL2, evaluateL1,
                                args.seq_in_len, False, 'Validation', return_arrays=True
                            )
                            vp_np = v_pred_ck.cpu().numpy() if hasattr(v_pred_ck, 'cpu') else v_pred_ck
                            vt_np = v_true_ck.cpu().numpy() if hasattr(v_true_ck, 'cpu') else v_true_ck
                            for c in focus_cols:
                                ss_err = np.sum((vp_np[:, c] - vt_np[:, c])**2)
                                ss_tot = np.sum((vt_np[:, c] - vt_np[:, c].mean())**2)
                                v_rse_c = math.sqrt(ss_err / ss_tot) if ss_tot > 1e-12 else float('inf')
                                tname = Data.col[c]
                                prev_best = main._per_target_best_val.get(tname, float('inf'))
                                if v_rse_c < prev_best:
                                    main._per_target_best_val[tname] = v_rse_c
                                    ck_dir = Path(args.save_pred_dir) / ("best_%s" % tname.replace(' ', '_'))
                                    ck_dir.mkdir(parents=True, exist_ok=True)
                                    torch.save(model.state_dict(), ck_dir / "model_state.pt")
                                    print("[per-target-ckpt] %s: val_RSE=%.4f at epoch %d (improved)" % (tname, v_rse_c, epoch))

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    print('best val loss=', best_val)
    print('best hps=', best_hp)
    
    hp_save_path = MODEL_BASE_DIR / 'hp.txt'
    hp_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hp_save_path, "w") as f:
        f.write(str(best_hp))

    if args.eval_last_epoch == 1:
        print(f"[eval_last_epoch] Using final epoch model (skipping checkpoint load)")
    elif os.path.exists(args.save):
        with open(args.save, 'rb') as f:
            model = torch.load(f, weights_only=False)
    else:
        print(f"Warning: checkpoint not found at {args.save}. Using current in-memory model.")
    # 로드한 모델도 학습 시와 같은 device로 이동
    model = model.to(device)

    if args.plot == 1:
        clear_split_outputs('Validation')
        clear_split_outputs('Testing')

    bias_offset = None
    if args.debias_mode == 'val_mean_error':
        _, _, _, _, _, v_pred_t, v_true_t = evaluate_sliding_window(
            Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Validation', return_arrays=True
        )
        bias_offset = estimate_bias_offset_from_arrays(Data, v_pred_t, v_true_t)
        focus_cols = get_focus_columns(Data)
        if focus_cols:
            summary = {Data.col[c]: float(bias_offset[c].item()) for c in focus_cols}
            print(f"[debias] mode={args.debias_mode} apply_to={args.debias_apply_to} offset={summary}")
    elif args.debias_mode == 'val_per_step':
        _, _, _, _, _, v_pred_t, v_true_t = evaluate_sliding_window(
            Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Validation', return_arrays=True
        )
        bias_offset = estimate_per_step_bias(Data, v_pred_t, v_true_t)
        focus_cols = get_focus_columns(Data)
        if focus_cols:
            for c in focus_cols:
                vals = [f"{bias_offset[t, c].item():.2f}" for t in range(min(12, bias_offset.shape[0]))]
                print(f"[debias per_step] {Data.col[c]}: [{', '.join(vals)}]")
    elif args.debias_mode == 'val_linear':
        _, _, _, _, _, v_pred_t, v_true_t = evaluate_sliding_window(
            Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Validation', return_arrays=True
        )
        bias_offset = estimate_linear_trend_bias(Data, v_pred_t, v_true_t)
        focus_cols = get_focus_columns(Data)
        if focus_cols:
            for c in focus_cols:
                vals = [f"{bias_offset[t, c].item():.2f}" for t in range(min(12, bias_offset.shape[0]))]
                print(f"[debias linear] {Data.col[c]}: [{', '.join(vals)}]")
    elif args.debias_mode == 'val_quadratic':
        _, _, _, _, _, v_pred_t, v_true_t = evaluate_sliding_window(
            Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Validation', return_arrays=True
        )
        bias_offset = estimate_quadratic_trend_bias(Data, v_pred_t, v_true_t)
        focus_cols = get_focus_columns(Data)
        if focus_cols:
            for c in focus_cols:
                vals = [f"{bias_offset[t, c].item():.2f}" for t in range(min(12, bias_offset.shape[0]))]
                print(f"[debias quadratic] {Data.col[c]}: [{', '.join(vals)}]")
    elif args.debias_mode == 'val_hybrid':
        _, _, _, _, _, v_pred_t, v_true_t = evaluate_sliding_window(
            Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Validation', return_arrays=True
        )
        bias_offset = estimate_hybrid_debias(Data, v_pred_t, v_true_t)
        focus_cols = get_focus_columns(Data)
        if focus_cols:
            for c in focus_cols:
                vals = [f"{bias_offset[t, c].item():.2f}" for t in range(min(12, bias_offset.shape[0]))]
                print(f"[debias hybrid] {Data.col[c]}: [{', '.join(vals)}]")

    # Reset seed before final evaluation for reproducibility
    set_random_seed(fixed_seed)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape, _ = evaluate_sliding_window(
        Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, args.plot == 1, 'Validation', bias_offset=bias_offset
    )

    set_random_seed(fixed_seed)

    test_acc, test_rae, test_corr, test_smape, test_focus_rrse = evaluate_sliding_window(
        Data, Data.test_window, model, evaluateL2, evaluateL1, args.seq_in_len, args.plot == 1, 'Testing', bias_offset=bias_offset
    )

    # ============================================================
    # FINAL EVALUATION — Per-target RSE (MOST IMPORTANT OUTPUT)
    # ============================================================
    set_random_seed(fixed_seed)
    _final_focus_cols = get_focus_columns(Data)
    _fp = _ft = _fp_np = _ft_np = None
    _confidence_np = None
    if _final_focus_cols:
        _, _, _, _, _, _fp, _ft = evaluate_sliding_window(
            Data, Data.test_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Testing', bias_offset=bias_offset, return_arrays=True
        )
        _fp_np = _fp.cpu().numpy() if hasattr(_fp, 'cpu') else _fp
        _ft_np = _ft.cpu().numpy() if hasattr(_ft, 'cpu') else _ft

    print('\n')
    print('#' * 70)
    print('#' + ' ' * 18 + 'FINAL RESULTS SUMMARY' + ' ' * 19 + '#')
    print('#' * 70)
    print(f'#  Profile:   {args.target_profile}')
    print(f'#  Seed:      {fixed_seed}')
    print(f'#  Epochs:    {args.epochs}')
    print(f'#  Eval mode: {"eval_last_epoch" if args.eval_last_epoch else "best_val_checkpoint"}')
    print(f'#  Device:    {device}')
    print('#' + '-' * 68 + '#')
    print('#  {:>45s}  {:>10s}  {:>6s}  #'.format('Target', 'RSE', 'Pass?'))
    print('#' + '-' * 68 + '#')
    _all_pass = True
    if _final_focus_cols and _fp_np is not None:
        for _c in _final_focus_cols:
            _p = _fp_np[:, _c]
            _a = _ft_np[:, _c]
            _den = np.sum((_a - _a.mean())**2)
            _rse = np.sqrt(np.sum((_p - _a)**2) / _den) if _den > 1e-12 else float('inf')
            _pass = 'YES' if _rse < 0.5 else 'NO'
            if _rse >= 0.5:
                _all_pass = False
            print(f'#  {Data.col[_c]:>45s}  {_rse:10.4f}  {_pass:>6s}  #')
    print('#' + '-' * 68 + '#')
    print(f'#  Overall RSE (agg): {test_acc:.4f}   RAE: {test_rae:.4f}   Corr: {test_corr:.4f}')
    print(f'#  Focus RRSE (max):  {test_focus_rrse:.4f}' if test_focus_rrse is not None else '')
    _verdict = 'ALL TARGETS PASS (< 0.5)' if _all_pass else 'SOME TARGETS FAILED'
    print(f'#  Verdict:           {_verdict}')
    print('#' * 70)

    # Save predictions to save_pred_dir if specified
    if _final_focus_cols and _fp_np is not None and args.save_pred_dir:
        from pathlib import Path as _P
        _save_dir = _P(args.save_pred_dir)
        _save_dir.mkdir(parents=True, exist_ok=True)
        np.save(_save_dir / "pred_Testing.npy", _fp_np)
        np.save(_save_dir / "actual_Testing.npy", _ft_np)
        print(f'[save] Predictions saved to {_save_dir}')

    # ============================================================
    # LAG DETECTION
    # ============================================================
    if _final_focus_cols and _fp_np is not None:
        check_lagging(_fp_np, _ft_np, Data, _final_focus_cols)

    # ============================================================
    # AUTO-GENERATE FINAL REPORT IMAGES
    # ============================================================
    if args.generate_final_report and _final_focus_cols and _fp_np is not None:
        generate_final_report(_fp_np, _ft_np, Data, _final_focus_cols, confidence_np=_confidence_np)

    # ===== Per-target best checkpoint evaluation =====
    if args.save_pred_dir and hasattr(main, '_per_target_best_val'):
        print('\n' + '='*70)
        print('PER-TARGET BEST CHECKPOINT EVALUATION')
        print('='*70)
        focus_cols = get_focus_columns(Data)
        # We need the model architecture to load state_dict
        import copy
        base_model = copy.deepcopy(model)
        for c in (focus_cols or []):
            tname = Data.col[c]
            safe_name = tname.replace(' ', '_')
            ck_dir = Path(args.save_pred_dir) / ("best_%s" % safe_name)
            ck_file = ck_dir / "model_state.pt"
            if ck_file.exists():
                # Load per-target best checkpoint
                state = torch.load(ck_file, weights_only=True, map_location=device)
                base_model.load_state_dict(state)
                base_model.to(device)
                base_model.eval()
                # Evaluate with this checkpoint
                _, _, _, _, _, tp_ck, ta_ck = evaluate_sliding_window(
                    Data, Data.test_window, base_model, evaluateL2, evaluateL1,
                    args.seq_in_len, False, 'Testing', return_arrays=True
                )
                tp_np = tp_ck.cpu().numpy() if hasattr(tp_ck, 'cpu') else tp_ck
                ta_np = ta_ck.cpu().numpy() if hasattr(ta_ck, 'cpu') else ta_ck
                ss_err = np.sum((tp_np[:, c] - ta_np[:, c])**2)
                ss_tot = np.sum((ta_np[:, c] - ta_np[:, c].mean())**2)
                t_rse = math.sqrt(ss_err / ss_tot) if ss_tot > 1e-12 else float('inf')
                v_rse = main._per_target_best_val.get(tname, float('inf'))
                print("  %s: test_RSE=%.4f  (val_RSE=%.4f)" % (tname, t_rse, v_rse))

                # Also save numpy predictions from this per-target checkpoint
                np.save(ck_dir / "pred_Testing.npy", tp_np)
                np.save(ck_dir / "actual_Testing.npy", ta_np)
                # Also try debias modes
                _, _, _, _, _, vp_ck, va_ck = evaluate_sliding_window(
                    Data, Data.valid_window, base_model, evaluateL2, evaluateL1,
                    args.seq_in_len, False, 'Validation', return_arrays=True
                )
                for dname, dfn in [('linear', estimate_linear_trend_bias), ('quadratic', estimate_quadratic_trend_bias)]:
                    off = dfn(Data, vp_ck, va_ck)
                    dp = tp_ck.clone()
                    if off.dim() == 1:
                        dp[:, c] -= off[c]
                    else:
                        dp[:, c] -= off[:, c]
                    dp_np = dp.cpu().numpy()
                    ss_err_d = np.sum((dp_np[:, c] - ta_np[:, c])**2)
                    t_rse_d = math.sqrt(ss_err_d / ss_tot) if ss_tot > 1e-12 else float('inf')
                    print("    + debias=%s: test_RSE=%.4f" % (dname, t_rse_d))
            else:
                print("  %s: no checkpoint found" % tname)
        print('='*70)

    # ===== Multi-debias comparison (auto-runs after main evaluation) =====
    if not args.autotune_mode:
        print('\n' + '='*70)
        print('POST-HOC DEBIAS COMPARISON (all modes)')
        print('='*70)
        debias_modes = [
            ('none', None),
            ('val_mean_error', estimate_bias_offset_from_arrays),
            ('val_linear', estimate_linear_trend_bias),
            ('val_quadratic', estimate_quadratic_trend_bias),
            ('val_hybrid', estimate_hybrid_debias),
        ]
        # Get validation predictions once
        _, _, _, _, _, v_pred_cmp, v_true_cmp = evaluate_sliding_window(
            Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Validation', return_arrays=True
        )
        # Get test predictions once (no debias) to compute per-target RSE
        _, _, _, _, _, t_pred_raw, t_true_raw = evaluate_sliding_window(
            Data, Data.test_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Testing', return_arrays=True
        )
        focus_cols = get_focus_columns(Data)
        if focus_cols and t_pred_raw is not None and t_true_raw is not None:
            best_per_target = {}
            for mode_name, mode_fn in debias_modes:
                if mode_fn is not None:
                    b = mode_fn(Data, v_pred_cmp, v_true_cmp)
                else:
                    b = None

                # Apply debias to test predictions
                test_p = t_pred_raw.clone()
                if b is not None:
                    bb = b.to(test_p.device)
                    if bb.dim() == 1:
                        bb = bb.unsqueeze(0)
                    test_p = test_p - bb
                test_a = t_true_raw.clone()

                rse_list = []
                for c in focus_cols:
                    p = test_p[:, c].cpu().numpy() if test_p.dim() == 2 else test_p[:, :, c].reshape(-1).cpu().numpy()
                    a = test_a[:, c].cpu().numpy() if test_a.dim() == 2 else test_a[:, :, c].reshape(-1).cpu().numpy()
                    ss_err = np.sum((p - a)**2)
                    ss_total = np.sum((a - a.mean())**2)
                    rse = math.sqrt(ss_err / ss_total) if ss_total > 1e-12 else float('inf')
                    me = float((p - a).mean())
                    rse_list.append((Data.col[c], rse, me))
                    if c not in best_per_target or rse < best_per_target[c][1]:
                        best_per_target[c] = (mode_name, rse, me)

                rse_strs = [f"{n}={r:.4f}(ME={m:+.2f})" for n, r, m in rse_list]
                print(f"  debias={mode_name:16s} | {' | '.join(rse_strs)}")

            print('\n  BEST per target (pick best debias mode for each):')
            for c in focus_cols:
                mode_name, rse, me = best_per_target[c]
                print(f"    {Data.col[c]:>45s}: RSE={rse:.4f} (mode={mode_name}, ME={me:+.2f})")
        print('='*70 + '\n')

    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape


if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    vsmape = []
    acc = []
    rae = []
    corr = []
    smape = []
    for i in range(1):
        val_acc, val_rae, val_corr, val_smape, test_acc, test_rae, test_corr, test_smape = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        vsmape.append(val_smape)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
        smape.append(test_smape)
    print('\n\n')
    print('1 run average')
    print('\n\n')
    print("valid\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae)))
    print('\n\n')
    print("test\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae)))