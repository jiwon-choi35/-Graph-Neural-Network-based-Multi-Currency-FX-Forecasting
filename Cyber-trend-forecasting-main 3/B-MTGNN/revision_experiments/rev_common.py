"""Shared helpers for the JIIS-2026-079 revision experiments.

Reproduces, outside train_test.py / forecast.py (which parse CLI args at import):
  - one_step_eval():  train_test.evaluate_sliding_window (teacher-forced, 10 MC runs,
                      eq.(7) anchoring, eq.(10) interval)  -> Table 2/3 setting
  - recursive_paths(): forecast.py MC-dropout recursive rollout (each MC run keeps its own
                      trajectory, all nodes fed back), with optional eq.(7) anchoring
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BMTGNN_DIR = Path(__file__).resolve().parents[1]
if str(BMTGNN_DIR) not in sys.path:
    sys.path.insert(0, str(BMTGNN_DIR))
import net  # noqa: E402,F401  (needed so torch.load can unpickle gtnet)

DATA_DIR = BMTGNN_DIR / 'data'
SM_PATH = DATA_DIR / 'sm_data.csv'
RAW_PATH = DATA_DIR / 'data.csv'

TARGETS = ['us_Trade Weighted Dollar Index', 'jp_fx', 'kr_fx']
TARGET_LABEL = {'us_Trade Weighted Dollar Index': 'USD Index', 'jp_fx': 'JPY/USD', 'kr_fx': 'KRW/USD'}

# triple_050 profile: anchor_focus_to_last=0.06, anchor_boost_map kr 1.8 / jp 1.0 / us 5.0
ANCHOR_ALPHA = {'us_Trade Weighted Dollar Index': 0.06 * 5.0, 'jp_fx': 0.06 * 1.0, 'kr_fx': 0.06 * 1.8}

SEQ_IN = 24
TRAIN_ROWS = 156          # 2011.01 ~ 2023.12
VALID_END = 168           # 2024.01 ~ 2024.12 validation; 2025 test = rows 168..179
N_ROWS = 180


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_frame(path=SM_PATH):
    df = pd.read_csv(path)
    dates = df.iloc[:, 0].astype(str).tolist()
    vals = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
    return dates, list(df.columns[1:]), vals


def train_scale(values, train_rows=TRAIN_ROWS):
    """Per-column max |x| over training rows, in float32 exactly like DataLoaderS(normalize=2)."""
    s = torch.from_numpy(values[:train_rows]).float().abs().max(dim=0).values
    s[s == 0] = 1.0
    return s


def _scaled(values, scale, device):
    # Same float32 arithmetic as DataLoaderS: RevIN divides by the window std, so near-constant
    # columns make predictions sensitive to 1-ulp differences in the scaled inputs.
    return (torch.from_numpy(values).float() / scale).to(device)


def build_paper_gtnet(num_nodes, device, adj_mode='learned'):
    """gtnet with the triple_050 hyper-parameters (for checkpoints saved as a state_dict)."""
    A = torch.eye(num_nodes, device=device) if adj_mode == 'identity' else None
    return net.gtnet(True, adj_mode == 'learned', 1, num_nodes, device, A, dropout=0.02, subgraph_size=40,
                     node_dim=30, dilation_exponential=2, conv_channels=16, residual_channels=128,
                     skip_channels=256, end_channels=1024, seq_length=SEQ_IN, in_dim=1, out_dim=1, layers=2,
                     propalpha=0.6, tanhalpha=0.1, layer_norm_affline=False)


def load_model(path, device='cpu', num_nodes=33, adj_mode='learned'):
    """Loads a full pickled gtnet (model.pt) or a state_dict (e.g. ensemble_runs/*/model_state.pt)."""
    device = torch.device(device)
    with open(path, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
    if isinstance(model, dict):
        state = model.get('state_dict', model)
        model = build_paper_gtnet(num_nodes, device, adj_mode)
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            raise SystemExit(f'{path}: state_dict does not match the paper (triple_050) architecture; '
                             f'use the full model.pt saved by train_test.py instead.\n{str(e)[:300]}')
    if hasattr(model, 'gc'):
        model.gc.device = device
    model.idx = model.idx.to(device)
    if isinstance(model.predefined_A, torch.Tensor):
        model.predefined_A = model.predefined_A.to(device)
    model = model.to(device)
    model.train()  # MC dropout, as in train_test/forecast
    return model


def _forward_level(model, window):
    """window: [T, N] tensor (scaled). Returns de-RevIN'd one-step prediction [N]."""
    X = window.T.unsqueeze(0).unsqueeze(0).float()           # [1,1,N,T]
    wm = X.mean(dim=-1, keepdim=True)
    ws = X.std(dim=-1, keepdim=True)
    ws[ws == 0] = 1
    out = model((X - wm) / ws)                               # [1, 1, N, 1]
    return out[-1, :, :, -1][0] * ws[0, 0, :, 0] + wm[0, 0, :, 0]


def _anchor(pred, last, cols, anchor):
    if not anchor:
        return pred
    pred = pred.clone()
    for c, a in cols:
        pred[c] = (1.0 - a) * pred[c] + a * last[c]
    return pred


def anchor_cols(cols, names=TARGETS):
    return [(cols.index(n), min(max(ANCHOR_ALPHA[n], 0.0), 0.98)) for n in names if n in cols]


@torch.no_grad()
def one_step_eval(model, values, cols, target_rows, scale, seed=1, n_runs=10, anchor=True, device='cpu'):
    """Teacher-forced one-step evaluation identical to train_test.evaluate_sliding_window.

    values: [T, N] smoothed levels (any rows needed for inputs must be present)
    target_rows: list of row indices to predict (inputs are the 24 rows before each)
    Returns dict with pred, actual, mc_var (level units), resid_var, eq10 half-width.
    """
    dev = torch.device(device)
    dat = _scaled(values, scale, dev)
    acols = anchor_cols(cols)
    first = target_rows[0]
    assert list(target_rows) == list(range(first, first + len(target_rows)))
    preds, mvars = [], []
    # CPU dropout fills its mask in memory order, so the window is built exactly as train_test does
    # (strided clone first, then row-major torch.cat) to reproduce its MC samples bit-for-bit.
    win = dat[first - SEQ_IN:first].clone()
    for r in target_rows:
        i = SEQ_IN + (r - first)                              # index inside train_test's test_window
        set_seed(seed + 9999 + i)
        outs = torch.stack([_forward_level(model, win) for _ in range(n_runs)])
        y = _anchor(outs.mean(0), win[-1], acols, anchor)
        preds.append(y)
        mvars.append(outs.var(0))
        win = torch.cat([win[1:].clone(), dat[r:r + 1].clone()], dim=0)
    s = scale.to(dev)
    pred = torch.stack(preds) * s
    mc_var = torch.stack(mvars) * s * s
    actual = torch.from_numpy(values[target_rows]).float().to(dev)
    resid_var = ((pred - actual) ** 2).mean(0, keepdim=True)
    hw10 = 0.6 * 2.58 * torch.sqrt(mc_var + resid_var)
    return {k: v.cpu().numpy() for k, v in
            dict(pred=pred, actual=actual, mc_var=mc_var, resid_var=resid_var, hw10=hw10).items()}


@torch.no_grad()
def one_step_eval_windows(model, windows, actual, cols, scale, seed=1, n_runs=10, anchor=True, device='cpu'):
    """One-step evaluation where every target has its own input window (e.g. real-time vintages).

    windows: list of [24, N] arrays; actual: [len(windows), N]. Every variant passed through this
    function gets identical dropout masks (same seed per step, same memory layout), so differences
    between variants come from the inputs only.
    """
    dev = torch.device(device)
    acols = anchor_cols(cols)
    preds = []
    for k, w in enumerate(windows):
        set_seed(seed + 9999 + SEQ_IN + k)
        win = _scaled(np.ascontiguousarray(w), scale, dev)
        outs = torch.stack([_forward_level(model, win) for _ in range(n_runs)])
        preds.append(_anchor(outs.mean(0), win[-1], acols, anchor))
    return (torch.stack(preds) * scale.to(dev)).cpu().numpy(), np.asarray(actual)


def double_exp_smooth(x, alpha=0.1, beta=0.3):
    """smoothing.py double_exponential_smoothing, vectorised over columns. x: [T, N]."""
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    level, trend = x[0].astype(np.float64), (x[1] - x[0]).astype(np.float64)
    for n in range(1, len(x)):
        last = level
        level = alpha * x[n] + (1 - alpha) * (level + trend)
        trend = beta * (level - last) + (1 - beta) * trend
        out[n] = level + trend
    return out


@torch.no_grad()
def recursive_paths(model, values, cols, origin_row, horizon, scale, n_runs=20, anchor=False,
                    seed=1, device='cpu'):
    """forecast.py-style recursive MC forecast.

    origin_row: last observed row (inputs = rows origin_row-23 .. origin_row)
    Returns paths [n_runs, horizon, N] in level units.
    """
    dev = torch.device(device)
    dat = _scaled(values[:origin_row + 1], scale, dev)
    acols = anchor_cols(cols)
    set_seed(seed)
    paths = []
    for _ in range(n_runs):
        win = dat[-SEQ_IN:].contiguous().clone()   # forecast.py feeds row-major numpy windows
        steps = []
        for _h in range(horizon):
            y = _anchor(_forward_level(model, win), win[-1], acols, anchor)
            steps.append(y)
            win = torch.cat([win[1:], y.unsqueeze(0)], dim=0)
        paths.append(torch.stack(steps))
    s = scale.to(dev)
    return (torch.stack(paths) * s).cpu().numpy()


def rrse(pred, actual):
    den = np.sum((actual - actual.mean()) ** 2)
    return float(np.sqrt(np.sum((actual - pred) ** 2) / den)) if den > 1e-12 else float('nan')


def rae(pred, actual):
    den = np.sum(np.abs(actual - actual.mean()))
    return float(np.sum(np.abs(actual - pred)) / den) if den > 1e-12 else float('nan')


def target_idx(cols):
    return {n: cols.index(n) for n in TARGETS if n in cols}
