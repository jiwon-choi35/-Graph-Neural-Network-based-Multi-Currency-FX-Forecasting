"""2026 forecast with the trained B-MTGNN (manuscript Section 4.2.2, Table 5, Table 6, Figure 4).

Procedure (manuscript Sections 3.4 and 3.4.1):
  1. Recursive 12-month forecast from 2025-12 (inputs: the last 24 months of data/sm_data.csv).
     Each MC-dropout run keeps its own trajectory and every node is fed back. 200 runs, seed 1.
  2. Eq. (7) anchoring at every step for the three target nodes:
        y_hat = (1 - a) * y_model + a * y_last,
        a = anchor_focus_to_last 0.06 x boost (US 5.0, KR 1.8, JP 1.0) = 0.300 / 0.108 / 0.060.
  3. Prediction band: 1.96 * sqrt(var_MC,h + var_res,h)
        var_MC,h  : variance of the 200 MC paths at horizon h
        var_res,h : mean squared h-step error of the same recursive procedure (50 runs) started at
                    the 13 origins 2023-12 .. 2024-12 (inputs scaled with the 2011-2023 training rows).
  4. Table 6 (Eq. 11): forecasts divided by the 2025-12 smoothed value, averaged per quarter / year,
     pairwise differences US-KR, JP-US, JP-KR.

Exact reproduction of the manuscript numbers requires CPU (MC dropout masks differ on GPU).

Usage (from B-MTGNN/):
    python forecast.py
    python forecast.py --fig_base 2025-12      # Figure 4 drawn with 2025-12 = 1

Outputs (model/Bayesian/forecast/):
    data/forecast_2026.csv             monthly forecast, MC sd, residual variance, 95% band (targets)
    data/forecast_2026_quarterly.csv   Table 5: quarterly / annual mean, half-width, width ratio
    data/backtest_residual_var.csv     var_res,h used for the band
    data/predict.csv, data/<var>.txt   forecast of every node
    gap/normalized_gap.csv             Table 6 (relative index gap, base 2025-12)
    gap/normalized_forecast.csv, gap/normalized_quarterly_summary.csv, gap/predict_by_country.csv
    plots/Multi_Country_Forecast_Normalized.png/.pdf   Figure 4
    pt_plots/<target>.png/.pdf         forecast of each target currency
"""
import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager, rc
from matplotlib import pyplot

import net  # noqa: F401  (needed so torch.load can unpickle gtnet)

pyplot.style.use("seaborn-v0_8-dark")
try:
    font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
except Exception:
    pass
pyplot.rcParams['axes.unicode_minus'] = False
pyplot.rcParams['savefig.dpi'] = 300

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'data', 'sm_data.csv')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'model', 'Bayesian', 'model.pt')
OUT_DIR = os.path.join(SCRIPT_DIR, 'model', 'Bayesian', 'forecast')

# plot order and colours (US, KR, JP)
TARGETS = ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx']
COLOURS = {'us_Trade Weighted Dollar Index': 'RoyalBlue', 'kr_fx': 'Crimson', 'jp_fx': 'DarkOrange'}
# Eq. (7) alpha for the 2026 forecast: triple_050 profile, anchor_focus_to_last 0.06 x anchor_boost_map
ANCHOR_ALPHA = {'us_Trade Weighted Dollar Index': 0.06 * 5.0, 'kr_fx': 0.06 * 1.8, 'jp_fx': 0.06 * 1.0}
# Table 6 pairs (A - B)
GAP_PAIRS = [('us_Trade Weighted Dollar Index', 'kr_fx'), ('jp_fx', 'us_Trade Weighted Dollar Index'),
             ('jp_fx', 'kr_fx')]
SHORT = {'us_Trade Weighted Dollar Index': 'US', 'kr_fx': 'KR', 'jp_fx': 'JP'}

SEQ_IN = 24
TRAIN_ROWS = 156                   # 2011-01 .. 2023-12
HORIZON = 12
ROLL_ORIGINS = range(155, 168)     # 2023-12 .. 2024-12 (13 origins)


# ==========================================
# Helpers
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def consistent_name(name):
    name = name.replace('_', ' ')
    return {'us Trade Weighted Dollar Index': 'US Dollar Index', 'kr fx': 'KRW/USD', 'jp fx': 'JPY/USD'}.get(name, name)


def to_month(label):
    """sm_data.csv dates are written like '11.Jan' (yy.Mon); also accepts '2011-01'."""
    for fmt in ('%y.%b', '%Y-%m', '%Y-%m-%d'):
        try:
            return pd.Period(pd.to_datetime(label, format=fmt), 'M')
        except (ValueError, TypeError):
            pass
    return pd.Period(label, 'M')


def load_frame(path=DATA_FILE):
    df = pd.read_csv(path)
    dates = [str(to_month(d)) for d in df.iloc[:, 0].astype(str)]
    vals = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
    return dates, list(df.columns[1:]), vals


def scale_from(values, rows):
    """Per-column max |x| over the first `rows` rows, in float32 like DataLoaderS(normalize=2)."""
    s = torch.from_numpy(values[:rows]).float().abs().max(dim=0).values
    s[s == 0] = 1.0
    return s


def load_model(path, device):
    with open(path, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
    if hasattr(model, 'gc'):
        model.gc.device = device
    model.idx = model.idx.to(device)
    if isinstance(getattr(model, 'predefined_A', None), torch.Tensor):
        model.predefined_A = model.predefined_A.to(device)
    model = model.to(device)
    model.train()  # MC dropout
    return model


def forward_level(model, window):
    """window: [T, N] scaled. RevIN as in train_test.py; returns the one-step prediction [N]."""
    X = window.T.unsqueeze(0).unsqueeze(0).float()           # [1, 1, N, T]
    wm = X.mean(dim=-1, keepdim=True)
    ws = X.std(dim=-1, keepdim=True)
    ws[ws == 0] = 1
    out = model((X - wm) / ws)                               # [1, 1, N, 1]
    return out[-1, :, :, -1][0] * ws[0, 0, :, 0] + wm[0, 0, :, 0]


def anchor(pred, last, acols):
    """Eq. (7)."""
    if not acols:
        return pred
    pred = pred.clone()
    for c, a in acols:
        pred[c] = (1.0 - a) * pred[c] + a * last[c]
    return pred


@torch.no_grad()
def recursive_paths(model, values, origin_row, scale, acols, n_runs, seed, device):
    """MC-dropout recursive forecast. Inputs are rows origin_row-23 .. origin_row.
    Returns paths [n_runs, HORIZON, N] in level units."""
    dat = (torch.from_numpy(values[:origin_row + 1]).float() / scale).to(device)
    set_seed(seed)
    paths = []
    for _ in range(n_runs):
        win = dat[-SEQ_IN:].contiguous().clone()
        steps = []
        for _h in range(HORIZON):
            y = anchor(forward_level(model, win), win[-1], acols)
            steps.append(y)
            win = torch.cat([win[1:], y.unsqueeze(0)], dim=0)
        paths.append(torch.stack(steps))
    return (torch.stack(paths) * scale.to(device)).cpu().numpy()


# ==========================================
# Outputs
# ==========================================

def save_node_files(V, mu, sd, band, cols, data_dir, month_fc):
    """predict.csv (every node) and one txt per node (history, forecast, 95% half-width, MC variance)."""
    df = pd.DataFrame(mu.T, columns=month_fc)
    df.insert(0, 'Node_Name', cols)
    df.to_csv(os.path.join(data_dir, 'predict.csv'), index=False)
    for i, name in enumerate(cols):
        with open(os.path.join(data_dir, name.replace('/', '_') + '.txt'), 'w') as ff:
            ff.write('Data: ' + str(V[:, i].tolist()) + '\n')
            ff.write('Forecast: ' + str(mu[:, i].tolist()) + '\n')
            ff.write('95% Confidence: ' + str(band[:, i].tolist()) + '\n')
            ff.write('Variance: ' + str((sd[:, i] ** 2).tolist()) + '\n')


def quarter_means(x):
    return [float(np.mean(x[3 * q:3 * q + 3])) for q in range(4)] + [float(np.mean(x))]


def save_gap_files(V, mu, cols, gap_dir, month_fc):
    """Table 6 (Eq. 11): relative index = forecast / smoothed value of 2025-12."""
    periods = ['2026-Q1', '2026-Q2', '2026-Q3', '2026-Q4', '2026-Annual']
    idx = {n: cols.index(n) for n in TARGETS}
    base = {n: float(V[-1, i]) for n, i in idx.items()}
    rel = {n: mu[:, i] / base[n] for n, i in idx.items()}

    with open(os.path.join(gap_dir, 'predict_by_country.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Country'] + month_fc)
        for n, i in idx.items():
            w.writerow([consistent_name(n)] + mu[:, i].tolist())
    with open(os.path.join(gap_dir, 'normalized_forecast.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Country', 'Base_Value_(2025-12)'] + month_fc)
        for n in TARGETS:
            w.writerow([consistent_name(n), base[n]] + rel[n].tolist())
    with open(os.path.join(gap_dir, 'normalized_quarterly_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Country', 'Base_Value_(2025-12)'] + periods)
        for n in TARGETS:
            w.writerow([consistent_name(n), base[n]] + quarter_means(rel[n]))
    rows = []
    with open(os.path.join(gap_dir, 'normalized_gap.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Pair'] + periods)
        for a, b in GAP_PAIRS:
            g = np.array(quarter_means(rel[a])) - np.array(quarter_means(rel[b]))
            w.writerow([f'{consistent_name(a)} − {consistent_name(b)}'] + g.tolist())
            rows.append((f'{SHORT[a]}-{SHORT[b]}', g[-1]))
    return rows


def plot_multi(V, mu, hw, cols, base_row, path_stem):
    """Figure 4: history and forecast divided by the value at base_row; band fans out from 2025-12."""
    n_hist = len(V)
    hist_x = np.arange(n_hist)
    bx = np.arange(n_hist - 1, n_hist + HORIZON)
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
    for k, n in enumerate(TARGETS):
        i = cols.index(n)
        base = V[base_row, i]
        lw = 2 if k == 0 else 1
        bm = np.r_[V[-1, i], mu[:, i]] / base
        bh = np.r_[0.0, hw[:, i]] / base
        ax.plot(hist_x, V[:, i] / base, '-', color=COLOURS[n], label=consistent_name(n), linewidth=lw, zorder=3)
        ax.plot(bx, bm, '-', color=COLOURS[n], linewidth=lw, zorder=3)
        ax.fill_between(bx, bm - bh, bm + bh, color=COLOURS[n], alpha=0.2, zorder=2)
    if base_row == n_hist - 1:
        ax.axhline(1.0, color='grey', linewidth=0.6, linestyle='--', zorder=1)
    years = [str(y) for y in range(2011, 2028)]
    ax.set_xticks([12 * i for i in range(len(years))], years)
    pyplot.yticks(fontsize=7)
    ax.legend(loc='upper left', prop={'size': 7})
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=7)
    fig.set_size_inches(7.487, 3.93)
    fig.savefig(path_stem + '.png', bbox_inches='tight')
    fig.savefig(path_stem + '.pdf', bbox_inches='tight', format='pdf')
    pyplot.close(fig)


def plot_single(V, mu, hw, name, i, path_stem):
    n_hist = len(V)
    bx = np.arange(n_hist - 1, n_hist + HORIZON)
    bm = np.r_[V[-1, i], mu[:, i]]
    bh = np.r_[0.0, hw[:, i]]
    lw = 2 if name.startswith('us_') else 1
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
    ax.plot(np.arange(n_hist), V[:, i], '-', color=COLOURS[name], label=consistent_name(name), linewidth=lw,
            zorder=3)
    ax.plot(bx, bm, '-', color=COLOURS[name], linewidth=lw, zorder=3)
    ax.fill_between(bx, bm - bh, bm + bh, color=COLOURS[name], alpha=0.2, label='95% PI', zorder=2)
    years = [str(y) for y in range(2011, 2028)]
    ax.set_xticks([12 * k for k in range(len(years))], years)
    ax.set_ylabel('Trend', fontsize=15)
    pyplot.yticks(fontsize=7)
    ax.legend(loc='upper left', prop={'size': 7})
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=7)
    pyplot.title(consistent_name(name), y=1.03, fontsize=18)
    fig.set_size_inches(7.487, 3.93)
    fig.savefig(path_stem + '.png', bbox_inches='tight')
    fig.savefig(path_stem + '.pdf', bbox_inches='tight', format='pdf')
    pyplot.close(fig)


# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL_FILE)
    p.add_argument('--n_runs', type=int, default=200, help='MC paths for the 2026 forecast')
    p.add_argument('--roll_runs', type=int, default=50, help='MC paths per origin for the residual variance')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--no_eq7', action='store_true', help='forecast without the Eq. (7) anchoring')
    p.add_argument('--fig_base', default='2011-01', choices=['2011-01', '2025-12'],
                   help='month set to 1 in Figure 4 (manuscript: 2011-01)')
    p.add_argument('--device', default='cpu', help='cpu reproduces the manuscript; cuda gives different MC draws')
    args = p.parse_args()

    device = torch.device(args.device)
    dirs = {k: os.path.join(OUT_DIR, k) for k in ['data', 'gap', 'plots', 'pt_plots']}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    dates, cols, V = load_frame()
    n_rows = len(V)
    origin = n_rows - 1                                          # 2025-12
    month_fc = [str(pd.Period(dates[-1], 'M') + h) for h in range(1, HORIZON + 1)]
    tix = {n: cols.index(n) for n in TARGETS}
    acols = [] if args.no_eq7 else [(cols.index(n), min(max(ANCHOR_ALPHA[n], 0.0), 0.98)) for n in TARGETS]
    print(f'Data: {n_rows} months ({dates[0]} ~ {dates[-1]}) x {len(cols)} variables')
    print(f'Eq. (7): {"off" if args.no_eq7 else {SHORT[n]: round(ANCHOR_ALPHA[n], 3) for n in TARGETS}}')

    model = load_model(args.model, device)
    sc_train = scale_from(V, TRAIN_ROWS)
    sc_all = scale_from(V, n_rows)

    # --- residual variance per horizon: recursive forecasts from the 13 origins 2023-12 .. 2024-12
    print(f'Residual variance: {len(ROLL_ORIGINS)} origins x {args.roll_runs} MC paths ...')
    err = np.zeros((len(ROLL_ORIGINS), HORIZON, len(TARGETS)))
    for k, o in enumerate(ROLL_ORIGINS):
        mu_o = recursive_paths(model, V, o, sc_train, acols, args.roll_runs, args.seed, device).mean(0)
        for j, n in enumerate(TARGETS):
            i = tix[n]
            err[k, :, j] = mu_o[:, i] - V[o + 1:o + 1 + HORIZON, i]
    res_var = (err ** 2).mean(0)                                 # [HORIZON, targets]

    # --- 2026 forecast
    print(f'2026 forecast: {args.n_runs} MC paths ...')
    paths = recursive_paths(model, V, origin, sc_all, acols, args.n_runs, args.seed, device)
    mu = paths.mean(0)
    sd = paths.std(0, ddof=1)
    hw = (1.96 * sd).astype(np.float64)                          # MC-only band for non-target nodes
    for j, n in enumerate(TARGETS):
        i = tix[n]
        hw[:, i] = 1.96 * np.sqrt(sd[:, i] ** 2 + res_var[:, j])

    # --- tables
    monthly, quarterly, resid = [], [], []
    for j, n in enumerate(TARGETS):
        i = tix[n]
        for h in range(HORIZON):
            monthly.append(dict(Country=consistent_name(n), Date=month_fc[h], Forecast=mu[h, i], MC_SD=sd[h, i],
                                Residual_Var_h=res_var[h, j], HW_95=hw[h, i], Lower_95=mu[h, i] - hw[h, i],
                                Upper_95=mu[h, i] + hw[h, i], HW_95_MC_only=1.96 * sd[h, i]))
            resid.append(dict(Country=consistent_name(n), h=h + 1, Residual_Var=res_var[h, j]))
        for period, m, w in zip(['Q1', 'Q2', 'Q3', 'Q4', 'Annual'], quarter_means(mu[:, i]), quarter_means(hw[:, i])):
            quarterly.append(dict(Country=consistent_name(n), Period=period, Forecast=m, HW_95=w,
                                  Lower_95=m - w, Upper_95=m + w, Width_pct=200 * w / m,
                                  Base_2025_12=V[-1, i]))
    pd.DataFrame(monthly).to_csv(os.path.join(dirs['data'], 'forecast_2026.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(resid).to_csv(os.path.join(dirs['data'], 'backtest_residual_var.csv'), index=False,
                               encoding='utf-8-sig')
    q = pd.DataFrame(quarterly)
    q.to_csv(os.path.join(dirs['data'], 'forecast_2026_quarterly.csv'), index=False, encoding='utf-8-sig')
    save_node_files(V, mu, sd, hw, cols, dirs['data'], month_fc)
    gaps = save_gap_files(V, mu, cols, dirs['gap'], month_fc)

    # --- figures
    base_row = 0 if args.fig_base == '2011-01' else n_rows - 1
    plot_multi(V, mu, hw, cols, base_row, os.path.join(dirs['plots'], 'Multi_Country_Forecast_Normalized'))
    for n in TARGETS:
        plot_single(V, mu, hw, n, tix[n], os.path.join(dirs['pt_plots'], n.replace(' ', '_')))

    # --- summary
    print('\nTable 5: 2026 forecast, mean ± 95% half-width (width / mean)')
    for n in TARGETS:
        cells = [f"{r.Forecast:,.2f} ± {r.HW_95:,.2f} ({r.Width_pct:.2f}%)"
                 for r in q[q.Country == consistent_name(n)].itertuples()]
        print(f'  {SHORT[n]}: ' + ' | '.join(cells))
    print('Table 6: relative index gap (annual, base 2025-12)')
    for pair, g in gaps:
        print(f'  {pair}: {g:.3f}')
    print(f'\nOutputs written to {OUT_DIR}')
