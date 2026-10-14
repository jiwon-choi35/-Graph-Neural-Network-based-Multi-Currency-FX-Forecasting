"""Experiments 1 & 2 (weaknesses 2, 3).

Exp 1  Recursive 12-month backtest: origin 2024-12 -> forecast 2025-01..12 exactly like forecast.py
       (MC-dropout trajectories, all 33 nodes fed back), with and without the eq.(7) anchoring.
       Also: rolling origins 2023-12..2024-12 (per-horizon error profile), MC-seed sensitivity,
       naive/drift 12-step baselines, and the one-step (Table 3) reference.
Exp 2  2026 forecast (origin 2025-12) for both variants with prediction bands that add the
       backtest residual variance to the MC variance.

Usage: python exp12_recursive.py --model <paper model.pt> [--n_runs 200]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rev_common import (TARGETS, TARGET_LABEL, load_frame, load_model, one_step_eval, recursive_paths,
                        rae, rrse, target_idx, train_scale)

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--n_runs', type=int, default=200, help='MC trajectories for the main estimates')
p.add_argument('--paper_runs', type=int, default=20, help='MC trajectories used by forecast.py')
p.add_argument('--n_seeds', type=int, default=10, help='seeds for the MC-sensitivity check')
p.add_argument('--roll_runs', type=int, default=50)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'recursive_backtest_2026_forecast'))
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
dates, cols, V = load_frame()
tix = target_idx(cols)
model = load_model(args.model)
sc_train = train_scale(V, 156)
sc_all = train_scale(V, 180)            # forecast.py scales by the full sample
H = 12
BT_ORIGIN = 167                          # 2024-12
FC_ORIGIN = 179                          # 2025-12
VARIANTS = {'no_eq7': False, 'eq7': True}
month_bt = [f'2025-{m:02d}' for m in range(1, 13)]
month_fc = [f'2026-{m:02d}' for m in range(1, 13)]


def summarize(paths):
    return paths.mean(0), paths.std(0, ddof=1)


# ------------------------------------------------------------------ Exp 1: single-origin backtest
actual_bt = V[BT_ORIGIN + 1:BT_ORIGIN + 1 + H]
rows, metrics = [], []
bt_mean = {}
for vname, anc in VARIANTS.items():
    paths = recursive_paths(model, V, cols, BT_ORIGIN, H, sc_train, n_runs=args.n_runs, anchor=anc, seed=args.seed)
    mu, sd = summarize(paths)
    bt_mean[vname] = mu
    for n, i in tix.items():
        a, f, s = actual_bt[:, i], mu[:, i], sd[:, i]
        e = f - a
        cov = np.mean(np.abs(e) <= 1.96 * s)
        metrics.append(dict(variant=vname, target=TARGET_LABEL[n], RRSE=rrse(f, a), RAE=rae(f, a),
                            MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)), ME=np.mean(e),
                            err_h1=e[0], err_h3=e[2], err_h6=e[5], err_h12=e[11],
                            pred_change=f[-1] - V[BT_ORIGIN, i], actual_change=a[-1] - V[BT_ORIGIN, i],
                            mc_hw_h1=1.96 * s[0], mc_hw_h12=1.96 * s[-1], mc_width_ratio_h12_h1=s[-1] / s[0],
                            mc95_coverage=cov))
        for h in range(H):
            rows.append(dict(variant=vname, target=TARGET_LABEL[n], month=month_bt[h], h=h + 1, actual=a[h],
                             pred=f[h], mc_sd=s[h], err=e[h]))

# 12-step baselines on the same task
for n, i in tix.items():
    a = actual_bt[:, i]
    last = V[BT_ORIGIN, i]
    slope = (V[BT_ORIGIN, i] - V[BT_ORIGIN - 23, i]) / 23.0
    for bname, f in [('naive_last', np.full(H, last)), ('drift', last + slope * np.arange(1, H + 1))]:
        e = f - a
        metrics.append(dict(variant=bname, target=TARGET_LABEL[n], RRSE=rrse(f, a), RAE=rae(f, a),
                            MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)), ME=np.mean(e),
                            err_h1=e[0], err_h3=e[2], err_h6=e[5], err_h12=e[11],
                            pred_change=f[-1] - last, actual_change=a[-1] - last))

# one-step (Table 3 setting) reference
for vname, anc in VARIANTS.items():
    r = one_step_eval(model, V, cols, list(range(168, 180)), sc_train, seed=args.seed, anchor=anc)
    for n, i in tix.items():
        f, a = r['pred'][:, i], r['actual'][:, i]
        e = f - a
        metrics.append(dict(variant=f'one_step_{vname}', target=TARGET_LABEL[n], RRSE=rrse(f, a), RAE=rae(f, a),
                            MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)), ME=np.mean(e)))

pd.DataFrame(rows).to_csv(out / 'exp1_backtest_monthly.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(metrics).to_csv(out / 'exp1_backtest_metrics.csv', index=False, encoding='utf-8-sig')
print(pd.DataFrame(metrics)[['variant', 'target', 'RRSE', 'RAE', 'MAE', 'err_h12', 'mc_width_ratio_h12_h1',
                             'mc95_coverage']].round(4).to_string())

# ------------------------------------------------------------------ Exp 1b: rolling origins
roll = []
for o in range(155, 168):                # 2023-12 .. 2024-12 (all have 12 future rows inside the data)
    for vname, anc in VARIANTS.items():
        mu, sd = summarize(recursive_paths(model, V, cols, o, H, sc_train, n_runs=args.roll_runs, anchor=anc,
                                           seed=args.seed))
        for n, i in tix.items():
            for h in range(H):
                roll.append(dict(origin=dates[o], variant=vname, target=TARGET_LABEL[n], h=h + 1,
                                 err=mu[h, i] - V[o + 1 + h, i], mc_sd=sd[h, i],
                                 naive_err=V[o, i] - V[o + 1 + h, i]))
roll = pd.DataFrame(roll)
roll.to_csv(out / 'exp1_rolling_errors.csv', index=False, encoding='utf-8-sig')
prof = roll.groupby(['variant', 'target', 'h']).agg(
    rmse=('err', lambda x: np.sqrt(np.mean(x ** 2))), naive_rmse=('naive_err', lambda x: np.sqrt(np.mean(x ** 2))),
    mc_sd=('mc_sd', 'mean'),
    cov95=('err', 'size')).reset_index()
cov = roll.assign(inside=lambda d: np.abs(d.err) <= 1.96 * d.mc_sd).groupby(['variant', 'target', 'h']).inside.mean()
prof['mc95_coverage'] = cov.values
prof = prof.drop(columns='cov95')
prof.to_csv(out / 'exp1_rolling_horizon_profile.csv', index=False, encoding='utf-8-sig')
summ = roll.groupby(['variant', 'target']).apply(
    lambda d: pd.Series(dict(rmse=np.sqrt(np.mean(d.err ** 2)), naive_rmse=np.sqrt(np.mean(d.naive_err ** 2)),
                             mc95_coverage=np.mean(np.abs(d.err) <= 1.96 * d.mc_sd))), include_groups=False)
summ.to_csv(out / 'exp1_rolling_summary.csv', encoding='utf-8-sig')
print(summ.round(4).to_string())

# ------------------------------------------------------------------ MC-seed sensitivity (20-run means)
sens = []
for origin, sc, label in [(BT_ORIGIN, sc_train, 'backtest_2025'), (FC_ORIGIN, sc_all, 'forecast_2026')]:
    for vname, anc in VARIANTS.items():
        for s in range(1, args.n_seeds + 1):
            mu, sd = summarize(recursive_paths(model, V, cols, origin, H, sc, n_runs=args.paper_runs, anchor=anc,
                                               seed=s))
            for n, i in tix.items():
                d = dict(run=label, variant=vname, seed=s, target=TARGET_LABEL[n], dec=mu[-1, i],
                         q4=mu[9:, i].mean(), annual=mu[:, i].mean(), q4_hw=(1.96 * sd[9:, i]).mean())
                if label == 'backtest_2025':
                    d['RRSE'] = rrse(mu[:, i], actual_bt[:, i])
                sens.append(d)
sens = pd.DataFrame(sens)
sens.to_csv(out / 'mc_seed_sensitivity.csv', index=False, encoding='utf-8-sig')
print(sens.groupby(['run', 'variant', 'target'])[['dec', 'q4', 'q4_hw', 'RRSE']].agg(['mean', 'min', 'max'])
      .round(3).to_string())

# ------------------------------------------------------------------ Exp 2: 2026 forecast + new bands
# residual variance from the 2024-12-origin backtest: constant (mean over 12 months, as eq.(10) does)
# and horizon-specific (rolling origins 2023-12..2024-12) as a sensitivity variant
bt = pd.DataFrame(rows)
res_const = bt.groupby(['variant', 'target']).err.apply(lambda x: np.mean(x ** 2))
res_h = roll.groupby(['variant', 'target', 'h']).err.apply(lambda x: np.mean(x ** 2))

fc_rows, fc_q = [], []
for vname, anc in VARIANTS.items():
    paths = recursive_paths(model, V, cols, FC_ORIGIN, H, sc_all, n_runs=args.n_runs, anchor=anc, seed=args.seed)
    mu, sd = summarize(paths)
    for n, i in tix.items():
        lab = TARGET_LABEL[n]
        mc_var = sd[:, i] ** 2
        rc = res_const[(vname, lab)]
        rh = np.array([res_h[(vname, lab, h)] for h in range(1, H + 1)])
        bands = {
            'mc_only_95': 1.96 * np.sqrt(mc_var),
            'mc_res_95': 1.96 * np.sqrt(mc_var + rc),
            'mc_res_eq10': 0.6 * 2.58 * np.sqrt(mc_var + rc),
            'mc_resh_95': 1.96 * np.sqrt(mc_var + rh),
        }
        for h in range(H):
            row = dict(variant=vname, target=lab, month=month_fc[h], forecast=mu[h, i], mc_sd=sd[h, i],
                       res_var_const=rc, res_var_h=rh[h])
            row.update({f'hw_{k}': v[h] for k, v in bands.items()})
            fc_rows.append(row)
        base = V[FC_ORIGIN, i]
        q = dict(variant=vname, target=lab, base_2025_12=base)
        for qi in range(4):
            sl = slice(3 * qi, 3 * qi + 3)
            q[f'Q{qi + 1}'] = mu[sl, i].mean()
            for k, v in bands.items():
                q[f'Q{qi + 1}_hw_{k}'] = v[sl].mean()
        q['Annual'] = mu[:, i].mean()
        for k, v in bands.items():
            q[f'Annual_hw_{k}'] = v.mean()
            q[f'ratio_Q4Q1_{k}'] = v[9:].mean() / v[:3].mean()
            q[f'ratio_DecJan_{k}'] = v[-1] / v[0]
        q['dec'] = mu[-1, i]
        q['annual_change_pct'] = (mu[:, i].mean() / base - 1) * 100
        fc_q.append(q)

pd.DataFrame(fc_rows).to_csv(out / 'exp2_forecast2026_monthly.csv', index=False, encoding='utf-8-sig')
fq = pd.DataFrame(fc_q)
fq.to_csv(out / 'exp2_forecast2026_quarterly.csv', index=False, encoding='utf-8-sig')
show = ['variant', 'target', 'Q1', 'Q2', 'Q3', 'Q4', 'Annual', 'dec', 'Q4_hw_mc_only_95', 'Q4_hw_mc_res_95',
        'ratio_Q4Q1_mc_only_95', 'ratio_Q4Q1_mc_res_95', 'ratio_Q4Q1_mc_resh_95']
print(fq[show].round(3).to_string())

# one-step 2025 (Table 2) band ratios for comparison with the 2026 ratios
r = one_step_eval(model, V, cols, list(range(168, 180)), sc_train, seed=args.seed, anchor=True)
t2 = {TARGET_LABEL[n]: dict(ratio_DecJan_eq10=float(r['hw10'][-1, i] / r['hw10'][0, i]),
                            ratio_Q4Q1_eq10=float(r['hw10'][9:, i].mean() / r['hw10'][:3, i].mean()),
                            mean_hw_eq10=float(r['hw10'][:, i].mean()),
                            coverage_eq10=int(np.sum(np.abs(r['pred'][:, i] - r['actual'][:, i]) <= r['hw10'][:, i])))
      for n, i in tix.items()}
json.dump(t2, open(out / 'table2_onestep_band_ratios.json', 'w'), indent=2)
print(json.dumps(t2, indent=2))
