"""Experiment 3 (weakness 4): ablations, evaluated exactly like Table 3.

Models are retrained with train_test.py (triple_050 profile, 180 epochs, best-validation checkpoint,
seeds 1..5) in three configurations:
  full      33 nodes, learned adjacency            (same as the paper model, retrained)
  nomacro   15 nodes (FX, returns, spreads, correlations, lags), learned adjacency
  identity  33 nodes, fixed A = I (graph propagation off; --adj_mode identity)
Each checkpoint is evaluated on 2024 (validation) and 2025 (test) with the one-step, eq.(7)-anchored
protocol, and on the 12-month recursive backtest from 2024-12 (eq.(7) on and off).

Usage: python exp3_ablation_eval.py --model_dir <dir with {cfg}_s{seed}.pt> [--paper <paper model.pt>]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rev_common import (TARGET_LABEL, load_frame, load_model, one_step_eval, recursive_paths, rae, rrse,
                        target_idx, train_scale)

p = argparse.ArgumentParser()
p.add_argument('--model_dir', required=True)
p.add_argument('--paper', default='')
p.add_argument('--rec_runs', type=int, default=50)
p.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'ablation'))
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

dates, cols_all, V_all = load_frame()
DATA = {'full': (cols_all, V_all), 'identity': (cols_all, V_all),
        'nomacro': (cols_all[:15], np.ascontiguousarray(V_all[:, :15])), 'paper': (cols_all, V_all)}

jobs = []
for f in sorted(Path(args.model_dir).glob('*_s*.pt')):
    m = re.match(r'(\w+)_s(\d+)\.pt', f.name)
    jobs.append((m.group(1), int(m.group(2)), f))
if args.paper:
    jobs.append(('paper', 1, Path(args.paper)))

rows = []
for cfg, seed, f in jobs:
    cols, V = DATA[cfg]
    sc = train_scale(V, 156)
    tix = target_idx(cols)
    model = load_model(f)
    n_params = sum(p.numel() for p in model.parameters())
    for split, trows in (('valid_2024', range(156, 168)), ('test_2025', range(168, 180))):
        r = one_step_eval(model, V, cols, list(trows), sc, seed=seed, anchor=True)
        for n, i in tix.items():
            pr, ac = r['pred'][:, i], r['actual'][:, i]
            cov = int(np.sum(np.abs(pr - ac) <= r['hw10'][:, i]))
            rows.append(dict(config=cfg, seed=seed, split=split, eval='one_step_eq7', target=TARGET_LABEL[n],
                             RRSE=rrse(pr, ac), RAE=rae(pr, ac), coverage_eq10=cov, n_params=n_params))
    actual = V[168:180]
    for vname, anc in (('recursive_no_eq7', False), ('recursive_eq7', True)):
        mu = recursive_paths(model, V, cols, 167, 12, sc, n_runs=args.rec_runs, anchor=anc, seed=seed).mean(0)
        for n, i in tix.items():
            rows.append(dict(config=cfg, seed=seed, split='test_2025', eval=vname, target=TARGET_LABEL[n],
                             RRSE=rrse(mu[:, i], actual[:, i]), RAE=rae(mu[:, i], actual[:, i]),
                             err_h12=mu[-1, i] - actual[-1, i], n_params=n_params))
    print(cfg, seed, 'done', flush=True)

res = pd.DataFrame(rows)
res.to_csv(out / 'exp3_per_seed.csv', index=False, encoding='utf-8-sig')

agg = res[res.config != 'paper'].groupby(['split', 'eval', 'target', 'config']).agg(
    RRSE_mean=('RRSE', 'mean'), RRSE_sd=('RRSE', 'std'), RAE_mean=('RAE', 'mean'), RAE_sd=('RAE', 'std'),
    n=('RRSE', 'size')).reset_index()

# paired differences vs. the full model of the same seed
piv = res[res.config != 'paper'].pivot_table(index=['split', 'eval', 'target', 'seed'], columns='config',
                                             values='RRSE')
diffs = []
for cfg in ('nomacro', 'identity'):
    if cfg not in piv:
        continue
    d = (piv[cfg] - piv['full']).dropna().groupby(level=[0, 1, 2])
    diffs.append(pd.DataFrame({'config': cfg, 'dRRSE_mean': d.mean(), 'dRRSE_sd': d.std(),
                               'n_worse': d.apply(lambda x: int((x > 0).sum())), 'n': d.size()}).reset_index())
pd.concat(diffs).to_csv(out / 'exp3_paired_vs_full.csv', index=False, encoding='utf-8-sig')
agg.to_csv(out / 'exp3_summary.csv', index=False, encoding='utf-8-sig')

with pd.option_context('display.width', 200):
    print(agg.round(4).to_string())
    print(pd.concat(diffs).round(4).to_string())
    print(res[res.config == 'paper'].round(4).to_string())
