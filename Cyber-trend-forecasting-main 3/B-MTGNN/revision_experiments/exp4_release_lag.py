"""Experiment 4 (weakness 5): re-evaluate 2025 with macro inputs restricted to what was published.

Forecast for month t is made at the end of month t-1. A macro value for reference month k is usable
only if it had been released by then:
  monthly indicators : released `lag` months after the reference month (table MONTHLY_LAG)
  quarterly series   : data.csv repeats one value over a quarter's three months (3-country GDP and
                       KR/JP employment, unemployment, trade balance). That value is released `d` months
                       after the quarter end (QUARTERLY_RELEASE), so all three rows become known at once
  FX-derived nodes, policy rates: no lag

Two ways of feeding the delayed information (both from data.csv, then the same double exponential
smoothing (alpha=0.1, beta=0.3) used to build sm_data.csv):
  shift  : every row holds the latest value published by the end of that month (the whole history is
           delayed, as a model run in real time would see it)
  ragged : per forecast origin, rows already published keep their true value; the unpublished tail
           is forward-filled from the last published value (standard ragged-edge treatment)
Plus `lag0` = the same pipeline without delays (control; equals the paper setting).

Usage: python exp4_release_lag.py --models paper=<path> [full_s1=<path> ...]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rev_common import (RAW_PATH, SEQ_IN, TARGET_LABEL, double_exp_smooth, load_frame, load_model,
                        one_step_eval_windows, rae, rrse, target_idx, train_scale)

MONTHLY_LAG = {
    'US_CPI': 1, 'KR_CPI': 1, 'JP_CPI': 1,
    'US_Employment': 1, 'US_Unemployment': 1,
    'US_Trade_Balance': 2,
    'US_Interest_Rate': 0, 'KR_Interest_Rate': 0, 'JP_Interest_Rate': 0,
}
# months after quarter end until the quarter's value is out (GDP: advance / 1st prelim.; KR/JP labour and
# trade: last month of the quarter is published one month later)
QUARTERLY_RELEASE = {
    'US_GDP': 1, 'KR_GDP': 1, 'JP_GDP': 2,
    'KR_Employment': 1, 'JP_Employment': 1,
    'KR_Unemployment': 1, 'JP_Unemployment': 1,
    'KR_Trade_Balance': 1, 'JP_Trade_Balance': 1,
}
TEST_ROWS = list(range(168, 180))                        # 2025-01 .. 2025-12


def known_mask(col, n_rows, s):
    """Boolean [n_rows]: is the value for reference row k published by the end of row s?"""
    k = np.arange(n_rows)
    if col in QUARTERLY_RELEASE:
        q_end = k - (k % 3) + 2                          # rows start at January, so quarters align
        return (q_end + QUARTERLY_RELEASE[col] <= s) & (k <= s)
    return k <= s - MONTHLY_LAG[col]


def latest_published(raw_col, col, s):
    known = np.flatnonzero(known_mask(col, len(raw_col), s))
    return raw_col[known[-1]] if known.size else raw_col[0]   # sample start: back-fill (2011 rows only)


def build_shift(raw, cols, macro):
    x = raw.copy()
    for c in macro:
        j = cols.index(c)
        x[:, j] = [latest_published(raw[:, j], c, s) for s in range(len(raw))]
    return x


def build_ragged(raw, cols, macro, s):
    x = raw[:s + 1].copy()
    for c in macro:
        j = cols.index(c)
        m = known_mask(c, s + 1, s)
        last = np.flatnonzero(m)[-1]
        x[last + 1:, j] = raw[last, j]
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True, help='name=path pairs')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'release_lag'))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dates, cols, SM = load_frame()
    _, rcols, RAW = load_frame(RAW_PATH)
    assert rcols == cols
    macro = [c for c in cols if c in MONTHLY_LAG or c in QUARTERLY_RELEASE]
    mj = [cols.index(c) for c in macro]
    # The release rule must match how data.csv stores each series: quarterly rules only for series that
    # repeat one value within every quarter (checked from 2015 on), monthly rules for the rest.
    for c in macro:
        q = RAW[48:180, cols.index(c)].reshape(-1, 3)
        quarter_constant = bool(np.all((q[:, 0] == q[:, 1]) & (q[:, 1] == q[:, 2])))
        assert quarter_constant == (c in QUARTERLY_RELEASE), f'{c}: release rule does not match data.csv'
    tix = target_idx(cols)
    scale = train_scale(SM, 156)

    # Non-macro columns always come from sm_data.csv, so only the macro inputs differ between variants.
    def with_macro(smoothed_macro_rows, r0, r1):
        w = SM[r0:r1].copy()
        w[:, mj] = smoothed_macro_rows[:, mj]
        return w

    sm_lag0 = double_exp_smooth(RAW)
    sm_shift = double_exp_smooth(build_shift(RAW, cols, macro))
    variants = {
        'paper_sm_data': [SM[r - SEQ_IN:r] for r in TEST_ROWS],
        'lag0_resmoothed': [with_macro(sm_lag0[r - SEQ_IN:r], r - SEQ_IN, r) for r in TEST_ROWS],
        'shift': [with_macro(sm_shift[r - SEQ_IN:r], r - SEQ_IN, r) for r in TEST_ROWS],
        'ragged': [with_macro(double_exp_smooth(build_ragged(RAW, cols, macro, r - 1))[r - SEQ_IN:r], r - SEQ_IN, r)
                   for r in TEST_ROWS],
    }
    actual = SM[TEST_ROWS]

    # How different are the delayed inputs? (last input row of each window, relative to sm_data)
    gaps = []
    for v in ('shift', 'ragged'):
        for c, j in zip(macro, mj):
            d = np.array([(w[-1, j] - SM[r - 1, j]) / (np.abs(SM[r - 1, j]) + 1e-12)
                          for w, r in zip(variants[v], TEST_ROWS)])
            gaps.append(dict(variant=v, node=c, mean_abs_rel_diff_last_row=np.mean(np.abs(d))))
    pd.DataFrame(gaps).to_csv(out / 'input_differences.csv', index=False, encoding='utf-8-sig')

    rows, monthly = [], []
    for spec in args.models:
        name, path = spec.split('=', 1)
        model = load_model(path)
        for v, wins in variants.items():
            pred, act = one_step_eval_windows(model, wins, actual, cols, scale, seed=args.seed)
            for n, i in tix.items():
                rows.append(dict(model=name, variant=v, target=TARGET_LABEL[n], RRSE=rrse(pred[:, i], act[:, i]),
                                 RAE=rae(pred[:, i], act[:, i]), MAE=np.mean(np.abs(pred[:, i] - act[:, i]))))
                for k, r in enumerate(TEST_ROWS):
                    monthly.append(dict(model=name, variant=v, target=TARGET_LABEL[n], month=dates[r],
                                        actual=act[k, i], pred=pred[k, i]))
        print(name, 'done', flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'exp4_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(monthly).to_csv(out / 'exp4_monthly.csv', index=False, encoding='utf-8-sig')
    print(res.pivot_table(index=['model', 'target'], columns='variant', values='RRSE').round(4).to_string())


if __name__ == '__main__':
    main()
