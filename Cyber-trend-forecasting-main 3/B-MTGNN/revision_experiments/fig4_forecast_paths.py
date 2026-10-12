"""그림 4: 2026년 다중 통화 예측 경로.

forecast.py의 plot_multi_node와 같은 형식(2011~2027, 색, 크기, 국가 간 음영)으로 그리되, 값은 표 5와 같게 맞춘다.
  - 과거 : sm_data.csv 평활 계열 (추가 평활 없음; forecast.py는 그리기 전에 α=0.3 지수평활을 한 번 더 했음)
  - 전망 : exp12_recursive.py의 식 (7) 적용 2026년 전망 (MC 경로 200개 평균)
  - 구간 : 1.96·sqrt(σ²_MC,h + σ²_res,h)  (hw_mc_resh_95, 표 5의 반폭과 같음)
  - 기준 : --base 2011-01 (기본, 원고 그림 4) 또는 2025-12 (식 (11)·표 6과 같은 기준)
  - 음영 : --gap_shading 1 이면 예측 기간의 미국–일본(파랑), 미국–한국(빨강) 곡선 사이를 옅게 칠함
           (원고 그림 4에는 음영이 없으므로 기본값 0)

Usage:
  python fig4_forecast_paths.py                                  # 원고 그림 4 (2011년 1월 = 1, 음영 없음)
  python fig4_forecast_paths.py --base 2025-12 --gap_shading 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rev_common import load_frame

HERE = Path(__file__).resolve().parent
SERIES = [  # (sm_data column, exp12 target label, legend label, colour, line width) — forecast.py order
    ('us_Trade Weighted Dollar Index', 'USD Index', 'US Dollar Index', 'RoyalBlue', 2),
    ('kr_fx', 'KRW/USD', 'KRW/USD', 'Crimson', 1),
    ('jp_fx', 'JPY/USD', 'JPY/USD', 'DarkOrange', 1),
]
BASE_ROW = {'2011-01': 0, '2025-12': -1}

p = argparse.ArgumentParser()
p.add_argument('--forecast', default=str(HERE / 'reference_results' / 'exp12' / 'exp2_forecast2026_monthly.csv'))
p.add_argument('--variant', default='eq7')
p.add_argument('--band', default='hw_mc_resh_95')
p.add_argument('--base', default='2011-01', choices=sorted(BASE_ROW), help='month whose smoothed value is set to 1')
p.add_argument('--gap_shading', type=int, default=0, choices=[0, 1], help='1: shade US-JP / US-KR gaps (forecast.py style)')
p.add_argument('--out', default=str(HERE / 'results' / 'fig4'))
args = p.parse_args()

dates, cols, V = load_frame()
fc = pd.read_csv(args.forecast)
fc = fc[fc.variant == args.variant]
n_hist = len(V)                                   # 180 months, 2011-01 .. 2025-12
hist_x = np.arange(n_hist)
fc_x = np.arange(n_hist, n_hist + 12)             # 2026-01 .. 2026-12

plt.style.use('seaborn-v0_8-dark')
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
norm = {}
for col, target, label, colour, lw in SERIES:
    hist = V[:, cols.index(col)]
    base = hist[BASE_ROW[args.base]]
    f = fc[fc.target == target].sort_values('month')
    mean = f.forecast.to_numpy() / base
    hw = f[args.band].to_numpy() / base
    norm[target] = mean
    ax.plot(hist_x, hist / base, '-', color=colour, label=label, linewidth=lw, zorder=3)
    ax.plot(np.r_[hist_x[-1], fc_x], np.r_[hist[-1] / base, mean], '-', color=colour, linewidth=lw, zorder=3)
    # band starts from the last observed month with zero width, so it fans out from the 2025-12 point
    bx, bm, bh = np.r_[hist_x[-1], fc_x], np.r_[hist[-1] / base, mean], np.r_[0.0, hw]
    ax.fill_between(bx, bm - bh, bm + bh, color=colour, alpha=0.2, zorder=2)

if args.gap_shading:
    ax.fill_between(fc_x, norm['USD Index'], norm['JPY/USD'], interpolate=True, color='RoyalBlue', alpha=0.08,
                    zorder=1)
    ax.fill_between(fc_x, norm['USD Index'], norm['KRW/USD'], interpolate=True, color='Crimson', alpha=0.08,
                    zorder=1)
if args.base == '2025-12':
    ax.axhline(1.0, color='grey', linewidth=0.6, linestyle='--', zorder=1)

years = [str(y) for y in range(2011, 2028)]
ax.set_xticks([12 * i for i in range(len(years))], years)
plt.yticks(fontsize=7)
ax.legend(loc='upper left', prop={'size': 7})
ax.axis('tight')
ax.grid(True)
plt.xticks(rotation=90, fontsize=7)
fig.set_size_inches(7.487, 3.93)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
stem = f'Multi_Country_Forecast_base{args.base.replace("-", "")}' + ('_shade' if args.gap_shading else '')
fig.savefig(out / f'{stem}.png', dpi=300, bbox_inches='tight')
fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', format='pdf')
print('saved:', out / f'{stem}.png')
for t, m in norm.items():
    print(f'{t}: 2026-12 = {m[-1]:.4f} (x {args.base})')
