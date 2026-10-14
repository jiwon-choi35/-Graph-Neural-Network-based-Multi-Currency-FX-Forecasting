"""그림 2: B-MTGNN 평가 구간(2025년) 성능 검증 (좌: 미국, 중: 한국, 우: 일본).

train_test.py의 plot_predicted_actual과 같은 형식으로 그리되, 값은 표 2와 같게 맞춘다.
  - 자료 : reference_results/table23_ep57/table23_seed1_monthly.csv
           (epoch 57 모형, CPU, MC 시드 1, one-step + 식 (7) 보정, split=test_2025)
  - 실선 : 평활된 실제값, 점선 : 예측값
  - 음영 : --band 95 (기본) : 예측값 ± 1.96·sqrt(σ²_MC + σ²_res)  → 범례 '95% Prediction Interval'
           --band eq10       : 식 (10) 구간 예측값 ± 0.6·2.58·sqrt(σ²_MC + σ²_res) → 범례 '식 (10) 예측구간'
           두 구간은 같은 sqrt(σ²_MC + σ²_res)를 쓰므로 95% 반폭 = 식 (10) 반폭 × 1.96/(0.6·2.58).
           (train_test.py export_interval_bounds의 lower_95/upper_95와 같은 구간; ep57 직접 재계산과 1e-4 이내 일치)

Usage: python fig2_eval_2025.py [--band 95|eq10] [--monthly <csv>] [--out results/fig2]
"""
import argparse
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
PANELS = [  # (target in csv, title as in the manuscript figure, file name)
    ('USD Index', 'Us_Trade Weighted Dollar Index', 'Us_Trade Weighted Dollar Index_Testing'),
    ('KRW/USD', 'Kr_fx', 'Kr_fx_Testing'),
    ('JPY/USD', 'Jp_fx', 'Jp_fx_Testing'),
]
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

p = argparse.ArgumentParser()
p.add_argument('--monthly', default=str(HERE / 'reference_results' / 'table23_ep57' / 'table23_seed1_monthly.csv'))
p.add_argument('--band', default='95', choices=['95', 'eq10'])
p.add_argument('--band_label', default='', help='legend text; default depends on --band')
p.add_argument('--out', default=str(HERE / 'results' / 'fig2'))
args = p.parse_args()

df = pd.read_csv(args.monthly)
df = df[(df.split == 'test_2025') & (df.variant == 'eq7')]
BAND_SCALE = {'95': 1.96 / (0.6 * 2.58), 'eq10': 1.0}
label = args.band_label or {'95': '95% Prediction Interval', 'eq10': '식 (10) 예측구간'}[args.band]
# keep the original legend font unless the label needs Hangul glyphs
legend_font = (font_manager.FontProperties(family='Malgun Gothic', size=11) if any(ord(ch) > 127 for ch in label)
               else {'size': 11})
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

plt.rcParams['savefig.dpi'] = 300
for target, title, stem in PANELS:
    d = df[df.target == target]
    assert len(d) == 12, f'{target}: expected 12 months, got {len(d)}'
    x = range(1, 13)
    plt.figure()
    plt.plot(x, d.actual.to_numpy(), 'b-', label='Actual')
    plt.plot(x, d.predict.to_numpy(), '--', color='purple', label='Predicted')
    pred = d.predict.to_numpy()
    hw = (d.upper_eq10.to_numpy() - pred) * BAND_SCALE[args.band]
    plt.fill_between(x, pred - hw, pred + hw, alpha=0.3, color='pink', label=label)
    plt.legend(loc='best', prop=legend_font)
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel('Trend', fontsize=15)
    plt.xlabel('Month', fontsize=15)
    plt.xticks(ticks=list(x), labels=[f'{m}-25' for m in MONTHS], rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(out / f'{stem}.png', bbox_inches='tight')
    plt.close()
    inside = int((abs(pred - d.actual.to_numpy()) <= hw).sum())
    print(f'{target}: saved {stem}.png | band {args.band} | coverage {inside}/12 | mean bounds '
          f'{(pred - hw).mean():.2f} ~ {(pred + hw).mean():.2f}')
print('saved:', out)
