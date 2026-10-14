"""체크포인트 하나로 원고 표 2·3 값을 다시 계산한다 (train_test.py와 같은 one-step + 식(7) 평가).

원고 값(0.3507 / 0.2881 / 0.2591)을 낸 모형을 찾기 위한 스크립트.
model.pt(모형 전체)와 model_state.pt(state_dict) 모두 읽는다. 정확한 재현은 CPU에서만 된다.

Usage:
  python check_model.py --model reference/paper_ep57.pt          # 환경 확인: 0.3181 / 0.2903 / 0.2557
  python check_model.py --model <체크포인트> --seed <학습 시드> --name <이름>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rev_common import TARGET_LABEL, load_frame, load_model, one_step_eval, rae, rrse, target_idx, train_scale

MANUSCRIPT_TEST = {'USD Index': 0.3507, 'JPY/USD': 0.2881, 'KRW/USD': 0.2591}
EP57_TEST = {'USD Index': 0.3181, 'JPY/USD': 0.2903, 'KRW/USD': 0.2557}

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--seed', type=int, default=1, help='MC 드롭아웃 시드 = 학습 때의 --seed (원고 모형은 1)')
p.add_argument('--name', default='')
p.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'check_model'))
args = p.parse_args()

name = args.name or Path(args.model).stem
dates, cols, V = load_frame()
model = load_model(args.model)
sc = train_scale(V, 156)
tix = target_idx(cols)

rows, monthly = [], []
for split, trows in (('test_2025', range(168, 180)), ('valid_2024', range(156, 168))):
    r = one_step_eval(model, V, cols, list(trows), sc, seed=args.seed, anchor=True)
    for n, i in tix.items():
        pr, ac, hw = r['pred'][:, i], r['actual'][:, i], r['hw10'][:, i]
        lab = TARGET_LABEL[n]
        rows.append(dict(model=name, split=split, target=lab, RRSE=rrse(pr, ac), RAE=rae(pr, ac),
                         lower_eq10_mean=float(np.mean(pr - hw)), upper_eq10_mean=float(np.mean(pr + hw)),
                         coverage_eq10=int(np.sum(np.abs(pr - ac) <= hw)),
                         manuscript_RRSE=MANUSCRIPT_TEST[lab] if split == 'test_2025' else np.nan))
        for k, row in enumerate(trows):
            monthly.append(dict(model=name, split=split, target=lab, month=dates[row], actual=ac[k], predict=pr[k],
                                lower_eq10=pr[k] - hw[k], upper_eq10=pr[k] + hw[k], in_eq10=abs(pr[k] - ac[k]) <= hw[k]))

res = pd.DataFrame(rows)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
res.to_csv(out / f'{name}_summary.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(monthly).to_csv(out / f'{name}_monthly.csv', index=False, encoding='utf-8-sig')

with pd.option_context('display.width', 200):
    print(res.drop(columns='model').round(4).to_string(index=False))
t = res[res.split == 'test_2025'].set_index('target').RRSE
for ref_name, ref in (('manuscript Table 3', MANUSCRIPT_TEST), ('epoch 57', EP57_TEST)):
    gap = max(abs(t[k] - v) for k, v in ref.items())
    print(f'max |RRSE diff| vs {ref_name}: {gap:.4f}' + ('  <- MATCH' if gap < 5e-4 else ''))
print('saved:', out)
