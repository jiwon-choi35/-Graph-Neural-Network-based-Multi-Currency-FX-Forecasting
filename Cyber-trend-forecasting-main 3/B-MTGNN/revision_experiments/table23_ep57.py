"""원고 표 2·3의 B-MTGNN 값을 epoch 57 모형(model.pt) 하나로 다시 계산한다 (CPU).

train_test.py의 triple_050 프로필은 eval_last_epoch=1이라 원고 표 2·3은 저장되지 않은 180 에폭 모형으로
평가되었다. 원고 전체를 저장된 model.pt(검증 기준 최고 에폭 57)로 통일하기 위해 다음을 산출한다.
  - 표 2: 2025년 RRSE, 식(10) 구간 상·하한의 12개월 평균, 피복 개월 수 (MC 시드 1)
  - 표 3: B-MTGNN 행(식(7) 적용)과 식(7) 미적용 행의 RRSE·RAE (MC 시드 1)
  - 2024년 검증 RRSE (모형 선택 기준: 세 통화 RRSE 최댓값)
  - MC 시드 1~30 민감도 (one-step, 두 변형)

Usage:
  python table23_ep57.py                      # reference/paper_ep57.pt, 결과는 results/table23_ep57/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rev_common import TARGET_LABEL, load_frame, load_model, one_step_eval, rae, rrse, target_idx, train_scale

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--model', default=str(HERE / 'reference' / 'paper_ep57.pt'))
p.add_argument('--seed', type=int, default=1, help='표 2·3에 쓰는 MC 시드 (train_test.py와 같게 학습 시드 1)')
p.add_argument('--n_seeds', type=int, default=30)
p.add_argument('--out', default=str(HERE / 'results' / 'table23_ep57'))
args = p.parse_args()

dates, cols, V = load_frame()
model = load_model(args.model)
sc = train_scale(V, 156)
tix = target_idx(cols)
SPLITS = {'test_2025': list(range(168, 180)), 'valid_2024': list(range(156, 168))}
VARIANTS = {'eq7': True, 'no_eq7': False}


def summarize(r, split, variant, seed):
    rows, monthly = [], []
    trows = SPLITS[split]
    for n, i in tix.items():
        pr, ac, hw = r['pred'][:, i], r['actual'][:, i], r['hw10'][:, i]
        rows.append(dict(split=split, variant=variant, seed=seed, target=TARGET_LABEL[n],
                         RRSE=rrse(pr, ac), RAE=rae(pr, ac),
                         lower_eq10_mean=float(np.mean(pr - hw)), upper_eq10_mean=float(np.mean(pr + hw)),
                         coverage_eq10=int(np.sum(np.abs(pr - ac) <= hw)), n_months=len(trows)))
        for k, row in enumerate(trows):
            monthly.append(dict(split=split, variant=variant, seed=seed, target=TARGET_LABEL[n], month=dates[row],
                                actual=ac[k], predict=pr[k], lower_eq10=pr[k] - hw[k], upper_eq10=pr[k] + hw[k],
                                in_eq10=bool(abs(pr[k] - ac[k]) <= hw[k])))
    return rows, monthly


main_rows, main_monthly, sens_rows = [], [], []
for split, trows in SPLITS.items():
    for vname, anc in VARIANTS.items():
        for s in range(1, args.n_seeds + 1):
            r = one_step_eval(model, V, cols, trows, sc, seed=s, anchor=anc)
            rows, monthly = summarize(r, split, vname, s)
            sens_rows += rows
            if s == args.seed:
                main_rows += rows
                main_monthly += monthly

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
main = pd.DataFrame(main_rows)
sens = pd.DataFrame(sens_rows)
main.to_csv(out / 'table23_seed1_summary.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(main_monthly).to_csv(out / 'table23_seed1_monthly.csv', index=False, encoding='utf-8-sig')
sens.to_csv(out / 'mc_seed_onestep_per_seed.csv', index=False, encoding='utf-8-sig')
agg = sens.groupby(['split', 'variant', 'target']).agg(
    RRSE_mean=('RRSE', 'mean'), RRSE_sd=('RRSE', 'std'), RRSE_min=('RRSE', 'min'), RRSE_max=('RRSE', 'max'),
    RAE_mean=('RAE', 'mean'), RAE_sd=('RAE', 'std'), coverage_mean=('coverage_eq10', 'mean'),
    coverage_min=('coverage_eq10', 'min'), coverage_max=('coverage_eq10', 'max')).reset_index()
agg.to_csv(out / 'mc_seed_onestep_summary.csv', index=False, encoding='utf-8-sig')

with pd.option_context('display.width', 220):
    print(f'== MC seed {args.seed} (표 2·3) ==')
    print(main.drop(columns=['seed', 'n_months']).round(4).to_string(index=False))
    v = main[(main.split == 'valid_2024') & (main.variant == 'eq7')]
    print(f'valid 2024 max RRSE (eq7): {v.RRSE.max():.4f}')
    print(f'\n== MC seeds 1..{args.n_seeds} ==')
    print(agg.round(4).to_string(index=False))
print('saved:', out)
