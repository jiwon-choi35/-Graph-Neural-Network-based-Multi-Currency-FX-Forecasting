"""제거 실험용 재학습 (3개 구성 x 시드).

train_test.py는 AXIS/model/Bayesian(hp.txt, Testing/, Validation/)에 결과를 쓰므로, B-MTGNN 코드와 데이터를
runs/ablation/box/ 로 복사해 그 안에서 학습한다 (저장소의 결과 파일을 덮어쓰지 않음).

  full      33노드, 학습 인접행렬 (원 모형과 같은 설정)
  nomacro   15노드 (거시변수 18개 제외)
  identity  33노드, A = I (--adj_mode identity)

학습 설정은 train_test.py 기본 프로필(triple_050: 180 epoch, 검증 최적 체크포인트 저장)과 같다.
GPU 기준 1회 약 8분, 3개 동시 실행 시 시드 5개 약 1시간.

Usage:
  python train_ablation.py                       # 3개 구성 x 시드 1~5
  python train_ablation.py --seeds 1 --configs full --epochs 2   # 동작 확인용
학습 후: python exp3_ablation_eval.py --model_dir runs/ablation/models --paper <원고 모형>
"""
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BMTGNN = HERE.parent
CODE_FILES = ['train_test.py', 'net.py', 'layer.py', 'util.py', 'trainer.py']

p = argparse.ArgumentParser()
p.add_argument('--configs', nargs='+', default=['full', 'nomacro', 'identity'])
p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
p.add_argument('--parallel', type=int, default=3, help='동시에 학습할 개수 (GPU 메모리에 맞게)')
p.add_argument('--epochs', type=int, default=0, help='0이면 프로필 기본값(180) 사용; 동작 확인용으로만 변경')
p.add_argument('--out', default=str(HERE / 'runs' / 'ablation'))
args = p.parse_args()

out = Path(args.out)
box = out / 'box' / 'B-MTGNN'
(box / 'data').mkdir(parents=True, exist_ok=True)
(box / 'data_nomacro').mkdir(parents=True, exist_ok=True)
(out / 'models').mkdir(parents=True, exist_ok=True)
(out / 'logs').mkdir(parents=True, exist_ok=True)
for f in CODE_FILES:
    shutil.copy2(BMTGNN / f, box / f)
for f in ('sm_data.csv', 'graph.csv'):
    shutil.copy2(BMTGNN / 'data' / f, box / 'data' / f)
shutil.copy2(BMTGNN / 'data' / 'graph.csv', box / 'data_nomacro' / 'graph.csv')
sm = pd.read_csv(box / 'data' / 'sm_data.csv')
sm.iloc[:, :16].to_csv(box / 'data_nomacro' / 'sm_data.csv', index=False)   # Date + 15개 비거시 노드

EXTRA = {
    'full': ['--data', 'data/sm_data.csv'],
    'nomacro': ['--data', 'data_nomacro/sm_data.csv'],
    'identity': ['--data', 'data/sm_data.csv', '--adj_mode', 'identity'],
}


def run(job):
    cfg, seed = job
    model_path = out / 'models' / f'{cfg}_s{seed}.pt'
    log_path = out / 'logs' / f'{cfg}_s{seed}.log'
    if model_path.exists() and log_path.exists() and '1 run average' in log_path.read_text(errors='ignore'):
        return f'{cfg} s{seed}: already done, skipped'
    cmd = [sys.executable, 'train_test.py', *EXTRA[cfg], '--seed', str(seed), '--plot', '0',
           '--generate_final_report', '0', '--save', str(model_path)]
    if args.epochs:
        cmd += ['--epochs', str(args.epochs)]
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    with open(log_path, 'w', encoding='utf-8') as log:
        code = subprocess.run(cmd, cwd=box, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    return f'{cfg} s{seed}: exit {code}'


jobs = [(c, s) for s in args.seeds for c in args.configs]
with ThreadPoolExecutor(max_workers=args.parallel) as ex:
    for msg in ex.map(run, jobs):
        print(msg, flush=True)
print('models:', out / 'models')
