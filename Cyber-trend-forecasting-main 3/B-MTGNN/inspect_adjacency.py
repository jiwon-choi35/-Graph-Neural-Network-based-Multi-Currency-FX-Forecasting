import sys
from pathlib import Path
import torch, numpy as np, pandas as pd

# ── 경로 자동 설정 (스크립트 위치 기준) ──────────────────────────
BASE = Path(__file__).resolve().parent          # .../B-MTGNN
sys.path.insert(0, str(BASE))                   # net.py 임포트용

MODEL_PATH = BASE.parent / 'AXIS' / 'model' / 'Bayesian' / 'model.pt'
DATA_PATH  = BASE / 'data' / 'sm_data.csv'

# 폴더 구조가 다르면 상위 3단계까지 올라가며 탐색
def find(pattern, start, up=3):
    for i in range(up + 1):
        hits = sorted((start.parents[i] if i else start).rglob(pattern))
        if hits:
            return hits[0]
    return None

if not MODEL_PATH.exists():
    MODEL_PATH = find('model.pt', BASE) or MODEL_PATH
if not DATA_PATH.exists():
    DATA_PATH = find('sm_data.csv', BASE) or DATA_PATH

for label, p in (('model', MODEL_PATH), ('data', DATA_PATH)):
    if not Path(p).exists():
        sys.exit(f'[오류] {label} 파일을 찾지 못했습니다: {p}')
print(f'model: {MODEL_PATH}\ndata : {DATA_PATH}\n')

TARGETS = ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx']
K = 40

# ── 인접행렬 추출 ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()
if hasattr(model, 'idx'):
    model.idx = model.idx.to(device)

cols = [c for c in pd.read_csv(DATA_PATH, nrows=0).columns if c.strip().lower() != 'date']

with torch.no_grad():
    idx = model.idx if hasattr(model, 'idx') else torch.arange(len(cols), device=device)
    A = model.gc(idx).detach().cpu().numpy()

print(f'adjacency shape: {A.shape} | columns: {len(cols)}')
if A.shape[0] != len(cols):
    sys.exit('[오류] 노드 수와 컬럼 수가 다릅니다.')

missing = [t for t in TARGETS if t not in cols]
if missing:
    print(f'[경고] 컬럼에 없는 타깃: {missing}')
    print('사용 가능한 컬럼 일부:', cols[:10])

# ── ③ 전체 희소도 ──────────────────────────────────────────────
print(f'\n③ 전체 비영 비율: {(A > 0).mean():.4f}  '
      f'(행당 평균 비영 개수 {(A > 0).sum(1).mean():.1f})')

# ── ①② 노드별 통계 + 상위 이웃 ────────────────────────────────
for name in TARGETS:
    if name not in cols:
        continue
    i = cols.index(name)
    row = A[i]
    top = np.sort(row)[::-1][:K]
    top = top[top > 0]

    print(f'\n── {name} (행 합 {row.sum():.4f})')
    for r, j in enumerate(np.argsort(row)[::-1][:8], 1):
        if row[j] <= 0:
            break
        print(f'  {r}. {cols[j]:<35s} {row[j]:.4f}')

    print(f'  ① 행 전체 합 {row.sum():.4f} | 상위 {K}개 합 {top.sum():.4f}')
    print(f'  ② 상위 {K}개  최대 {top.max():.4f} | 중앙값 {np.median(top):.4f} | 최소 {top.min():.4f}')
    for other in TARGETS:
        if other == name or other not in cols:
            continue
        j = cols.index(other)
        v = row[j]
        rank = int((row > v).sum()) + 1
        pct = (top < v).mean() * 100 if len(top) else 0
        print(f'     ←{other}: {v:.4f}  (행 내 {rank}위, 상위 {K}개 중 하위에서 {pct:.0f}% 지점)')

# ── 결과 저장 (논문 표 작성용) ────────────────────────────────
np.save(BASE / 'learned_adj.npy', A)
pd.DataFrame(A, index=cols, columns=cols).to_csv(BASE / 'learned_adj.csv', encoding='utf-8-sig')
print(f'\n저장: {BASE / "learned_adj.npy"}\n      {BASE / "learned_adj.csv"}')
