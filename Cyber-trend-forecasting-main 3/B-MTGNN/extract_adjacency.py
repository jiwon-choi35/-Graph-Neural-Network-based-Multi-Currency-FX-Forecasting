"""Extract the adaptive (learned) graph-adjacency matrix from a trained B-MTGNN checkpoint.

Direction convention (verified against net.py / layer.py, not assumed):
    A[i, j] is the edge weight for information flowing FROM source node j
    TO target node i  ->  "j -> i".

    Why:
      1) layer.graph_constructor.forward() does
             s1, t1 = (adj + noise).topk(self.k, dim=1)
             mask.scatter_(1, t1, 1)
         topk is taken along dim=1 (columns) FOR EACH ROW. So row i keeps
         its k highest-scoring COLUMNS j -> row i is "the k nodes node i
         listens to".
      2) layer.mixprop.forward(x, adj) row-normalizes:
             d = adj.sum(1)          # per-ROW sum
             a = adj / d.view(-1, 1) # divide each ROW by its own sum
         so rows (not columns) are the ones that sum toward being a
         probability distribution over sources.
      3) layer.nconv.forward(x, A) does
             out = einsum('ncwl,vw->ncvl', x, A)
         i.e. out[..., v, ...] = sum_w  x[..., w, ...] * A[v, w]
         Output index v = A's first/row dimension = TARGET node.
         Contracted index w = A's second/column dimension = SOURCE node.
      4) net.gtnet.forward() calls
             x = gconv1(x, adp) + gconv2(x, adp.transpose(1, 0))
         gconv1 uses A exactly as extracted here (row=target, col=source).
         gconv2 uses A^T, which is MTGNN's standard "second direction"
         (models outflow from a node, using the same weights transposed).

    So: for a given target node, its top-weighted "incoming" neighbors are
    found by reading its ROW (not its column) in the matrix returned here.

Sparsification note:
    graph_constructor.__init__ clamps k = min(subgraph_size, num_nodes).
    This checkpoint was trained with subgraph_size=40 but num_nodes=33
    (see AXIS/model/Bayesian/hp.txt), so k = min(40, 33) = 33 = num_nodes.
    A top-33-out-of-33 selection keeps every column -> the "sparsified"
    matrix (graph_constructor.forward / model.gc(idx)) is numerically
    identical to the raw matrix (graph_constructor.fullA / model.gc.fullA(idx)).
    This script computes BOTH and explicitly checks/reports whether they
    are actually identical for the loaded checkpoint, rather than assuming it.

Normalization note:
    The matrix returned by graph_constructor is NOT row-normalized (rows do
    not sum to 1). Row-normalization (plus a +I self-loop) only happens
    transiently inside layer.mixprop.forward at propagation time and is
    never written back to any stored adjacency. This script reports the
    actual row sums of the extracted matrix so this can be verified rather
    than assumed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_BASE_DIR = PROJECT_DIR / 'AXIS' / 'model' / 'Bayesian'

DEFAULT_CHECKPOINT = MODEL_BASE_DIR / 'model.pt'
DEFAULT_DATA = SCRIPT_DIR / 'data' / 'sm_data.csv'
DEFAULT_OUTPUT_DIR = MODEL_BASE_DIR / 'graph_analysis'

# (display_name, exact_column_name_in_csv) - order matters for reporting.
FOCUS_NODES = [
    ('US Trade Weighted Dollar Index', 'us_Trade Weighted Dollar Index'),
    ('KRW/USD', 'kr_fx'),
    ('JPY/USD', 'jp_fx'),
]

TOPK_NEIGHBORS = 5
ROUND_DECIMALS = 4
SEED = 777


def parse_args():
    p = argparse.ArgumentParser(description='Extract adaptive adjacency matrix from a B-MTGNN checkpoint')
    p.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument('--data', type=str, default=str(DEFAULT_DATA),
                    help='CSV whose header column order defines the node order used at training time')
    p.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument('--top-k', type=int, default=TOPK_NEIGHBORS)
    p.add_argument('--seed', type=int, default=SEED)
    return p.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load the pickled gtnet model and eval() it under no_grad-safe settings."""
    # net.py / layer.py define the classes this pickle references, so they
    # must be importable. SCRIPT_DIR is where this file lives (same dir as
    # net.py), so make sure it's on sys.path before unpickling.
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from net import gtnet  # noqa: F401  (needed for unpickling)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    with open(checkpoint_path, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)

    model = model.to(device)

    # `self.idx` and `graph_constructor.device` are plain Python attributes
    # (not nn.Parameters/buffers), so `model.to(device)` does NOT move them.
    # Force them onto the same device we actually have, otherwise
    # graph_constructor.forward()'s internal `.to(self.device)` calls can
    # try to reach a CUDA device that doesn't exist on this machine.
    if hasattr(model, 'idx'):
        model.idx = model.idx.to(device)
    if hasattr(model, 'gc'):
        model.gc.device = device

    model.eval()
    return model


def load_node_names(data_path: Path):
    """Read the node/column order exactly as util.create_columns() does at training time."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from util import create_columns
    return create_columns(str(data_path))


def verify_node_alignment(node_names, model):
    """Fail loudly if node order/count can't be trusted - a silent mismatch
    would make every downstream number meaningless."""
    print('=' * 70)
    print('NODE ALIGNMENT CHECK')
    print('=' * 70)
    print(f'  model.num_nodes      = {model.num_nodes}')
    print(f'  len(node_names)      = {len(node_names)}  (from CSV header order)')

    if len(node_names) != model.num_nodes:
        raise ValueError(
            f'Node count mismatch: checkpoint expects {model.num_nodes} nodes but '
            f'the data file header has {len(node_names)} columns after dropping Date. '
            f'Wrong --data file, or the checkpoint was trained on a different column set.'
        )

    print('  index -> node name:')
    for i, name in enumerate(node_names):
        print(f'    [{i:2d}] {name}')

    name_to_idx = {}
    dupes = set()
    for i, name in enumerate(node_names):
        if name in name_to_idx:
            dupes.add(name)
        name_to_idx[name] = i
    if dupes:
        raise ValueError(f'Duplicate column names found, index mapping is ambiguous: {dupes}')

    focus_idx = {}
    missing = []
    for display_name, col_name in FOCUS_NODES:
        if col_name not in name_to_idx:
            missing.append(col_name)
        else:
            focus_idx[col_name] = name_to_idx[col_name]
    if missing:
        raise ValueError(
            f'Focus node(s) not found in CSV header: {missing}. '
            f'Available columns: {node_names}'
        )

    print('  focus node index mapping:')
    for display_name, col_name in FOCUS_NODES:
        print(f'    {display_name!r:35s} -> column {col_name!r:35s} -> index {focus_idx[col_name]}')
    print('=' * 70)
    return name_to_idx, focus_idx


def extract_adjacency(model, device, seed):
    """Return (A_topk, A_raw) as numpy [N, N] arrays, both computed under
    model.eval() + torch.no_grad(), exactly the way net.gtnet.forward() would
    obtain the adjacency (buildA_true branch: `adp = self.gc(self.idx)`)."""
    idx = model.idx.to(device)
    torch.manual_seed(seed)  # graph_constructor.forward adds tiny random noise for topk tie-breaking
    with torch.no_grad():
        a_topk = model.gc(idx)          # sparsified: what forward() actually uses when buildA_true=True
        a_raw = model.gc.fullA(idx)     # pre-sparsification raw compatibility scores
    return a_topk.cpu().numpy(), a_raw.cpu().numpy()


def report_sparsification_and_normalization(a_topk, a_raw, model):
    print('=' * 70)
    print('SPARSIFICATION CHECK (top-k vs raw)')
    print('=' * 70)
    k_eff = model.gc.k
    n = model.gc.nnodes
    print(f'  graph_constructor.k (after min(subgraph_size, num_nodes) clamp) = {k_eff}')
    print(f'  num_nodes                                                       = {n}')
    if k_eff >= n:
        print(f'  -> k >= num_nodes: top-k selects EVERY column per row, i.e. NO actual')
        print(f'     sparsification is applied. The "sparsified" matrix should equal the raw one.')
    else:
        print(f'  -> k < num_nodes: real sparsification. Each row keeps its top {k_eff} columns only.')

    identical = np.allclose(a_topk, a_raw, atol=1e-6)
    max_diff = float(np.max(np.abs(a_topk - a_raw)))
    print(f'  max |topk - raw| over all entries = {max_diff:.8f}')
    print(f'  topk and raw matrices are numerically identical: {identical}')

    nonzero_per_row = (np.abs(a_topk) > 1e-12).sum(axis=1)
    print(f'  nonzero entries per row in topk matrix: min={nonzero_per_row.min()}, '
          f'max={nonzero_per_row.max()}, mean={nonzero_per_row.mean():.2f} (out of {n} possible)')
    print('=' * 70)

    print('=' * 70)
    print('ROW-NORMALIZATION CHECK (is the stored weight already a probability, i.e. row sum == 1?)')
    print('=' * 70)
    row_sums = a_topk.sum(axis=1)
    print(f'  row sums (as extracted, BEFORE the +I self-loop / division mixprop does at propagate-time):')
    print(f'    min={row_sums.min():.4f}  max={row_sums.max():.4f}  mean={row_sums.mean():.4f}')
    print(f'  -> NOT normalized to 1. Row-stochastic normalization (d = adj.sum(1); a = adj/d)')
    print(f'     happens dynamically inside layer.mixprop.forward() at propagation time and is')
    print(f'     never written back into any stored adjacency matrix.')
    print('=' * 70)


def report_diagonal(a_topk, a_raw, node_names):
    print('=' * 70)
    print('SELF-LOOP (diagonal) CHECK')
    print('=' * 70)
    diag_topk = np.diag(a_topk)
    diag_raw = np.diag(a_raw)
    max_diag = float(np.max(np.abs(diag_topk)))
    print(f'  max |diagonal| in topk matrix = {max_diag:.8f}')
    if max_diag > 1e-6:
        print('  [FOUND] Non-zero self-loop weight(s) present in the extracted adjacency:')
        for i, v in enumerate(diag_topk):
            if abs(v) > 1e-6:
                print(f'    node[{i}] {node_names[i]!r}: self-loop weight = {v:.6f}')
        print('  These are EXCLUDED from graph_top_neighbors.csv rankings (a node cannot be its')
        print('  own "neighbor"), but their presence is flagged here as requested.')
    else:
        print('  No self-loops found (diagonal is ~0 for every node).')
        print('  This is expected by construction: graph_constructor computes')
        print('    a = nodevec1 @ nodevec2.T - nodevec2 @ nodevec1.T')
        print('  whose diagonal is always exactly 0 (a[i,i] = <v1_i,v2_i> - <v2_i,v1_i> = 0),')
        print('  and relu(tanh(x)) at x=0 is still 0, so no self-loop mass can appear.')
    print('=' * 70)
    return max_diag > 1e-6


def save_top_neighbors_csv(a_topk, node_names, focus_idx, top_k, output_path):
    rows = []
    for display_name, col_name in FOCUS_NODES:
        t = focus_idx[col_name]
        row = a_topk[t, :].copy()
        row[t] = -np.inf  # exclude self from neighbor ranking (self-loop is reported separately)
        order = np.argsort(-row)[:top_k]
        for rank, j in enumerate(order, start=1):
            weight = row[j]
            rows.append({
                'target_node': col_name,
                'rank': rank,
                'source_node': node_names[j],
                'edge_weight': round(float(weight), ROUND_DECIMALS),
                # direction verified in module docstring: column j is source, row i is target
                'direction': f'{node_names[j]} -> {col_name} (source -> target)',
            })
    df = pd.DataFrame(rows, columns=['target_node', 'rank', 'source_node', 'edge_weight', 'direction'])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'[saved] {output_path}')
    return df


def save_3x3_csv(a_topk, a_raw, focus_idx, output_path):
    """Long-format 3x3 submatrix among the 3 focus currencies: every ordered
    pair (row_node, col_node) with A[row,col], A[col,row], their difference
    (asymmetry), and whether it's a diagonal (self) cell."""
    cols = [c for _, c in FOCUS_NODES]
    rows = []
    for row_name in cols:
        i = focus_idx[row_name]
        for col_name in cols:
            j = focus_idx[col_name]
            a_ij_topk = float(a_topk[i, j])
            a_ji_topk = float(a_topk[j, i])
            a_ij_raw = float(a_raw[i, j])
            rows.append({
                'row_node': row_name,
                'col_node': col_name,
                'A_row_to_col_topk': round(a_ij_topk, ROUND_DECIMALS),
                'A_col_to_row_topk': round(a_ji_topk, ROUND_DECIMALS),
                'asymmetry_topk': round(a_ij_topk - a_ji_topk, ROUND_DECIMALS),
                'A_row_to_col_raw': round(a_ij_raw, ROUND_DECIMALS),
                'is_diagonal_self_loop': bool(i == j),
            })
    df = pd.DataFrame(rows, columns=[
        'row_node', 'col_node', 'A_row_to_col_topk', 'A_col_to_row_topk',
        'asymmetry_topk', 'A_row_to_col_raw', 'is_diagonal_self_loop',
    ])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'[saved] {output_path}')
    return df


def save_full_matrices(a_topk, a_raw, node_names, output_dir: Path):
    """Bonus: save the complete N x N matrices (both variants) with labels,
    so the top-5/3x3 extracts above can be independently audited."""
    df_topk = pd.DataFrame(a_topk, index=node_names, columns=node_names).round(ROUND_DECIMALS)
    df_raw = pd.DataFrame(a_raw, index=node_names, columns=node_names).round(ROUND_DECIMALS)
    p_topk = output_dir / 'graph_adjacency_full_topk.csv'
    p_raw = output_dir / 'graph_adjacency_full_raw.csv'
    df_topk.to_csv(p_topk, encoding='utf-8-sig')
    df_raw.to_csv(p_raw, encoding='utf-8-sig')
    print(f'[saved] {p_topk}  (sparsified / actually used by forward())')
    print(f'[saved] {p_raw}  (pre-sparsification raw compatibility scores)')


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')  # analysis-only extraction; CPU is enough and avoids device-mismatch issues

    print(f'[load] checkpoint = {checkpoint_path}')
    model = load_checkpoint(checkpoint_path, device)

    print(f'[load] data (for node/column order) = {data_path}')
    node_names = load_node_names(data_path)

    name_to_idx, focus_idx = verify_node_alignment(node_names, model)

    a_topk, a_raw = extract_adjacency(model, device, args.seed)

    report_sparsification_and_normalization(a_topk, a_raw, model)
    report_diagonal(a_topk, a_raw, node_names)

    save_top_neighbors_csv(a_topk, node_names, focus_idx, args.top_k, output_dir / 'graph_top_neighbors.csv')
    save_3x3_csv(a_topk, a_raw, focus_idx, output_dir / 'graph_3x3.csv')
    save_full_matrices(a_topk, a_raw, node_names, output_dir)

    print('=' * 70)
    print(f'Done. All outputs written to: {output_dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
