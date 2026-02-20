import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import csv
from collections import defaultdict
from pathlib import Path
import pandas as pd 

def create_columns(file_path):
    if not os.path.exists(file_path):
        return []

    # Read the CSV file of the dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            # Read the first row
            col = [c for c in next(reader)]
            if col and col[0].strip() == 'Date':
                return col[1:]
            return col
        except StopIteration:
            return []

def build_predefined_adj(columns, graph_file='data/graph.csv', exclude_nodes=None):
    # 회의 결정: graph에서 중국·영국 제외 (노드 이름에 포함되면 엣지에 사용 안 함)
    if exclude_nodes is None:
        exclude_nodes = ('china', 'uk', '영국', '중국', 'cn_', 'uk_')  # 소문자 비교용
    def _excluded(name):
        if not name:
            return True
        n = str(name).strip().lower()
        return any(n.startswith(e.lower()) or e.lower() in n for e in exclude_nodes if e)

    graph = defaultdict(list)
    try:
        with open(graph_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                key_node = row[0]
                if _excluded(key_node):
                    continue
                adjacent_nodes = [node for node in row[1:] if node and not _excluded(node)]
                graph[key_node].extend(adjacent_nodes)
        print('Graph loaded with', len(graph), 'attacks (China/UK excluded per meeting).')
    except FileNotFoundError:
        print(f"Warning: Graph file not found at {graph_file}. Returning zero matrix.")
        return torch.zeros((len(columns), len(columns)))

    print(len(columns), 'columns loaded')
    n_nodes = len(columns)
    
    if n_nodes == 0:
        return torch.zeros(0, 0)

    col_to_idx = {col: i for i, col in enumerate(columns)}

    row_indices = []
    col_indices = []

    for node_name, neighbors in graph.items():
        if node_name in col_to_idx:
            i = col_to_idx[node_name]
            for neighbor_name in neighbors:
                if neighbor_name in col_to_idx:
                    j = col_to_idx[neighbor_name]
                    # Undirected graph (Symmetric)
                    row_indices.append(i); col_indices.append(j)
                    row_indices.append(j); col_indices.append(i)

    if not row_indices:
        print("No edges found in the graph.")
        # 데이터 CSV 헤더와 graph.csv 노드명이 완전히 일치해야 함 (공백·대소문자 포함)
        in_data_not_graph = [c for c in columns if c not in graph and not _excluded(c)]
        in_graph_not_data = [k for k in graph.keys() if k not in col_to_idx]
        if in_data_not_graph:
            print("  Hint: columns in data but not in graph (first 5):", in_data_not_graph[:5])
        if in_graph_not_data:
            print("  Hint: nodes in graph but not in data (first 5):", in_graph_not_data[:5])
        return torch.zeros(n_nodes, n_nodes)
    
    data = np.ones(len(row_indices), dtype=np.float32)
    adj_sp = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes)).astype(np.float32)
    
    adj_dense = np.array(adj_sp.todense())
    np.clip(adj_dense, 0, 1, out=adj_dense)  # Clip values to 1
    
    adj = torch.from_numpy(adj_dense).float()
    print('Adjacency created...')

    return adj

def normal_std(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    n = len(x) if x.ndim >= 1 else 0
    if n == 0:
        return 1.0  # 테스트 샘플 0개 시 division by zero 방지
    return x.std() * np.sqrt((n - 1.) / n)


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2, out=1, col_file=None):
        self.P = window
        self.h = horizon
        self.out_len = out
        self.device = device

        try:
            print(f"Loading data from {file_name}...")
            df = pd.read_csv(file_name)
            
            # 날짜 기준 분할용: Date 컬럼 보존 후 파싱
            date_series = None
            if 'Date' in df.columns:
                date_series = df['Date'].copy()
                df = df.drop(columns=['Date'])
            elif df.columns[0].lower() == 'date':
                date_series = df.iloc[:, 0].copy()
                df = df.iloc[:, 1:]

            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(0)
            self.rawdat_np = df.values.astype(float)
            print("Data loaded and converted to numeric successfully.")

        except Exception as e:
            print(f"Pandas load failed: {e}. Trying np.loadtxt fallback...")
            try:
                self.rawdat_np = np.loadtxt(file_name, delimiter=',', skiprows=1)
            except Exception as e2:
                raise ValueError(f"Failed to load data: {e2}")
            date_series = None

        # 플롯 시작 날짜용: 행 인덱스 → (year, month) 조회
        self.date_series = date_series  # pandas Series or None, 길이 = n
        self.rawdat = torch.from_numpy(self.rawdat_np).float()

        self.shift = 0
        self.min_data = torch.min(self.rawdat)
        if(self.min_data < 0):
            self.shift = (self.min_data * -1) + 1
        elif (self.min_data == 0):
            self.shift = 1

        self.dat = torch.zeros_like(self.rawdat)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        
        self.scale = torch.ones(self.m)
        self.shift = torch.zeros(self.m)

        self._normalized(normalize)
        # 회의 결정: valid=24년 1~12월, test=25년 1~12월 → 날짜 기준 분할 사용 시
        train_end, valid_end = self._resolve_split_indices(train, valid, date_series)
        self._split(train_end, valid_end, self.n)

        target_col_file = col_file if col_file else file_name
        try:
            self.col = create_columns(target_col_file)
            if len(self.col) != self.m:
                 self.col = [str(i) for i in range(self.m)]
        except:
             self.col = [str(i) for i in range(self.m)]

        # Target-weighted loss: FX 변수에 극대화된 가중치 적용 (loss 압도)
        # [수정] kr_fx 가중치를 낮춰 과적합 방지 및 다른 변수와의 균형 회복
        # 기존: kr=50, jp=30, us=30 → kr 압도적으로 높아 학습 불안정
        # 변경: kr=20, jp=20, us=20 → 균형 잡힌 가중치로 전반적 예측력 향상
        self.target_weight = torch.ones(self.m, device=device) * 0.1  # 나머지 변수는 0.1
        for v in range(self.m):
            if self.col[v] == 'kr_fx':
                self.target_weight[v] = 20.0
            elif self.col[v] == 'jp_fx':
                self.target_weight[v] = 20.0
            elif self.col[v] == 'us_Trade Weighted Dollar Index':
                self.target_weight[v] = 20.0
        fx_total = 20.0 + 20.0 + 20.0
        others_total = (self.m - 3) * 0.1
        fx_ratio = fx_total / (fx_total + others_total) * 100
        print(f"[Target Weight] kr_fx=20.0, jp_fx=20.0, us=20.0, others=0.1 (FX 비중: {fx_ratio:.1f}%)")

        # 그래프 경로: 데이터 파일과 같은 폴더의 graph.csv, 없으면 (데이터파일 부모)/data/graph.csv
        data_path = Path(file_name).resolve()
        same_dir_graph = data_path.parent / 'graph.csv'
        data_subdir_graph = data_path.parent / 'data' / 'graph.csv'
        if same_dir_graph.exists():
            graph_file = str(same_dir_graph)
        elif data_subdir_graph.exists():
            graph_file = str(data_subdir_graph)
        else:
            graph_file = 'data/graph.csv'
        self.adj = build_predefined_adj(self.col, graph_file=graph_file)

        # Calculate metrics using Test set (CPU에서 수행)
        n_test = self.test[1].size(0)
        if n_test == 0:
            # test 배치 0개 (예: 날짜 기준 test=12개월 + seq_out_len=12 → _batchify에서 0개)
            self.rse = 1.0
            self.rae = 1.0
        else:
            scale_exp = self.scale.expand(n_test, self.test[1].size(1), self.m)
            shift_exp = self.shift.expand(n_test, self.test[1].size(1), self.m)
            tmp = self.test[1] * scale_exp + shift_exp
            self.rse = normal_std(tmp)
            self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        # 초기화 완료 후 scale, shift를 GPU로 이동
        self.scale = self.scale.to(device)
        self.shift = self.shift.to(device)


    def _resolve_split_indices(self, train_ratio, valid_ratio, date_series):
        """회의 결정: valid=24년 1~12월, test=25년 1~12월. Date 있으면 날짜 기준, 없으면 비율.
        train_ratio=1.0, valid_ratio=0.0 이면 전체를 train으로 사용 (원본 train.py 전체 데이터 학습용)."""
        if train_ratio >= 1.0 or (train_ratio + valid_ratio) >= 1.0:
            print("Full-data mode: using 100% of data for training (no valid/test split).")
            return self.n, self.n
        if date_series is None or len(date_series) != self.n:
            return int(train_ratio * self.n), int((train_ratio + valid_ratio) * self.n)
        try:
            # "2011-01", "2024-12" 형태 파싱
            parts = date_series.astype(str).str.strip().str.split('-', expand=True)
            if parts.shape[1] < 2:
                return int(train_ratio * self.n), int((train_ratio + valid_ratio) * self.n)
            years = parts[0].astype(int).to_numpy()
            months = parts[1].astype(int).to_numpy()
            mask_2024_01 = (years == 2024) & (months == 1)
            mask_2025_01 = (years == 2025) & (months == 1)
            if mask_2024_01.any() and mask_2025_01.any():
                valid_start = int(np.where(mask_2024_01)[0][0])
                test_start = int(np.where(mask_2025_01)[0][0])
                if valid_start < test_start:
                    print(f"Date-based split: valid 2024-01~12 (rows {valid_start}~{test_start-1}), test 2025-01~12 (rows {test_start}~{self.n-1})")
                    return valid_start, test_start
        except Exception as e:
            print(f"Date-based split failed ({e}), using ratio.")
        return int(train_ratio * self.n), int((train_ratio + valid_ratio) * self.n)

    def get_date_at_index(self, idx):
        """행 인덱스 idx에 해당하는 날짜를 (year, month)로 반환. Date 없으면 None."""
        if not hasattr(self, 'date_series') or self.date_series is None or idx < 0 or idx >= len(self.date_series):
            return None
        try:
            s = str(self.date_series.iloc[idx]).strip()
            parts = s.split('-')
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return None

    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / torch.max(self.rawdat)

        # normalized by the maximum value of each row(sensor).
        if (normalize == 2):
            # Optimized: Vectorized operation using torch
            max_abs_val = torch.max(torch.abs(self.rawdat), dim=0).values
            self.scale = max_abs_val

            mask = max_abs_val > 0
            self.dat = self.rawdat.clone()
            # Avoid division by zero
            self.dat[:, mask] = self.rawdat[:, mask] / max_abs_val[mask]

        # z-score normalization: (x - mean) / std (global, 모든 시점이 같은 기준)
        if (normalize == 3):
            col_mean = self.rawdat.mean(dim=0)
            col_std = self.rawdat.std(dim=0)
            col_std[col_std == 0] = 1  # avoid division by zero
            self.scale = col_std
            self.shift = col_mean
            self.dat = (self.rawdat - col_mean) / col_std
            
            # 역정규화용: per-step rolling 통계 저장 (그래프 출력 정확도 유지)
            window_size = 24
            self.rolling_mean = torch.zeros(self.n, self.m)
            self.rolling_std = torch.ones(self.n, self.m)
            
            for t in range(self.n):
                start_idx = max(0, t - window_size + 1)
                window_data = self.rawdat[start_idx:t+1, :]
                
                if window_data.size(0) < 2:
                    self.rolling_mean[t, :] = col_mean
                    self.rolling_std[t, :] = col_std
                else:
                    rolling_mean_t = window_data.mean(dim=0)
                    rolling_std_t = window_data.std(dim=0)
                    rolling_std_t[rolling_std_t == 0] = 1
                    self.rolling_mean[t, :] = rolling_mean_t
                    self.rolling_std[t, :] = rolling_std_t
            
            print(f"[Global Z-score] 정규화: 전체 기간 통계 사용, 역정규화: per-step rolling 통계 사용 (window={window_size})")

        # rolling z-score normalization: 각 시점에서 직전 24개월(또는 가능한 만큼)의 평균/표준편차 사용
        if (normalize == 4):
            window_size = 24
            self.dat = torch.zeros_like(self.rawdat)
            
            col_mean_global = self.rawdat.mean(dim=0)
            col_std_global = self.rawdat.std(dim=0)
            col_std_global[col_std_global == 0] = 1
            
            # per-step rolling 통계 저장 (정확한 역정규화용)
            self.rolling_mean = torch.zeros(self.n, self.m)
            self.rolling_std = torch.ones(self.n, self.m)
            
            for t in range(self.n):
                start_idx = max(0, t - window_size + 1)
                window_data = self.rawdat[start_idx:t+1, :]
                
                if window_data.size(0) < 2:
                    col_mean = col_mean_global
                    col_std = col_std_global
                else:
                    col_mean = window_data.mean(dim=0)
                    col_std = window_data.std(dim=0)
                    col_std[col_std == 0] = 1
                
                self.rolling_mean[t, :] = col_mean
                self.rolling_std[t, :] = col_std
                self.dat[t, :] = (self.rawdat[t, :] - col_mean) / col_std
            
            self.scale = col_std_global
            self.shift = col_mean_global
            print(f"[Rolling Z-score] window={window_size}, per-step rolling_mean/std 저장 완료")

    def _split(self, train, valid, test):
        # util.py Logic: Strictly separates Train / Valid / Test ranges
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set  = range(valid, self.n)

        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test  = self._batchify(test_set,  self.h)

        # test_window: 직전 P개(실제 과거) + test 구간 전체
        num_test_points  = len(test_set)
        test_window_len  = num_test_points + self.P
        self.test_window = self.dat[-test_window_len:, :].clone()

        # [추가] valid_window: 직전 P개(train 끝) + valid 구간 전체
        # → evaluate_sliding_window에서 validation에도 오토리그레시브 방식 적용 가능
        num_valid_points  = len(valid_set)
        valid_window_len  = num_valid_points + self.P
        # train 끝(=valid 시작) 직전 P개부터 valid 끝까지
        valid_window_start = max(train - self.P, 0)
        self.valid_window  = self.dat[valid_window_start:valid, :].clone()

        self.valid_start_idx = train
        self.test_start_idx  = valid

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        if n == 0:
            return torch.zeros((0, self.P, self.m)), torch.zeros((0, self.out_len, self.m))

        num_samples = max(n - self.out_len + 1, 0)
        if num_samples == 0:
            return torch.zeros((0, self.P, self.m)), torch.zeros((0, self.out_len, self.m))

        valid_samples = []
        for i in range(num_samples):
            y_end = idx_set[i] + self.out_len
            if y_end > self.n:
                break
            end = idx_set[i] - self.h + 1
            start = end - self.P
            if start < 0:
                continue
            valid_samples.append(i)

        if len(valid_samples) == 0:
            return torch.zeros((0, self.P, self.m)), torch.zeros((0, self.out_len, self.m))

        X = torch.zeros((len(valid_samples), self.P, self.m))
        Y = torch.zeros((len(valid_samples), self.out_len, self.m))

        for j, i in enumerate(valid_samples):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[j, :, :] = self.dat[start:end, :]
            Y[j, :, :] = self.dat[idx_set[i]:idx_set[i]+self.out_len, :]

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield X, Y 
            start_idx += batch_size