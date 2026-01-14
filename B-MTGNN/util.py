import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import csv
from collections import defaultdict
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

def build_predefined_adj(columns, graph_file='data/graph.csv'):
    # Initialize an empty dictionary
    graph = defaultdict(list)

    # Read the graph CSV file
    try:
        with open(graph_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                key_node = row[0]
                # Extract adjacent nodes (skipping empty strings)
                adjacent_nodes = [node for node in row[1:] if node]
                graph[key_node].extend(adjacent_nodes)
        print('Graph loaded with', len(graph), 'attacks...')
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
        return torch.zeros(n_nodes, n_nodes)
    
    data = np.ones(len(row_indices), dtype=np.float32)
    adj_sp = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes)).astype(np.float32)
    
    adj_dense = adj_sp.todense()
    adj_dense[adj_sp > 1] = 1 # Clip values to 1
    
    adj = torch.from_numpy(adj_dense).float()
    print('Adjacency created...')

    return adj

def normal_std(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


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
            
            if 'Date' in df.columns:
                df = df.drop(columns=['Date'])
            elif df.columns[0].lower() == 'date':
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

        self.rawdat = torch.from_numpy(self.rawdat_np).float()

        self.shift = 0
        self.min_data = torch.min(self.rawdat)
        if(self.min_data < 0):
            self.shift = (self.min_data * -1) + 1
        elif (self.min_data == 0):
            self.shift = 1

        self.dat = torch.zeros_like(self.rawdat)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        
        self.scale = torch.ones(self.m)
        self.scale = self.scale.to(device)

        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        target_col_file = col_file if col_file else file_name
        try:
            self.col = create_columns(target_col_file)
            if len(self.col) != self.m:
                 self.col = [str(i) for i in range(self.m)]
        except:
             self.col = [str(i) for i in range(self.m)]

        self.adj = build_predefined_adj(self.col) 
        # 만약 그래프 파일 경로가 다르다면 build_predefined_adj(self.col, '경로') 로 수정 필요

        # Calculate metrics using Test set
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.test[1].size(1), self.m)
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))


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

    def _split(self, train, valid, test):
        # util.py Logic: Strictly separates Train / Valid / Test ranges
        train_set = range(self.P + self.h - 1, train) 
        valid_set = range(train, valid) 
        test_set = range(valid, self.n)
        
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test =  self._batchify(test_set, self.h)
        
        self.test_window = self.dat[-(36+self.P):, :].clone()

    def _batchify(self, idx_set, horizon):
        n = len(idx_set) 
        X = torch.zeros((n - self.out_len, self.P, self.m)) 
        Y = torch.zeros((n - self.out_len, self.out_len, self.m)) 

        for i in range(n - self.out_len): 
            end = idx_set[i] - self.h + 1 
            start = end - self.P 
            
            # Optimized: Direct tensor slicing
            X[i, :, :] = self.dat[start:end, :]
            Y[i, :, :] = self.dat[idx_set[i]:idx_set[i]+self.out_len, :] 
            
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
