import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import csv
from collections import defaultdict

#by Zaid et al.
# returns column names within dataset    
def create_columns(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Column file not found at: {file_path}")

    # Read the CSV file of the dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Read the first row
        col = [c for c in next(reader)]
            
        if 'Date' in col[0]:
            return col[1:]
            
        return col
    
#by Zaid et al.
def build_predefined_adj(columns, graph_files='data/graph.csv'):
    # Initialize an empty dictionary with default value as an empty list
    graph = defaultdict(list)

    # Read the graph CSV file
    try:
        with open('data/graph.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Iterate over each row in the CSV file
            for row in reader:
                # Extract the key node from the first column
                if not row: continue
                key_node = row[0]
                # Extract the adjacent nodes from the remaining columns
                adjacent_nodes =  [node for node in row[1:] if node] #does not include empty columns
                # Add the adjacent nodes to the graph dictionary
                graph[key_node].extend(adjacent_nodes)
        print('Graph loaded with',len(graph),'attacks...')
    except FileNotFoundError:
        print("Warning: file not found. Retruning zero matrix")
        return torch.zeros((len(columns), len(columns)))

    
    print(len(columns), 'columns loaded')

    n_nodes = len(columns)

    col_to_idx = {col: i for i, col in enumerate(columns)}

    row_indices = []
    col_indices = []

    for node_name, neighbors in graph.items() :
        if node_name in col_to_idx:
            i = col_to_idx[node_name]
            for neighbor_name in neighbors:
                if neighbor_name in col_to_idx:
                    j = col_to_idx[neighbor_name]

                    row_indices.append(i)
                    col_indices.append(j)

                    row_indices.append(j)
                    col_indices.append(i)

    if not row_indices:
        print("No edges found in the graph.")
        return torch.zeros(n_nodes, n_nodes)
    
    data = np.ones(len(row_indices), dtype=np.float32)
    
    adj_sp = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes)).astype(np.float32)
    
    adj_dense = adj_sp.todense()
    adj_dense[adj_sp > 1] = 1
    
    adj = torch.from_numpy(adj_dense).float()
        
    print('Adjacency created...')

    return adj

def normal_std(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, adj, normalize=2, out=1, col_file='data/data.csv'):
        self.P = window
        self.h = horizon
        self.out_len = out

        with open(file_name, 'r', encoding='utf-8') as fin:
            self.rawdat_np = np.loadtxt(fin, delimiter='\t')

        self.rawdat = torch.from_numpy(self.rawdat_np).float()

        self.shift=0
        self.min_data=torch.min(self.rawdat)
        if(self.min_data<0):
            self.shift=(self.min_data*-1)+1
        elif (self.min_data==0):
            self.shift=1

        self.dat = torch.zeros_like(self.rawdat)

        self.n, self.m = self.dat.shape
        self.normalize = 2

        self.scale = torch.ones(self.m)

        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        tmp = self.test[1] * self.scale.expand(self.test[1].size(0),self.test[1].size(1), self.m)

        self.scale = self.scale.to(device)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

        self.adj = adj 

        try:
            cols = create_columns(col_file)

            if len(cols) != self.m:
                print(f"Warning: Column count mismatch! Data has {self.m}, but {col_file} has {len(cols)} columns.")
            DataLoaderS.col = cols
        except Exception as e:
            print(f"Error loading columns from {col_file}: {e}")
            DataLoaderS.col = [str(i) for i in range(self.m)]

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / torch.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            '''
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))
            '''
            max_abs_val = torch.max(torch.abs(self.rawdat), dim=0).values
            self.scale = max_abs_val

            mask = max_abs_val > 0
                
            self.dat = self.rawdat.clone()
                
            self.dat[:, mask] = self.rawdat[:, mask] / max_abs_val[mask]  

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1,self.n) #full data (for final training)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test =  self._batchify(test_set, self.h)
        
        self.test_window = self.dat[-(36+self.P):, :].clone()

    def _batchify(self, idx_set, horizon):
        n = len(idx_set) 
        X = torch.zeros((n-self.out_len, self.P, self.m)) #n samples x P time steps lookback x number of columns.
        Y = torch.zeros((n-self.out_len, self.out_len, self.m)) 

        for i in range(n-self.out_len): 
            end = idx_set[i] - self.h + 1 
            start = end - self.P 
            
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

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def _create_mask(labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()

    mask /= torch.mean(mask)

    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    return mask


def masked_mse(preds, labels, null_val=np.nan):
    mask = _create_mask(labels, null_val)

    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    mask = _create_mask(labels, null_val)
    
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    mask = _create_mask(labels, null_val)
    
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_node_feature(path):
    return z

