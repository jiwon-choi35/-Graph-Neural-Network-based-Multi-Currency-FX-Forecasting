import ast
import argparse
import math
import time
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
AXIS_DIR = PROJECT_DIR / 'AXIS'
MODEL_BASE_DIR = PROJECT_DIR / 'model' / 'Bayesian'

from net import gtnet
from util import *
from trainer import Optim
from random import randrange
from matplotlib import pyplot as plt

# 원본 README: 전체 데이터로 최종 모델 학습, hp.txt의 최적 HP 사용. 출력 o_model.pt (운용 모델).
import numpy as np
import importlib
import random


plt.rcParams['savefig.dpi'] = 1200

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    # ===== Target-Weighted Loss =====
    # 목표: Testing kr_fx 0.4~0.5, Validation kr_fx 0.7~1.0
    # Validation 일반화 개선을 위해 kr_fx 가중치를 약간 낮춤 (40 → 38)
    target_weight = torch.ones(data.m, device=device) 
    for i, col_name in enumerate(data.col):
        if col_name == 'us_Trade Weighted Dollar Index':
            target_weight[i] = 20.0
        elif col_name == 'kr_fx':
            target_weight[i] = 38.0  # 40 → 38 (Validation 일반화 개선, Testing도 유지)
        elif col_name == 'jp_fx':
            target_weight[i] = 25.0
    print(f"[Target-Weighted Loss] weights applied: "
          f"{ {data.col[i]: target_weight[i].item() for i in range(data.m) if target_weight[i] > 1} }")
    # ==================================

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)  # [B, 1, N, T]

        # ===== RevIN: Per-Window Normalization =====
        w_mean = X.mean(dim=-1, keepdim=True)  # [B, 1, N, 1]
        w_std = X.std(dim=-1, keepdim=True)    # [B, 1, N, 1]
        w_std[w_std == 0] = 1
        X = (X - w_mean) / w_std

        # Normalize target Y using same window stats
        wm = w_mean[:, 0, :, 0]  # [B, N]
        ws = w_std[:, 0, :, 0]   # [B, N]
        Y = (Y - wm.unsqueeze(1)) / ws.unsqueeze(1)  # [B, T_out, N]
        # ============================================

        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.tensor(id).to(device)
            tx = X[:, :, :, :]
            ty = Y[:, :, :]
            
            # ===== A: Scheduled Sampling (15% 확률로 자기회귀 학습, Validation 일반화 개선) =====
            use_autoregressive = random.random() < 0.15  # 0.1 → 0.15 (Validation 개선)
            if use_autoregressive and iter > 0:
                # 이전 예측값을 다음 입력에 일부 사용 (자기회귀 학습)
                with torch.no_grad():
                    prev_output = model(tx)
                    prev_pred = torch.squeeze(prev_output, 3)  # [B, T_out, N]
                    # 예측값을 RevIN 통계로 정규화
                    prev_pred_norm = (prev_pred - wm.unsqueeze(1)) / ws.unsqueeze(1)  # [B, T_out, N]
                    # 입력의 마지막 부분을 예측값으로 대체 (자기회귀 시뮬레이션)
                    if tx.size(3) > 1:
                        # 마지막 1 step을 예측값으로 교체
                        tx_autoregressive = tx.clone()
                        # 예측값을 입력 형태로 변환 [B, 1, N, 1]에 맞춤
                        # prev_pred_norm[:, -1:, :] → [B, 1, N], unsqueeze(-1) → [B, 1, N, 1]
                        prev_pred_reshaped = prev_pred_norm[:, -1:, :].unsqueeze(-1)  # [B, 1, N, 1]
                        tx_autoregressive = torch.cat([tx_autoregressive[:, :, :, :-1], prev_pred_reshaped], dim=3)
                        tx = tx_autoregressive
            # ============================================================
            
            output = model(tx)
            output = torch.squeeze(output, 3)

            # ===== Target-Weighted Loss =====
            diff = torch.abs(output - ty) if args.L1Loss else (output - ty) ** 2
            w = target_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
            loss = (diff * w).sum()
            # ==================================
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)
            
            grad_norm = optim.step()

        if iter % 1 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / (output.size(0) * output.size(1) * data.m)))
        iter += 1
    return total_loss / n_samples


# train_test와 동일: data/data.csv 우선, 없으면 루트 CSV
DEFAULT_DATA_PATH = PROJECT_DIR / 'data' / 'data.csv'
if not DEFAULT_DATA_PATH.exists():
    DEFAULT_DATA_PATH = PROJECT_DIR / 'ExchangeRate_dataset.csv'
DEFAULT_MODEL_SAVE = MODEL_BASE_DIR / 'o_model.pt'  # README: operational model

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default=str(DEFAULT_DATA_PATH), help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default=str(DEFAULT_MODEL_SAVE), help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=3)
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=142, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
parser.add_argument('--end_channels', type=int, default=128, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=36, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length (match train_test that produced hp)')
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--layers', type=int, default=5, help='number of layers')
parser.add_argument('--batch_size', type=int, default=4, help='batch size (match train_test)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00000, help='weight decay rate')
parser.add_argument('--clip', type=int, default=10, help='clip')
parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')
parser.add_argument('--epochs', type=int, default=50, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')
parser.add_argument('--patience', type=int, default=100, help='scheduler patience')


try:
    args = parser.parse_args()
except:
    args = parser.parse_args(args=[])
device = torch.device('cpu')
torch.set_num_threads(3)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixed_seed = 123
set_random_seed(fixed_seed)


# read hyper-parameters (same path as train_test_cyberattack_index save)
filename = str(MODEL_BASE_DIR / 'hp.txt')
with open(filename, 'r') as file:
    content = file.read()
    hp = ast.literal_eval(content)

print('hp',hp)

#train the model
gcn_depth=hp[0]
lr=hp[1]
conv=hp[2]
res=hp[3]
skip=hp[4]
end=hp[5]
layer=hp[-2]
k=hp[6]
dropout=hp[7]
dilation_ex=hp[8]
node_dim=hp[9]
prop_alpha=hp[10]
tanh_alpha=hp[11]
epochs=hp[-1]


# 원본 README: 전체 데이터(train+valid+test)로 재학습 → o_model.pt
Data = DataLoaderS(args.data, 1.0, 0.0, device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len)

print("Data loaded. Checking shape...")
if len(Data.train[0].shape) == 4: # (Samples, C, N, T)
    args.num_nodes = Data.train[0].shape[2]
elif len(Data.train[0].shape) == 3: # (Samples, T, N) usually
    args.num_nodes = Data.train[0].shape[2]

print(f"Auto-detected num_nodes: {args.num_nodes}")

model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
            device, Data.adj, dropout=dropout, subgraph_size=k,
            node_dim=node_dim, dilation_exponential=dilation_ex,
            conv_channels=conv, residual_channels=res,
            skip_channels=skip, end_channels= end,
            seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
            layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)



print(args)
print('The recpetive field size is', model.receptive_field)
nParams = sum([p.nelement() for p in model.parameters()])
print('Number of model parameters is', nParams, flush=True)

if args.L1Loss:
    criterion = nn.L1Loss(reduction='sum').to(device)
else:
    criterion = nn.MSELoss(reduction='sum').to(device)
evaluateL2 = nn.MSELoss(reduction='sum').to(device) #MSE
evaluateL1 = nn.L1Loss(reduction='sum').to(device) #MAE


optim = Optim(
    model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay
)

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training')
    for epoch in range(1, epochs + 1):
        print('epoch:',epoch)
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)

        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, 'wb') as f:
            torch.save(model, f)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')