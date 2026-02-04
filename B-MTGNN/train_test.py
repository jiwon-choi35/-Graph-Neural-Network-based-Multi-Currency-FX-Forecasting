import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import random
from util import *
from trainer import Optim
import sys
from random import randrange
from matplotlib import pyplot as plt
import time
import copy
import os

plt.rcParams['savefig.dpi'] = 1200

# ======================================================================================================================
# 1. Custom ExpandingDataLoader (Hybrid: Rolling Logic + Fixed End Support)
class ExpandingDataLoader(DataLoaderS):
    def __init__(self, file_name, train_end_idx, device, horizon, window, normalize=3, out=1, raw_preset=None):
        self.P = window
        self.h = horizon
        
        # Use preset raw data if available for consistency
        if raw_preset is not None:
            self.rawdat = raw_preset
        else:
            self.rawdat = np.loadtxt(file_name)

        self.n, self.m = self.rawdat.shape
        
        # Handle zeros / negatives for Log-Return
        self.min_data = np.min(self.rawdat, axis=0)
        self.shift = np.zeros(self.m)
        for i in range(self.m):
            if self.min_data[i] <= 0:
                self.shift[i] = abs(self.min_data[i]) + 1.0
        
        self.shifted_rawdat = self.rawdat + self.shift
        self.log_rawdat = np.log(self.shifted_rawdat)
        self.returns = np.diff(self.log_rawdat, axis=0) 
        
        self.dat = np.zeros(self.returns.shape)
        self.normalize = normalize
        self.out_len = out
        self.scale = np.ones(self.m)
        
        # Normalize based on TRAINING DATA ONLY (0 ~ train_end_idx)
        adj_train_end = train_end_idx - 1 
        
        self._normalized_custom(normalize, adj_train_end)
        self._split_custom(adj_train_end)

        self.scale_t = torch.from_numpy(self.scale).float().to(device)
        self.device = device
        self.adj = self.build_predefined_adj() 
        self.col = self.create_columns()
        self.mean_t = torch.from_numpy(self.mean).float().to(device)
        self.shift_t = torch.from_numpy(self.shift).float().to(device)

    def _normalized_custom(self, normalize, train_end_idx):
        if normalize == 3: # StandardScaler on Returns
            self.mean = np.zeros(self.m)
            for i in range(self.m):
                train_series = self.returns[:train_end_idx, i]
                m, s = np.mean(train_series), np.std(train_series)
                self.mean[i] = m
                if s > 1e-8:
                    self.dat[:, i] = (self.returns[:, i] - m) / s
                    self.scale[i] = s
                else:
                    self.dat[:, i], self.scale[i] = 0, 1
        else:
            for i in range(self.m):
                m_val = np.max(np.abs(self.returns[:train_end_idx, i]))
                self.scale[i] = m_val if m_val > 0 else 1.0
                self.dat[:, i] = self.returns[:, i] / self.scale[i]
            self.mean = np.zeros(self.m)

    def _split_custom(self, train_end_idx):
        # Training set: from the beginning up to train_end_idx
        self.train = self._batchify(range(self.P + self.h - 1, train_end_idx), self.h)
        pass

    def _batchify(self, idx_set, horizon):
        num_samples = max(0, len(idx_set) - self.out_len + 1)
        X, Y = torch.zeros((num_samples, self.P, self.m)), torch.zeros((num_samples, self.out_len, self.m)) 
        for i in range(num_samples): 
            end = idx_set[i] - self.h + 1 
            start = end - self.P 
            if start < 0: continue
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :]) 
            Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i]:idx_set[i]+self.out_len, :])
        return [X, Y]

# ======================================================================================================================
# 2. Plotting & Metrics
def consistent_name(name):
    name_map = {
        'kr_fx': 'Korean Won',
        'us_Trade Weighted Dollar Index': 'US Dollar Index',
        'jp_fx': 'Japanese Yen',
        'us_fx': 'US FX'
    }
    if name in name_map: return name_map[name]

    if not name.isupper():
        words = name.replace('_', ' ').split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: result+=word
            else: result+=word[0].upper()+word[1:]
            if i<len(words)-1: result+=' '
        return result
    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 in word: result+=word
        else: result+=word[0]+(word[1:].lower())
        if i<len(words)-1: result+=' '
    return result

def save_metrics_1d(predict, test, title, type_):
    if not torch.is_tensor(predict): predict = torch.from_numpy(predict)
    if not torch.is_tensor(test): test = torch.from_numpy(test)
    
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared = math.sqrt(sum_squared_diff)
    sum_absolute_diff = torch.sum(torch.abs(test - predict))
    mean_all = torch.mean(test) 
    diff_r = test - mean_all 
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    if root_sum_squared_r == 0 or root_sum_squared_r < 1e-10: rrse = 0.0
    else: rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_r = torch.sum(torch.abs(diff_r))
    if sum_absolute_r == 0 or sum_absolute_r < 1e-10: rae = 0.0
    else: rae = (sum_absolute_diff / sum_absolute_r).item()

    title = title.replace('/','_')
    os.makedirs(f'model/Bayesian/{type_}/', exist_ok=True)
    with open(f'model/Bayesian/{type_}/{title}_{type_}.txt',"w") as f:
      f.write('rse:'+str(rrse)+'\n')
      f.write('rae:'+str(rae)+'\n')
      f.close()

def plot_predicted_actual(predicted, actual, title, type_, variance, confidence_95, total_data_len, is_y_fx=True):
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M=[]
    # Extend year range manually to cover future predictions (e.g., up to 2030)
    for year in range(11, 31):   
        for month in months:
            M.append(month+'-'+str(year))   
    if len(M) < total_data_len:
         for month in months:
             if 'Jul' in month: break 
             
    M2=[]; p=[]
    
    # Dynamic Slicing based on prediction length
    x_len = len(predicted)
    
    if type_=='Validation':
        end_idx = total_data_len - x_len
        start_idx = end_idx - x_len
        if start_idx < 0: start_idx = 0 # Safety clamp
        M_slice = M[start_idx:end_idx]
    elif type_=='Testing': 
        end_idx = total_data_len
        start_idx = end_idx - x_len
        if start_idx < 0: start_idx = 0 # Safety clamp
        M_slice = M[start_idx:end_idx]
    else: 
        M_slice = M[-x_len:]

    for index, value in enumerate(M_slice):
        if 'Dec' in value or 'Mar' in value or 'Jun' in value or 'Sep' in value:
            M2.append(value)
            p.append(index+1)

    x=range(1,len(predicted)+1)
    plt.plot(x,actual,'b-',label='Actual')
    plt.plot(x,predicted,'--', color='purple',label='Predicted')
    plt.fill_between(x, predicted-confidence_95.numpy(), predicted+confidence_95.numpy(), alpha=0.5, color='pink', label='95% Confidence')
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Exchange Rate" if is_y_fx else "Log Return", fontsize=15)
    plt.xlabel("Month",fontsize=15)
    plt.xticks(ticks = p ,labels = M2, rotation='vertical',fontsize=13)
    plt.yticks(fontsize=13)
    
    title=title.replace('/','_')
    os.makedirs(f'model/Bayesian/{type_}/', exist_ok=True)
    plt.savefig('model/Bayesian/'+type_+'/'+title+'_'+type_+'.png', bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# ======================================================================================================================
# 3. Training & Expanding Logic
def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    
    length = X.shape[0]
    indices = np.arange(length)
    np.random.shuffle(indices) 
    
    for i in range(0, length, batch_size):
        idx = indices[i:i+batch_size]
        tx = X[idx].unsqueeze(1).transpose(2,3).to(device) # [Batch, C, N, P]
        ty = Y[idx].to(device)
        
        model.zero_grad()
        output = model(tx).squeeze(3)
        
        # Adaptation for 1-step fine-tuning if dimension differs
        if output.shape[1] > ty.shape[1]:
            output = output[:, :ty.shape[1], :]

        loss = criterion(output, ty) 
        loss.backward()
        total_loss += loss.item()
        
        # Verbose Logging (Iter Level) / toooo long & many XD
        #if iter % 1 == 0:
        #     print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item()))

        n_samples += output.numel()
        optim.step()
        iter += 1

    return total_loss / n_samples

def evaluate_expanding_window(raw_full, start_idx, months, model, criterion, batch_size, type_, is_plot=False, features_to_plot=[0,1,2]):
    print(f'\nRunning Expanding Window ({type_}) - Start Index: {start_idx}, Duration: {months} months')
    
    pred_history = []
    actual_history = []
    conf_history = []
    pred_ret_history = []
    actual_ret_history = []
    conf_ret_history = []
    
    # Random column for display debugging
    r_col = 0 
    print(f'testing r={r_col}')
    
    for i in range(months):
        curr_idx = start_idx + i
        dl = ExpandingDataLoader(args.data, curr_idx, device, args.horizon, args.seq_in_len, 3, 1, raw_preset=raw_full)
        
        # Print Window Feature (Input) Analysis
        in_win_dat = dl.dat[curr_idx - args.seq_in_len - 1 : curr_idx - 1, :]
        print('Test Window Feature:', torch.from_numpy(in_win_dat[:, r_col]))
        print('**************x_input*******************')
        print(torch.from_numpy(in_win_dat[:, r_col]).to(device))
        print('**************-------*******************')

        input_seq = torch.from_numpy(in_win_dat).float().unsqueeze(0).unsqueeze(0).transpose(2,3).to(device)
        
        # Predict
        outputs = []
        # model.eval() # To get Bayesian estimation (MC Dropout), do NOT call eval() here!
        for _ in range(10):
            with torch.no_grad():
                out = model(input_seq) 
                if out.shape[3] > 1: # If model predicts 12 steps, take 1st
                     out = out[:, :, :, 0]
                else: 
                     out = out.squeeze(3)
                outputs.append(out[0, 0, :,]) 
        outputs = torch.stack(outputs)
        
        # Process Results
        # 여러 번 예측한 결과(outputs)의 평균을 내어 예측값(pred)과 신뢰구간(ci) 산출
        pred_ret_norm = torch.mean(outputs, 0)
        ci_ret_norm = 1.96 * torch.std(outputs, 0) / math.sqrt(10)

        # 정규화된 값을 원래의 '로그 수익률' 단위로 복원 (값 * 표준편차 + 평균)
        scale_t, mean_t = dl.scale_t, dl.mean_t
        pred_ret = pred_ret_norm * scale_t + mean_t
        ci_ret = ci_ret_norm * scale_t
        
        prev_p_raw = raw_full[curr_idx - 1, :]  # 이전 시점의 실제 환율
        shift_t = dl.shift_t                    # 음수 방지를 위한 보정값(보통 음수는 없긴함)
        # 이전 가격에 보정값을 더해 기준점을 잡음
        prev_p_sh = torch.from_numpy(prev_p_raw).float().to(device) + shift_t
        # 로그 수익률을 지수함수(exp)로 풀어준 뒤 이전 가격에 곱함
        pred_p = (prev_p_sh * torch.exp(pred_ret)) - shift_t
        # 상한선 예측 (신뢰구간 적용)
        pred_p_up = (prev_p_sh * torch.exp(pred_ret + ci_ret)) - shift_t
        ci_p = torch.abs(pred_p_up - pred_p)
        # 이번 시점의 진짜 환율 (정답지)
        actual_p = raw_full[curr_idx, :]
        
        pred_history.append(pred_p.cpu().numpy())
        actual_history.append(actual_p)
        conf_history.append(ci_p.cpu().numpy())
        
        # Store returns for togglable plotting
        pred_ret_history.append(pred_ret.cpu().numpy())
        # Calculate actual log return: log(P_t + shift) - log(P_{t-1} + shift)
        actual_ret = np.log(actual_p + dl.shift) - np.log(prev_p_raw + dl.shift)
        actual_ret_history.append(actual_ret)
        conf_ret_history.append(ci_ret.cpu().numpy())
        
        # Explicit Tensor Dump for User Verification
        print('EVALUATE RESULTS:')
        print(pred_p)
        print(actual_p)

        # Detailed Table Print
        print(f'\n----------------------------Predicted month {i+1}--------------------------------------------------')
        print(f'Pred: {pred_p[r_col]:.4f} | Act: {actual_p[r_col]:.4f}')
        print('------------------------------------------------------------------------------------------------------------')

        # Retrain (Fine-tuning)
        ft_dl = ExpandingDataLoader(args.data, curr_idx + 1, device, args.horizon, args.seq_in_len, 3, 1, raw_preset=raw_full)
        ft_optim = Optim(model.parameters(), args.optim, args.lr * 0.5, args.clip, lr_decay=args.weight_decay)
        for _ in range(10): 
            train(ft_dl, ft_dl.train[0], ft_dl.train[1], model, criterion, ft_optim, args.batch_size)
            
    preds, acts, confs = np.array(pred_history), np.array(actual_history), np.array(conf_history)
    preds_ret, acts_ret, confs_ret = np.array(pred_ret_history), np.array(actual_ret_history), np.array(conf_ret_history)
    
    if is_plot:
        is_y_fx = getattr(args, 'is_y_fx', True)
        for v in features_to_plot:
            col_name = consistent_name(dl.col[v])
            p_data = preds[:, v] if is_y_fx else preds_ret[:, v]
            a_data = acts[:, v] if is_y_fx else acts_ret[:, v]
            c_data = confs[:, v] if is_y_fx else confs_ret[:, v]
            
            save_metrics_1d(p_data, a_data, col_name, type_)
            plot_predicted_actual(p_data, a_data, col_name, type_, torch.from_numpy(c_data), torch.from_numpy(c_data), len(raw_full), is_y_fx=is_y_fx)
            
    # Metric Calculation (Matching train_test.py logic exactly)
    # Convert to Torch for RSE/RAE (as train_test.py uses Torch for these)
    predict_t = torch.from_numpy(preds).float().to(device)
    test_t = torch.from_numpy(acts).float().to(device)

    # 1. RSE (Relative Squared Error) - using Torch
    sum_squared_diff = torch.sum(torch.pow(test_t - predict_t, 2))
    sum_absolute_diff = torch.sum(torch.abs(test_t - predict_t))

    root_sum_squared = math.sqrt(sum_squared_diff)
    
    test_s = test_t
    mean_all = torch.mean(test_s, dim=0) 
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1))
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    if root_sum_squared_r < 1e-10: final_rse = 0.0
    else: final_rse = root_sum_squared / root_sum_squared_r

    # 2. RAE (Relative Absolute Error) - using Torch
    sum_absolute_r = torch.sum(torch.abs(diff_r))
    
    if sum_absolute_r < 1e-10: final_rae = 0.0
    else: final_rae = (sum_absolute_diff / sum_absolute_r).item()

    # 3. Correlation - using Numpy (as train_test.py switches to numpy for this values)
    # preds and acts are already numpy arrays here
    sigma_p = preds.std(axis=0)
    sigma_g = acts.std(axis=0)
    mean_p = preds.mean(axis=0)
    mean_g = acts.mean(axis=0)
    
    index = (sigma_g != 0)
    numerator = ((preds - mean_p) * (acts - mean_g)).mean(axis=0)
    denominator = (sigma_p * sigma_g)
    
    corr_per_node = np.zeros_like(mean_g)
    valid_mask = index & (denominator != 0)
    
    if valid_mask.any():
        corr_per_node[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        final_corr = corr_per_node[valid_mask].mean()
    else:
        final_corr = 0.0

    return final_rse, final_rae, final_corr

# ======================================================================================================================
# 4. Mainz
# Hyperparameters and Arguments
parser = argparse.ArgumentParser()
# Dateset and Paths
parser.add_argument('--data', type=str, default='./data/sm_data.txt') # 데이터 파일 경로: 학습에 사용할 시계열 데이터
parser.add_argument('--save', type=str, default='model/Bayesian/model.pt') # 모델 저장 경로: 학습 완료된 모델의 가중치를 저장할 위치
parser.add_argument('--device',type=str,default='cuda:1' if torch.cuda.is_available() else 'cpu') # 하드웨어 장치: CPU 또는 GPU 사용 여부 지정
# Model Hyperparameters
parser.add_argument('--num_nodes',type=int,default=33) # 노드 개수: 모델링할 대상 변수(Time Series)의 수
parser.add_argument('--seq_in_len',type=int,default=12) # 입력 시퀀스 길이: 과거 데이터를 얼마나 길게 참조할지 결정 (Receptive Field 관련)
parser.add_argument('--seq_out_len',type=int,default=12) # 출력 시퀀스 길이: 모델이 한 번에 예측하도록 훈련되는 기간
parser.add_argument('--horizon', type=int, default=1) # 예측 범위(Horizon): 실제 예측하고자 하는 미래 시점 (예: 1개월 뒤)
parser.add_argument('--batch_size',type=int,default=8) # 배치 크기: 한 번의 가중치 업데이트에 사용할 데이터 샘플 수
parser.add_argument('--lr',type=float,default=0.002) # 학습률(Learning Rate): 모델 가중치를 업데이트하는 속도 조절
parser.add_argument('--weight_decay',type=float,default=0.001) # 가중치 감쇠(L2 Regularization): 모델의 과적합(Overfitting) 방지
parser.add_argument('--epochs',type=int,default=1000) # 학습 에폭 수: 전체 데이터셋을 반복 학습할 횟수
parser.add_argument('--clip',type=int,default=10) # Gradient Clipping: 기울기 폭주를 막아 학습 안정성을 높임
parser.add_argument('--step_size',type=int,default=100) # 학습률 스케줄링: 일정 에폭마다 학습률을 감소시키는 주기
parser.add_argument('--num_split',type=int,default=1) # 데이터 분할 수: 데이터를 몇 개의 덩어리로 나눌지 설정
# GNN Hyperparameters
parser.add_argument('--gcn_true', type=bool, default=True) # GCN 사용 여부: 그래프 합성곱 신경망을 통해 변수 간 관계 학습
parser.add_argument('--buildA_true', type=bool, default=True) # 동적 그래프 학습: 데이터로부터 변수 간 관계(Adjacency Matrix)를 학습할지 여부
parser.add_argument('--gcn_depth',type=int,default=2) # GCN 깊이/층 수: 그래프 특징을 얼마나 깊게 추출할지 결정
parser.add_argument('--dropout',type=float,default=0.2) # Dropout: 일부 뉴런을 비활성화하여 과적합 방지 및 일반화 성능 향상
parser.add_argument('--subgraph_size',type=int,default=10) # 서브그래프 크기: 동적 그래프 학습 시 고려할 주요 이웃 노드의 수
parser.add_argument('--node_dim',type=int,default=40) # 노드 임베딩 차원: 각 변수(노드)를 표현하는 잠재 벡터의 크기
parser.add_argument('--dilation_exponential',type=int,default=2) # Dilation 증가율: 층이 깊어질수록 수용 영역(Receptive Field)을 지수적으로 넓혀 장기 의존성 학습
parser.add_argument('--conv_channels',type=int,default=32) # 합성곱 채널 수: 입력 데이터에서 추출할 특징(Feature Map)의 개수
parser.add_argument('--residual_channels',type=int,default=64) # Residual 채널 수: 층과 층 사이를 연결하는 지름길(Skip Connection)의 정보통로 크기
parser.add_argument('--skip_channels',type=int,default=128) # Skip Connection 채널 수: 다양한 층의 정보를 최종 출력단으로 전달하여 정보 손실 방지 및 풍부한 표현 학습
parser.add_argument('--end_channels',type=int,default=128) # 최종 출력 채널 수: 예측 값을 생성하기 직전 층의 표현력 결정
parser.add_argument('--in_dim',type=int,default=1) # 입력 차원: 시계열 데이터의 특성 수 (보통 1, 단변량)
parser.add_argument('--layers',type=int,default=3) # TCN 층 수: 시간적 패턴을 학습하는 합성곱 층을 얼마나 쌓을지 결정
parser.add_argument('--propalpha',type=float,default=0.1) # MixHop 비율: GCN에서 정보 전파 시 유지할 원본 정보의 비율
parser.add_argument('--tanhalpha',type=float,default=4) # 그래프 활성화 계수: 학습된 그래프 구조의 포화도를 조절하는 파라미터
parser.add_argument('--optim', type=str, default='adam') # 최적화 알고리즘: 손실 함수를 최소화하기 위한 방법 (예: Adam)
parser.add_argument('--normalize', type=int, default=3) # 정규화 방식: 데이터 스케일링 방법 (예: StandardScaler)
parser.add_argument('--is_y_fx', type=lambda x: (str(x).lower() == 'true'), default=True) # Y축 표시 방식: True(실제 환율), False(로그 수익률.변화량)

args = parser.parse_args()
device = torch.device(args.device.split(':')[0])

def set_random_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

fixed_seed = 123

def main():
    start_time = time.time()
    set_random_seed(fixed_seed)
    raw_full = np.loadtxt(args.data)
    # Setting Forecasting Month
    forecast_months = 36
    total_len = len(raw_full)
    # Setting training period
    test_start_idx = total_len - forecast_months
    val_start_idx = test_start_idx - forecast_months
    pretrain_end_idx = val_start_idx
    
    # Check for trigger file 'model/Bayesian/hp.txt'
    trigger_path = 'model/Bayesian/hp.txt'
    is_random_search = os.path.exists(trigger_path)

    if is_random_search:
        iterations = 60
        print(f"Iterations: {iterations}")
    else:
        iterations = 1
        print(f"{trigger_path} is doesn't exist. Using Default Hyperparameters (Best HPs).")

    # Hyperparameter Lists (Random Search Ranges)
    # Best Default Huperparameter: [depth=2, lr=0.002, conv=32, res=64, skip=128, end=128, k=10, drop=0.2, dil=2, dim=40, prop=0.1, tanh=4, layer=3]
    gcn_depths=[2,3]               # GCN 깊이: 그래프 합성곱 층의 수 (2~3층 탐색)
    lrs=[0.0005,0.001,0.002]       # 학습률: 가중치 업데이트 속도 (너무 크면 발산, 작으면 느림)
    convs=[32,64,128]              # 합성곱 채널 수: 입력 특징을 추출하는 필터의 개수 (표현력 결정)
    ress=[32,64,128]               # Residual 채널 수: 정보 보존을 위한 스킵 연결의 용량
    skips=[64,128,256]             # Skip Connection 채널: 최종 출력단으로 전달되는 정보의 양
    ends=[128,256,512]             # 최종 출력 채널: 예측 직전 층의 뉴런 수 (모델의 최종 용량)
    layers=[3,4,5]                 # TCN 층 수: 시계열 패턴을 학습하는 깊이 (깊을수록 장기 패턴 학습 유리)
    ks=[5,10,15,20]                # Subgraph Size (k): 동적 그래프 학습 시 고려할 Top-k 이웃 노드 수 (작을수록 국소적 관계 집중)
    dropouts=[0.1,0.2,0.3]         # Dropout 비율: 과적합 방지를 위해 학습 중 일부 뉴런 비활성화
    dilation_exs=[1,2]             # Dilation 증가율: Receptive Field 확장 속도 (1=선형, 2=지수적)
    node_dims=[30,40,50,60]        # 노드 임베딩 차원: 각 변수(환율/지표)를 표현하는 벡터 크기
    prop_alphas=[0.1,0.2,0.3]      # MixHop 비율: GCN 정보 전파 시 원본 정보 유지 및 이웃 정보 혼합 비율
    tanh_alphas=[2,3,4]            # 그래프 활성화 계수: 학습된 Adjacency Matrix의 포화도(Saturation) 조절

    best_val_score = float('inf')  # Metric: RSE + RAE - Corr
    best_hp = []
    
    print(f" - Forecast Duration: {forecast_months} months")
    print(f" - Validation Period: Idx {val_start_idx} ~ {test_start_idx} (Data[{val_start_idx}:])")
    print(f" - Testing Period   : Idx {test_start_idx} ~ {total_len} (Data[{test_start_idx}:])")
    print("=" * 80)
    
    # 1. Search Loop (Random or Fixed)
    for q in range(iterations):
        # Parameter Assignment
        if is_random_search:
            # Sample Randomly
            gcn_depth = gcn_depths[randrange(len(gcn_depths))]
            lr = lrs[randrange(len(lrs))]
            conv = convs[randrange(len(convs))]
            res = ress[randrange(len(ress))]
            skip = skips[randrange(len(skips))]
            end = ends[randrange(len(ends))]
            layer = layers[randrange(len(layers))]
            subgraph_size = ks[randrange(len(ks))]
            dropout = dropouts[randrange(len(dropouts))]
            dilation_exponential = dilation_exs[randrange(len(dilation_exs))]
            node_dim = node_dims[randrange(len(node_dims))]
            propalpha = prop_alphas[randrange(len(prop_alphas))]
            tanhalpha = tanh_alphas[randrange(len(tanh_alphas))]
        else:
            # Use Fixed Defaults (Best HPs from parser)
            gcn_depth = args.gcn_depth
            lr = args.lr
            conv = args.conv_channels
            res = args.residual_channels
            skip = args.skip_channels
            end = args.end_channels
            layer = args.layers
            subgraph_size = args.subgraph_size
            dropout = args.dropout
            dilation_exponential = args.dilation_exponential
            node_dim = args.node_dim
            propalpha = args.propalpha
            tanhalpha = args.tanhalpha


        print(f'\n[Iter {q+1}] HP: [depth={gcn_depth}, lr={lr}, conv={conv}, res={res}, skip={skip}, end={end}, k={subgraph_size}, drop={dropout}, dil={dilation_exponential}, dim={node_dim}, prop={propalpha}, tanh={tanhalpha}, layer={layer}]')

        # Initialize Data Loader
        dl_pre = ExpandingDataLoader(args.data, pretrain_end_idx, device, args.horizon, args.seq_in_len, 3, args.seq_out_len, raw_preset=raw_full)
        
        # Initialize Model with Sampled HPs
        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes, device, dl_pre.adj, None, dropout, subgraph_size,
                    node_dim, dilation_exponential, conv, res, skip, end,
                    args.seq_in_len, args.in_dim, args.seq_out_len, layer, propalpha, tanhalpha, False).to(device)
        
        if hasattr(model, 'idx'): model.idx = model.idx.to(device)
        
        criterion = nn.L1Loss(reduction='sum').to(device)
        optim = Optim(model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay)
        
        # Phase 1: Pre-training
        print("\n",("="*80))
        print('  Pre-training...')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            t_loss = train(dl_pre, dl_pre.train[0], dl_pre.train[1], model, criterion, optim, args.batch_size)
            if epoch % 50 == 0: # Log every 50 epochs to reduce clutter
                 print(f'| end of epoch {epoch:3d} | time: {time.time()-epoch_start_time:5.2f}s | train_loss {t_loss:5.4f}')
                
        # Phase 2: Expanding Validation
        print("\n", ("="*80))
        print(f'  Validating ({forecast_months} months)...')
        val_res = evaluate_expanding_window(raw_full, val_start_idx, forecast_months, model, criterion, args.batch_size, 'Validation', is_plot=True)
        
        # Handle unpacking depending on return size (2 or 3)
        if len(val_res) == 3:
             val_rse, val_rae, val_corr = val_res
        else:
             val_rse, val_corr = val_res
             val_rae = 0.0

        current_score = val_rse + val_rae - val_corr
        print(f" - Validation Score: {current_score:.4f} (RSE: {val_rse:.4f}, RAE: {val_rae:.4f}, Corr: {val_corr:.4f})")
        print(f" - Now Best Score: {best_val_score:.4f}")
        
        # Track Best
        if current_score < best_val_score:
            best_val_score = current_score
            best_hp = [gcn_depth, lr, conv, res, skip, end, subgraph_size, dropout, dilation_exponential, node_dim, propalpha, tanhalpha, layer]
            
            # Save Best Model State and HPs
            torch.save(model.state_dict(), args.save)
            with open('model/Bayesian/hp.txt', "w") as f:
                f.write(str(best_hp))
            
            print(f'  - NEW BEST SCORE FOUND: {best_val_score:.4f} ***')

    # Phase 3: Final Testing with Best HPs
    print("\n", ("="*80))
    print(f'Phase 3: Final Testing ({forecast_months} months) with Best Model...')
    print('Best HPs:', best_hp)
    
    # Re-initialize model with Best HPs
    if best_hp:
        gcn_depth, lr, conv, res, skip, end, subgraph_size, dropout, dilation_exponential, node_dim, propalpha, tanhalpha, layer = best_hp

        dl_final = ExpandingDataLoader(args.data, val_start_idx, device, args.horizon, args.seq_in_len, 3, args.seq_out_len, raw_preset=raw_full)
        
        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes, device, dl_final.adj, None, dropout, subgraph_size,
                    node_dim, dilation_exponential, conv, res, skip, end,
                    args.seq_in_len, args.in_dim, args.seq_out_len, layer, propalpha, tanhalpha, False).to(device)
        
        if hasattr(model, 'idx'): model.idx = model.idx.to(device)
        model.load_state_dict(torch.load(args.save, weights_only=True))
        
        criterion = nn.L1Loss(reduction='sum').to(device) # Re-init criterion just in case
        
        test_res = evaluate_expanding_window(raw_full, test_start_idx, forecast_months, model, criterion, args.batch_size, 'Testing', is_plot=True)
         
        if len(test_res) == 3:
             test_rse, test_rae, test_corr = test_res
        else:
             test_rse, test_corr = test_res
             test_rae = 0.0

        print('********************************************************************************************************')
        print(f"Best Val Score: {best_val_score:.4f} (RSE: {val_rse:.4f}, RAE: {val_rae:.4f}, Corr: {val_corr:.4f})")
        print(f"Test RSE: {test_rse:.4f} | Test RAE: {test_rae:.4f} | Test Corr: {test_corr:.4f}")
        print('Best HPs:', best_hp)
        print('********************************************************************************************************')

        # Save Evaluation Result
        with open('model/Bayesian/evaluation.txt', 'w') as f:
            f.write(f"Best Val Score: {best_val_score:.4f} (RSE: {val_rse:.4f}, RAE: {val_rae:.4f}, Corr: {val_corr:.4f})\n")
            f.write(f"Test RSE: {test_rse:.4f} | Test RAE: {test_rae:.4f} | Test Corr: {test_corr:.4f}\n")
            f.write(f"Best HPs: {best_hp}\n")

    end_time = time.time()
    excute_time = end_time - start_time
    print(f"\nTotal Execution Time: {excute_time:.2f} seconds ({excute_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()
