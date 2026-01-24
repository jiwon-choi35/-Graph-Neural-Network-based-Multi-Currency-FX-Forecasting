import argparse
import math
import time
import torch
import torch.nn as nn
import sys
import os
from net import gtnet
import numpy as np
import importlib
import random
from util import *
from trainer import Optim
import sys
from random import randrange
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import time
import os
from pathlib import Path

plt.rcParams['savefig.dpi'] = 1200


# ==========================================
# PATCH: Integrated RSE/RAE (ALL vs FX-only)
# ==========================================
def _to_numpy(x):
    """Convert torch.Tensor to numpy, handle various types."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _get_node_names_fallback(real_2d, node_names):
    """Fallback node name generation if node_names is None or wrong length."""
    if node_names is not None and len(node_names) == int(real_2d.shape[1]):
        return list(node_names)
    return [f"node_{i}" for i in range(int(real_2d.shape[1]))]

def _get_fx_indices(node_names, fx_keyword="fx"):
    """Return indices where 'fx_keyword' appears in node name (case-insensitive)."""
    fx_keyword = (fx_keyword or "fx").lower()
    idx = []
    for i, n in enumerate(node_names):
        s = str(n).lower()
        if fx_keyword in s:
            idx.append(i)
    return idx

def compute_rse_rae(pred_2d, real_2d, eps=1e-12):
    """
    pred_2d, real_2d: shape [T, N] (numpy or torch)

    정의를 evaluate()의 RRSE/RAE와 일치시키기 위해
    열별 평균(column-wise mean) 기반으로 분모를 계산합니다.
    """
    pred = _to_numpy(pred_2d)
    real = _to_numpy(real_2d)

    pred = np.asarray(pred, dtype=float)
    real = np.asarray(real, dtype=float)

    if pred.ndim != 2 or real.ndim != 2:
        raise ValueError(f"Expected 2D [T,N]. Got pred={pred.shape}, real={real.shape}")
    if pred.shape != real.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape} vs real={real.shape}")

    diff = pred - real

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))

    real_mean = real.mean(axis=0, keepdims=True)
    var = float(np.mean((real - real_mean) ** 2))
    mad = float(np.mean(np.abs(real - real_mean)))

    rse = (mse / max(var, eps)) ** 0.5
    rae = (mae / max(mad, eps))

    return float(rse), float(rae)

def _safe_makedirs(p):
    """Create directory if not exists."""
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_integrated_metrics_txt(save_dir, split_name, scope_tag, rse, rae, node_names=None, fx_indices=None):
    """
    Save integrated metrics to: <save_dir>/Integrated_<scope_tag>_<split_name>.txt
    
    Args:
        save_dir: Output directory
        split_name: 'Training', 'Validation', 'Testing', etc.
        scope_tag: 'ALL' or 'FX'
        rse, rae: Metric values
        node_names: List of column names (for reference in file)
        fx_indices: Indices of FX nodes (for reference)
    """
    _safe_makedirs(save_dir)
    fname = f"Integrated_{scope_tag}_{split_name}.txt"
    fpath = os.path.join(save_dir, fname)
    
    nodes_line = ""
    if node_names is not None:
        if fx_indices is not None:
            used = [str(node_names[i]) for i in fx_indices]
        else:
            used = [str(x) for x in node_names]
        nodes_line = "nodes=" + ",".join(used)
    
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(f"{scope_tag}_{split_name}: rse={rse:.6f}, rae={rae:.6f}\n")
        if nodes_line:
            f.write(nodes_line + "\n")

# ==========================================
# [개선 2] 상관계수 기반 그래프 생성 함수
def build_corr_adj(raw_TN: torch.Tensor, topk: int = 8, add_self_loops: bool = True, row_norm: bool = True):
    """
    raw_TN: [T, N] (훈련 구간 데이터)
    returns: [N, N] adjacency matrix (상관계수 기반, Top-k, 대칭화, (옵션) self-loop, (옵션) row-normalize)
    """
    x = raw_TN - raw_TN.mean(dim=0, keepdim=True)
    num = x.t() @ x

    denom_vec = torch.sqrt((x ** 2).sum(dim=0, keepdim=False))
    den = denom_vec.unsqueeze(1) @ denom_vec.unsqueeze(0)

    corr = num / (den + 1e-8)
    corr = corr.abs()
    corr.fill_diagonal_(0)

    N = corr.size(0)
    k = max(1, min(int(topk), N - 1))

    vals, idx = corr.topk(k, dim=1)
    A = torch.zeros_like(corr)
    A.scatter_(1, idx, vals)

    A = torch.maximum(A, A.t())

    if add_self_loops:
        A = A + torch.eye(N, device=A.device, dtype=A.dtype)

    if row_norm:
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)

    return A
def inverse_diff_2d(output, I,shift):
    output[0,:]=torch.exp(output[0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[0]):
        output[i,:]= torch.exp(output[i,:]+torch.log(output[i-1,:]+shift))-shift
    return output

def inverse_diff_3d(output, I,shift):
    output[:,0,:]=torch.exp(output[:,0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[1]):
        output[:,i,:]=torch.exp(output[:,i,:]+torch.log(output[:,i-1,:]+shift))-shift
    return output


def plot_data(data,title):
    x=range(1,len(data)+1)
    plt.plot(x,data,'b-',label='Actual')
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Trend",fontsize=15)
    plt.xlabel("Month",fontsize=15)
    locs, labs = plt.xticks() 
    plt.xticks(rotation='vertical',fontsize=13) 
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    plt.close()


# for figure display, we rename columns
def consistent_name(name):

    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name

    #e.g., University of london
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    

    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result

#computes and saves validation/testing error to a text file given a single node's prediction and actual curve values
def save_metrics_1d(predict, test, title, type):
    """Save metrics using consistent compute_rse_rae method"""
    # Ensure 2D shape for compute_rse_rae
    if predict.dim() == 1:
        predict = predict.unsqueeze(1)
    if test.dim() == 1:
        test = test.unsqueeze(1)
    
    rse, rae = compute_rse_rae(predict.numpy(), test.numpy())

    from pathlib import Path
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    save_path = str(PROJECT_DIR / 'AXIS' / 'model' / 'Bayesian' / type)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    title = title.replace('/', '_')
    file_path = Path(save_path) / (title + '_' + type + '.txt')
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f'rse:{rse}\n')
        f.write(f'rae:{rae}\n')
        f.close()


#plots predicted curve with actual curve. The x axis can be adjusted as needed
def plot_predicted_actual(predicted, actual, title, type, variance=None, confidence_95=None, scale=None, col_name=None):
    import numpy as np

    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M=[]
    for year in range (11, 26):
        for month in months:
            if year == 25 and month in ['Oct', 'Nov', 'Dec']:
                continue
            M.append(month+'-'+str(year))

    M2=[]
    p=[]

    if type=='Testing':
        M = M[-len(predicted):]
        for index, value in enumerate(M):
            if 'Dec' in value or 'Mar' in value or 'Jun' in value or 'Sep' in value:
                M2.append(value)
                p.append(index+1)
    else:
        M = M[144:180]
        for index, value in enumerate(M):
            if 'Dec' in value or 'Mar' in value or 'Jun' in value or 'Sep' in value:
                M2.append(value)
                p.append(index+1)

    # --- numpy로 강제 변환 + 1D 정리 ---
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    act  = np.asarray(actual, dtype=float).reshape(-1)
    L = min(len(pred), len(act))
    pred = pred[:L]
    act  = act[:L]
    x = np.arange(1, L+1)

    plt.figure(figsize=(12, 6))
    plt.plot(x, act, 'b-', label='Actual')
    plt.plot(x, pred, '--', color='purple', label='Predicted')

    # --- 95% Confidence: confidence_95만 있으면 그린다 ---
    if confidence_95 is not None:
        ci = np.asarray(confidence_95, dtype=float).reshape(-1)
        L_ci = min(len(pred), len(ci))
        pred = pred[:L_ci]
        act  = act[:L_ci]
        ci   = ci[:L_ci]
        x = np.arange(1, L_ci + 1)

        lower = pred - ci
        upper = pred + ci

        # 밴드(채움) + 경계선 - 미세 스타일 조정
        plt.fill_between(x, lower, upper, alpha=0.18, color='pink', label='95% Confidence', zorder=1)
        plt.plot(x, lower, color='pink', linewidth=0.5, alpha=0.9, zorder=2)
        plt.plot(x, upper, color='pink', linewidth=0.5, alpha=0.9, zorder=2)

    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Months (Time Step)", fontsize=15)
    
    step = max(1, L // 10)
    plt.xticks(np.arange(1, L+1, step), rotation=45, fontsize=10)

    # from pathlib import Path
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    save_path = str(PROJECT_DIR / 'AXIS' / 'model' / 'Bayesian' / type)
    Path(save_path).mkdir(parents=True, exist_ok=True)


    title = title.replace('/', '_')
    file_path = Path(save_path) / (title + '_' + type + '.png')
    plt.savefig(str(file_path), bbox_inches="tight")
    plt.close()


#symmetric mean absolute percentage error (optional)
def s_mape(yTrue,yPred):
  mape=0
  for i in range(len(yTrue)):
    den = abs(yTrue[i]) + abs(yPred[i]) + 1e-8
    mape+= abs(yTrue[i]-yPred[i])/ den
  mape/=len(yTrue)

  return mape

# === MC Dropout for Uncertainty Estimation ===
@torch.no_grad()
def mc_dropout_forward(model, x, n_samples=20):
    """
    F.dropout 기반 모델에서 MC Dropout을 '진짜로' 켜기 위한 함수.
    - 샘플링 동안만 model.train()로 바꿔서 self.training=True를 만들고,
    - 끝나면 원래 모드로 복구합니다.
    """
    was_training = model.training
    try:
        model.train()  # 중요: F.dropout 활성화 (self.training=True)
        preds = []
        for _ in range(int(n_samples)):
            preds.append(model(x).detach())
        preds = torch.stack(preds, dim=0)  # [S, B, out_len, m, 1] (대부분 이렇게 나옴)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0, unbiased=False)
        return mean, std
    finally:
        model.train(was_training)  # was_training=False이면 eval 모드로 복구됨

#for testing the model on unseen data, a sliding window can be used when the output period of the model is smaller than the target period to be forecasted.
#The sliding window uses the output from previous step as input of the next step.
#In our case, the window was not slided (we predicted 36 months and the model by default predicts 36 months)
def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot):
    # [개선 3] eval() 모드로 고정 (드롭아웃 비활성화, 성능 최우선)
    model.eval()
    
    # MC Dropout 및 95% CI 설정
    MC_SAMPLES = 20
    Z_95 = 1.96
    
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    std_series = None
    predictions = []
    sum_squared_diff=0
    sum_absolute_diff=0
    r=random.randint(0, 141)
    r=0 # we can choose any random node index for printing
    print('testing r=',str(r))
    scale = data.scale.expand(test_window.size(0), data.m) #scale will have the max of each column (142 max values)
    print('Test Window Feature:',test_window[:,r])
    
    x_input = test_window[0:n_input, :].clone() # Generate input sequence

    for i in range(n_input, test_window.shape[0], data.out_len):

        print('**************x_input*******************')
        print(x_input[:,r])#prints 1 random column in the sliding window
        print('**************-------*******************')

        X = torch.unsqueeze(x_input,dim=0)
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        X = X.to(torch.float)

        # 현재 배치에서 추출할 진짜 길이 (마지막 배치는 짤릴 수 있음)
        actual_out_len = min(data.out_len, test_window.shape[0] - i)
        y_true = test_window[i: i+actual_out_len,:].clone() 

        # [개선 3] torch.no_grad()로 감싸기 (메모리 효율 + 속도)
        with torch.no_grad():
            if is_plot:
                mean_output, std_output = mc_dropout_forward(model, X, n_samples=MC_SAMPLES)
                out_delta = mean_output[0, :, :, 0].clone()       # [out_len, m] (Δ)
                y_std  = std_output[0, :, :, 0].clone()   # [out_len, m]
            else:
                output = model(X)
                out_delta = output[0, :, :, -1].clone()  # [out_len, m] (Δ)
                y_std  = None

            # [Δ 학습 정렬] Δ 출력 → 레벨 복원
            last = X[0, 0, :, -1].clone()  # [m]
            y_pred = out_delta + last.unsqueeze(0).repeat(out_delta.size(0), 1)  # [out_len, m] (레벨)

            # 마지막 구간에서 y_true 길이에 맞춰 두 다 자르기
            L = y_true.shape[0]
            if y_pred.shape[0] > L:
                y_pred = y_pred[:L, :]
                if y_std is not None:
                    y_std = y_std[:L, :]

            # std 누적
            if is_plot and (y_std is not None):
                if std_series is None:
                    std_series = y_std
                else:
                    std_series = torch.cat((std_series, y_std), dim=0)
        
        # [개선 3] outputs는 단순 단일 예측값으로 사용
        outputs = y_pred
        
        # [개선 3] variance/confidence는 선택사항 (성능 최우선)
        var = None
        std_dev = None
        confidence = None

        #shift the sliding window (y_pred는 이미 레벨로 복원됨)
        # if data.P<=data.out_len:
        #     x_input = y_pred[-data.P:].clone()
        # else:
        #     x_input = torch.cat([x_input[ -(data.P-data.out_len):, :].clone(), y_pred.clone()], dim=0)

        if n_input <= data.out_len:
             x_input = y_pred[-n_input:].clone()
        else:
             x_input = torch.cat([x_input[data.out_len:, :].clone(), y_pred.clone()], dim=0)

        print('----------------------------Predicted months',str(i-n_input+1),'to',str(i-n_input+actual_out_len),'--------------------------------------------------')
        print(y_pred.shape,y_true.shape)
        
        # [방어] 출력 길이가 실제 라벨과 다를 수 있으므로 min으로 제한
        print_len = min(y_pred.shape[0], y_true.shape[0])
        for z in range(print_len):
            print(y_pred[z,r],y_true[z,r]) #only one col
        print('------------------------------------------------------------------------------------------------------------')

        if predict is None:
            predict = y_pred
            test = y_true
            variance=None
            confidence_95=None
        else:
            predict = torch.cat((predict, y_pred))
            test = torch.cat((test, y_true))


    scale_2d = data.scale.expand(test.size(0), data.m)

    predict = predict * scale_2d
    test = test * scale_2d

    ci_series = None
    if is_plot and (std_series is not None):
        std_series = std_series * scale_2d
        Z_95 = 1.96
        ci_series = Z_95 * std_series
        print("CI min/max:", ci_series.min().item(), ci_series.max().item())

    print("\n[DEBUG] Scale Check at Integrated computation:")
    print(f"  predict min/max: {predict.min().item():.4f} / {predict.max().item():.4f}")
    print(f"  test min/max:    {test.min().item():.4f} / {test.max().item():.4f}")

    node_names_test = None
    if hasattr(data, 'col') and data.col is not None:
        node_names_test = list(data.col)
    node_names_test = _get_node_names_fallback(_to_numpy(test), node_names_test)

    fx_keyword = getattr(args, 'fx_keyword', 'fx')
    fx_idx = _get_fx_indices(node_names_test, fx_keyword=fx_keyword)

    pred_np = _to_numpy(predict)
    real_np = _to_numpy(test)

    rse_all, rae_all = compute_rse_rae(pred_np, real_np)

    rse_fx, rae_fx = None, None
    if len(fx_idx) > 0:
        rse_fx, rae_fx = compute_rse_rae(pred_np[:, fx_idx], real_np[:, fx_idx])

    print(f"\n[TEST] RSE/RAE - ALL: rse={rse_all:.6f}, rae={rae_all:.6f} (N={real_np.shape[1]})")
    if rse_fx is not None:
        print(f"[TEST] RSE/RAE - FX : rse={rse_fx:.6f}, rae={rae_fx:.6f} (FX_N={len(fx_idx)})")
        print(f"  FX nodes: {[node_names_test[i] for i in fx_idx]}")
    else:
        print(f"[TEST] FX nodes not found by keyword='{fx_keyword}'. Using ALL as primary.")

    if rse_fx is not None:
        rrse = float(rse_fx)
        rae = float(rae_fx)
        use_idx = fx_idx
    else:
        rrse = float(rse_all)
        rae = float(rae_all)
        use_idx = None

    split_name = "Testing"
    integrated_scope = getattr(args, 'integrated_scope', 'both')

    if getattr(args, 'save_integrated_txt', False):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_save_dir = os.path.normpath(os.path.join(script_dir, '..', 'model', 'Bayesian', 'Testing'))

        if integrated_scope in ('all', 'both'):
            save_integrated_metrics_txt(test_save_dir, split_name, 'ALL', rse_all, rae_all, node_names=node_names_test)

        if integrated_scope in ('fx', 'both') and (rse_fx is not None):
            save_integrated_metrics_txt(test_save_dir, split_name, 'FX', rse_fx, rae_fx,
                                       node_names=node_names_test, fx_indices=fx_idx)

    pred_Tm = pred_np
    real_Tm = real_np
    if use_idx is not None:
        pred_Tm = pred_Tm[:, use_idx]
        real_Tm = real_Tm[:, use_idx]

    sigma_p = pred_Tm.std(axis=0) + 1e-8
    sigma_g = real_Tm.std(axis=0) + 1e-8
    mean_p = pred_Tm.mean(axis=0)
    mean_g = real_Tm.mean(axis=0)

    corr_each = ((pred_Tm - mean_p) * (real_Tm - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = float(np.mean(corr_each))

    smape = 0.0
    for z in range(real_Tm.shape[1]):
        smape += s_mape(real_Tm[:, z], pred_Tm[:, z])
    smape /= max(1, real_Tm.shape[1])

    predict = _to_numpy(predict)
    Ytest = _to_numpy(test)

    counter = 0
    if is_plot:
        print("\n[Plotting] Saving graphs to Testing folder...")
        
        from pathlib import Path
        PROJECT_DIR = Path(__file__).resolve().parents[1]
        save_path = str(PROJECT_DIR / 'AXIS' / 'model' / 'Bayesian' / 'Testing')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 먼저 개별 FX 파일들 저장 및 평균 계산
        fx_rses = []
        fx_raes = []
        avg_file = os.path.join(save_path, 'Average_of_each_call.txt')
        
        with open(avg_file, 'w', encoding='utf-8') as f:
            # 개별 FX 계산
            for col in range(data.m):
                node_name = data.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
                node_name = consistent_name(node_name)
                
                if 'fx' not in node_name.lower():
                    continue
                
                col_rse, col_rae = compute_rse_rae(
                    predict[:, col].reshape(-1, 1),
                    Ytest[:, col].reshape(-1, 1)
                )
                fx_rses.append(col_rse)
                fx_raes.append(col_rae)
                f.write(f'{node_name}: rse:{col_rse:.3f}, rae:{col_rae:.3f}\n')
            
            # 평균값 계산 및 저장
            avg_rse = np.mean(fx_rses)
            avg_rae = np.mean(fx_raes)
            f.write(f'\nAverage RSE/RAE: rse:{avg_rse:.3f}, rae:{avg_rae:.3f}\n')
        
        # 이제 그래프 저장
        for col in range(data.m):
            
            node_name = data.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name = consistent_name(node_name)
            
            if 'fx' not in node_name.lower():
                continue 

            save_metrics_1d(torch.from_numpy(predict[:,col]), torch.from_numpy(Ytest[:,col]), node_name, 'Testing')

            ci_col = None
            if ci_series is not None:
                ci_col = ci_series[:, col].detach().cpu().numpy()

            plot_predicted_actual(
                predict[:,col],
                Ytest[:,col],
                node_name,
                'Testing',
                variance=None,
                confidence_95=ci_col
            )
            counter += 1
            
        print(f"[Done] Saved {counter} graphs.")

    return rrse, rae, correlation, smape, predict, Ytest



def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    # RSE 최적화: 빈 배치 방어
    if X is None or X.size(0) == 0:
        return float('inf'), float('inf'), float('nan'), float('inf')
    
    model.eval()  # 1순위 수정: Validation은 결정론적으로(성능 선택 안정화)
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    sum_squared_diff=0
    sum_absolute_diff=0
    r=0 #we choose any node index for printing (debugging)
    print('validation r=',str(r))

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)

        # Bayesian estimation
        num_runs = 10

        # Create a list to store the outputs
        outputs = []

        # Run the model multiple times (10)
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(X)
                output = output.squeeze(3)
                outputs.append(output)
            

        # Stack the outputs along a new dimension
        outputs = torch.stack(outputs)

        # Calculate mean, variance, and standard deviation
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)#variance
        std_dev = torch.std(outputs, dim=0)#standard deviation

        # Calculate 95% confidence interval
        z=1.96
        confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))

        output=mean #we will consider the mean to be the prediction

        # [Δ 학습 정렬] Δ 출력 → 레벨 복원
        out_delta = output  # [B, out_len, m] (Δ 예측)
        last = X[:, 0, :, -1]  # [B, m]
        last_rep = last.unsqueeze(1).repeat(1, out_delta.size(1), 1)  # [B, out_len, m]
        output = out_delta + last_rep  # [B, out_len, m] (레벨로 복원)
        
        # 신뢰도도 동일하게 처리 (Δ의 불확실성은 그대로 레벨에 적용)
        var = var  # Δ의 분산 그대로 사용

        scale = data.scale.expand(Y.size(0), Y.size(1), data.m) #scale will have the max of each column (142 max values)
        
        #inverse normalisation
        output*=scale
        Y*=scale
        var*=scale
        confidence*=scale

        if predict is None:
            predict = output
            test = Y
            variance=var
            confidence_95=confidence
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            variance= torch.cat((variance, var))
            confidence_95=torch.cat((confidence_95,confidence))


        print('EVALUATE RESULTS:')
        scale = data.scale.expand(Y.size(0), Y.size(1), data.m) #scale will have the max of each column (142 max values)
        y_pred_o=output
        y_true_o=Y
        for z in range(Y.shape[1]):
            print(y_pred_o[0,z,r],y_true_o[0,z,r]) #only one col
        
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * output.size(1) * data.m)

        #RRSE according to Lai et.al
        sum_squared_diff += torch.sum(torch.pow(Y - output, 2))
        #Relative Absolute Error RAE - numerator
        sum_absolute_diff+=torch.sum(torch.abs(Y - output))

    #The below 2 lines are not used
    rse = math.sqrt(total_loss / n_samples) / data.rse 
    rae = (total_loss_l1 / n_samples) / data.rae 

    #RRSE according to Lai et.al - numerator
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    #RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s, dim=(0,1)) # calculate the mean of each column in test
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m) # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r #RRSE

    #Relative Absolute Error RAE
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements - denominator
    rae=sum_absolute_diff/sum_absolute_r # RAE
    rae=rae.item()

    # ==========================================
    # PATCH: Compute Integrated ALL/FX metrics for Validation (shape-safe)
    # ==========================================
    pred_Tm = predict.reshape(-1, data.m).detach().cpu().numpy()
    real_Tm = test.reshape(-1, data.m).detach().cpu().numpy()

    split_name_val = "Validation"

    node_names_val = None
    if hasattr(data, 'col') and data.col is not None and len(data.col) == int(real_Tm.shape[1]):
        node_names_val = list(data.col)
    node_names_val = _get_node_names_fallback(real_Tm, node_names_val)

    fx_keyword_val = getattr(args, 'fx_keyword', 'fx')
    fx_idx_val = _get_fx_indices(node_names_val, fx_keyword=fx_keyword_val)

    rse_all_val, rae_all_val = compute_rse_rae(pred_Tm, real_Tm)

    if len(fx_idx_val) > 0:
        pred_fx_val = pred_Tm[:, fx_idx_val]
        real_fx_val = real_Tm[:, fx_idx_val]
        rse_fx_val, rae_fx_val = compute_rse_rae(pred_fx_val, real_fx_val)

        print(f"\n[Integrated ALL] {split_name_val}: rse={rse_all_val:.6f}, rae={rae_all_val:.6f} (N={real_Tm.shape[1]})")
        print(f"[Integrated FX]  {split_name_val}: rse={rse_fx_val:.6f}, rae={rae_fx_val:.6f} (FX_N={len(fx_idx_val)})")
        print(f"  FX nodes: {[node_names_val[i] for i in fx_idx_val]}")
    else:
        print(f"\n[Integrated ALL] {split_name_val}: rse={rse_all_val:.6f}, rae={rae_all_val:.6f}")
        print(f"[Integrated FX]  {split_name_val}: skipped (no fx nodes found)")
        print(f"[Integrated FX]  {split_name_val}: skipped (no fx nodes found)")

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0)/ (sigma_p * sigma_g) #Pearson's correlation coefficient?
    correlation = (correlation[index]).mean()

    #s-mape
    smape=0
    for x in range(Ytest.shape[0]):
        for z in range(Ytest.shape[2]):
            smape+=s_mape(Ytest[x,:,z],predict[x,:,z])
    smape/=Ytest.shape[0]*Ytest.shape[2]


    #plot actual vs predicted curves and save errors to file
    counter = 0
    if is_plot:
        print("\n[Plotting] Saving Validation graphs (FX only)...")
        
        # 전체 컬럼을 순회 (하드코딩 범위 제거)
        for col in range(data.m):
            
            # 노드 이름(변수명) 가공
            node_name = data.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name = consistent_name(node_name)
            
            # 'fx'만 플롯 (FX만 그리기)
            if 'fx' not in node_name.lower(): 
                continue

            # [추가된 부분] 'fx'가 포함되지 않은 변수는 건너뛰기 (저장 안 함)
            if 'fx' not in node_name.lower(): 
                continue

            if predict.ndim > 2:
                # 보고 싶은 미래 시점 설정 (0: 1개월뒤, 2: 3개월뒤, 5: 6개월뒤)
                target_step = 5
                
                # 에러 방지: 모델 출력 길이(seq_out_len)가 target_step보다 긴지 확인
                if predict.shape[1] > target_step:
                    idx = target_step
                else:
                    # 만약 출력 길이가 짧다면, 그냥 가장 마지막 달 예측값을 사용
                    idx = -1
                
                pred_save = predict[:, idx, col].flatten()
                y_save = Ytest[:, idx, col].flatten()
                
                # 그래프 그릴 때 필요한 분산/신뢰구간도 동일한 시점(idx)으로 가져오기
                var_save = variance[:, idx, col].cpu().numpy().flatten() if variance is not None else None
                conf_save = confidence_95[:, idx, col].cpu().numpy().flatten() if confidence_95 is not None else None

            else:
                # (기존 코드 유지) 차원이 낮을 경우
                pred_save = predict[:, col]
                y_save = Ytest[:, col]
                var_save = variance[:, col].cpu().numpy() if variance is not None else None
                conf_save = confidence_95[:, col].cpu().numpy() if confidence_95 is not None else None

            # 1. 텍스트 파일로 오차율 저장
            save_metrics_1d(torch.from_numpy(pred_save), torch.from_numpy(y_save), node_name, 'Validation')
            
            # 2. 그래프 이미지 저장
            plot_predicted_actual(pred_save, y_save, node_name, 'Validation', var_save, conf_save)
            
            counter += 1
            
        print(f"[Done] Saved {counter} FX graphs to Validation folder.")

    return rrse, rae, correlation, smape


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0.0
    n_batches = 0
    iter = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1).transpose(2, 3)  # [B,1,m,P]
        ty = Y  # [B,out_len,m]  (레벨)

        # 마지막 관측 레벨 (정규화 스케일)
        last = X[:, 0, :, -1]  # [B,m]
        last_rep = last.unsqueeze(1).repeat(1, ty.size(1), 1)  # [B,out_len,m]

        # Δ 타깃
        ty_delta = ty - last_rep

        # 모델은 Δ를 출력하도록 통일
        out_delta = model(X).squeeze(3)  # [B,out_len,m]

        loss_mse = (out_delta - ty_delta).pow(2)

        # 방향이 다르면(곱이 음수면) 페널티 부여
        # sign_loss: 예측과 실제의 부호가 다르면 큰 값, 같으면 0
        diff_sign = -1.0 * torch.sign(out_delta) * torch.sign(ty_delta)
        loss_direction = torch.relu(diff_sign) 

        # 최종 Loss = MSE + 1.0 * 방향성_Loss
        loss_raw = loss_mse + 1.0 * loss_direction

        if hasattr(data, 'fx_idx') and len(data.fx_idx) > 0 and args.fx_weight > 1.0:
            w = torch.ones((1, 1, data.m), device=loss_raw.device)
            w[..., data.fx_idx] = args.fx_weight
            loss = (loss_raw * w).mean()
        else:
            loss = loss_raw.mean()

        loss.backward()
        optim.step()
        total_loss += float(loss.item())
        n_batches += 1

        print('iter:{:3d} | loss: {:.6f}'.format(iter, loss.item()))
        iter += 1

    return total_loss / max(1, n_batches)


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'AXIS', 'ExchangeRate_dataset.csv'),
                    help='location of the data file')
parser.add_argument('--save', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'AXIS', 'model', 'Bayesian', 'model.pt'),
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
# parser.add_argument('--device',type=str,default='cuda:0',help='device (cuda:0 or cpu)')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', action=argparse.BooleanOptionalAction, default=False, help='use predefined adjacency (correlation graph) - use --no-buildA_true to disable')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=142,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout rate (권장: 0.15->0.05, 36개월)')
parser.add_argument('--subgraph_size',type=int,default=8,help='k (권장: 10->20, 36개월)')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential (권장: 1, receptive field 축소)')
parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels (권장: 16->32, 36개월)')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels (권장: 16->32, 36개월)')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels (권장: 32->64, 36개월)')
parser.add_argument('--end_channels',type=int,default=128,help='end channels (권장: 64->128, 36개월)')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=36,help='input sequence length (receptive_field과 일치 필수)')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length (권장: 12, 3회 roll=36개월)')
parser.add_argument('--horizon', type=int, default=6, help='lead time (권장: 1 = next-step)')
parser.add_argument('--layers',type=int,default=3,help='number of layers')

parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate (권장: 0.001->0.0005, 36개월 직접예측)')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate (0.001 or 0.0005)') 

parser.add_argument('--clip',type=int,default=10,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

# [2순위 추가] FX 가중치
parser.add_argument('--fx_weight', type=float, default=5.0,
                    help='loss weight multiplier for FX columns (권장: 5.0)')

# [4순위 추가] 학습률 감쇠
parser.add_argument('--lr_gamma', type=float, default=0.97,
                    help='ExponentialLR gamma for lr schedule (0<gamma<1, 권장: 0.97)')

# [개선 4] 학습 설정 권장값으로 변경 (36개월 직접예측)
parser.add_argument('--epochs',type=int,default=200,help='epochs (권장: 50->200, 36개월 학습 안정화)')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')

# ==========================================
# PATCH: Integrated metrics scope control
# ==========================================
parser.add_argument(
    '--integrated_scope',
    type=str,
    default='both',
    choices=['all', 'fx', 'both'],
    help='Integrated metrics scope: all=all nodes only, fx=fx-only only, both=write both.'
)
parser.add_argument(
    '--fx_keyword',
    type=str,
    default='fx',
    help='Keyword to select FX nodes from node_names. Default: fx'
)
parser.add_argument(
    '--save_integrated_txt',
    action='store_true',
    help='If set, save integrated metrics txt files in the split folder (Training/Validation/Testing).'
)


# ==========================================
# ENSEMBLE (multi-seed training) controls
# ==========================================
parser.add_argument(
    '--ensemble_runs',
    type=int,
    default=1,
    help='Number of independent training runs (different seeds) for ensemble. Set 5 for 5-run ensemble.'
)
parser.add_argument(
    '--base_seed',
    type=int,
    default=123,
    help='Base random seed. Each ensemble run uses base_seed + run_id.'
)
parser.add_argument(
    '--save_per_run_models',
    action='store_true',
    help='If set, save each run model to <save>_runK.pt (recommended when ensemble_runs>1).'
)

args = parser.parse_args()

# ---- FX tuning overrides (safe defaults; CLI flags still take precedence when explicitly provided) ----
# These values are chosen to slow down training (more epochs, smaller batch) and to strengthen graph learning.
if not hasattr(args, "_fx_tuned"):
    args._fx_tuned = True
    # If user did not explicitly set these via CLI, apply safer defaults.
    if "--epochs" not in " ".join(sys.argv):
        args.epochs = 200
    if "--batch_size" not in " ".join(sys.argv):
        args.batch_size = 16
    if "--subgraph_size" not in " ".join(sys.argv):
        args.subgraph_size = 8
    if "--dropout" not in " ".join(sys.argv):
        args.dropout = 0.2
    if "--buildA_true" not in " ".join(sys.argv):
        args.buildA_true = True
    if "--gcn_true" not in " ".join(sys.argv):
        args.gcn_true = True

# device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
torch.set_num_threads(3)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixed_seed = None  # set from args.base_seed

def main(experiment, seed=None):
    # Set per-run random seed for ensemble diversity
    if seed is None:
        seed = int(getattr(args, "base_seed", 123)) + int(experiment)
    set_random_seed(int(seed))
    print(f"[RUN {experiment}] seed={seed}")

    # run-specific save path (avoid overwrite when ensemble_runs>1)
    save_path_run = args.save
    if int(getattr(args, "ensemble_runs", 1)) > 1 or getattr(args, "save_per_run_models", False):
        root, ext = os.path.splitext(args.save)
        if not ext:
            ext = ".pt"
        save_path_run = f"{root}_run{experiment}{ext}"


    #model hyper-parameters
    gcn_depths=[1,2,3]
    lrs=[0.01,0.001,0.0005,0.0008,0.0001,0.0003,0.005]#[0.00001,0.0001,0.0002,0.0003]
    convs=[4,8,16]
    ress=[16,32,64]
    skips=[64,128,256]
    ends=[256,512,1024]
    layers=[1,2]
    ks=[20,30,40,50,60,70,80,90,100]
    dropouts=[0.2,0.3,0.4,0.5,0.6,0.7]
    dilation_exs=[1,2,3]
    node_dims=[20,30,40,50,60,70,80,90,100]
    prop_alphas=[0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8]
    tanh_alphas=[0.05,0.1,0.5,1,2,3,5,7,9]


    best_val = 10000000
    best_rse=  10000000
    best_rae=  10000000
    best_corr= -10000000
    best_smape=10000000
    
    best_test_rse=10000000
    best_test_corr=-10000000

    best_hp=[]


    #random search
    for q in range(1):

        #hps - args 값을 직접 사용 (랜덤 서치 제거)
        gcn_depth = args.gcn_depth
        lr = args.lr
        conv = args.conv_channels
        res = args.residual_channels
        skip = args.skip_channels
        end = args.end_channels
        layer = args.layers
        k = args.subgraph_size
        dropout = args.dropout
        dilation_ex = args.dilation_exponential
        node_dim = args.node_dim
        prop_alpha = args.propalpha
        tanh_alpha = args.tanhalpha
        

        Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, normalize=args.normalize, out=args.seq_out_len)

        # [4순위 안정화] epochs=0 대기 - 데이터 체크전용
        if args.epochs == 0:
            print("[INFO] epochs=0: data/model check only. Exiting.")
            print(f"  Train shape: {Data.train[0].shape}")
            print(f"  Valid shape: {Data.valid[0].shape}")
            print(f"  Test shape: {Data.test[0].shape}")
            return 0, 0, 0, 0, 0, 0, 0, 0

        args.num_nodes = Data.train[0].shape[2]
        
        # [개선 2] 그래프가 비어있으면 상관계수 기반 그래프 생성
        train_end = int(0.6 * Data.n)
        if (Data.adj is None) or (Data.adj.numel() == 0) or (Data.adj.sum().item() == 0):
            print("\n[GRAPH] predefined graph is empty -> building correlation graph from training data")
            A_corr = build_corr_adj(Data.dat[:train_end, :], topk=8)
            Data.adj = A_corr
            print(f"[GRAPH] correlation adjacency created: shape={Data.adj.shape}, edges={Data.adj.nonzero().size(0)}")
        
        # [데이터 품질 점검] 30초 진단
        print("\n=== Data Quality Check ===")
        print("rawdat shape:", Data.rawdat.shape)
        print("rawdat min/max:", Data.rawdat.min().item(), "/", Data.rawdat.max().item())
        col_zero_ratio = (Data.rawdat == 0).float().mean(dim=0)
        print("zero ratio per column (top5):", col_zero_ratio.topk(5).values)
        print("================================\n")
    

        print('train X:',Data.train[0].shape)
        print('train Y:', Data.train[1].shape)
        print('valid X:',Data.valid[0].shape)
        print('valid Y:',Data.valid[1].shape)
        print('test X:',Data.test[0].shape)
        print('test Y:',Data.test[1].shape)
        print('test window:', Data.test_window.shape)

        print('length of training set=',Data.train[0].shape[0])
        print('length of validation set=',Data.valid[0].shape[0])
        print('length of testing set=',Data.test[0].shape[0])
        print('valid=',int((0.43 + 0.3) * Data.n))
        
       
        
        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                    device, Data.adj, dropout=dropout, subgraph_size=k,
                    node_dim=node_dim, dilation_exponential=dilation_ex,
                    conv_channels=conv, residual_channels=res,
                    skip_channels=skip, end_channels= end,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)
        

        print(args)
        print('\n' + '='*60)
        print('[GNN CHECK] --gcn_true:', args.gcn_true)
        print('[GNN CHECK] --buildA_true:', args.buildA_true)
        print('[GNN CHECK] Graph adjacency matrix shape:', Data.adj.shape)
        print('[GNN CHECK] Graph edges (non-zero):', (Data.adj > 0).sum().item())
        print('='*60 + '\n')
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        # [개선 4] SmoothL1Loss (Huber) 사용: outlier에 덜 민감
        # criterion = nn.SmoothL1Loss(reduction='sum').to(device)
        criterion = nn.MSELoss(reduction='sum').to(device)
        evaluateL2 = nn.MSELoss(reduction='sum').to(device) #MSE
        evaluateL1 = nn.L1Loss(reduction='sum').to(device) #MAE

        optim = Optim(
            model.parameters(),
            args.optim,
            lr,
            args.clip
        )

        # 2. [Fix] weight_decay 수동 적용
        # Optim 클래스가 직접 받지 못하므로, 내부 optimizer에 직접 설정해줍니다.
        if hasattr(args, 'weight_decay') and args.weight_decay > 0:
            for param_group in optim.optimizer.param_groups:
                param_group['weight_decay'] = args.weight_decay

        # 3. 스케줄러 설정 (기존 코드 유지)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim.optimizer, mode='min', factor=0.5, patience=5
        )
        
        es_counter=0 #early stopping
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            print('begin training')
            for epoch in range(1, args.epochs + 1):
                print('Experiment:',(experiment+1))
                print('Iter:',q)
                print('epoch:',epoch)
                print('hp=',[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch])
                print('best sum=',best_val)
                print('best rrse=',best_rse)
                print('best rrae=',best_rae)
                print('best corr=',best_corr)
                print('best smape=',best_smape)       
                print('best hps=',best_hp)
                print('best test rse=',best_test_rse)
                print('best test corr=',best_test_corr)

                
                es_counter+=1 # feel free to use this for early stopping (not used)

                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr, val_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                 args.batch_size,False)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape), flush=True)
                
                # [4순위 추가] 매 epoch마다 lr 감소 (학습률 스케줄)
                # optim.lr_step()
                current_lr = optim.optimizer.param_groups[0]['lr']
                print(f"  → current lr: {current_lr:.6f}", flush=True)
                
                # Save the model if the validation loss is the best we've seen so far.
                sum_loss=val_loss+val_rae-val_corr
                if (not math.isnan(val_corr)) and val_loss < best_rse:
                    with open(save_path_run, 'wb') as f:
                        torch.save(model, f)
                    best_val = sum_loss
                    best_rse= val_loss
                    best_rae= val_rae
                    best_corr= val_corr
                    best_smape=val_smape

                    best_hp=[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]
                    
                    es_counter=0
                    
                    test_acc, test_rae, test_corr, test_smape, _, _ = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                           args.seq_in_len, False) 
                    scheduler.step(val_loss)
                    print('********************************************************************************************************')
                    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape), flush=True)
                    print('********************************************************************************************************')
                    best_test_rse=test_acc
                    best_test_corr=test_corr

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    print('best val loss=',best_val)
    print('best hps=',best_hp)
    #save best hp to desk
    # from pathlib import Path
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    hp_path = PROJECT_DIR / 'AXIS' / 'model' / 'Bayesian' / 'hp.txt'
    hp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hp_path, "w", encoding="utf-8") as f:
        f.write(str(best_hp))
    
    # Load the best saved model.
    with open(save_path_run, 'rb') as f:
        model = torch.load(f, weights_only=False)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size, True)

    test_acc, test_rae, test_corr, test_smape, test_pred, test_true = evaluate_sliding_window(
        Data, Data.test_window, model, evaluateL2, evaluateL1, args.seq_in_len, True
    )
    print('********************************************************************************************************')    
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print('********************************************************************************************************')
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape, test_pred, test_true, Data.col, getattr(Data, "fx_idx", [])

if __name__ == "__main__":
    # NOTE:
    # - ensemble_runs=1  : single training run (default)
    # - ensemble_runs>1  : train multiple runs with different seeds and average predictions (true ensemble)
    ensemble_runs = int(getattr(args, "ensemble_runs", 1))

    if ensemble_runs <= 1:
        # Single run (keeps backward compatibility)
        main(0, seed=int(getattr(args, "base_seed", 123)))
        sys.exit(0)

    print(f"\n=== Starting Ensemble Training ({ensemble_runs} runs) ===\n")

    all_preds = []
    final_true = None
    node_names = None
    fx_idx = None

    # Per-run metric logs
    acc = []
    rae = []
    corr = []
    smape = []

    for run_id in range(ensemble_runs):
        seed = int(getattr(args, "base_seed", 123)) + int(run_id)
        val_acc, val_rae, val_corr, val_smape, test_acc, test_rae, test_corr, test_smape, test_pred, test_true, cols, fx_indices = main(run_id, seed=seed)

        acc.append(test_acc); rae.append(test_rae); corr.append(test_corr); smape.append(test_smape)

        all_preds.append(test_pred)
        if final_true is None:
            final_true = test_true
        if node_names is None:
            node_names = cols
        if fx_idx is None:
            fx_idx = fx_indices

    print('\n\n' + '='*60)
    print(f'   ENSEMBLE RESULT ({ensemble_runs} runs)')
    print('='*60)

    print("\n[Individual Models Average]")
    print("Test RSE mean: {:5.4f} (std: {:5.4f})".format(np.mean(acc), np.std(acc)))
    print("Test RAE mean: {:5.4f} (std: {:5.4f})".format(np.mean(rae), np.std(rae)))
    print("Test Corr mean: {:5.4f} (std: {:5.4f})".format(np.mean(corr), np.std(corr)))
    print("Test sMAPE mean: {:5.4f} (std: {:5.4f})".format(np.mean(smape), np.std(smape)))

    # Ensemble prediction (mean)
    ensemble_pred = np.mean(np.stack(all_preds, axis=0), axis=0)

    # Integrated metrics (ALL and FX-only)
    ens_rse_all, ens_rae_all = compute_rse_rae(ensemble_pred, final_true)
    print("\n[Ensemble Model Performance] (Lower is Better)")
    print(f"ALL Nodes -> RSE: {ens_rse_all:.6f} | RAE: {ens_rae_all:.6f}")

    if fx_idx is not None and len(fx_idx) > 0:
        ens_rse_fx, ens_rae_fx = compute_rse_rae(ensemble_pred[:, fx_idx], final_true[:, fx_idx])
        print(f"FX Nodes  -> RSE: {ens_rse_fx:.6f} | RAE: {ens_rae_fx:.6f} (FX_N={len(fx_idx)})")
    else:
        print("FX Nodes  -> skipped (no fx columns detected)")

    improvement_rse = float(np.mean(acc) - ens_rse_all)
    print(f"\n>> Improvement by Ensemble (vs mean single-run): {improvement_rse:.6f} (RSE reduction)")

    # Save ensemble summary
    from pathlib import Path
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    out_dir = PROJECT_DIR / 'AXIS' / 'model' / 'Bayesian' / 'Testing'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'Ensemble_Result.txt'
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Ensemble Runs: {ensemble_runs}\n")
        f.write(f"ALL RSE: {ens_rse_all:.6f}\n")
        f.write(f"ALL RAE: {ens_rae_all:.6f}\n")
        if fx_idx is not None and len(fx_idx) > 0:
            f.write(f"FX RSE: {ens_rse_fx:.6f}\n")
            f.write(f"FX RAE: {ens_rae_fx:.6f}\n")
    print(f"\n[Saved] {out_path}\n")
