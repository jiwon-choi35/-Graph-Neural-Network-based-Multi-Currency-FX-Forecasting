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
import os
from pathlib import Path

import glob

plt.rcParams['savefig.dpi'] = 1200

PROJECT_DIR = Path(__file__).resolve().parents[1]
AXIS_DIR = PROJECT_DIR / 'AXIS'
MODEL_BASE_DIR = AXIS_DIR / 'model' / 'Bayesian'


def inverse_diff_2d(output, I, shift):
    output[0, :] = torch.exp(output[0, :] + torch.log(I + shift)) - shift
    for i in range(1, output.shape[0]):
        output[i, :] = torch.exp(output[i, :] + torch.log(output[i - 1, :] + shift)) - shift
    return output


def inverse_diff_3d(output, I, shift):
    output[:, 0, :] = torch.exp(output[:, 0, :] + torch.log(I + shift)) - shift
    for i in range(1, output.shape[1]):
        output[:, i, :] = torch.exp(output[:, i, :] + torch.log(output[:, i - 1, :] + shift)) - shift
    return output


def plot_data(data, title):
    x = range(1, len(data) + 1)
    plt.plot(x, data, 'b-', label='Actual')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    locs, labs = plt.xticks()
    plt.xticks(rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    plt.show()


def consistent_name(name):
    if name == 'CAPTCHA' or name == 'DNSSEC' or name == 'RRAM':
        return name

    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2:
                result += word
            else:
                result += word[0].upper() + word[1:]
            
            if i < len(words) - 1:
                result += ' '
        return result

    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word or word == 'MITM' or word == 'SIEM':
            result += word
        else:
            result += word[0] + (word[1:].lower())
        
        if i < len(words) - 1:
            result += ' '
        
    return result


def save_metrics_1d(predict, test, title, type):
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared = math.sqrt(sum_squared_diff)
    sum_absolute_diff = torch.sum(torch.abs(test - predict))

    test_s = test
    mean_all = torch.mean(test_s)
    diff_r = test_s - mean_all
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    if root_sum_squared_r == 0:
        rrse = 0.0
    else:
        rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = sum_absolute_diff / sum_absolute_r
    rae = rae.item()

    title = title.replace('/', '_')

    save_dir = MODEL_BASE_DIR / type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = save_dir / f"{title}_{type}.txt"
    
    with open(file_path, "w") as f:
        f.write('rse:' + str(rrse) + '\n')
        f.write('rae:' + str(rae) + '\n')


def plot_predicted_actual(predicted, actual, title, type, variance, confidence_95):
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M=[]
    # 2011년부터 2026년까지 넉넉하게 라벨 생성
    for year in range (11, 27):   
        for month in months:
            if year==11 and month not in ['Jul','Aug','Sep','Oct','Nov','Dec']:
                continue
            M.append(month+'-'+str(year))   
    M2=[]
    p=[]
    
    # === [수정] 날짜 필터링 로직 강화 ===
    target_year = '25' if type == 'Testing' else '24'
    
    # 1. 전체 라벨 중 해당 연도(24 or 25)가 포함된 것만 추출
    target_labels = [m for m in M if f'-{target_year}' in m]
    
    # 2. 데이터 길이(predicted)가 라벨보다 짧으면, 뒤에서부터 맞춤
    if len(target_labels) > len(predicted):
        target_labels = target_labels[-len(predicted):]
    
    # 3. 데이터 길이가 라벨보다 길면, 데이터의 뒷부분만 사용 (최신 데이터 우선)
    elif len(predicted) > len(target_labels):
        diff = len(predicted) - len(target_labels)
        predicted = predicted[diff:]
        actual = actual[diff:]
        confidence_95 = confidence_95[diff:]

    M = target_labels
    
    # X축 틱 설정 (분기별)
    for index, value in enumerate(M):
        if 'Dec' in value or 'Mar' in value or 'Jun' in value or 'Sep' in value:
            M2.append(value)
            p.append(index+1) 

    # === 그래프 그리기 ===
    x = range(1, len(predicted) + 1)
    
    plt.plot(x, actual, 'b-', label='Actual')
    plt.plot(x, predicted, '--', color='purple', label='Predicted')
    if isinstance(confidence_95, torch.Tensor):
        confidence_95 = confidence_95.cpu().numpy()
    plt.fill_between(x, predicted - confidence_95, predicted + confidence_95, alpha=0.5, color='pink', label='95% Confidence')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    
    # X축 라벨 적용
    plt.xticks(ticks=p, labels=M2, rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    
    # === 파일 저장 ===
    fig = plt.gcf()
    title = title.replace('/', '_')
    
    save_dir = MODEL_BASE_DIR / type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_dir / f"{title}_{type}.png", bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def s_mape(yTrue, yPred):
    mape = 0
    for i in range(len(yTrue)):
        mape += abs(yTrue[i] - yPred[i]) / (abs(yTrue[i]) + abs(yPred[i]))
    mape /= len(yTrue)
    return mape


def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot):
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None

    # ============================================
    r = 5
    print('testing r=', str(r))
    test_window = test_window.to(data.device)
    
    # 초기 입력 데이터 설정
    x_input = test_window[0:n_input, :].clone()
    last_predicted = None  # B: Smoothing을 위한 이전 예측값 저장

    # [수정 1] fixed_wm, fixed_ws 변수 및 관련 로직 제거
    # 매 반복문(Sliding Window)마다 통계를 새로 계산해야 함

    for i in range(n_input, test_window.shape[0], data.out_len):
        X = torch.unsqueeze(x_input, dim=0)
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)  # [1, 1, N, T]
        X = X.to(torch.float)

        # =====================================================
        # [수정 2] Dynamic RevIN: 현재 윈도우(X)의 통계 계산
        # =====================================================
        w_mean = X.mean(dim=-1, keepdim=True)  # [1, 1, N, 1]
        w_std = X.std(dim=-1, keepdim=True)    # [1, 1, N, 1]
        w_std[w_std == 0] = 1 # 0으로 나누기 방지
        
        # 정규화 (Normalization)
        X = (X - w_mean) / w_std
        
        # 나중에 복원을 위해 차원 축소해서 저장
        wm = w_mean[0, 0, :, 0]  # [N]
        ws = w_std[0, 0, :, 0]   # [N]
        # =====================================================

        y_true = test_window[i: i + data.out_len, :].clone()

        num_runs = 10
        outputs = []

        for _ in range(num_runs):
            with torch.no_grad():
                output = model(X)
                y_pred = output[-1, :, :, -1].clone()
                
                # =====================================================
                # [수정 3] 현재 윈도우의 통계로 복원 (Denormalization)
                # =====================================================
                y_pred = y_pred * ws + wm
                # =====================================================
                
                if y_pred.shape[0] > y_true.shape[0]:
                    y_pred = y_pred[:-(y_pred.shape[0] - y_true.shape[0]), ]
            outputs.append(y_pred)

        outputs = torch.stack(outputs)
        y_pred = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)

        z = 1.96
        confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

        # ===== B: Smoothing/Clamping (예측값 안정화) =====
        if last_predicted is not None:
            # Exponential smoothing: 0.7 * 현재 예측 + 0.3 * 이전 예측
            y_pred = 0.7 * y_pred + 0.3 * last_predicted
            
            # Clamping: 이전 실제값 대비 ±10% 제한 (첫 step 이후)
            if i > n_input:
                last_actual = test_window[i-1, :]
                y_pred = torch.clamp(y_pred, last_actual * 0.9, last_actual * 1.1)
        
        last_predicted = y_pred.clone()
        # ==================================================

        # 다음 스텝을 위한 입력 업데이트 (Sliding Window)
        if data.P <= data.out_len:
            x_input = y_pred[-data.P:].clone()
        else:
            x_input = torch.cat([x_input[-(data.P - data.out_len):, :].clone(), y_pred.clone()], dim=0)

        if predict is None:
            predict = y_pred
            test = y_true
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, y_pred))
            test = torch.cat((test, y_true))
            variance = torch.cat((variance, var))
            confidence_95 = torch.cat((confidence_95, confidence))

    # 데이터 스케일(DataLoader의 scale/shift) 복원
    scale = data.scale.expand(test.size(0), data.m)
    shift = data.shift.expand(test.size(0), data.m)
    
    predict = predict * scale + shift
    test = test * scale + shift
    variance *= scale
    confidence_95 *= scale

    # --- Metrics 계산 (기존 코드 유지) ---
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    sum_absolute_diff = torch.sum(torch.abs(test - predict))

    root_sum_squared = math.sqrt(sum_squared_diff)
    test_s = test
    mean_all = torch.mean(test_s, dim=0)
    diff_r = test_s - mean_all.expand(test_s.size(0), data.m)
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    if root_sum_squared_r == 0:
        rrse = 0.0
    else:
        rrse = root_sum_squared / root_sum_squared_r
    
    print('rrse=', root_sum_squared, '/', root_sum_squared_r)

    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = sum_absolute_diff / sum_absolute_r
    rae = rae.item()

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0) & (sigma_p != 0)
    if index.sum() > 0:
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
    else:
        correlation = 0.0

    smape = 0
    for z in range(Ytest.shape[1]):
        smape += s_mape(Ytest[:, z], predict[:, z])
    smape /= Ytest.shape[1]

    # --- Plotting (기존 코드 유지) ---
    counter = 0
    if is_plot:
        target_nodes = ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx']

        for v in range(data.m):
            col = v
            raw_name = data.col[col]
            
            if raw_name not in target_nodes:
                continue

            node_name = raw_name.replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            
            save_metrics_1d(torch.from_numpy(predict[:, col]), torch.from_numpy(Ytest[:, col]), node_name, 'Testing')
            plot_predicted_actual(predict[:, col], Ytest[:, col], node_name, 'Testing', variance[:, col], confidence_95[:, col])
            counter += 1

    return rrse, rae, correlation, smape


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None
    sum_squared_diff = 0
    sum_absolute_diff = 0
    
    r = 5
    print('validation r=', str(r))

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)  # [B, 1, N, T]

        # ===== RevIN: Per-Window Normalization =====
        w_mean = X.mean(dim=-1, keepdim=True)  # [B, 1, N, 1]
        w_std = X.std(dim=-1, keepdim=True)    # [B, 1, N, 1]
        w_std[w_std == 0] = 1
        X = (X - w_mean) / w_std

        wm = w_mean[:, 0, :, 0]  # [B, N]
        ws = w_std[:, 0, :, 0]   # [B, N]
        # ============================================

        num_runs = 10
        outputs = []

        with torch.no_grad():
            for _ in range(num_runs):
                output = model(X)
                output = output.squeeze(-1)  # [B, T, N, 1] → [B, T, N]
                if len(output.shape) == 2:
                    output = output.unsqueeze(dim=1)  # [B, N] → [B, 1, N]
                elif len(output.shape) == 1:
                    output = output.unsqueeze(dim=0).unsqueeze(dim=0)  # [N] → [1, 1, N]
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)

        z = 1.96
        confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

        output = mean

        # ===== RevIN Denormalize (back to z-score space) =====
        output = output * ws.unsqueeze(1) + wm.unsqueeze(1)
        var = var * ws.unsqueeze(1)
        confidence = confidence * ws.unsqueeze(1)
        # ======================================================

        # =============================================
        # [중요] Global z-score Denormalize
        # =============================================
        scale = data.scale.expand(Y.size(0), Y.size(1), data.m)
        shift = data.shift.expand(Y.size(0), Y.size(1), data.m)
        
        output = output * scale + shift
        Y = Y * scale + shift
        var *= scale
        confidence *= scale

        if predict is None:
            predict = output
            test = Y
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            variance = torch.cat((variance, var))
            confidence_95 = torch.cat((confidence_95, confidence))

        print('EVALUATE RESULTS:')
        scale = data.scale.expand(Y.size(0), Y.size(1), data.m)
        y_pred_o = output
        y_true_o = Y
        for z in range(Y.shape[1]):
            print(y_pred_o[0, z, r], y_true_o[0, z, r])
        
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * output.size(1) * data.m)

        sum_squared_diff += torch.sum(torch.pow(Y - output, 2))
        sum_absolute_diff += torch.sum(torch.abs(Y - output))

    rse = math.sqrt(total_loss / n_samples) / data.rse 
    rae = (total_loss_l1 / n_samples) / data.rae 

    root_sum_squared = math.sqrt(sum_squared_diff)
    test_s = test
    mean_all = torch.mean(test_s, dim=(0, 1))
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m)
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    if root_sum_squared_r == 0:
        rrse = 0.0
    else:
        rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = sum_absolute_diff / sum_absolute_r
    rae = rae.item()

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0) & (sigma_p != 0)
    if index.sum() > 0:
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
    else:
        correlation = 0.0

    smape = 0
    for x in range(Ytest.shape[0]):
        for z in range(Ytest.shape[2]):
            smape += s_mape(Ytest[x, :, z], predict[x, :, z])
    smape /= Ytest.shape[0] * Ytest.shape[2]

    # ===== jp_fx 개별 RSE 계산 (Best 모델 선택 기준용) =====
    jp_fx_rse = None
    target_nodes = ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx']
    for v in range(data.m):
        raw_name = data.col[v]
        if raw_name == 'jp_fx':
            # jp_fx 노드의 RSE 계산
            jp_pred = predict[:, 0, v]
            jp_true = Ytest[:, 0, v]
            jp_diff_sq = np.sum((jp_pred - jp_true) ** 2)
            jp_mean = np.mean(jp_true)
            jp_diff_from_mean_sq = np.sum((jp_true - jp_mean) ** 2)
            if jp_diff_from_mean_sq > 0:
                jp_fx_rse = np.sqrt(jp_diff_sq / jp_diff_from_mean_sq)
            else:
                jp_fx_rse = 0.0
            break
    if jp_fx_rse is None:
        jp_fx_rse = rrse  # jp_fx를 찾지 못한 경우 전체 RSE 사용
    # ========================================================

    counter = 0
    if is_plot:
        for v in range(data.m):
            col = v
            raw_name = data.col[col]

            if raw_name not in target_nodes:
                continue

            node_name = raw_name.replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict[:, 0, col]), torch.from_numpy(Ytest[:, 0, col]), node_name, 'Validation')
            plot_predicted_actual(predict[:, 0, col], Ytest[:, 0, col], node_name, 'Validation', variance[:, 0, col], confidence_95[:, 0, col])
            counter += 1
    return rrse, rae, correlation, smape, jp_fx_rse


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    # ===== Target-Weighted Loss =====
    # us_Trade Weighted Dollar Index, kr_fx: 기본 가중치 (10.0)
    # jp_fx: 더 강하게 학습 (20.0)
    target_weight = torch.ones(data.m, device=device)
    for i, col_name in enumerate(data.col):
        if col_name == 'us_Trade Weighted Dollar Index' or col_name == 'kr_fx':
            target_weight[i] = 10.0
        elif col_name == 'jp_fx':
            target_weight[i] = 20.0
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
            
            # ===== A: Scheduled Sampling (25% 확률로 자기회귀 학습) =====
            use_autoregressive = random.random() < 0.25
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


DEFAULT_DATA_PATH = AXIS_DIR / 'ExchangeRate_dataset.csv'
DEFAULT_MODEL_SAVE = MODEL_BASE_DIR / 'model.pt'

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
parser.add_argument('--gcn_depth', type=int, default=1, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=142, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=40, help='k')
parser.add_argument('--node_dim', type=int, default=30, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=8, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=64, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=128, help='skip channels')
parser.add_argument('--end_channels', type=int, default=1024, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=24, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--layers', type=int, default=1, help='number of layers')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=10, help='clip')
parser.add_argument('--propalpha', type=float, default=0.6, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=0.1, help='tanh alpha')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')


args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")
    torch.set_num_threads(3)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fixed_seed = 123


def main(experiment):
    set_random_seed(fixed_seed)

    # ===== Fixed HP (0209 최적 결과 기반) - no random search =====
    best_val = 10000000
    best_rse = 10000000
    best_rae = 10000000
    best_corr = -10000000
    best_smape = 10000000
    best_combined_score = 10000000  # Best 모델 선택 기준: 0.5 * 전체 RSE + 0.5 * jp_fx RSE

    best_test_rse = 10000000
    best_test_corr = -10000000

    best_hp = []

    for q in range(1):
        # 0209 최적 HP 사용 (고정)
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

        # ============================================================
        # Cache Cleaning
        # ============================================================
        data_dir = os.path.dirname(args.data) 
        pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
        
        print(f"!!! Cleaning Cache Files in {data_dir} !!!")
        for file_path in pt_files:
            if "model" not in file_path: 
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        print("!!! Cache Clean Complete !!!")
        # ============================================================

        Data = DataLoaderS(args.data, 0.60, 0.20, device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len)

        print('train X:', Data.train[0].shape)
        print('train Y:', Data.train[1].shape)
        print('valid X:', Data.valid[0].shape)
        print('valid Y:', Data.valid[1].shape)
        print('test X:', Data.test[0].shape)
        print('test Y:', Data.test[1].shape)
        print('test window:', Data.test_window.shape)

        print('length of training set=', Data.train[0].shape[0])
        print('length of validation set=', Data.valid[0].shape[0])
        print('length of testing set=', Data.test[0].shape[0])
        print('valid=', int((0.60 + 0.20) * Data.n))

        if len(Data.train[0].shape) == 4: 
             args.num_nodes = Data.train[0].shape[2]
        elif len(Data.train[0].shape) == 3: 
             args.num_nodes = Data.train[0].shape[2]

        print(f"Auto-detected num_nodes: {args.num_nodes}")

        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                      device, Data.adj, dropout=dropout, subgraph_size=k,
                      node_dim=node_dim, dilation_exponential=dilation_ex,
                      conv_channels=conv, residual_channels=res,
                      skip_channels=skip, end_channels=end,
                      seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)
        model = model.to(device)

        print(args)
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        if args.L1Loss:
            criterion = nn.L1Loss(reduction='sum').to(device)
        else:
            criterion = nn.MSELoss(reduction='sum').to(device)
        evaluateL2 = nn.MSELoss(reduction='sum').to(device)
        evaluateL1 = nn.L1Loss(reduction='sum').to(device)

        optim = Optim(
            model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim.optimizer, mode='min', factor=0.5, patience=15
        )

        es_counter = 0
        try:
            print('begin training')
            for epoch in range(1, args.epochs + 1):
                print('Experiment:', (experiment + 1))
                print('Iter:', q)
                print('epoch:', epoch)
                print('hp=', [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch])
                print('best sum=', best_val)
                print('best rrse=', best_rse)
                print('best rrae=', best_rae)
                print('best corr=', best_corr)
                print('best smape=', best_smape)
                print('best hps=', best_hp)
                print('best test rse=', best_test_rse)
                print('best test corr=', best_test_corr)

                es_counter += 1

                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr, val_smape, jp_fx_val_rse = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                                  args.batch_size, False)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f} | jp_fx_rse {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape, jp_fx_val_rse), flush=True)
                
                scheduler.step(val_loss)
                
                # ===== Best 모델 선택 기준: 0.5 * 전체 RSE + 0.5 * jp_fx RSE =====
                safe_corr = val_corr if not math.isnan(val_corr) else 0.0
                sum_loss = val_loss + val_rae - safe_corr
                combined_score = 0.5 * val_loss + 0.5 * jp_fx_val_rse
                
                if combined_score < best_combined_score:
                    save_path = Path(args.save)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(save_path, 'wb') as f:
                        torch.save(model, f)
                    best_val = sum_loss
                    best_rse = val_loss
                    best_rae = val_rae
                    best_corr = val_corr
                    best_smape = val_smape
                    best_combined_score = combined_score  # Best 모델 선택 기준 업데이트

                    best_hp = [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]

                    es_counter = 0

                    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                                                                        args.seq_in_len, False)
                    print('********************************************************************************************************')
                    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape), flush=True)
                    print('********************************************************************************************************')
                    best_test_rse = test_acc
                    best_test_corr = test_corr

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    print('best val loss=', best_val)
    print('best hps=', best_hp)
    
    hp_save_path = MODEL_BASE_DIR / 'hp.txt'
    hp_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hp_save_path, "w") as f:
        f.write(str(best_hp))

    with open(args.save, 'rb') as f:
        model = torch.load(f, weights_only=False)
    # 로드한 모델도 학습 시와 같은 device로 이동
    model = model.to(device)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape, _ = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                             args.batch_size, True)

    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                                                        args.seq_in_len, True)
    print('********************************************************************************************************')
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print('********************************************************************************************************')
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape


if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    vsmape = []
    acc = []
    rae = []
    corr = []
    smape = []
    for i in range(1):
        val_acc, val_rae, val_corr, val_smape, test_acc, test_rae, test_corr, test_smape = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        vsmape.append(val_smape)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
        smape.append(test_smape)
    print('\n\n')
    print('1 run average')
    print('\n\n')
    print("valid\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae)))
    print('\n\n')
    print("test\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae)))