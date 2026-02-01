import argparse
import os
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

plt.rcParams['savefig.dpi'] = 1200

try:
    PROJECT_DIR = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_DIR = Path(os.getcwd())
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
    # === X축 날짜 매핑: Testing/Validation = Aug/22 ~ Jul/25 (값/학습은 그대로, 라벨만) ===
    import pandas as pd
    END = pd.Timestamp("2025-07-01")  # 최종 관측: 25/Jul
    dates = pd.date_range(end=END, periods=len(predicted), freq="MS")
    labels_all = [d.strftime('%b-%y') for d in dates]

    # === 완전 균등 간격 + 시작(Aug) / 끝(Jul) 반드시 포함 ===
    # Aug-22(0) ~ Jul-25(35) => (len-1)=35를 나누는 step만 균등 가능
    step = 5   # 추천: 라벨 과밀 방지 + 완전 균등
    total = len(labels_all) - 1
    if total % step != 0:
        step = 1  # 안전장치(데이터 길이가 바뀌면 월별로라도 균등)
    idxs = list(range(0, len(labels_all), step))  # total%step==0이면 자동으로 마지막 포함
    if idxs[-1] != total:
        idxs.append(total)

    M2 = [labels_all[i] for i in idxs]
    p = [i + 1 for i in idxs]  # x가 1부터 시작

    x = range(1, len(predicted) + 1)
    plt.plot(x, actual, 'b-', label='Actual')
    plt.plot(x, predicted, '--', color='purple', label='Predicted')
    plt.fill_between(x, predicted - confidence_95.numpy(), predicted + confidence_95.numpy(), alpha=0.5, color='pink', label='95% Confidence')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    locs, labs = plt.xticks()
    plt.xticks(ticks=p, labels=M2, rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)

    fig = plt.gcf()
    title = title.replace('/', '_')

    save_dir = MODEL_BASE_DIR / type
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir / f"{title}_{type}.png", bbox_inches="tight")
    # plt.savefig(save_dir / f"{title}_{type}.pdf", bbox_inches="tight", format='pdf')

    plt.show(block=False)
    plt.pause(2)
    plt.close()


def s_mape(yTrue, yPred, eps=1e-8):
    """Symmetric MAPE.

    eps를 더해 0/0 상황에서 NaN이 나오는 것을 방지한다.
    """
    mape = 0.0
    n = len(yTrue)
    if n == 0:
        return float('nan')
    for i in range(n):
        denom = abs(yTrue[i]) + abs(yPred[i]) + eps
        mape += abs(yTrue[i] - yPred[i]) / denom
    mape /= n
    return float(mape)


def _select_metric_nodes(data, fallback_m):
    """메트릭 계산에 사용할 노드 인덱스 선택.

    - 우선순위 1) 'fx' 포함 & 'us_fx' 제외
    - 우선순위 2) 'fx' 포함
    - 우선순위 3) 전체 노드
    """
    col_names = getattr(data, 'col', None)
    if col_names is None:
        return list(range(fallback_m))

    # pandas Index/리스트/튜플 등 대응
    names = [str(x) for x in list(col_names)]
    fx_wo_us = [i for i, n in enumerate(names) if ('fx' in n.lower()) and ('us_fx' not in n.lower())]
    if len(fx_wo_us) > 0:
        return fx_wo_us

    fx_all = [i for i, n in enumerate(names) if ('fx' in n.lower())]
    if len(fx_all) > 0:
        return fx_all

    return list(range(min(len(names), fallback_m)))

# ========================================================
# [수정] horizon 파라미터 추가, y_true 인덱싱 시 horizon 반영
# ========================================================
def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot, horizon=1):
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None
    r = 0 
    print('testing r=', str(r))

    #test_window : 전체 시계열 데이터
    x_input = test_window[0:n_input, :].clone()

    # ===================================================
    # [수정] 루프 범위 : y_true가 범위를 벗어나지 않도록 조정
    # i는 입력 윈도우의 끝 시점
    # ===================================================
    for i in range(n_input, test_window.shape[0], data.out_len):

        X = torch.unsqueeze(x_input, dim=0)
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        X = X.to(torch.float)

        # ==================================================================================
        # [중요 수정] 정답(y_true)은 입력이 끝난 시점(i)에서 horizon만큼 떨어진 시점부터 시작해야 함(멀티스텝)
        # ex) horizon=12 => 입력(1~10) -> 예측(11+12 ~ 11+12+out_len) (23부터 시작)
        # 데이터셋 구성에 따라 인덱스 조정 필요 (현재 수정은 i가 현재 시점 t라고 가정 시 t+horizon
        # ==================================================================================
        y_true_start = i + horizon - 1
        y_true_end = y_true_start + data.out_len

        y_true = test_window[y_true_start : y_true_end, :].clone()

        num_runs = 10
        outputs = []

        for _ in range(num_runs):
            with torch.no_grad():
                output = model(X)
                # =============================================================
                # output shape: (Batch, N, T_out, 1) -> (Batch, N, T_out) 가정
                # 모델 구조에 따라 차원 확인 필요, 보통 squeeze 후 마지막 차원 가져옴
                # =============================================================
                y_pred = output[-1, :, :, -1].clone()

                if y_pred.shape[0] > y_true.shape[0]:
                    y_pred = y_pred[:y_true.shape[0], :]
            outputs.append(y_pred)

        outputs = torch.stack(outputs)
        y_pred = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)

        z = 1.96
        confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

        # ==================================================================
        # [Autoregressive Update]
        # 다음 스텝의 입력을 위해 예측값을 사용하여 x_input 업데이트
        # P(입력 길이)가 out_len(출력 길이)보다 작거나 같으면 예측값의 끝부분 사용
        # ==================================================================
        if data.P <= data.out_len:
            x_input = y_pred[-data.P:].clone()
        else:
            # 입력 길이가 더 길면 기존 입력의 뒷부분 + 예측값 겷합
            x_input = torch.cat([x_input[-(data.P - data.out_len):, :].clone(), y_pred.clone()], dim=0)

        print('----------------------------Predicted months', str(i - n_input + 1), 'to', str(i - n_input + data.out_len), '--------------------------------------------------')
        print(y_pred.shape, y_true.shape)
        y_pred_o = y_pred
        y_true_o = y_true
        for z in range(y_true.shape[0]):
            print(y_pred_o[z, r], y_true_o[z, r])
        print('------------------------------------------------------------------------------------------------------------')

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

    scale = data.scale.expand(test.size(0), data.m)
    predict *= scale
    test *= scale
    variance *= scale
    confidence_95 *= scale

    # === metrics: per-node(RRSE/RAE) 평균 (txt 파일과 동일한 기준) ===
    idx = _select_metric_nodes(data, data.m)

    rrse_list = []
    rae_list = []
    for i in idx:
        yt = test[:, i].contiguous().view(-1)
        yp = predict[:, i].contiguous().view(-1)
        diff = yt - yp
        rrse_num = torch.sqrt(torch.sum(diff ** 2))
        rrse_den = torch.sqrt(torch.sum((yt - yt.mean()) ** 2))
        rrse_list.append((rrse_num / (rrse_den + 1e-12)).item())

        rae_num = torch.sum(torch.abs(diff))
        rae_den = torch.sum(torch.abs(yt - yt.mean()))
        rae_list.append((rae_num / (rae_den + 1e-12)).item())

    rrse = float(np.mean(rrse_list))
    rae = float(np.mean(rae_list))

    # correlation/smape도 동일한 노드 집합으로 계산
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()

    corr_list = []
    for i in idx:
        yp = predict[:, i].reshape(-1)
        yt = Ytest[:, i].reshape(-1)
        if np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
            corr_list.append(float(np.corrcoef(yp, yt)[0, 1]))
    correlation = float(np.mean(corr_list)) if len(corr_list) else float('nan')

    smape_list = [float(s_mape(Ytest[:, i], predict[:, i])) for i in idx]
    smape = float(np.mean(smape_list)) if len(smape_list) else float('nan')

    # counter = 0
    from pathlib import Path
    import os
    
    # 경로 설정 (train_test.py 위치 기준)
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    save_path = PROJECT_DIR / 'AXIS' / 'model' / 'Bayesian' / 'Testing'
    save_path.mkdir(parents=True, exist_ok=True)
    
    avg_file = save_path / 'Average_of_each_call.txt'
    
    # 파일 저장 (전체 평균값 기록)
    with open(avg_file, 'w', encoding='utf-8') as f:
        f.write(f"rse: {rrse:.6f}\n")
        f.write(f"rae: {rae:.6f}\n")
        f.write(f"correlation: {correlation:.6f}\n")
        f.write(f"smape: {smape:.6f}\n")

    if is_plot:
        loop_end = min(r + 142, data.m) 
        for v in range(r, loop_end):
            col = v % data.m
            # node_name 처리 로직 유지
            try:
                node_name = data.col[col] # data.col 접근 방식 확인 필요
            except:
                node_name = str(col)

            node_name = node_name.replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            
            save_metrics_1d(torch.from_numpy(predict[:, col]), torch.from_numpy(Ytest[:, col]), node_name, 'Testing')
            plot_predicted_actual(predict[:, col], Ytest[:, col], node_name, 'Testing', variance[:, col], confidence_95[:, col])
            # counter += 1

    return rrse, rae, correlation, smape


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    # Ensure model is in eval mode during validation (dropout/BN fixed)
    model.eval()

    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None
    sum_squared_diff = 0
    sum_absolute_diff = 0
    r = 0
    print('validation r=', str(r))

    with torch.no_grad():
        for X, Y in data.get_batches(X, Y, batch_size, False):
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)

        num_runs = 10
        outputs = []

        with torch.no_grad():
            for _ in range(num_runs):
                output = model(X)
                output = torch.squeeze(output)
                if len(output.shape) == 1 or len(output.shape) == 2:
                    output = output.unsqueeze(dim=0)
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)

        z = 1.96
        confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

        output = mean
        scale = data.scale.expand(Y.size(0), Y.size(1), data.m)
        
        output *= scale
        Y *= scale
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

    # --------------------------
    # Metric 산출 기준 통일
    # - 개별 노드 txt(save_metrics_1d)와 동일하게: "노드별 RSE/RAE"를 구한 뒤 평균
    # - 가능하면 FX 노드만(US FX는 있으면 제외)로 평가
    # --------------------------
    idx = _select_metric_nodes(data, data.m)
    eps = 1e-12

    rrse_list = []
    rae_list = []
    for i in idx:
        yt = test[:, :, i].contiguous().view(-1)
        yp = predict[:, :, i].contiguous().view(-1)
        diff = yt - yp

        rrse_num = torch.sqrt(torch.sum(diff ** 2))
        rrse_den = torch.sqrt(torch.sum((yt - yt.mean()) ** 2))
        rrse_i = (rrse_num / (rrse_den + eps)).item()

        rae_num = torch.sum(torch.abs(diff))
        rae_den = torch.sum(torch.abs(yt - yt.mean()))
        rae_i = (rae_num / (rae_den + eps)).item()

        rrse_list.append(rrse_i)
        rae_list.append(rae_i)

    rrse = float(np.mean(rrse_list)) if len(rrse_list) else float('nan')
    rae = float(np.mean(rae_list)) if len(rae_list) else float('nan')

    # correlation / smape (같은 idx 기준)
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()

    corr_list = []
    for i in idx:
        yp = predict[:, :, i].reshape(-1)
        yt = Ytest[:, :, i].reshape(-1)
        if np.std(yp) > eps and np.std(yt) > eps:
            corr_list.append(float(np.corrcoef(yp, yt)[0, 1]))
    correlation = float(np.mean(corr_list)) if len(corr_list) else float('nan')

    smape_sum = 0.0
    smape_cnt = 0
    for x in range(Ytest.shape[0]):
        for z in idx:
            smape_sum += s_mape(Ytest[x, :, z], predict[x, :, z])
            smape_cnt += 1
    smape = (smape_sum / smape_cnt) if smape_cnt else float('nan')

    counter = 0
    if is_plot:
        # [수정] loop 범위 수정 (에러 방지)
        loop_end = min(r + 142, data.m)
        for v in range(r, loop_end):
            col = v % data.m
            node_name = data.col[col].replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict[-1, :, col]), torch.from_numpy(Ytest[-1, :, col]), node_name, 'Validation')
            plot_predicted_actual(predict[-1, :, col], Ytest[-1, :, col], node_name, 'Validation', variance[-1, :, col], confidence_95[-1, :, col])
            counter += 1
    # restore train mode
    model.train()
    return rrse, rae, correlation, smape


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
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
            output = model(tx)
            output = torch.squeeze(output, 3)
            scale = data.scale.expand(output.size(0), output.size(1), data.m)
            scale = scale[:, :, :]

            output *= scale
            ty *= scale

            loss = criterion(output, ty)
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
parser.add_argument('--data', type=str, 
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ExchangeRate_dataset.csv'),
    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default=str(DEFAULT_MODEL_SAVE),
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=142, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=10, help='input sequence length')

# ========================================================
# [핵심 수정] horizon과 Output Length를 늘려 Multi-step 설정
# ========================================================
parser.add_argument('--seq_out_len', type=int, default=36, help='output sequence length')
parser.add_argument('--horizon', type=int, default=1)

parser.add_argument('--layers', type=int, default=5, help='number of layers')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=10, help='clip')
parser.add_argument('--lr_decay', type=float, default=1.0, help='learning rate decay (gamma for ExponentialLR)')
parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')


args = parser.parse_args()
device = torch.device('cpu')
torch.set_num_threads(3)

# ---- 데이터 경로 자동 보정 (어느 디렉토리에서 실행해도 동작) ----
from pathlib import Path
_here = Path(__file__).resolve().parent
_p = Path(args.data)
if (not _p.is_absolute()) and (not _p.exists()):
    cand = _here / args.data
    if cand.exists():
        args.data = str(cand)
    else:
        cand2 = _here / 'data' / _p.name
        if cand2.exists():
            args.data = str(cand2)


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

    gcn_depths = [1, 2, 3]
    lrs = [0.01, 0.001, 0.0005, 0.0008, 0.0001, 0.0003, 0.005]
    convs = [4, 8]
    ress = [8, 16]
    skips = [64, 128, 256]
    ends = [256, 512, 1024]
    layers = [1, 2]
    ks = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    dropouts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dilation_exs = [1, 2, 3]
    node_dims = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    prop_alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]
    tanh_alphas = [0.05, 0.1, 0.5, 1, 2, 3, 5, 7, 9]

    best_val = 10000000
    best_rse = 10000000
    best_rae = 10000000
    best_corr = -10000000
    best_smape = 10000000

    best_test_rse = 10000000
    best_test_corr = -10000000

    best_hp = []

    for q in range(80):
        gcn_depth = gcn_depths[randrange(len(gcn_depths))]
        lr = lrs[randrange(len(lrs))]
        conv = convs[randrange(len(convs))]
        res = ress[randrange(len(ress))]
        skip = skips[randrange(len(skips))]
        end = ends[randrange(len(ends))]
        layer = layers[randrange(len(layers))]
        k = ks[randrange(len(ks))]
        dropout = dropouts[randrange(len(dropouts))]
        dilation_ex = dilation_exs[randrange(len(dilation_exs))]
        node_dim = node_dims[randrange(len(node_dims))]
        prop_alpha = prop_alphas[randrange(len(prop_alphas))]
        tanh_alpha = tanh_alphas[randrange(len(tanh_alphas))]

        Data = DataLoaderS(args.data, 0.43, 0.30, device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len)

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
        print('valid=', int((0.43 + 0.3) * Data.n))

        # [중요 수정] 실제 데이터에 맞춰 노드 개수 재설정 (하드코딩된 142 -> 32로 자동 변경)
        # Data.train[0] 형태가 (Samples, Time, Nodes)인 경우 2번 인덱스가 Node 수입니다.
        if len(Data.train[0].shape) == 4: # (Samples, C, N, T)인 경우
             args.num_nodes = Data.train[0].shape[2]
        elif len(Data.train[0].shape) == 3: # (Samples, T, N) 혹은 (Samples, N, T)
             # Transpose 로직상 Data.train[0]는 (Samples, T, N) 형태일 확률이 높음 (N=32)
             # 따라서 마지막 차원을 사용합니다.
             args.num_nodes = Data.train[0].shape[2]

        print(f"Auto-detected num_nodes: {args.num_nodes}")

        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                      device, Data.adj, dropout=dropout, subgraph_size=k,
                      node_dim=node_dim, dilation_exponential=dilation_ex,
                      conv_channels=conv, residual_channels=res,
                      skip_channels=skip, end_channels=end,
                      seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)

        print(args)
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        criterion = nn.SmoothL1Loss(reduction='mean').to(device)

       # 평가지표용(L1, L2)은 그대로 두셔도 됩니다
        evaluateL2 = nn.MSELoss(reduction='mean').to(device)
        evaluateL1 = nn.L1Loss(reduction='mean').to(device)

        optim = Optim(
            model.parameters(), args.optim, lr, args.clip, lr_decay=args.lr_decay, weight_decay=args.weight_decay
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
                val_loss, val_rae, val_corr, val_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                                  args.batch_size, False)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape), flush=True)
                
                sum_loss = val_loss + val_rae - val_corr
                if (not math.isnan(val_corr)) and val_loss < best_rse:
                    # [수정] 모델 저장 경로 디렉토리 확인 및 생성
                    save_path = Path(args.save)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(save_path, 'wb') as f:
                        torch.save(model, f)
                    best_val = sum_loss
                    best_rse = val_loss
                    best_rae = val_rae
                    best_corr = val_corr
                    best_smape = val_smape

                    best_hp = [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]

                    es_counter = 0

                    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                                                                        args.seq_in_len, False, horizon=args.horizon)
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
    
    # [수정] hp.txt 저장 경로 수정
    hp_save_path = MODEL_BASE_DIR / 'hp.txt'
    hp_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hp_save_path, "w") as f:
        f.write(str(best_hp))

    with open(args.save, 'rb') as f:
        model = torch.load(f, weights_only=False)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                             args.batch_size, True)

    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                                                        args.seq_in_len, True, horizon=args.horizon)
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
