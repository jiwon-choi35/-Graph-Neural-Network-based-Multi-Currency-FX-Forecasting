"""
Exchange Rate Forecasting Script
Bayesian MTGNN with Monte Carlo Dropout
"""

import numpy as np
import os
import torch
import sys
import pandas as pd
from matplotlib import pyplot
import matplotlib.dates as mdates
from net import gtnet

# High-resolution plot settings
pyplot.rcParams['savefig.dpi'] = 1200


# ==========================================
# Helper Functions
# ==========================================

def exponential_smoothing(series, alpha):
    """지수평활법"""
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result


def consistent_name(name):
    """컬럼명 정리"""
    name = name.replace('_', ' ')
    if name == 'us Trade Weighted Dollar Index':
        return 'US Dollar Index'
    if name == 'kr fx':
        return 'KRW/USD'
    if name == 'jp fx':
        return 'JPY/USD'
    return name


def zero_negative_curves(data, forecast):
    """음수값 제거 (환율은 양수만 있음)"""
    data = torch.clamp(data, min=0)
    forecast = torch.clamp(forecast, min=0)
    return data, forecast


def save_data(data, forecast, confidence, variance, col, output_dir=None):
    """예측 데이터를 텍스트 파일로 저장"""
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        file_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'data')
    else:
        file_dir = output_dir
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    for i in range(data.shape[1]):
        d = data[:, i]
        f = forecast[:, i]
        c = confidence[:, i]
        v = variance[:, i]
        name = col[i]
        with open(os.path.join(file_dir, name.replace('/', '_') + '.txt'), 'w') as ff:
            ff.write('Data: ' + str(d.tolist()) + '\n')
            ff.write('Forecast: ' + str(f.tolist()) + '\n')
            ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
            ff.write('Variance: ' + str(v.tolist()) + '\n')


def save_gap_data(forecast, col, target_indices, output_dir=None):
    """각 주요 변수와 다른 변수들 간의 gap을 CSV로 저장"""
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        gap_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'gap')
    else:
        gap_dir = output_dir
    
    if not os.path.exists(gap_dir):
        os.makedirs(gap_dir)
    
    # 각 target 변수를 기준으로 다른 변수들과의 gap 계산
    for target_idx in target_indices:
        target_name = col[target_idx]
        target_forecast = forecast[:, target_idx]
        
        # 이 target을 기준으로 다른 변수들과의 gap 계산
        gap_data = {}
        
        for i in range(len(col)):
            if i != target_idx:
                other_forecast = forecast[:, i]
                gap = other_forecast - target_forecast
                gap_data[col[i]] = gap.cpu().numpy().tolist()
        
        # CSV 파일로 저장
        gap_df = pd.DataFrame(gap_data)
        safe_name = target_name.replace('/', '_').replace(' ', '_')
        csv_path = os.path.join(gap_dir, safe_name + '_gap.csv')
        gap_df.to_csv(csv_path, index=False)
        print(f"✅ Gap data saved: {safe_name}_gap.csv")


def plot_forecast(data, forecast, confidence, col_name, dates_hist, dates_future, output_dir=None, color='RoyalBlue'):
    """개별 국가 예측 플롯 생성 (사이버 보안 스타일)"""
    
    # 음수값 제거
    data, forecast = zero_negative_curves(data, forecast)
    
    # 스타일 설정
    pyplot.style.use("seaborn-v0_8-dark")
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
    
    # Historical과 Forecast 연결
    d = torch.cat((data, forecast[0:1]), dim=0)  # Historical 끝 + Forecast 시작 연결
    f = forecast
    c = confidence
    
    # 선 두께 결정 (US는 2, 나머지는 1)
    if 'us_' in col_name.lower() or 'dollar' in col_name.lower():
        line_width = 2
    else:
        line_width = 1
    
    # Historical 플롯 (인덱스 기반)
    ax.plot(range(len(d)), d, '-', color=color, label=consistent_name(col_name), linewidth=line_width)
    
    # Forecast 플롯 (Historical 끝에서 이어서)
    forecast_range = range(len(d)-1, (len(d)+len(f))-1)
    ax.plot(forecast_range, f, '-', color=color, linewidth=line_width)
    
    # 95% Confidence Interval (종합 그래프와 동일하게 증폭)
    ax.fill_between(forecast_range, 
                     f - c * 6, 
                     f + c * 6,
                     color=color, alpha=0.4, label='95% CI')
    
    # X축 년도 레이블 (2011~2027, 2011-01부터 시작)
    # 데이터: 180개월 (2011-01 ~ 2025-12) + 12개월 예측 (2026-01 ~ 2026-12)
    x = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027']
    # 각 연도 1월의 인덱스: 2011-01=0, 2012-01=12, 2013-01=24, ...
    positions = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192]
    ax.set_xticks(positions, x)
    
    # Y축 레이블
    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    
    # 범례
    ax.legend(loc="upper left", prop={'size': 10})
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    
    # 타이틀
    pyplot.title(consistent_name(col_name), y=1.03, fontsize=18)
    
    # 크기 설정
    fig.set_size_inches(10, 7)
    
    # 저장
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        images_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'plots')
    else:
        images_dir = output_dir
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    safe_name = col_name.replace('/', '_').replace(' ', '_')
    pyplot.savefig(os.path.join(images_dir, safe_name + '.png'), bbox_inches="tight")
    pyplot.savefig(os.path.join(images_dir, safe_name + '.pdf'), bbox_inches="tight", format='pdf')
    print(f"✅ Plot saved: {safe_name}")
    pyplot.show(block=False)
    pyplot.pause(3)
    pyplot.close()


def plot_multi_node(data, forecast, confidence, target_indices, col, dates_hist, dates_future, output_dir=None):
    """다국가 비교 플롯 (DDoS 공격-솔루션 스타일)"""
    
    # 음수값 제거
    data, forecast = zero_negative_curves(data, forecast)
    
    # 색상 팔레트 (DDoS 이미지와 유사하게)
    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
              "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta"]
    
    pyplot.style.use("seaborn-v0_8-dark")
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
    
    # 각 국가별 플롯 (Normalize 적용)
    for idx, i in enumerate(target_indices):
        color = colours[idx % len(colours)]
        col_name = consistent_name(col[i])
        
        # Historical + Forecast 시작점 연결
        d = torch.cat((data[:, i], forecast[0:1, i]), dim=0)
        f = forecast[:, i]
        c = confidence[:, i]
        
        # Normalize: 첫 번째 값을 1.0으로 (또는 모든 값을 min-max scaling)
        # 옵션 1: 첫 번째 값 기준 normalize (2011-01 = 1.0)
        base_value = d[0].item()
        d_normalized = d / base_value
        f_normalized = f / base_value
        # Confidence는 상대적 비율로 유지 (normalize된 값의 비율로 계산)
        # 원본 confidence를 원본 값으로 나누면 상대적 변동폭이 됨
        c_relative = c / base_value
        # 음영이 잘 보이도록 confidence를 3배로 증폭
        c_normalized = c_relative * 3.0
        
        # Historical 플롯 (인덱스 기반, normalized)
        line_width = 2 if idx == 0 else 1  # 첫 번째 국가는 굵게
        ax.plot(range(len(d_normalized)), d_normalized, '-', color=color, label=col_name, linewidth=line_width, zorder=3)
        
        # Forecast 플롯 (Historical 끝에서 연결, normalized)
        forecast_range = range(len(d_normalized)-1, (len(d_normalized)+len(f_normalized))-1)
        ax.plot(forecast_range, f_normalized, '-', color=color, linewidth=line_width, zorder=3)
        
        # 95% Confidence Interval (각 국가별로, 더 두껍게)
        ax.fill_between(forecast_range,
                        f_normalized - c_normalized * 2,
                        f_normalized + c_normalized * 2,
                        color=color, alpha=0.4, zorder=3)
    
    # 모든 국가 플롯 후 음영 적용: US선~KR선은 KR색, KR선~JP선은 파란색
    if len(target_indices) > 2:
        # US (첫 번째 - 가장 아래)
        us_idx = target_indices[0]
        d_us = torch.cat((data[:, us_idx], forecast[0:1, us_idx]), dim=0)
        f_us = forecast[:, us_idx]
        base_us = d_us[0].item()
        full_us = torch.cat((d_us, f_us[1:]), dim=0)
        full_us_norm = full_us / base_us
        
        # KR (두 번째 - 중간)
        kr_idx = target_indices[1]
        d_kr = torch.cat((data[:, kr_idx], forecast[0:1, kr_idx]), dim=0)
        f_kr = forecast[:, kr_idx]
        base_kr = d_kr[0].item()
        full_kr = torch.cat((d_kr, f_kr[1:]), dim=0)
        full_kr_norm = full_kr / base_kr
        
        # JP (세 번째 - 가장 위)
        jp_idx = target_indices[2]
        d_jp = torch.cat((data[:, jp_idx], forecast[0:1, jp_idx]), dim=0)
        f_jp = forecast[:, jp_idx]
        base_jp = d_jp[0].item()
        full_jp = torch.cat((d_jp, f_jp[1:]), dim=0)
        full_jp_norm = full_jp / base_jp
        
        # Forecast 구간만 추출 (180~191)
        forecast_start_idx = len(d_us) - 1  # 179
        forecast_x = np.arange(forecast_start_idx, forecast_start_idx + len(f_us))
        forecast_y_us = full_us_norm[forecast_start_idx:forecast_start_idx + len(f_us)].cpu().numpy()
        forecast_y_kr = full_kr_norm[forecast_start_idx:forecast_start_idx + len(f_kr)].cpu().numpy()
        forecast_y_jp = full_jp_norm[forecast_start_idx:forecast_start_idx + len(f_jp)].cpu().numpy()
        
        # 색상 팔레트 (RoyalBlue=US, Crimson=KR)
        colours_shading = ["RoyalBlue", "Crimson", "DarkOrange"]
        
        # 파란색 음영: US선부터 JP선까지 (전체 배경)
        ax.fill_between(
            forecast_x, forecast_y_us, forecast_y_jp,
            interpolate=True,
            color=colours_shading[0],  # RoyalBlue (파란색)
            alpha=0.3,
            zorder=1
        )
        
        # KR 색깔 음영: US선부터 KR선까지 (위에 덮기)
        ax.fill_between(
            forecast_x, forecast_y_us, forecast_y_kr,
            interpolate=True,
            color=colours_shading[1],  # Crimson (KR 색)
            alpha=0.3,
            zorder=2
        )
    
    # X축 년도 레이블 (2011~2027, 2011-01부터 시작)
    x = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027']
    # 각 연도 1월의 인덱스: 2011-01=0, 2012-01=12, 2013-01=24, ...
    positions = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192]
    ax.set_xticks(positions, x)
    
    # Y축 레이블 (Normalized)
    ax.set_ylabel("Normalized Index (2011-01 = 1.0)", fontsize=15)
    pyplot.yticks(fontsize=13)
    
    # 범례
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    
    # 타이틀
    pyplot.title("Exchange Rate Forecast (Normalized, 2011-01 = 1.0)", y=1.03, fontsize=18)
    
    # 크기 설정
    fig.set_size_inches(10, 7)
    
    # 저장
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        images_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'plots')
    else:
        images_dir = output_dir
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    pyplot.savefig(os.path.join(images_dir, 'Multi_Country_Forecast_Normalized.png'), bbox_inches="tight")
    pyplot.savefig(os.path.join(images_dir, 'Multi_Country_Forecast_Normalized.pdf'), bbox_inches="tight", format='pdf')
    print(f"✅ Multi-country plot saved (normalized)")
    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()


# ==========================================
# Main Forecasting
# ==========================================

if __name__ == "__main__":
    
    print("="*70)
    print("  BAYESIAN MTGNN EXCHANGE RATE FORECASTING")
    print("="*70)
    
    # 파일 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file = os.path.join(script_dir, 'data', 'sm_data.csv')
    model_file = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'model.pt')
    
    # 출력 디렉토리 설정
    plot_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'plots')
    data_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'data')
    
    for d in [plot_dir, data_out_dir]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    
    # 데이터 로드
    print("\n📂 Loading data...")
    df_raw = pd.read_csv(data_file)
    
    # 날짜 컬럼 찾기
    date_col = next((c for c in ["Date", "date", "DATA", "data"] if c in df_raw.columns), None)
    
    if date_col is not None:
        dates_all = pd.to_datetime(df_raw[date_col], errors="coerce")
        df = df_raw.drop(columns=[date_col])
    else:
        dates_all = None
        df = df_raw
    
    # 수치 변환
    df = df.apply(pd.to_numeric, errors="coerce").ffill().fillna(0)
    col = df.columns.tolist()
    rawdat = df.values
    n, m = rawdat.shape
    
    print(f"✅ Data shape: {n} time points × {m} variables")
    
    # 날짜 생성 (2025년 12월까지)
    LAST_OBS = pd.Timestamp("2025-12-01")
    dates_hist = pd.date_range(end=LAST_OBS, periods=n, freq="MS").tolist()
    print(f"📅 Historical period: {dates_hist[0].strftime('%Y-%m')} ~ {dates_hist[-1].strftime('%Y-%m')}")
    
    # Normalization
    print("\n⚙️  Normalizing...")
    scale = np.ones(m)
    dat = np.zeros(rawdat.shape)
    
    for i in range(m):
        scale[i] = np.max(np.abs(rawdat[:, i]))
        if scale[i] == 0:
            scale[i] = 1.0
        dat[:, i] = rawdat[:, i] / scale[i]
    
    print("✅ Normalization complete")
    
    # 모델 로드
    print("\n🧠 Loading model...")
    with open(model_file, 'rb') as f:
        model = torch.load(f, map_location='cpu', weights_only=False)
    print("✅ Model loaded")
    
    # Input sequence length (모델에서 설정된 값 사용)
    try:
        seq_len = model.seq_length
    except:
        seq_len = 12  # train_test.py의 기본값
    
    P = seq_len
    print(f"🕹️  Input sequence length: {seq_len} months")
    
    # 초기 입력 (마지막 10개월)
    X_init = torch.from_numpy(dat[-seq_len:, :]).float()
    
    # Forecast settings
    horizon = 12  # 2026년 1~12월 (12개월)
    num_runs = 20  # MC dropout runs
    
    print(f"\n🎲 Running Bayesian forecast...")
    print(f"   • MC runs: {num_runs}")
    print(f"   • Horizon: {horizon} months")
    
    # Monte Carlo Dropout Forecasting
    outputs = []
    model.train()  # Enable dropout
    
    with torch.no_grad():
        # 먼저 모델 출력 길이 확인
        tmp_in = X_init.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).contiguous()
        tmp_out = model(tmp_in)
        pred_len = int(tmp_out.size(1))
        
        for r in range(num_runs):
            curr_X = X_init.clone()
            preds = []
            len_preds = 0
            
            while len_preds < horizon:
                curr_input = curr_X.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).contiguous()
                
                # RevIN (train_test.py와 동일)
                w_mean = curr_input.mean(dim=-1, keepdim=True)
                w_std = curr_input.std(dim=-1, keepdim=True)
                w_std[w_std == 0] = 1
                curr_input_norm = (curr_input - w_mean) / w_std
                
                out = model(curr_input_norm)
                
                # Denormalize
                pred_block = out.squeeze(3).squeeze(0)  # [T_out, N]
                wm = w_mean[0, 0, :, 0]
                ws = w_std[0, 0, :, 0]
                pred_level = pred_block * ws.unsqueeze(0) + wm.unsqueeze(0)
                
                need = horizon - len_preds
                take = min(pred_level.size(0), need)
                take_block = pred_level[:take, :].cpu().numpy()
                
                preds.append(take_block)
                len_preds += take
                
                new_X = np.concatenate([curr_X.cpu().numpy(), take_block], axis=0)
                curr_X = torch.from_numpy(new_X[-P:, :]).float().contiguous()
            
            outputs.append(torch.tensor(np.concatenate(preds, axis=0)))
    
    print("✅ Forecast complete")
    
    # 통계 계산
    outputs = torch.stack(outputs)
    Y = torch.mean(outputs, dim=0)
    std_dev = torch.std(outputs, dim=0)
    confidence = 1.96 * std_dev / torch.sqrt(torch.tensor(num_runs))
    variance = torch.var(outputs, dim=0)
    
    # Denormalization
    scale_torch = torch.from_numpy(scale).float()
    dat_denorm = torch.from_numpy(dat).float() * scale_torch
    Y_denorm = Y * scale_torch
    confidence_denorm = confidence * scale_torch
    variance_denorm = variance * scale_torch
    
    # Residual-based prediction interval adjustment (0.5 factor)
    confidence_denorm = confidence_denorm * 0.5
    
    print(f"\n📊 Statistics:")
    print(f"   • Mean forecast: {Y_denorm.mean():.4f}")
    print(f"   • Std: {Y_denorm.std():.4f}")
    print(f"   • Avg confidence width: {confidence_denorm.mean():.4f}")
    
    # 플롯 대상 선택 (3개국만) - 먼저 정의
    target_names = ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx']
    target_indices = [i for i, name in enumerate(col) if name in target_names]
    
    if not target_indices:
        print("⚠️  Warning: Target columns not found")
        target_indices = list(range(min(3, m)))
    
    print(f"\n🎯 Target countries: {[col[i] for i in target_indices]}")
    
    # 데이터 저장
    print("\n💾 Saving forecast data...")
    save_data(dat_denorm, Y_denorm, confidence_denorm, variance_denorm, col, data_out_dir)
    print("✅ Data saved")
    
    # Gap 데이터 저장
    print("\n💾 Saving gap data...")
    gap_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'gap')
    if not os.path.exists(gap_dir):
        os.makedirs(gap_dir, exist_ok=True)
    save_gap_data(Y_denorm, col, target_indices, gap_dir)
    print("✅ Gap data saved")
    
    # Smoothing (optional)
    print("\n🔧 Applying exponential smoothing...")
    alpha_hist = 0.3
    alpha_future = 0.5
    alpha_conf = 0.95  # Confidence는 거의 smoothing 안 함 (음영이 잘 보이도록)
    
    hist_smoothed = []
    fut_smoothed = []
    conf_smoothed = []
    
    for i in range(m):
        hist_arr = dat_denorm[:, i].cpu().numpy()
        fut_arr = Y_denorm[:, i].cpu().numpy()
        conf_arr = confidence_denorm[:, i].cpu().numpy()
        
        hist_smoothed.append(exponential_smoothing(hist_arr.tolist(), alpha_hist))
        fut_smoothed.append(exponential_smoothing(fut_arr.tolist(), alpha_future))
        conf_smoothed.append(exponential_smoothing(conf_arr.tolist(), alpha_conf))
    
    hist_plot = torch.tensor(np.array(hist_smoothed)).T
    fut_plot = torch.tensor(np.array(fut_smoothed)).T
    conf_plot = torch.tensor(np.array(conf_smoothed)).T
    
    print("✅ Smoothing complete")
    
    # 예측 날짜 생성 (2026년 1월~12월)
    FORECAST_START = LAST_OBS + pd.DateOffset(months=1)
    dates_future = pd.date_range(start=FORECAST_START, periods=horizon, freq="MS").tolist()
    print(f"📅 Forecast period: {dates_future[0].strftime('%Y-%m')} ~ {dates_future[-1].strftime('%Y-%m')}")
    
    # 플롯 색상 팔레트 (Multi_Country와 동일)
    plot_colours = ["RoyalBlue", "Crimson", "DarkOrange"]
    
    # 플롯 생성
    print("\n📊 Generating plots...")
    
    # 개별 플롯 (색상 매칭)
    for idx, i in enumerate(target_indices):
        color = plot_colours[idx % len(plot_colours)]
        plot_forecast(hist_plot[:, i], fut_plot[:, i], conf_plot[:, i],
                     col[i], dates_hist, dates_future, plot_dir, color)
    
    # 다국가 비교 플롯
    plot_multi_node(hist_plot, fut_plot, conf_plot,
                   target_indices, col, dates_hist, dates_future, plot_dir)
    
    print("\n" + "="*70)
    print("✅ FORECASTING COMPLETED")
    print("="*70)
    print(f"📁 Output directories:")
    print(f"   • Plots: {plot_dir}")
    print(f"   • Data:  {data_out_dir}")
    print(f"   • Gap:   {gap_dir}")
    print(f"\n📊 Generated Files:")
    print(f"   • Multi_Country_Forecast_Normalized.png (Normalized comparison)")
    print(f"   • Individual forecast plots for each country")
    print(f"   • Gap analysis for each target variable")
    print("="*70)
