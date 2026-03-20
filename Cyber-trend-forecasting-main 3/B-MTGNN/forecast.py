import numpy as np
import os
import torch
import sys
import csv
import pandas as pd
from matplotlib import pyplot
import matplotlib.dates as mdates
from net import gtnet
from matplotlib import font_manager, rc
pyplot.style.use("seaborn-v0_8-dark")


# --- [추가] 한글 폰트 및 마이너스 기호 설정 (Windows 맑은 고딕 기준) ---
try:
    font_path = "C:/Windows/Fonts/malgun.ttf" # Windows 맑은 고딕 경로
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
except:
    # 경로가 다를 경우 시스템 폰트명으로 직접 설정 시도
    pyplot.rcParams['font.family'] = 'Malgun Gothic'
pyplot.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
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


def save_country_gap(forecast_denorm, col, target_names, dates_future, output_dir=None):
    """국가 간 환율 예측값 격차를 CSV로 저장.

    사이버 연구의 save_gap() 로직을 환율 데이터에 맞게 재구성:
      - 사이버: attack − solution (월별→연평균 변환 후 gap)
      - 환율: country_A − country_B (월별→분기평균 + 연평균 변환 후 gap)

    forecast_denorm: (horizon, num_vars) denormalized 예측 텐서
    col: 컬럼명 리스트
    target_names: 비교 대상 국가 컬럼명 리스트  (예: ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx'])
    dates_future: 예측 기간 날짜 리스트
    output_dir: 저장 디렉토리
    """
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'gap')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    target_indices = {name: col.index(name) for name in target_names if name in col}
    if len(target_indices) < 2:
        print("Warning: Need at least 2 target columns for gap calculation")
        return

    horizon = forecast_denorm.shape[0]
    month_labels = [d.strftime('%Y-%m') for d in dates_future]

    # --- 1) 월별 원본 예측값 테이블 (predict_by_country.csv) ---
    rows_predict = []
    for name, idx in target_indices.items():
        vals = forecast_denorm[:, idx].detach().cpu().numpy().tolist()
        rows_predict.append([consistent_name(name)] + vals)

    predict_path = os.path.join(output_dir, 'predict_by_country.csv')
    with open(predict_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Country'] + month_labels)
        for row in rows_predict:
            writer.writerow(row)

    # --- 2) 분기 평균 변환 (사이버 연구의 12개월→연평균 방식을 분기로 확장) ---
    quarter_labels = []
    quarter_data = {}  # {country_name: [Q1_avg, Q2_avg, ...]}

    months_per_q = 3
    num_quarters = horizon // months_per_q

    for q in range(num_quarters):
        start = q * months_per_q
        end = start + months_per_q
        qlabel = f"{dates_future[start].year}-Q{(dates_future[start].month - 1) // 3 + 1}"
        quarter_labels.append(qlabel)

    # 연간 평균
    year_label = f"{dates_future[0].year}-Annual"

    for name, idx in target_indices.items():
        vals = forecast_denorm[:, idx].detach().cpu().numpy()
        q_avgs = []
        for q in range(num_quarters):
            start = q * months_per_q
            end = start + months_per_q
            q_avgs.append(float(np.mean(vals[start:end])))
        quarter_data[name] = q_avgs

    # --- 3) Pairwise gap 계산 (A − B) ---
    names = list(target_indices.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))

    gap_rows = []
    for name_a, name_b in pairs:
        display_a = consistent_name(name_a)
        display_b = consistent_name(name_b)
        pair_label = f"{display_a} − {display_b}"

        q_a = np.array(quarter_data[name_a])
        q_b = np.array(quarter_data[name_b])

        vals_a = forecast_denorm[:, target_indices[name_a]].detach().cpu().numpy()
        vals_b = forecast_denorm[:, target_indices[name_b]].detach().cpu().numpy()
        annual_gap = float(np.mean(vals_a) - np.mean(vals_b))

        q_gaps = (q_a - q_b).tolist()
        gap_rows.append([pair_label] + q_gaps + [annual_gap])

    gap_path = os.path.join(output_dir, 'country_gap.csv')
    with open(gap_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Pair'] + quarter_labels + [year_label])
        for row in gap_rows:
            writer.writerow(row)

    # --- 4) Normalized gap (기준국 대비 %) ---
    gap_pct_rows = []
    for name_a, name_b in pairs:
        display_a = consistent_name(name_a)
        display_b = consistent_name(name_b)
        pair_label = f"{display_a} − {display_b}"

        q_a = np.array(quarter_data[name_a])
        q_b = np.array(quarter_data[name_b])

        base = np.where(q_a != 0, q_a, 1.0)
        q_pct = ((q_a - q_b) / base * 100).tolist()

        vals_a = forecast_denorm[:, target_indices[name_a]].detach().cpu().numpy()
        vals_b = forecast_denorm[:, target_indices[name_b]].detach().cpu().numpy()
        annual_mean_a = float(np.mean(vals_a))
        annual_pct = float((np.mean(vals_a) - np.mean(vals_b)) / annual_mean_a * 100) if annual_mean_a != 0 else 0.0

        gap_pct_rows.append([pair_label] + q_pct + [annual_pct])

    gap_pct_path = os.path.join(output_dir, 'country_gap_pct.csv')
    with open(gap_pct_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Pair'] + quarter_labels + [year_label])
        for row in gap_pct_rows:
            writer.writerow(row)

    # --- 5) 개별 국가 분기 요약 ---
    summary_rows = []
    for name, idx in target_indices.items():
        vals = forecast_denorm[:, idx].detach().cpu().numpy()
        q_avgs = quarter_data[name]
        annual_avg = float(np.mean(vals))
        summary_rows.append([consistent_name(name)] + q_avgs + [annual_avg])

    summary_path = os.path.join(output_dir, 'country_quarterly_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Country'] + quarter_labels + [year_label])
        for row in summary_rows:
            writer.writerow(row)

    print(f"  -> predict_by_country.csv  (monthly forecasts per country)")
    print(f"  -> country_gap.csv         (pairwise gaps: quarterly + annual)")
    print(f"  -> country_gap_pct.csv     (pairwise gaps as %)")
    print(f"  -> country_quarterly_summary.csv (quarterly + annual averages)")

    return gap_path


def save_normalized_gap(hist_denorm, forecast_denorm, col, target_names,
                        dates_future, output_dir=None):
    """단위가 다른 환율을 비교 가능하게 정규화한 뒤 격차를 계산.

    각 국가 시계열을 해당 국가의 첫 번째 관측값(historical t=0)으로 나누어
    base=1.0 지수로 변환한 뒤, pairwise gap을 산출한다.

      normalized_i[t] = series_i[t] / series_i[0]
      gap_ij          = mean(normalized_i) − mean(normalized_j)

    양수 → i가 기준 시점 대비 더 많이 상승, 음수 → j가 더 많이 상승.

    hist_denorm:     (T_hist, num_vars) denormalized 과거 데이터
    forecast_denorm: (horizon, num_vars) denormalized 예측 텐서
    """
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    target_indices = {name: col.index(name) for name in target_names if name in col}
    if len(target_indices) < 2:
        print("Warning: Need at least 2 target columns")
        return

    horizon = forecast_denorm.shape[0]
    month_labels = [d.strftime('%Y-%m') for d in dates_future]
    months_per_q = 3
    num_quarters = horizon // months_per_q

    quarter_labels = []
    for q in range(num_quarters):
        start = q * months_per_q
        qlabel = f"{dates_future[start].year}-Q{(dates_future[start].month - 1) // 3 + 1}"
        quarter_labels.append(qlabel)
    year_label = f"{dates_future[0].year}-Annual"

    # --- 1) Normalize: series / series[0]  (base = historical first observation) ---
    norm_monthly = {}   # {country_name: np.array of normalized forecast}
    norm_quarterly = {} # {country_name: [Q1_avg, Q2_avg, ...]}
    base_values = {}

    for name, idx in target_indices.items():
        base_val = float(hist_denorm[0, idx].detach().cpu().numpy())
        if base_val == 0:
            base_val = 1.0
        base_values[name] = base_val

        raw_forecast = forecast_denorm[:, idx].detach().cpu().numpy()
        normed = raw_forecast / base_val
        norm_monthly[name] = normed

        q_avgs = []
        for q in range(num_quarters):
            s = q * months_per_q
            e = s + months_per_q
            q_avgs.append(float(np.mean(normed[s:e])))
        norm_quarterly[name] = q_avgs

    # --- 2) Normalized 월별 예측값 CSV ---
    norm_month_path = os.path.join(output_dir, 'normalized_forecast.csv')
    with open(norm_month_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Country', 'Base_Value_(t0)'] + month_labels)
        for name in target_indices:
            row = [consistent_name(name), base_values[name]] + norm_monthly[name].tolist()
            writer.writerow(row)

    # --- 3) Pairwise normalized gap (quarterly + annual) ---
    names = list(target_indices.keys())
    pairs = [(names[i], names[j])
             for i in range(len(names)) for j in range(i + 1, len(names))]

    gap_rows = []
    for name_a, name_b in pairs:
        pair_label = f"{consistent_name(name_a)} − {consistent_name(name_b)}"
        q_a = np.array(norm_quarterly[name_a])
        q_b = np.array(norm_quarterly[name_b])
        q_gaps = (q_a - q_b).tolist()
        annual_gap = float(np.mean(norm_monthly[name_a]) - np.mean(norm_monthly[name_b]))
        gap_rows.append([pair_label] + q_gaps + [annual_gap])

    norm_gap_path = os.path.join(output_dir, 'normalized_gap.csv')
    with open(norm_gap_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Pair'] + quarter_labels + [year_label])
        for row in gap_rows:
            writer.writerow(row)

    # --- 4) Normalized 분기 요약 ---
    norm_summary_path = os.path.join(output_dir, 'normalized_quarterly_summary.csv')
    with open(norm_summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Country', 'Base_Value_(t0)'] + quarter_labels + [year_label])
        for name in target_indices:
            annual_avg = float(np.mean(norm_monthly[name]))
            row = [consistent_name(name), base_values[name]] + norm_quarterly[name] + [annual_avg]
            writer.writerow(row)

    print(f"  -> normalized_forecast.csv          (monthly, base=1.0 index)")
    print(f"  -> normalized_gap.csv               (pairwise normalized gap)")
    print(f"  -> normalized_quarterly_summary.csv  (quarterly averages, normalized)")
    return norm_gap_path


def plot_forecast(data, forecast, confidence, col_name, dates_hist, dates_future, output_dir=None, color='RoyalBlue'):
    """개별 국가 예측 플롯 생성 (사이버 보안 스타일)"""
    
    # 음수값 제거
    data, forecast = zero_negative_curves(data, forecast)
    
    # 스타일 설정
    
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
    ax.plot(range(len(d)), d, '-', color=color, label=consistent_name(col_name), linewidth=line_width, zorder=3)
    
    # Forecast 플롯 (Historical 끝에서 이어서)
    forecast_range = range(len(d)-1, (len(d)+len(f))-1)
    ax.plot(forecast_range, f, '-', color=color, linewidth=line_width, zorder=3)
    
    # 95% Confidence Interval
    ax.fill_between(forecast_range, 
                     f - c, 
                     f + c,
                     color=color, alpha=0.2, label='95% CI', zorder=2)
    
    # X축 년도 레이블 (2011~2027, 2011-01부터 시작)
    # 데이터: 180개월 (2011-01 ~ 2025-12) + 12개월 예측 (2026-01 ~ 2026-12)
    x = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027']
    # 각 연도 1월의 인덱스: 2011-01=0, 2012-01=12, 2013-01=24, ...
    positions = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192]
    ax.set_xticks(positions, x)
    
    # Y축 레이블
    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=7)
    
    # 범례
    ax.legend(loc="upper left", prop={'size': 7})
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=7)
    
    # 타이틀
    pyplot.title(consistent_name(col_name), y=1.03, fontsize=18)
    
    # 크기 설정
    fig.set_size_inches(7.487, 3.93)
    
    # 저장
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        images_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'pt_plots')
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
        c_normalized = c / base_value
        
        # Historical 플롯 (인덱스 기반, normalized)
        line_width = 2 if idx == 0 else 1  # 첫 번째 국가는 굵게
        ax.plot(range(len(d_normalized)), d_normalized, '-', color=color, label=col_name, linewidth=line_width, zorder=3)
        
        # Forecast 플롯 (Historical 끝에서 연결, normalized)
        forecast_range = range(len(d_normalized)-1, (len(d_normalized)+len(f_normalized))-1)
        ax.plot(forecast_range, f_normalized, '-', color=color, linewidth=line_width, zorder=3)
        
        # 95% Confidence Interval
        ax.fill_between(forecast_range,
                        f_normalized - c_normalized,
                        f_normalized + c_normalized,
                        color=color, alpha=0.2, zorder=2)
    
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
        
        # Gap shading (background layer)
        ax.fill_between(
            forecast_x, forecast_y_us, forecast_y_jp,
            interpolate=True,
            color=colours_shading[0],
            alpha=0.08,
            zorder=1
        )
        
        ax.fill_between(
            forecast_x, forecast_y_us, forecast_y_kr,
            interpolate=True,
            color=colours_shading[1],
            alpha=0.08,
            zorder=1
        )
    
    # X축 년도 레이블 (2011~2027, 2011-01부터 시작)
    x = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027']
    # 각 연도 1월의 인덱스: 2011-01=0, 2012-01=12, 2013-01=24, ...
    positions = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192]
    ax.set_xticks(positions, x)
    
    # Y축 레이블 (Normalized)
    #ax.set_ylabel("Normalized Index (2011-01 = 1.0)", fontsize=15)
    pyplot.yticks(fontsize=7)
    
    # 범례
    ax.legend(loc="upper left", prop={'size': 7})
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=7)
    
   
    # 크기 설정
    fig.set_size_inches(7.487, 3.93)
    
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
    gap_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'gap')
    pt_plots_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'pt_plots')
    
    for d in [plot_dir, data_out_dir, gap_out_dir, pt_plots_dir]:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(model_file, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
    
    # Ensure all stored device references are consistent
    if hasattr(model, 'gc') and hasattr(model.gc, 'device'):
        model.gc.device = device
    if hasattr(model, 'idx'):
        model.idx = model.idx.to(device)
    for module in model.modules():
        if hasattr(module, 'device') and not isinstance(module.device, torch.device):
            module.device = device
    
    model = model.to(device)
    print(f"✅ Model loaded on {device}")
    
    # Input sequence length (모델에서 설정된 값 사용)
    try:
        seq_len = model.seq_length
    except:
        seq_len = 12  # train_test.py의 기본값
    
    P = seq_len
    print(f"🕹️  Input sequence length: {seq_len} months")
    
    # 초기 입력 (마지막 10개월)
    X_init = torch.from_numpy(dat[-seq_len:, :]).float().to(device)
    
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
                curr_X = torch.from_numpy(new_X[-P:, :]).float().contiguous().to(device)
            
            outputs.append(torch.tensor(np.concatenate(preds, axis=0)))
    
    print("✅ Forecast complete")
    
    # 통계 계산
    outputs = torch.stack(outputs)
    Y = torch.mean(outputs, dim=0)
    std_dev = torch.std(outputs, dim=0)
    confidence = 1.96 * std_dev
    variance = torch.var(outputs, dim=0)
    
    # Denormalization
    scale_torch = torch.from_numpy(scale).float()
    dat_denorm = torch.from_numpy(dat).float() * scale_torch
    Y_denorm = Y * scale_torch
    confidence_denorm = confidence * scale_torch
    variance_denorm = variance * scale_torch
    
    print(f"\n📊 Statistics:")
    print(f"   • Mean forecast: {Y_denorm.mean():.4f}")
    print(f"   • Std: {Y_denorm.std():.4f}")
    print(f"   • Avg confidence width: {confidence_denorm.mean():.4f}")
    
    # 데이터 저장
    print("\n💾 Saving forecast data...")
    save_data(dat_denorm, Y_denorm, confidence_denorm, variance_denorm, col, data_out_dir)
    print("✅ Data saved")
    
    # Smoothing: smooth hist+forecast as one continuous series to avoid junction jump,
    # then apply the same smoothing to confidence so band always wraps the plotted line.
    print("\n🔧 Applying exponential smoothing...")
    alpha_smooth = 0.3
    
    hist_smoothed = []
    fut_smoothed = []
    conf_smoothed = []
    
    for i in range(m):
        hist_arr = dat_denorm[:, i].cpu().numpy().tolist()
        fut_arr = Y_denorm[:, i].cpu().numpy().tolist()
        conf_arr = confidence_denorm[:, i].cpu().numpy().tolist()
        
        full_series = hist_arr + fut_arr
        full_sm = exponential_smoothing(full_series, alpha_smooth)
        hist_smoothed.append(full_sm[:len(hist_arr)])
        fut_smoothed.append(full_sm[len(hist_arr):])
        conf_smoothed.append(exponential_smoothing(conf_arr, alpha_smooth))
    
    hist_plot = torch.tensor(np.array(hist_smoothed)).T
    fut_plot = torch.tensor(np.array(fut_smoothed)).T
    conf_plot = torch.tensor(np.array(conf_smoothed)).T
    
    print("✅ Smoothing complete")
    
    # 예측 날짜 생성 (2026년 1월~12월)
    FORECAST_START = LAST_OBS + pd.DateOffset(months=1)
    dates_future = pd.date_range(start=FORECAST_START, periods=horizon, freq="MS").tolist()
    print(f"📅 Forecast period: {dates_future[0].strftime('%Y-%m')} ~ {dates_future[-1].strftime('%Y-%m')}")
    
    # 플롯 대상 선택 (3개국만)
    target_names = ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx']
    target_indices = [i for i, name in enumerate(col) if name in target_names]
    
    if not target_indices:
        print("⚠️  Warning: Target columns not found")
        target_indices = list(range(min(3, m)))
    
    print(f"\n🎯 Target countries: {[col[i] for i in target_indices]}")
    
    # 플롯 색상 팔레트 (Multi_Country와 동일)
    plot_colours = ["RoyalBlue", "Crimson", "DarkOrange"]
    
    # 플롯 생성
    print("\n📊 Generating plots...")
    
    # 개별 플롯 (색상 매칭)
    for idx, i in enumerate(target_indices):
        color = plot_colours[idx % len(plot_colours)]
        plot_forecast(hist_plot[:, i], fut_plot[:, i], conf_plot[:, i],
                     col[i], dates_hist, dates_future, pt_plots_dir, color)
    
    # 다국가 비교 플롯
    plot_multi_node(hist_plot, fut_plot, conf_plot,
                   target_indices, col, dates_hist, dates_future, plot_dir)
    
    # --- Node-wise CSV (전체 변수 predict/actual) ---
    print("\n📊 Generating node-wise CSV tables...")
    try:
        time_labels = [d.strftime('%Y-%m-%d') for d in dates_future]
        pred_matrix = Y_denorm.detach().cpu().numpy().T

        def save_formatted_csv(data_mat, filename, node_names, time_headers):
            df_table = pd.DataFrame(data=data_mat, columns=time_headers)
            df_table.insert(0, 'Node_Name', node_names)
            out_path = os.path.join(data_out_dir, filename)
            df_table.to_csv(out_path, index=False)
            return out_path

        save_formatted_csv(pred_matrix, 'predict.csv', col, time_labels)
        print(f"  -> predict.csv saved")
    except Exception as e:
        print(f"  CSV 생성 중 오류: {e}")

    # --- 국가 간 환율 예측 격차 (Country Gap, raw) ---
    print("\n📊 Generating country-wise forecast gap CSV files...")
    try:
        save_country_gap(Y_denorm, col, target_names, dates_future, gap_out_dir)
        print("✅ Country gap CSV files saved (raw)")
    except Exception as e:
        print(f"  Country gap 생성 중 오류: {e}")
        import traceback
        traceback.print_exc()

    # --- Normalized gap (단위 통일: base=1.0 index) ---
    print("\n📊 Generating normalized gap CSV files...")
    try:
        save_normalized_gap(dat_denorm, Y_denorm, col, target_names,
                            dates_future, gap_out_dir)
        print("✅ Normalized gap CSV files saved")
    except Exception as e:
        print(f"  Normalized gap 생성 중 오류: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("✅ FORECASTING COMPLETED")
    print("="*70)
    print(f"📁 Output directories:")
    print(f"   • Plots (multi-country): {plot_dir}")
    print(f"   • Individual plots:     {pt_plots_dir}")
    print(f"   • Data:                 {data_out_dir}")
    print(f"   • Gap data:             {gap_out_dir}")
    print(f"\n📊 Generated Files:")
    print(f"   • Multi_Country_Forecast_Normalized.png")
    print(f"   • Individual forecast plots for each country")
    print(f"   • predict_by_country.csv             (monthly forecasts)")
    print(f"   • country_gap.csv                    (pairwise raw gap)")
    print(f"   • country_gap_pct.csv                (pairwise gap as %)")
    print(f"   • country_quarterly_summary.csv      (quarterly averages)")
    print(f"   • normalized_forecast.csv            (base=1.0 index)")
    print(f"   • normalized_gap.csv                 (pairwise normalized gap)")
    print(f"   • normalized_quarterly_summary.csv   (normalized quarterly)")
    print("="*70)