from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
from pathlib import Path

def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
  
def plot_exponential_smoothing(series, alphas, attack):
 
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing - "+attack)
    plt.grid(True);

def double_exponential_smoothing(series, alpha, beta):
    """Holt 이중 지수평활. series는 1차원 배열/리스트. 반환 길이 len(series)+1 (마지막은 한 스텝 예측)."""
    series = np.asarray(series, dtype=float)
    if len(series) == 0:
        return []
    if len(series) == 1:
        return [series[0], series[0]]

    result = [series[0]]
    level, trend = series[0], series[1] - series[0]
    for n in range(1, len(series) + 1):
        if n >= len(series):  # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def smooth_csv_file(input_path, output_path, alpha=0.1, beta=0.3):
    """
    Date + 숫자 컬럼 CSV에 이중 지수평활 적용 후 동일 형식으로 저장.
    학습 파이프라인에서 호출해 data/data.csv -> data/sm_data.csv 생성 시 사용.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_csv(input_path)

    date_col = None
    if "Date" in df.columns:
        date_col = df["Date"].copy()
        data = df.drop(columns=["Date"])
    elif df.columns[0].lower() == "date":
        date_col = df.iloc[:, 0].copy()
        data = df.iloc[:, 1:]
    else:
        data = df

    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)
    smoothed = {}
    for col in data.columns:
        ser = data[col].values
        res = double_exponential_smoothing(ser, alpha, beta)
        smoothed[col] = res[:-1]  # forecasting 한 스텝 제외

    out_df = pd.DataFrame(smoothed)
    if date_col is not None:
        out_df.insert(0, "Date", date_col.values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Smoothed data saved: {output_path} (alpha={alpha}, beta={beta})")
    return str(output_path)

def plot_double_exponential_smoothing(series, alphas, betas, attack):
     
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
    plt.plot(series, label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Double Exponential Smoothing - "+attack)
    plt.grid(True)
    plt.show()

# 스크립트 직접 실행 시: data/data.txt 있으면 탭 구분 스무딩, 없으면 data/data.csv -> data/sm_data.csv
if __name__ == "__main__":
    alpha = float(0.1)
    beta = float(0.3)
    data_dir = Path("data")
    txt_file = data_dir / "data.txt"
    csv_file = data_dir / "data.csv"
    out_csv = data_dir / "sm_data.csv"

    if txt_file.exists():
        fin = open(txt_file)
        rawdat = np.loadtxt(fin, delimiter="\t")
        fin.close()
        print(rawdat.shape)
        smoothed = []
        for r in rawdat.transpose():
            smoothed.append(double_exponential_smoothing(r, alpha, beta))
        smoothed = list(map(list, zip(*smoothed)))
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(smoothed[:-1])
        print(f"Smoothed (from data.txt) saved: {out_csv}")
    elif csv_file.exists():
        smooth_csv_file(csv_file, out_csv, alpha=alpha, beta=beta)
    else:
        print("Neither data/data.txt nor data/data.csv found. Run with input path or create data file.")


