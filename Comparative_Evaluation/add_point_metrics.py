"""Add 2025 test MAE, RMSE, MAPE and sMAPE to comparison_table_2025.csv.

All values are computed from the saved monthly predictions in predictions/<model>/.
RRSE and RAE are recomputed from the same files and checked against the table first.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRED = HERE / "predictions"
TABLE = HERE / "comparison_table_2025.csv"
CURRENCIES = ["US", "KR", "JP"]
METRICS = ["MAE", "RMSE", "MAPE", "sMAPE"]

FILES = {
    "Naive-last": "naive_last/predictions_test_{c}.csv",
    "Drift": "drift/predictions_test_{c}.csv",
    "AR (diff, p=7/7/1)": "ar",
    "LSTM": "lstm_33var/predictions_test_{c}.csv",
    "GRU": "gru/predictions_test_{c}.csv",
    "TCN (causal)": "tcn_33var_causal/predictions_test_{c}.csv",
    "Transformer": "transformer_33var/predictions_test_{c}.csv",
    "MTGNN": "mtgnn_deterministic/predictions_test_{c}.csv",
    "MTGNN + Eq.(7), val alpha": "mtgnn_eq7_val_alpha/predictions_test_val_alpha_{c}.csv",
    "B-MTGNN (Eq.7, val alpha)": "b_mtgnn/paper_ep57_mc10_val_alpha_test_{c}.csv",
    "B-MTGNN (w/o Eq.7)": "b_mtgnn/paper_ep57_mc10_raw_test_{c}.csv",
}


def load(model, c):
    spec = FILES.get(model)
    if spec is None:
        return None
    if spec == "ar":
        df = pd.read_csv(PRED / "ar/ar_predictions_test.csv", encoding="utf-8-sig")
        df = df[df["target"] == c]
    else:
        df = pd.read_csv(PRED / spec.format(c=c), encoding="utf-8-sig")
    return df.sort_values("time").reset_index(drop=True)


def metrics(model, c):
    df = load(model, c)
    y, p = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    e = y - p
    return {
        "RRSE": np.sqrt(np.sum(e**2) / np.sum((y - y.mean()) ** 2)),
        "RAE": np.sum(np.abs(e)) / np.sum(np.abs(y - y.mean())),
        "MAE": np.mean(np.abs(e)),
        "RMSE": np.sqrt(np.mean(e**2)),
        "MAPE": 100 * np.mean(np.abs(e / y)),
        "sMAPE": 100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p))),
    }


def main():
    table = pd.read_csv(TABLE, encoding="utf-8-sig")
    new_cols = [f"test2025_{m}_{c}" for m in METRICS for c in CURRENCIES]
    table = table.drop(columns=[col for col in new_cols if col in table.columns])

    rows = []
    for model in table["model"]:
        row = {}
        for c in CURRENCIES:
            if model not in FILES:
                continue
            m = metrics(model, c)
            for k in ("RRSE", "RAE"):
                saved = table.loc[table["model"] == model, f"test2025_{k}_{c}"].item()
                assert abs(m[k] - saved) < 1e-6, (model, c, k, m[k], saved)
            for k in METRICS:
                row[f"test2025_{k}_{c}"] = m[k]
        rows.append(row)
    added = pd.DataFrame(rows, columns=new_cols)

    insert_at = table.columns.get_loc("test2025_RAE_JP") + 1
    out = pd.concat([table.iloc[:, :insert_at], added, table.iloc[:, insert_at:]], axis=1)
    out.to_csv(TABLE, index=False)
    print(out[["model"] + new_cols].to_string())


if __name__ == "__main__":
    main()
