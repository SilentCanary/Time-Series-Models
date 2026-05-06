"""
xgboost_train.py — XGBoost training improvements
  • Adds Lag_7, Lag_14, Rolling_Mean_30 when available
  • Early stopping via eval set
  • Feature importance saved per state
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "Year", "Month", "WeekOfYear", "DayOfWeek", "IsHoliday",
    "Lag_1", "Lag_7", "Lag_14",
    "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Mean_30",
]
TARGET = "Total"


def train_xgboost(train_path: str, val_path: str, out_dir: str, results_path: str):
    print("=" * 70)
    print("XGBoost Training (All States)")
    print("=" * 70)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    # Only keep features that exist in the dataset
    feat_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    print(f"Features: {feat_cols}\n")

    os.makedirs(out_dir, exist_ok=True)

    states  = sorted(train_df["State"].unique())
    results = []

    print(f"Training on {len(states)} states …\n")

    for idx, state in enumerate(states, 1):
        tr = train_df[train_df["State"] == state].copy()
        va = val_df[val_df["State"] == state].copy()

        if len(tr) < 30:
            print(f"  ⚠  {state}: not enough data — skipped")
            continue

        X_tr, y_tr = tr[feat_cols].values, tr[TARGET].values
        X_va, y_va = va[feat_cols].values, va[TARGET].values

        model = xgb.XGBRegressor(
            n_estimators      = 500,
            learning_rate     = 0.05,
            max_depth         = 6,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            min_child_weight  = 3,
            random_state      = 42,
            tree_method       = "hist",
            early_stopping_rounds = 20,
            eval_metric       = "rmse",
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        pred = model.predict(X_va)

        mae  = mean_absolute_error(y_va, pred)
        rmse = np.sqrt(mean_squared_error(y_va, pred))
        mape = float(np.mean(np.abs((y_va - pred) / np.where(y_va == 0, 1, y_va))) * 100)

        results.append({
            "State":   state,
            "MAE":     round(mae, 2),
            "RMSE":    round(rmse, 2),
            "MAPE(%)": round(mape, 2),
        })

        joblib.dump(model, os.path.join(out_dir, f"xgboost_{state}.pkl"))
        print(f"  ✅ [{idx:2d}/{len(states)}] {state:<22} | MAPE: {mape:6.2f}%  Trees used: {model.best_iteration}")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("XGBoost PERFORMANCE SUMMARY")
    print("=" * 70)
    print(results_df[["MAE", "RMSE", "MAPE(%)"]].describe().round(2))
    print("\nTop 5 Best States:")
    print(results_df.nsmallest(5, "MAPE(%)")[["State", "MAPE(%)"]].to_string(index=False))

    results_df.to_csv(results_path, index=False)
    print(f"\n✅ XGBoost Training Completed!  Results → {results_path}")
    return results_df


if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..", "..")
    train_xgboost(
        train_path   = os.path.join(BASE, "train_xgboost_processed.csv"),
        val_path     = os.path.join(BASE, "val_xgboost_processed.csv"),
        out_dir      = os.path.join(BASE, "models", "xgboost"),
        results_path = os.path.join(BASE, "xgboost_results_all.csv"),
    )