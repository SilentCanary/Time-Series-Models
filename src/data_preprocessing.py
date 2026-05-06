"""
data_preprocessing.py — Unified preprocessing for LSTM & XGBoost
Improvements over original:
  • Reads directly from the raw Excel file
  • Fills missing dates per state (ensures contiguous weekly series)
  • Forward-fills / back-fills missing values
  • Adds Lag_7, Lag_14, Rolling_Mean_30 (required by improved LSTM)
  • Saves separate processed CSVs for LSTM and XGBoost
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import holidays

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_COLS_LSTM = [
    "Lag_1", "Lag_7", "Lag_14",
    "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Mean_30",
    "Year", "Month", "WeekOfYear", "DayOfWeek", "IsHoliday",
]
FEATURE_COLS_XGB = [
    "Year", "Month", "WeekOfYear", "DayOfWeek", "IsHoliday",
    "Lag_1", "Rolling_Mean_7", "Rolling_Std_7",
]
TARGET = "Total"


def load_and_clean(xlsx_path: str) -> pd.DataFrame:
    """Load raw Excel, aggregate to weekly-state level, fill missing dates."""
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]

    # Detect date column
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise ValueError("No date column found in Excel file")
    df[date_col] = pd.to_datetime(df[date_col])
    df.rename(columns={date_col: "Date"}, inplace=True)

    # Detect state column
    state_col = next((c for c in df.columns if "state" in c.lower()), None)
    if state_col is None:
        raise ValueError("No state column found in Excel file")
    df.rename(columns={state_col: "State"}, inplace=True)

    # Detect sales column (numeric, non-date, non-state)
    skip = {"Date", "State"}
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in skip]
    if "Total" in num_cols:
        pass
    elif len(num_cols) == 1:
        df.rename(columns={num_cols[0]: "Total"}, inplace=True)
    else:
        # Use the column with the highest variance as target
        best = max(num_cols, key=lambda c: df[c].std())
        df.rename(columns={best: "Total"}, inplace=True)

    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df = df[["Date", "State", "Total"]].dropna(subset=["Total"])

    # Aggregate to weekly level (sum per state per date)
    df = df.groupby(["Date", "State"], as_index=False)["Total"].sum()
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)

    # Fill missing dates per state with forward-fill
    states = df["State"].unique()
    min_date, max_date = df["Date"].min(), df["Date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="W")

    filled = []
    for state in states:
        s = df[df["State"] == state].set_index("Date").reindex(all_dates)
        s["Total"] = s["Total"].fillna(method="ffill").fillna(method="bfill")
        s["State"] = state
        s.index.name = "Date"
        filled.append(s.reset_index())
    df = pd.concat(filled, ignore_index=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time + lag + rolling features. No data leakage between states."""
    us_hols = holidays.US(years=range(df["Date"].dt.year.min(),
                                      df["Date"].dt.year.max() + 2))

    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["IsHoliday"]  = df["Date"].dt.date.apply(lambda d: d in us_hols).astype(int)

    def per_state_features(g):
        g = g.sort_values("Date").copy()
        g["Lag_1"]          = g["Total"].shift(1)
        g["Lag_7"]          = g["Total"].shift(7)
        g["Lag_14"]         = g["Total"].shift(14)
        g["Rolling_Mean_7"] = g["Total"].rolling(7,  min_periods=1).mean()
        g["Rolling_Std_7"]  = g["Total"].rolling(7,  min_periods=1).std().fillna(0)
        g["Rolling_Mean_30"]= g["Total"].rolling(30, min_periods=1).mean()
        return g

    df = df.groupby("State", group_keys=False).apply(per_state_features)
    df = (df.groupby("State", group_keys=False)
            .apply(lambda x: x.fillna(method="bfill").fillna(method="ffill").fillna(0)))

    return df.reset_index(drop=True)


def time_split(df: pd.DataFrame, n_val_periods: int = 8):
    """Chronological train/val split (no leakage)."""
    dates = sorted(df["Date"].unique())
    cutoff = dates[-(n_val_periods + 1)]
    train = df[df["Date"] <= cutoff].copy()
    val   = df[df["Date"] >  cutoff].copy()
    return train, val


def preprocess_for_lstm(train: pd.DataFrame, val: pd.DataFrame,
                        models_dir: str, out_dir: str):

    cols = [c for c in FEATURE_COLS_LSTM + [TARGET] if c in train.columns]

    train_out = train[["State", "Date"] + cols].copy()
    val_out   = val[["State", "Date"] + cols].copy()

    train_out.to_csv(os.path.join(out_dir, "train_lstm_processed.csv"), index=False)
    val_out.to_csv(  os.path.join(out_dir, "val_lstm_processed.csv"),   index=False)

    print("✅ LSTM processed data saved (NO scaling — handled in training)")


def preprocess_for_xgboost(train: pd.DataFrame, val: pd.DataFrame,
                            models_dir: str, out_dir: str):
    """XGBoost uses raw (unscaled) features."""
    cols = [c for c in FEATURE_COLS_XGB + [TARGET, "State", "Date"] if c in train.columns]

    scaler = MinMaxScaler()
    scale_cols = [c for c in FEATURE_COLS_XGB + [TARGET] if c in train.columns]
    train_s = train[cols].copy()
    val_s   = val[cols].copy()
    train_s[scale_cols] = scaler.fit_transform(train[scale_cols])
    val_s[scale_cols]   = scaler.transform(val[scale_cols])

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler_xgboost.pkl"))

    train_s.to_csv(os.path.join(out_dir, "train_xgboost_processed.csv"), index=False)
    val_s.to_csv(  os.path.join(out_dir, "val_xgboost_processed.csv"),   index=False)
    print("✅ XGBoost processed data saved")


if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..")

    xlsx =  "../preprocess/Forecasting Case- Study.xlsx"
    print("Loading raw data …")
    df = load_and_clean(xlsx)
    print(f"  Shape after cleaning: {df.shape}")

    print("Engineering features …")
    df = add_features(df)

    df.to_csv("../preprocess/featured_sales_data.csv", index=False)
    print("  featured_sales_data.csv saved")

    train, val = time_split(df, n_val_periods=8)
    train.to_csv(os.path.join(BASE, "preprocess", "train_data.csv"), index=False)
    val.to_csv(  os.path.join(BASE, "preprocess", "val_data.csv"),   index=False)
    print(f"  Train: {train.shape}  Val: {val.shape}")

    preprocess_for_lstm(train, val,
                        models_dir=os.path.join(BASE, "models"),
                        out_dir=BASE)
    preprocess_for_xgboost(train, val,
                           models_dir=os.path.join(BASE, "models"),
                           out_dir=BASE)

    print("\n✅ All preprocessing complete!")