"""
LSTM Model Training — Fixed & Improved
Key fixes:
  1. Correct inverse-transform: only scale the target column independently
  2. Per-state MinMaxScaler (no data leakage across states)
  3. EarlyStopping + ReduceLROnPlateau callbacks
  4. Lag_7 and Lag_14 added as features (matches feature_engineering.py)
  5. Robust fallback when val sequences are too short
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.get_logger().setLevel("ERROR")

# ── Config ──────────────────────────────────────────────────────────────────
SEQ_LEN      = 8        # look-back window (weeks)
EPOCHS       = 60
BATCH_SIZE   = 16
MIN_TRAIN    = 50       # minimum rows to attempt training

FEATURE_COLS = [
    "Lag_1", "Lag_7", "Lag_14",
    "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Mean_30",
    "Year", "Month", "WeekOfYear", "DayOfWeek", "IsHoliday",
]
TARGET = "Total"

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Return (X_seq, y_seq) arrays of shape (N, seq_len, features) and (N,)."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def train_lstm(train_path: str, val_path: str, out_dir: str, results_path: str):
    print("=" * 70)
    print("LSTM Training  (Fixed & Improved)")
    print("=" * 70)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    # Keep only columns that exist in both dataframes
    avail_features = [c for c in FEATURE_COLS if c in train_df.columns]
    print(f"Features used: {avail_features}\n")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    results = []
    states  = sorted(train_df["State"].unique())
    print(f"Training LSTM on {len(states)} states …\n")

    for idx, state in enumerate(states, 1):
        tr = train_df[train_df["State"] == state].copy().reset_index(drop=True)
        va = val_df[val_df["State"] == state].copy().reset_index(drop=True)

        if len(tr) < MIN_TRAIN:
            print(f"  ⚠  {state}: too little training data ({len(tr)} rows) — skipped")
            continue

        # ── Per-state scaling (no leakage between states) ──────────────────
        feat_scaler   = MinMaxScaler()
        target_scaler = MinMaxScaler()

        X_tr_scaled = feat_scaler.fit_transform(tr[avail_features].values)
        y_tr_scaled = target_scaler.fit_transform(tr[[TARGET]].values).ravel()

        X_va_scaled = feat_scaler.transform(va[avail_features].values)
        y_va_scaled = target_scaler.transform(va[[TARGET]].values).ravel()

        # ── Build sequences ────────────────────────────────────────────────
        X_train, y_train = make_sequences(X_tr_scaled, y_tr_scaled, SEQ_LEN)
        X_val,   y_val   = make_sequences(X_va_scaled, y_va_scaled, SEQ_LEN)

        if len(X_train) < 10:
            print(f"  ⚠  {state}: not enough sequences after windowing — skipped")
            continue

        # ── Train ──────────────────────────────────────────────────────────
        model = build_model((SEQ_LEN, len(avail_features)))

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=0),
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
            shuffle=False,
        )

        # ── Evaluate ───────────────────────────────────────────────────────
        if len(X_val) == 0:
            print(f"  ⚠  {state}: no validation sequences — model saved without metrics")
            model.save(os.path.join(out_dir, f"lstm_{state}.h5"))
            joblib.dump(feat_scaler,   os.path.join(out_dir, f"feat_scaler_{state}.pkl"))
            joblib.dump(target_scaler, os.path.join(out_dir, f"target_scaler_{state}.pkl"))
            continue

        pred_scaled = model.predict(X_val, verbose=0).ravel()

        # ── Correct inverse transform (target-only scaler) ─────────────────
        pred   = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        actual = target_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()

        mae  = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = float(np.mean(np.abs((actual - pred) / np.where(actual == 0, 1, actual))) * 100)

        results.append({
            "State":    state,
            "MAE":      round(mae, 2),
            "RMSE":     round(rmse, 2),
            "MAPE(%)":  round(mape, 2),
        })

        # ── Save model + scalers ───────────────────────────────────────────
        model.save(os.path.join(out_dir, f"lstm_{state}.h5"))
        joblib.dump(feat_scaler,   os.path.join(out_dir, f"feat_scaler_{state}.pkl"))
        joblib.dump(target_scaler, os.path.join(out_dir, f"target_scaler_{state}.pkl"))

        print(f"  ✅ [{idx:2d}/{len(states)}] {state:<20} | MAPE: {mape:6.2f}%  MAE: {mae:.0f}  RMSE: {rmse:.0f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print("\n" + "=" * 70)
        print("LSTM PERFORMANCE SUMMARY")
        print("=" * 70)
        print(results_df.describe().round(2))
        print("\nTop 5 Best States (lowest MAPE):")
        print(results_df.nsmallest(5, "MAPE(%)")[["State", "MAPE(%)"]].to_string(index=False))

    results_df.to_csv(results_path, index=False)
    print(f"\n✅ LSTM Training Completed!  Results → {results_path}")
    return results_df


if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..", "..")
    train_lstm(
        train_path   = os.path.join(BASE, "train_lstm_processed.csv"),
        val_path     = os.path.join(BASE, "val_lstm_processed.csv"),
        out_dir      = os.path.join(BASE, "models", "lstm"),
        results_path = os.path.join(BASE, "lstm_results.csv"),
    )