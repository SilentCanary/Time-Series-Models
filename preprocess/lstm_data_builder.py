"""
lstm_data_builder.py
────────────────────
Builds LSTM-ready sequences from weekly state-level sales data.

Flow (sequence-first, then split):
    raw data per state
        ↓
    create ALL sequences per state   (window=12 → ~244 seq from 256 pts)
        ↓
    80/20 split on sequences         (chronological, not on raw points!)
        ↓
    save per-state scalers           (feat_scaler_{state}.pkl + target_scaler_{state}.pkl)
    save per-state .npy arrays       (X_train_{state}.npy etc.)  ← per-state eval
    save global stacked arrays       (X_train_lstm.npy etc.)     ← for training

Matches main.py which loads per state:
    lstm_{state}.h5
    feat_scaler_{state}.pkl
    target_scaler_{state}.pkl

Usage:
    python lstm_data_builder.py
    python lstm_data_builder.py --seq_len 8
    python lstm_data_builder.py --seq_len 12 --val_split 0.2 --scalers_dir ../models/lstm
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ── Must match FEATURE_COLS_LSTM in main.py ───────────────────────────────────
FEATURE_COLS = [
    "Lag_1", "Lag_7", "Lag_14",
    "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Mean_30",
    "Year", "Month", "WeekOfYear", "DayOfWeek", "IsHoliday",
]
TARGET_COL = "Total"


def build_sequences(features: np.ndarray, target: np.ndarray, seq_len: int):
    """
    Sliding window: each sample = seq_len timesteps → predict next step.
    features : (T, F) — already scaled
    target   : (T,)   — already scaled
    returns X (N, seq_len, F), y (N,)
    """
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(target[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(
    train_csv:    str   = "train_data.csv",
    val_csv:      str   = "val_data.csv",
    seq_len:      int   = 12,
    val_split:    float = 0.2,
    out_dir:      str   = "lstm_data",
    scalers_dir:  str   = None,   # defaults to out_dir; set to models/lstm/ for API
):
    scalers_dir = scalers_dir or out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(scalers_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  LSTM Data Builder  (sequence-first, then split)")
    print(f"  seq_len={seq_len} | val_split={val_split:.0%}")
    print(f"{'='*55}\n")

    # ── 1. Load & combine (we re-split properly per state) ────────────────────
    train = pd.read_csv(train_csv, parse_dates=["Date"])
    val   = pd.read_csv(val_csv,   parse_dates=["Date"])
    df    = pd.concat([train, val], ignore_index=True)
    df    = df.sort_values(["State", "Date"]).reset_index(drop=True)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    pts = df.groupby("State").size()
    print(f"Combined shape  : {df.shape}")
    print(f"States          : {df['State'].nunique()}")
    print(f"Points/state    : {pts.min()} – {pts.max()}")
    print(f"Sequences/state : ~{pts.min() - seq_len}  (with window={seq_len})")
    print()

    # ── 2. Per state: scale → all sequences → split ───────────────────────────
    X_train_all, y_train_all = [], []
    X_val_all,   y_val_all   = [], []
    summary = []

    for state, grp in df.groupby("State"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        T   = len(grp)

        if T < seq_len + 2:
            print(f"  ⚠  {state}: only {T} points — skipping (need > {seq_len + 2})")
            continue

        # Fit scalers only on training portion of raw points (no look-ahead)
        fit_end = int(T * (1 - val_split))
        feat_scaler   = MinMaxScaler()
        target_scaler = MinMaxScaler()
        feat_scaler.fit(grp[FEATURE_COLS].values[:fit_end])
        target_scaler.fit(grp[[TARGET_COL]].values[:fit_end])

        feats_scaled  = feat_scaler.transform(grp[FEATURE_COLS].values)
        target_scaled = target_scaler.transform(grp[[TARGET_COL]].values).flatten()

        # Build ALL sequences first
        X_all, y_all = build_sequences(feats_scaled, target_scaled, seq_len)
        N = len(X_all)

        # Then split chronologically
        split_idx = int(N * (1 - val_split))
        X_tr, y_tr = X_all[:split_idx], y_all[:split_idx]
        X_vl, y_vl = X_all[split_idx:], y_all[split_idx:]

        # Save per-state scalers — main.py loads these by name
        joblib.dump(feat_scaler,   os.path.join(scalers_dir, f"feat_scaler_{state}.pkl"))
        joblib.dump(target_scaler, os.path.join(scalers_dir, f"target_scaler_{state}.pkl"))

        # Save per-state sequences — useful for per-state MAPE evaluation
        np.save(os.path.join(out_dir, f"X_train_{state}.npy"), X_tr)
        np.save(os.path.join(out_dir, f"y_train_{state}.npy"), y_tr)
        np.save(os.path.join(out_dir, f"X_val_{state}.npy"),   X_vl)
        np.save(os.path.join(out_dir, f"y_val_{state}.npy"),   y_vl)

        X_train_all.append(X_tr);  y_train_all.append(y_tr)
        X_val_all.append(X_vl);    y_val_all.append(y_vl)

        summary.append({
            "State":           state,
            "total_points":    T,
            "total_sequences": N,
            "train_sequences": len(X_tr),
            "val_sequences":   len(X_vl),
        })

    # ── 3. Stack global arrays for training ───────────────────────────────────
    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_val   = np.concatenate(X_val_all,   axis=0)
    y_val   = np.concatenate(y_val_all,   axis=0)

    np.save(os.path.join(out_dir, "X_train_lstm.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train_lstm.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val_lstm.npy"),   X_val)
    np.save(os.path.join(out_dir, "y_val_lstm.npy"),   y_val)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "lstm_sequence_summary.csv"), index=False)

    # ── 4. Print summary ──────────────────────────────────────────────────────
    print(f"{'─'*50}")
    print(f"  model input_shape     : (None, {seq_len}, {len(FEATURE_COLS)})")
    print(f"  Features              : {FEATURE_COLS}")
    print()
    print(f"  Global train          : {X_train.shape}  →  {len(X_train):,} sequences")
    print(f"  Global val            : {X_val.shape}   →  {len(X_val):,} sequences")
    print()
    print(f"  Per-state train seqs  : ~{summary_df['train_sequences'].mean():.0f} (mean)")
    print(f"  Per-state val seqs    : ~{summary_df['val_sequences'].mean():.0f}  (mean)")
    print(f"{'─'*50}")
    print(f"\n  Output: {out_dir}/")
    print(f"    X_train_lstm.npy / y_train_lstm.npy    ← global training")
    print(f"    X_val_lstm.npy   / y_val_lstm.npy      ← global validation")
    print(f"    X_train_{{state}}.npy / X_val_{{state}}.npy  ← per-state eval")
    print(f"  Scalers: {scalers_dir}/")
    print(f"    feat_scaler_{{state}}.pkl                ← loaded by main.py")
    print(f"    target_scaler_{{state}}.pkl              ← loaded by main.py")
    print(f"\n✅ Done!")

    return X_train, y_train, X_val, y_val


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prepare_data(
        train_csv="train_data.csv",
        val_csv="val_data.csv",
        seq_len=12,
        val_split=0.2,
        out_dir="lstm_data",
        scalers_dir="models/lstm"
    )