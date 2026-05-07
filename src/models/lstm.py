
import os
import warnings
import numpy as np
import pandas as pd
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

tf.get_logger().setLevel("ERROR")

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN      = 12       # must match what you used in lstm_data_builder.py
EPOCHS       = 60
BATCH_SIZE   = 32       # larger batch is fine now — we have 8k+ sequences
N_FEATURES   = 11       # (None, 12, 11) — from lstm_data_builder output

DATA_DIR     = "../../preprocess/lstm_data"               # output of lstm_data_builder.py
SCALERS_DIR  = os.path.join("../../preprocess", "models", "lstm")   # where scalers were saved
OUT_DIR      = os.path.join("../../", "models", "lstm")   # where to save .h5 files
RESULTS_PATH = "lstm_results.csv"


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(seq_len: int, n_features: int):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(data_dir: str, scalers_dir: str, out_dir: str, results_path: str):
    print("=" * 65)
    print("  LSTM Training  (loads pre-built sequences)")
    print(f"  seq_len={SEQ_LEN} | epochs={EPOCHS} | batch={BATCH_SIZE}")
    print("=" * 65)

    os.makedirs(out_dir, exist_ok=True)

    # ── Load global arrays (used to infer which states exist) ─────────────────
    X_train_global = np.load(os.path.join(data_dir, "X_train_lstm.npy"))
    y_train_global = np.load(os.path.join(data_dir, "y_train_lstm.npy"))
    X_val_global   = np.load(os.path.join(data_dir, "X_val_lstm.npy"))
    y_val_global   = np.load(os.path.join(data_dir, "y_val_lstm.npy"))

    print(f"\n  Global train : {X_train_global.shape}")
    print(f"  Global val   : {X_val_global.shape}\n")

    # ── Phase 1: Train ONE global LSTM on all states ──────────────────────────
    print("── Phase 1: Global LSTM training ──")
    global_model = build_model(SEQ_LEN, N_FEATURES)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=0),
    ]

    global_model.fit(
        X_train_global, y_train_global,
        validation_data=(X_val_global, y_val_global),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,       # safe to shuffle globally — state info is in features
        verbose=1,
    )
    print("  ✅ Global model trained\n")

    # ── Phase 2: Evaluate & fine-tune per state ───────────────────────────────
    print("── Phase 2: Per-state fine-tune & evaluate ──\n")

    # Discover states from saved per-state .npy files
    state_files = [
        f.replace("X_train_", "").replace(".npy", "")
        for f in os.listdir(data_dir)
        if f.startswith("X_train_") and not f.startswith("X_train_lstm")
    ]
    states = sorted(state_files)
    print(f"  States found: {len(states)}\n")

    results = []

    for idx, state in enumerate(states, 1):
        X_tr = np.load(os.path.join(data_dir, f"X_train_{state}.npy"))
        y_tr = np.load(os.path.join(data_dir, f"y_train_{state}.npy"))
        X_vl = np.load(os.path.join(data_dir, f"X_val_{state}.npy"))
        y_vl = np.load(os.path.join(data_dir, f"y_val_{state}.npy"))

        # Load per-state target scaler (for inverse transform)
        target_scaler_path = os.path.join(scalers_dir, f"target_scaler_{state}.pkl")
        if not os.path.exists(target_scaler_path):
            print(f"  ⚠  {state}: scaler not found — skipping")
            continue

        target_scaler = joblib.load(target_scaler_path)

        # Fine-tune: start from global weights, brief training on state data
        state_model = build_model(SEQ_LEN, N_FEATURES)
        state_model.set_weights(global_model.get_weights())   # warm start

        if len(X_tr) >= 10:
            state_model.fit(
                X_tr, y_tr,
                validation_data=(X_vl, y_vl) if len(X_vl) > 0 else None,
                epochs=20,          # short fine-tune — global model already learned patterns
                batch_size=16,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
                shuffle=False,      # keep chronological order within a state
                verbose=0,
            )

        # Evaluate on val sequences
        if len(X_vl) == 0:
            print(f"  ⚠  {state}: no val sequences — saved without metrics")
            state_model.save(os.path.join(out_dir, f"lstm_{state}.h5"))
            continue

        pred_scaled = state_model.predict(X_vl, verbose=0).ravel()
        pred   = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        actual = target_scaler.inverse_transform(y_vl.reshape(-1, 1)).ravel()

        mae  = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = float(np.mean(np.abs((actual - pred) / np.where(actual == 0, 1, actual))) * 100)

        results.append({"State": state, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE(%)": round(mape, 2)})

        state_model.save(os.path.join(out_dir, f"lstm_{state}.h5"))

        print(f"  ✅ [{idx:2d}/{len(states)}] {state:<22} MAPE: {mape:6.2f}%  RMSE: {rmse:,.0f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print("\n" + "=" * 65)
        print("  LSTM RESULTS SUMMARY")
        print("=" * 65)
        print(f"  Mean MAPE : {results_df['MAPE(%)'].mean():.2f}%")
        print(f"  Best state: {results_df.loc[results_df['MAPE(%)'].idxmin(), 'State']}  "
              f"({results_df['MAPE(%)'].min():.2f}%)")
        print(f"  Worst state: {results_df.loc[results_df['MAPE(%)'].idxmax(), 'State']}  "
              f"({results_df['MAPE(%)'].max():.2f}%)")

    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Done!  Results saved → {results_path}")
    return results_df


if __name__ == "__main__":
    train(
        data_dir     = DATA_DIR,
        scalers_dir  = SCALERS_DIR,
        out_dir      = OUT_DIR,
        results_path = RESULTS_PATH,
    )