"""
main.py — Production-ready FastAPI forecasting service

Endpoints
---------
GET  /health          — liveness check
GET  /states          — list all available states
GET  /models          — best model per state with metrics
POST /predict         — 8-week forecast for a given state
POST /predict/batch   — forecast for multiple states in one call

LSTM inference uses per-state scalers saved during training.
XGBoost inference builds recursive lag features from the last
known values instead of the placeholder 1% trend used before.
"""

import os
import pickle
import warnings
from datetime import datetime, timedelta
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Paths (relative to this file) ─────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.join(HERE, "..", "..")
MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR  = os.path.join(ROOT, "preprocess")
BEST_CSV  = os.path.join(ROOT, "best_model_per_state.csv")


# ── Load metadata ─────────────────────────────────────────────────────────────
best_models_df = pd.read_csv(BEST_CSV)
best_models    = dict(zip(best_models_df["State"], best_models_df["Model"]))

# Pre-load last known values for XGBoost recursive inference
_raw_train = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"), parse_dates=["Date"])

FEATURE_COLS_XGB = [
    "Year", "Month", "WeekOfYear", "DayOfWeek", "IsHoliday",
    "Lag_1", "Lag_7", "Lag_14",
    "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Mean_30",
]
FEATURE_COLS_LSTM = [
    "Lag_1", "Lag_7", "Lag_14",
    "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Mean_30",
    "Year", "Month", "WeekOfYear", "DayOfWeek", "IsHoliday",
]
SEQ_LEN = 8


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Beverage Sales Forecast API",
    description = "Production-ready multi-model forecasting system (SARIMA / Prophet / XGBoost / LSTM)",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    state: str
    weeks: int = Field(default=8, ge=1, le=52)
    model_override: Optional[str] = Field(
        default=None,
        description="Force a specific model: SARIMA | Prophet | XGBoost | LSTM",
    )


class BatchPredictRequest(BaseModel):
    states: List[str]
    weeks:  int = Field(default=8, ge=1, le=52)


class WeekPrediction(BaseModel):
    week:              int
    date:              str
    forecasted_sales:  float


class PredictResponse(BaseModel):
    state:       str
    best_model:  str
    predictions: List[WeekPrediction]


# ── Inference helpers ─────────────────────────────────────────────────────────

def _xgboost_predict(state: str, n_weeks: int) -> List[float]:
    """Recursive multi-step XGBoost forecast using real lag features."""
    model = joblib.load(os.path.join(MODEL_DIR, "xgboost", f"xgboost_{state}.pkl"))

    state_df = (_raw_train[_raw_train["State"] == state]
                .sort_values("Date").copy())

    # Determine which feature cols this model was trained on
    try:
        trained_feat_cols = model.get_booster().feature_names
        if trained_feat_cols is None:
            raise ValueError
    except Exception:
        trained_feat_cols = [c for c in FEATURE_COLS_XGB if c in state_df.columns]

    recent = state_df["Total"].values.tolist()
    preds  = []
    last_date = state_df["Date"].iloc[-1]

    for i in range(1, n_weeks + 1):
        next_date = last_date + timedelta(weeks=i)

        row = {
            "Year":      next_date.year,
            "Month":     next_date.month,
            "WeekOfYear":next_date.isocalendar()[1],
            "DayOfWeek": next_date.weekday(),
            "IsHoliday": 0,  # simplification; can integrate holidays pkg
            "Lag_1":     recent[-1]  if len(recent) >= 1  else 0,
            "Lag_7":     recent[-7]  if len(recent) >= 7  else 0,
            "Lag_14":    recent[-14] if len(recent) >= 14 else 0,
            "Rolling_Mean_7":  float(np.mean(recent[-7:]))   if len(recent) >= 7  else float(np.mean(recent)),
            "Rolling_Std_7":   float(np.std(recent[-7:]))    if len(recent) >= 7  else 0.0,
            "Rolling_Mean_30": float(np.mean(recent[-30:]))  if len(recent) >= 30 else float(np.mean(recent)),
        }

        X = pd.DataFrame([row])
        X = X[[c for c in trained_feat_cols if c in X.columns]]

        pred = float(model.predict(X)[0])
        preds.append(pred)
        recent.append(pred)

    return preds


def _sarima_predict(state: str, n_weeks: int) -> List[float]:
    model = joblib.load(os.path.join(MODEL_DIR, "arima", f"sarima_{state}.pkl"))
    return model.forecast(steps=n_weeks).tolist()


def _prophet_predict(state: str, n_weeks: int) -> List[float]:
    with open(os.path.join(MODEL_DIR, "prophet", f"prophet_{state}.pkl"), "rb") as f:
        model = pickle.load(f)
    future   = model.make_future_dataframe(periods=n_weeks, freq="W")
    forecast = model.predict(future)
    return forecast["yhat"].tail(n_weeks).tolist()


def _lstm_predict(state: str, n_weeks: int) -> List[float]:
    """LSTM inference using per-state scalers from improved training."""
    import tensorflow as tf

    lstm_dir      = os.path.join(MODEL_DIR, "lstm")
    model_path    = os.path.join(lstm_dir, f"lstm_{state}.h5")
    feat_scaler   = joblib.load(os.path.join(lstm_dir, f"feat_scaler_{state}.pkl"))
    target_scaler = joblib.load(os.path.join(lstm_dir, f"target_scaler_{state}.pkl"))

    model = tf.keras.models.load_model(model_path, compile=False)

    state_df = (_raw_train[_raw_train["State"] == state]
                .sort_values("Date").copy())

    feat_cols = [c for c in FEATURE_COLS_LSTM if c in state_df.columns]
    recent_X = feat_scaler.transform(state_df[feat_cols].values[-SEQ_LEN:])

    preds      = []
    recent_raw = state_df["Total"].values.tolist()
    last_date  = state_df["Date"].iloc[-1]

    for i in range(1, n_weeks + 1):
        X_seq = recent_X[-SEQ_LEN:].reshape(1, SEQ_LEN, len(feat_cols))
        pred_scaled = model.predict(X_seq, verbose=0)[0, 0]
        pred_val    = float(target_scaler.inverse_transform([[pred_scaled]])[0, 0])
        preds.append(pred_val)

        next_date = last_date + timedelta(weeks=i)
        next_row_raw = {
            "Lag_1":          recent_raw[-1]  if len(recent_raw) >= 1  else 0,
            "Lag_7":          recent_raw[-7]  if len(recent_raw) >= 7  else 0,
            "Lag_14":         recent_raw[-14] if len(recent_raw) >= 14 else 0,
            "Rolling_Mean_7": float(np.mean(recent_raw[-7:]))  if len(recent_raw) >= 7  else float(np.mean(recent_raw)),
            "Rolling_Std_7":  float(np.std(recent_raw[-7:]))   if len(recent_raw) >= 7  else 0.0,
            "Rolling_Mean_30":float(np.mean(recent_raw[-30:])) if len(recent_raw) >= 30 else float(np.mean(recent_raw)),
            "Year":      next_date.year,
            "Month":     next_date.month,
            "WeekOfYear":next_date.isocalendar()[1],
            "DayOfWeek": next_date.weekday(),
            "IsHoliday": 0,
        }
        next_row = pd.DataFrame([[next_row_raw.get(c, 0) for c in feat_cols]], columns=feat_cols)
        next_scaled = feat_scaler.transform(next_row.values)[0]

        recent_X = np.vstack([recent_X, next_scaled])
        recent_raw.append(pred_val)

    return preds


DISPATCH = {
    "SARIMA":  _sarima_predict,
    "Prophet": _prophet_predict,
    "XGBoost": _xgboost_predict,
    "LSTM":    _lstm_predict,
}


def _run_forecast(state: str, weeks: int, model_name: str) -> List[dict]:
    fn = DISPATCH.get(model_name)
    if fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")
    raw_preds = fn(state, weeks)

    today = datetime.now()
    return [
        {
            "week":             i + 1,
            "date":             (today + timedelta(weeks=i + 1)).strftime("%Y-%m-%d"),
            "forecasted_sales": round(float(p), 2),
        }
        for i, p in enumerate(raw_preds)
    ]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":          "✅ API is running",
        "total_states":    len(best_models),
        "models_available": list(DISPATCH.keys()),
        "timestamp":       datetime.now().isoformat(),
    }


@app.get("/states")
async def list_states():
    return {"states": sorted(best_models.keys())}


@app.get("/models")
async def model_summary():
    records = best_models_df[["State", "Model", "MAPE(%)"]].to_dict(orient="records")
    return {
        "model_distribution": best_models_df["Model"].value_counts().to_dict(),
        "details": records,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    state = req.state.strip().title()
    if state not in best_models:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found. Check GET /states")

    model_name = req.model_override.strip() if req.model_override else best_models[state]

    try:
        predictions = _run_forecast(state, req.weeks, model_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"state": state, "best_model": model_name, "predictions": predictions}


@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    results = []
    errors  = []
    for s in req.states:
        state = s.strip().title()
        if state not in best_models:
            errors.append({"state": state, "error": "not found"})
            continue
        try:
            preds = _run_forecast(state, req.weeks, best_models[state])
            results.append({"state": state, "best_model": best_models[state], "predictions": preds})
        except Exception as e:
            errors.append({"state": state, "error": str(e)})
    return {"results": results, "errors": errors}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)