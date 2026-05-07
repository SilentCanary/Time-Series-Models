# 🍶 Beverage Sales Forecasting System
**End-to-End Time Series Forecasting + Production REST API**
*By Advitiya Prakash*

---

## 1. Objective

Build a **production-ready forecasting system** that:
- Trains multiple forecasting algorithms — SARIMA, Prophet, XGBoost, LSTM
- Automatically selects the best model per US state
- Handles missing dates, irregular intervals, seasonality & trend
- Exposes predictions via a clean **REST API** (FastAPI)

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| File | `Forecasting Case-Study.xlsx` |
| Records | 8,084 |
| States | 43 |
| Time Range | 2019-01-12 → 2023-12-03 |
| Unique Dates | 188 (mostly weekly, irregular gaps) |
| Target | `Total` (Beverage Sales in USD) |

**Key Challenges:**
- Irregular time intervals (gaps ranging from 1 to 91 days between records)
- Missing dates across all states requiring reconstruction
- Large variation in sales volume across states (~10M to ~900M)
- Short series per state — only 188–256 points, too few for naive LSTM training

---

## 3. Exploratory Data Analysis

- Confirmed 43 states with exactly 188 raw records each
- Visualized sales trends for top states (California, Texas, Florida, New York)
- Observed strong **upward trend** + clear **annual seasonality**
- Confirmed mostly weekly cadence with some irregular gaps

---

## 4. Data Preprocessing & Feature Engineering

### 4.1 Cleaning
- Converted `Date` to datetime, sorted by State + Date
- Per-state date reindexing to reconstruct a clean weekly series
- Forward-fill + backward-fill for missing values
- After filling: **256 points per state** (up from 188)

### 4.2 Feature Engineering

| Feature Type | Features Created |
|---|---|
| Time-based | `Year`, `Month`, `WeekOfYear`, `DayOfWeek`, `IsWeekend`, `IsHoliday` |
| Lag Features | `Lag_1`, `Lag_7`, `Lag_14` |
| Rolling Stats | `Rolling_Mean_7`, `Rolling_Std_7`, `Rolling_Mean_30` |

US holidays added via the `holidays` library.

### 4.3 Correlation Analysis

Correlation matrix revealed very high multicollinearity (0.95–0.99) between:
- `Lag_1`, `Lag_7`, `Lag_14`
- `Rolling_Mean_7`, `Rolling_Mean_30`

**Action taken:** Dropped redundant features for XGBoost to reduce noise and improve generalization.

**Final XGBoost features:** `Year`, `Month`, `WeekOfYear`, `DayOfWeek`, `IsHoliday`, `Lag_1`, `Rolling_Mean_7`, `Rolling_Std_7`

**Final LSTM features:** `Lag_1`, `Lag_7`, `Lag_14`, `Rolling_Mean_7`, `Rolling_Std_7`, `Rolling_Mean_30`, `Year`, `Month`, `WeekOfYear`, `DayOfWeek`, `IsHoliday`

---

## 5. Train-Validation Split

Time-based chronological split (no data leakage):
- **Train:** 2019-01-13 → 2023-10-08 (~244 points per state)
- **Val:** remaining ~12 weeks per state

---

## 6. LSTM — Sequence Strategy

Early LSTM attempts failed because of the short per-state series. The fix was a two-step approach:

**Old (broken) flow:**
```
split raw points first → only 8 val points per state
→ create sequences → val has almost nothing → training fails
```

**New (fixed) flow:**
```
combine train + val per state (256 points)
→ create ALL sequences with window=12  (~244 sequences/state)
→ split sequences 80/20 chronologically
→ ~195 train + ~49 val sequences per state
→ stack all states → 8,385 train / 2,107 val sequences globally
```

**Two-phase training:**
1. **Global LSTM** trained on all 8,385 sequences across 43 states — learns general weekly sales patterns
2. **Per-state fine-tuning** — warm-starts from global weights, fine-tunes for 20 epochs on each state's ~195 sequences

Per-state scalers (`feat_scaler_{state}.pkl`, `target_scaler_{state}.pkl`) saved separately so the API can do clean inverse-transform per state at inference time.

---

## 7. Model Performance

| Model | States Selected | Avg MAPE | Notes |
|---|---|---|---|
| **LSTM** | 16 states | **2.90%** | Global train + per-state fine-tune |
| **XGBoost** | 14 states | **3.05%** | Recursive multi-step with real lag updates |
| **SARIMA** | 13 states | **3.09%** | Seasonal ARIMA per state |
| Prophet | 0 states | 8.42% | Trained but outperformed by others |

Best model automatically selected per state based on lowest validation MAPE.

### Overall Best States (across all models)
Top performers include Vermont (0.38%), Kansas (0.90%), Washington (0.97%) — all under 1% MAPE.

---

## 8. Production REST API

Built with **FastAPI**. All models dispatched by best-model-per-state lookup.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/states` | List all 43 available states |
| GET | `/models` | Best model per state with MAPE |
| POST | `/predict` | 8-week forecast for a single state |
| POST | `/predict/batch` | Forecast for multiple states in one call |

### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"state": "California", "weeks": 8}'
```

### Example Response
```json
{
  "state": "California",
  "best_model": "SARIMA",
  "predictions": [
    {"week": 1, "date": "2024-01-07", "forecasted_sales": 312847291.45},
    {"week": 2, "date": "2024-01-14", "forecasted_sales": 318204819.12},
    ...
  ]
}
```

---

## 9. How to Run

### Install dependencies
```bash
pip install fastapi uvicorn pandas joblib prophet xgboost statsmodels \
            scikit-learn holidays tensorflow
```

### Step 1 — Preprocess & build sequences
```bash
cd preprocess
python analysis.py
python feature_engineering.py
python train_test_split.py
python data_preprocessing.py
python lstm_data_builder.py --scalers_dir ../models/lstm
```

### Step 2 — Train models
```bash
cd src/models
python train_arima.py
python train_xgboost.py
python train_prophet.py
python train_lstm.py          # uses sequences built in step 1
```

### Step 3 - Run the Infernece 
```bash
cd src
python inference.py
```

### Step 3 — Run API
```bash
cd src/api
python main.py
```

API live at: http://127.0.0.1:8000
Interactive docs: http://127.0.0.1:8000/docs

---

## 10. Project Structure

```
TIM SERIES/
├── models/
│   ├── arima/                   # sarima_{state}.pkl
│   ├── prophet/                 # prophet_{state}.pkl
│   ├── xgboost/                 # xgboost_{state}.pkl
│   └── lstm/                    # lstm_{state}.h5
│                                # feat_scaler_{state}.pkl
│                                # target_scaler_{state}.pkl
│
├── preprocess/
│   ├── data_preprocessing.py    # cleaning + feature engineering
│   ├── feature_engineering.py
│   ├── train_test_split.py
│   ├── lstm_data_builder.py     # sequence-first builder for LSTM
│   ├── cleaned_sales_data.csv
│   ├── featured_sales_data.csv
│   ├── train_data.csv
│   └── val_data.csv
│
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI entrypoint
│   └── models/
│       ├── train_arima.py
│       ├── xgboost.py
│       ├── prophet.py
│       ├── lstm.py
│       └── inference.py         # best model selector
│
├── best_model_per_state.csv
├── requirements.txt
└── README.md
```

---

## 11. What Was Implemented

- ✅ Missing date reconstruction + forward/backward fill per state
- ✅ Full feature engineering — lags, rolling stats, time features, US holidays
- ✅ Correlation analysis + feature selection for XGBoost
- ✅ SARIMA, Prophet, XGBoost trained per state
- ✅ LSTM with sequence-first build strategy + global + per-state fine-tuning
- ✅ Automatic best model selection per state (lowest val MAPE)
- ✅ Production FastAPI with recursive XGBoost inference and per-state LSTM scalers
- ✅ Clean project structure with separation of preprocessing, training, and serving

---

## 12. Challenges & Solutions

| Challenge | Solution |
|---|---|
| Irregular dates (1–91 day gaps) | Per-state date reindexing to weekly frequency |
| Short series per state (188 pts) | Date filling → 256 pts; sequence-first LSTM split |
| LSTM failing on 8 val points | Sequence-first build → 49 val sequences per state |
| Multicollinearity in features | Correlation matrix → dropped redundant features for XGBoost |
| Scale differences across states | Per-state MinMaxScaler (no cross-state leakage) |
| API using stale lag features | Merged train + val before computing inference lags |
