# Time-Series-Models

**End-to-End Time Series Forecasting + REST API**

---

## 🎯 Objective
Forecast **next 8 weeks** of Beverage sales for each state using historical data.

---

## 📊 Final Model Performance

| Model      | Selected For   | Avg MAPE |
|------------|----------------|----------|
| **SARIMA**     | 24 states      | **3.09%** |
| **XGBoost**    | 19 states      | **3.05%** |
| Prophet        | -              | 8.42%    |

**Best model automatically selected per state** based on lowest MAPE on validation data.

---

# 📂 Project Structure - Time Series Forecasting
```
TIM SERIES/
├── models/                     # Trained models storage
│   ├── arima/                  # ARIMA model artifacts
│   ├── prophet/                # Prophet model artifacts
│   ├── xgboost/                # XGBoost model artifacts
│   └── XGBOOST_1/              # Alternate XGBoost run
│       ├── scaler_lstm.pkl
│       └── scaler_xgboost.pkl
│
├── preprocess/                 # Data preprocessing & analysis
│   ├── analysis.py
│   ├── corr.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_test_split.py
│   ├── cleaned_sales_data.csv
│   ├── featured_sales_data.csv
│   ├── train_data.csv
│   ├── val_data.csv
│   ├── Figure_1.png
│   └── Forecasting Case-Study.txt
│
├── src/                        # Core source code
│   ├── api/
│   │   └── main.py             # FastAPI entrypoint
│   └── models/
│       ├── lstm.py
│       ├── prophet.py
│       ├── train_arima.py
│       ├── xgboost.py
│       ├── data_preprocess_xg.py
│       ├── data_preprocessing.py
│       └── inference.py
│
├── best_model_per_state.csv     # Model selection results
├── train_data.csv               # Training dataset
├── val_data.csv                 # Validation dataset
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```
## 🚀 How to Run

### 1. Install Requirements
```bash
pip install fastapi uvicorn pandas joblib prophet xgboost statsmodels scikit-learn holidays
```
### 2. Start the apo
```bash
cd src
cd api
python main.py
```
API runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
Interactive Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Test
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"state\": \"California\", \"weeks\": 8}"
```
📋 What Was Implemented

- ✅ Missing dates & values handling
- ✅ Full Feature Engineering (lags, rolling, time, holidays)
- ✅ SARIMA, Prophet, XGBoost
- ✅ Automatic best model selection per state
- ✅ Production-ready FastAPI
- ✅ Clean project structure



## Author: Advitiya Prakash

