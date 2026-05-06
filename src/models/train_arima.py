import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from itertools import product

warnings.filterwarnings('ignore')

print("=== SARIMA Training Started (ALL STATES) ===")

# Load data
train_df = pd.read_csv('../../preprocess/train_data.csv', parse_dates=['Date'])
val_df = pd.read_csv('../../preprocess/val_data.csv', parse_dates=['Date'])

# All states
all_states = train_df['State'].unique().tolist()

os.makedirs('../../models/arima', exist_ok=True)

results = []

print(f"Training SARIMA on {len(all_states)} states...\n")

# Small parameter grid (fast but effective)
p = d = q = [0, 1]
P = D = Q = [0, 1]
s = 7  # weekly seasonality (change if needed)

param_grid = list(product(p, d, q))
seasonal_grid = list(product(P, D, Q))

for state in all_states:
    train_state = train_df[train_df['State'] == state].sort_values('Date').copy()
    val_state = val_df[val_df['State'] == state].sort_values('Date').copy()

    y_train = train_state['Total'].values
    y_val = val_state['Total'].values

    if len(y_train) < 30:
        print(f"⚠️ Skipping {state} (too little data)")
        continue

    best_mape = np.inf
    best_model = None
    best_order = None

    try:
        # Grid search
        for order in param_grid:
            for seasonal in seasonal_grid:
                try:
                    model = SARIMAX(
                        y_train,
                        order=order,
                        seasonal_order=(seasonal[0], seasonal[1], seasonal[2], s),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    fit = model.fit(disp=False, maxiter=50)

                    forecast = fit.forecast(steps=len(y_val))

                    mape = np.mean(np.abs((y_val - forecast) / y_val)) * 100

                    if mape < best_mape:
                        best_mape = mape
                        best_model = fit
                        best_order = (order, seasonal)

                except:
                    continue

        if best_model is None:
            print(f"❌ No valid model for {state}")
            continue

        # Final evaluation
        forecast = best_model.forecast(steps=len(y_val))
        mae = mean_absolute_error(y_val, forecast)
        rmse = np.sqrt(mean_squared_error(y_val, forecast))

        results.append({
            'State': state,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE(%)': round(best_mape, 2),
            'Order': best_order
        })

        joblib.dump(best_model, f'../../models/arima/sarima_{state}.pkl')

        print(f"✅ {state} | MAPE: {round(best_mape, 2)}% | Order: {best_order}")

    except Exception as e:
        print(f"❌ Failed for {state}: {e}")
        continue

# Results summary
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("SARIMA PERFORMANCE (ALL STATES)")
print("="*70)
print(results_df.describe())

results_df.to_csv('../../sarima_results.csv', index=False)

print("\n✅ SARIMA Training Completed!")