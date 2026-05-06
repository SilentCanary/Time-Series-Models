import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

train_path = '../../train_data.csv'      # Go up 2 levels
val_path = '../../val_data.csv'
# Load data
train_df = pd.read_csv(train_path, parse_dates=['Date'])
val_df = pd.read_csv(val_path, parse_dates=['Date'])

print("Training Facebook Prophet...")

# Prepare data for Prophet (needs columns: ds, y)
def prepare_for_prophet(df):
    return df.rename(columns={'Date': 'ds', 'Total': 'y'}).copy()

# We'll train one model per state (or sample some big states first)
states = train_df['State'].unique()
results = []

# For faster testing, let's train on top 5 states first
top_states = ['California', 'Texas', 'Florida', 'New York', 'Georgia']

for state in top_states:
    train_state = train_df[train_df['State'] == state].copy()
    val_state = val_df[val_df['State'] == state].copy()
    
    if len(train_state) < 20:
        continue
    
    # Prepare data
    train_prophet = prepare_for_prophet(train_state)
    val_prophet = prepare_for_prophet(val_state)
    
    # Train Prophet
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train_prophet)
    
    # Forecast
    future = model.make_future_dataframe(periods=len(val_state), freq='D')
    forecast = model.predict(future)
    
    # Evaluate on validation
    forecast_val = forecast.tail(len(val_state))['yhat'].values
    actual_val = val_prophet['y'].values
    
    mae = mean_absolute_error(actual_val, forecast_val)
    rmse = np.sqrt(mean_squared_error(actual_val, forecast_val))
    
    results.append({
        'State': state,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': np.mean(np.abs((actual_val - forecast_val) / actual_val)) * 100
    })

# Show results
results_df = pd.DataFrame(results)
print("\n=== Prophet Performance on Top States ===")
print(results_df.round(2))

# Save one model example (California)
print("\nProphet training completed for top states!")