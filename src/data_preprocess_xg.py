import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

print("=== XGBoost Data Preprocessing Started ===")

# Load data
train_df = pd.read_csv('../preprocess/train_data.csv', parse_dates=['Date'])
val_df = pd.read_csv('../preprocess/val_data.csv', parse_dates=['Date'])

# Optimized Features (after correlation analysis)
feature_cols = ['Year', 'Month', 'WeekOfYear', 'DayOfWeek', 'IsHoliday',
                'Lag_1', 'Rolling_Mean_7', 'Rolling_Std_7']

print(f"Using {len(feature_cols)} features")

# Prepare features
X_train = train_df[feature_cols].copy()
y_train = train_df['Total'].copy()

X_val = val_df[feature_cols].copy()
y_val = val_df['Total'].copy()

# Feature Scaling (Recommended for XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save Scaler
os.makedirs('../models', exist_ok=True)
joblib.dump(scaler, '../models/scaler_xgboost.pkl')
print("✅ Scaler saved!")

# Create processed DataFrames (keeping State and Date for easy grouping)
train_processed = train_df[['State', 'Date']].copy()
val_processed = val_df[['State', 'Date']].copy()

train_processed[feature_cols] = X_train_scaled
val_processed[feature_cols] = X_val_scaled

train_processed['Total'] = y_train
val_processed['Total'] = y_val

# Save processed data
train_processed.to_csv('../train_xgboost_processed.csv', index=False)
val_processed.to_csv('../val_xgboost_processed.csv', index=False)

print(f"✅ Preprocessing Completed!")
print(f"Train shape: {train_processed.shape}")
print(f"Val shape: {val_processed.shape}")
print("Files saved: train_xgboost_processed.csv and val_xgboost_processed.csv")