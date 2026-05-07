import pandas as pd

print("=== Creating FINAL Best Model Selection ===")

# Load the three model results
sarima = pd.read_csv('../sarima_results.csv')
xgboost = pd.read_csv('../xgboost_results_all (1).csv')
prophet = pd.read_csv('../prophet_results_all.csv')
lstm=pd.read_csv('../lstm_results.csv')

# Standardize MAPE column name
for df in [sarima, xgboost, prophet,lstm]:
    if 'MAPE' in df.columns and 'MAPE(%)' not in df.columns:
        df.rename(columns={'MAPE': 'MAPE(%)'}, inplace=True)

# Add Model name
sarima['Model'] = 'SARIMA'
xgboost['Model'] = 'XGBoost'
prophet['Model'] = 'Prophet'
lstm['Model']='LSTM'

# Combine all
all_results = pd.concat([sarima, xgboost, prophet,lstm], ignore_index=True)

# Select the BEST model for each state (lowest MAPE)
best_per_state = all_results.loc[all_results.groupby('State')['MAPE(%)'].idxmin()]

# Clean and sort
best_per_state = best_per_state[['State', 'Model', 'MAPE(%)']].sort_values('MAPE(%)').reset_index(drop=True)

print("\n🎯 FINAL BEST MODEL PER STATE (Top 15):")
print(best_per_state.head(42))

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(best_per_state['Model'].value_counts())

best_per_state.to_csv('../best_model_per_state.csv', index=False)
print("\n✅ Correct Best Model Selection Saved!")