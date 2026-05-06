import pandas as pd
import numpy as np
import holidays

df = pd.read_csv('cleaned_sales_data.csv', parse_dates=['Date'])
df = df.sort_values(['State', 'Date']).reset_index(drop=True)

print("Starting Improved Feature Engineering...")

# 1. Basic Time Features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# 2. Holiday Flag
us_holidays = holidays.US(years=range(2019, 2025))
df['IsHoliday'] = df['Date'].dt.date.apply(lambda x: x in us_holidays).astype(int)

# 3. Lag & Rolling Features (per state)
def create_features(group):
    group = group.sort_values('Date').copy()
    
    # Lags
    group['Lag_1'] = group['Total'].shift(1)
    group['Lag_7'] = group['Total'].shift(7)
    group['Lag_14'] = group['Total'].shift(14)
    
    # Rolling
    group['Rolling_Mean_7'] = group['Total'].rolling(window=7, min_periods=1).mean()
    group['Rolling_Std_7'] = group['Total'].rolling(window=7, min_periods=1).std()
    group['Rolling_Mean_30'] = group['Total'].rolling(window=30, min_periods=1).mean()
    
    return group

df = df.groupby('State', group_keys=False).apply(create_features)

# Fill NaN values (important for models)
df = df.groupby('State', group_keys=False).apply(lambda x: x.fillna(method='bfill').fillna(method='ffill').fillna(0))

print("✅ Feature Engineering Done!")
print("Shape:", df.shape)
print("\nSample columns:", df.columns.tolist())

df.to_csv('featured_sales_data.csv', index=False)
print("Saved as 'featured_sales_data.csv'")