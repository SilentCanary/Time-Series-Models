import pandas as pd
from datetime import datetime

# Load featured data
df = pd.read_csv('featured_sales_data.csv', parse_dates=['Date'])

df = df.sort_values(['State', 'Date']).reset_index(drop=True)


print("Total Data Shape:", df.shape)

unique_dates = sorted(df['Date'].unique())
split_date = unique_dates[-5]

train_df = df[df['Date'] <= split_date].copy()
val_df = df[df['Date'] > split_date].copy()


print(f"Train Shape : {train_df.shape} | Train Period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
print(f"Val Shape   : {val_df.shape}   | Val Period  : {val_df['Date'].min().date()} to {val_df['Date'].max().date()}")

# Save splits
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)

print("\n✅ Train & Validation split completed!")
print("Files saved: train_data.csv and val_data.csv")