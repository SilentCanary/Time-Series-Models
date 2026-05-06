import pandas as pd

# Load original data
df = pd.read_excel('Forecasting Case- Study.xlsx')
df['Date'] = pd.to_datetime(df['Date'])

print("Original Shape:", df.shape)
print("Unique Dates:", df['Date'].nunique())

# Sort and remove any duplicates if present
df = df.sort_values(['State', 'Date']).drop_duplicates(subset=['State', 'Date'])

# For now, we'll work with available data only (no artificial filling)
# We'll handle missing values later during modeling

print("\nFinal Cleaned Data Info:")
print(df.info())
print("\nRecords per state (top 10):")
print(df.groupby('State').size().sort_values(ascending=False).head(10))

# Save cleaned version
df.to_csv('cleaned_sales_data.csv', index=False)
print("\nSaved as 'cleaned_sales_data.csv'")