import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv('cleaned_sales_data.csv', parse_dates=['Date'])
df = df.sort_values(['State', 'Date'])

print("=== Final Data Overview ===")
print(f"Shape: {df.shape}")
print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Unique Dates: {df['Date'].nunique()}")

# Check time frequency
date_counts = df['Date'].value_counts().sort_index()
print("\nTop 10 most frequent dates:")
print(date_counts.head(10))

print("\nDate differences (in days):")
diffs = pd.Series(df['Date'].unique()).sort_values().diff().dt.days.value_counts()
print(diffs)

# Plot sales trend for a few big states
big_states = ['California', 'Texas', 'Florida', 'New York']

plt.figure(figsize=(12, 6))
for state in big_states:
    state_data = df[df['State'] == state].copy()
    plt.plot(state_data['Date'], state_data['Total']/1e6, label=state)

plt.title('Sales Trend (in Millions) - Top States')
plt.xlabel('Date')
plt.ylabel('Sales (Millions USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()