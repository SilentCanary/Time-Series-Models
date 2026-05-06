import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('featured_sales_data.csv')

feature_cols = ['Year', 'Month', 'WeekOfYear', 'DayOfWeek', 'IsWeekend', 
                'IsHoliday', 'Lag_1', 'Lag_7', 'Lag_14', 
                'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_30']

corr = df[feature_cols + ['Total']].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()