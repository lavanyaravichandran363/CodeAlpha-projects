import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Strip leading/trailing spaces in column names
data.columns = data.columns.str.strip()

# Display the first 5 rows
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Drop rows with missing values (if any)
data = data.dropna()

# Convert the correct date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Display basic information
print("\nDataset Info:")
print(data.info())

# Describe the dataset statistically
print("\nStatistical Summary:")
print(data.describe())

# Check unique regions
print("\nRegions in Dataset:")
print(data['Region'].unique())

# Data visualization - Line Plot (Unemployment Over Time)
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=data)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Plot - Average unemployment rate by region
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=data)
plt.xticks(rotation=90)
plt.title('Average Unemployment Rate by Region')
plt.tight_layout()
plt.show()

# Filter data before and after 2020
pre_covid = data[data['Date'] < '2020-01-01']
post_covid = data[data['Date'] >= '2020-01-01']

# Mean comparison
print("\n--- COVID Impact on Unemployment ---")
print("Average Unemployment Rate Before COVID-19:", pre_covid['Estimated Unemployment Rate (%)'].mean())
print("Average Unemployment Rate After COVID-19:", post_covid['Estimated Unemployment Rate (%)'].mean())