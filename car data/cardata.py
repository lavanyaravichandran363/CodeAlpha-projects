import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("car data.csv")
print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Understand the Data
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Drop missing values (if any)
df.dropna(inplace=True)

# Step 4: Drop irrelevant column (like Car_Name) and handle categorical columns
df.drop('Car_Name', axis=1, inplace=True)

# Step 5: Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Step 6: Define features and target
target_col = "Selling_Price"  # ✅ corrected column name

X = df.drop(target_col, axis=1)
y = df[target_col]

# Step 7: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predict
y_pred = model.predict(X_test)

# Step 10: Evaluate Model
print("\nModel Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 11: Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Car Prices")
plt.ylabel("Predicted Car Prices")
plt.title("Actual vs Predicted Car Prices")
plt.show()