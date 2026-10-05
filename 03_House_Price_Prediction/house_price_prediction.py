import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate synthetic dataset
np.random.seed(42)

n = 100

house_df = pd.DataFrame({
    "Area_sqft": np.random.randint(600, 3000, n),
    "Bedrooms": np.random.randint(1, 6, n),
    "Bathrooms": np.random.randint(1, 4, n),
    "Age_years": np.random.randint(0, 30, n)
})

# Generate synthetic house prices
house_df["Price"] = (
    house_df["Area_sqft"] * 3000
    + house_df["Bedrooms"] * 500000
    + house_df["Bathrooms"] * 300000
    - house_df["Age_years"] * 50000
    + np.random.normal(0, 500000, n)
)

print("House Price Dataset:")
print(house_df.head())

print("\nDataset Shape:", house_df.shape)

# Features and target
X = house_df[
    ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years"]
]

y = house_df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# Actual vs Predicted
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(comparison.head(10))

# Visualization
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()

print("\nHouse Price Prediction Completed Successfully.")
