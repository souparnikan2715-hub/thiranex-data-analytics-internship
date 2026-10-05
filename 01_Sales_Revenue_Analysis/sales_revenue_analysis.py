import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("sales_dataset.csv")

# Display dataset information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

# Total Sales and Profit
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()

print("\nTotal Sales:", round(total_sales, 2))
print("Total Profit:", round(total_profit, 2))

# Profit Margin
profit_margin = (total_profit / total_sales) * 100
print("Profit Margin:", round(profit_margin, 2), "%")

# Category-wise analysis
category_summary = df.groupby("Category")[["Sales", "Profit"]].sum()

print("\nCategory-wise Sales and Profit:")
print(category_summary)

# Region-wise analysis
region_summary = df.groupby("Region")[["Sales", "Profit"]].sum()

print("\nRegion-wise Sales and Profit:")
print(region_summary)

# Monthly sales analysis
df["Order Date"] = pd.to_datetime(df["Order Date"])

monthly_sales = df.groupby(
    df["Order Date"].dt.to_period("M")
)["Sales"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

# Category-wise Sales Visualization
plt.figure(figsize=(8, 5))
category_summary["Sales"].plot(kind="bar")

plt.title("Category-wise Sales")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Region-wise Sales Visualization
plt.figure(figsize=(8, 5))
region_summary["Sales"].plot(kind="bar")

plt.title("Region-wise Sales")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nSales and Revenue Analysis Completed Successfully.")
