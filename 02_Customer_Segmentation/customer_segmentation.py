import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("sales_dataset.csv")

# Create segment-level summary
segment_summary = df.groupby("Segment").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Quantity=("Quantity", "sum")
).reset_index()

print("Segment Summary:")
print(segment_summary)

# Select clustering features
X = segment_summary[
    ["Total_Sales", "Total_Profit", "Total_Quantity"]
]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

segment_summary["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means Cluster Results:")
print(segment_summary)

# Display cluster summary
cluster_summary = segment_summary.groupby("Cluster")[
    ["Total_Sales", "Total_Profit", "Total_Quantity"]
].mean()

print("\nCluster Summary:")
print(cluster_summary)

# Visualization
plt.figure(figsize=(8, 5))

plt.scatter(
    segment_summary["Total_Sales"],
    segment_summary["Total_Profit"],
    c=segment_summary["Cluster"],
    s=100
)

plt.xlabel("Total Sales")
plt.ylabel("Total Profit")
plt.title("K-Means Clusters of Customer Segments")
plt.tight_layout()
plt.show()

print("\nCustomer Segmentation Completed Successfully.")
