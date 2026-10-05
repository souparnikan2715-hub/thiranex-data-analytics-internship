import pandas as pd

# Load dataset
df = pd.read_csv("sales_dataset.csv")

print("DATA CLEANING AND REPORTING")
print("=" * 40)

# Original dataset shape
print("\nOriginal Dataset Shape:")
print(df.shape)

# Missing value check
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate check
duplicate_count = df.duplicated().sum()

print("\nDuplicate Records:", duplicate_count)

# Data types
print("\nData Types:")
print(df.dtypes)

# Remove duplicate records
df_cleaned = df.drop_duplicates()

# Handle missing numerical values
numeric_columns = df_cleaned.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:
    df_cleaned[column] = df_cleaned[column].fillna(
        df_cleaned[column].median()
    )

# Quantity validation
if "Quantity" in df_cleaned.columns:
    df_cleaned = df_cleaned[
        df_cleaned["Quantity"] > 0
    ]

# Final dataset shape
print("\nCleaned Dataset Shape:")
print(df_cleaned.shape)

# Create summary report
summary = pd.DataFrame({
    "Original Rows": [len(df)],
    "Cleaned Rows": [len(df_cleaned)],
    "Original Columns": [df.shape[1]],
    "Cleaned Columns": [df_cleaned.shape[1]],
    "Duplicate Records": [duplicate_count]
})

print("\nSummary Report:")
print(summary)

# Save outputs
df_cleaned.to_csv(
    "cleaned_dataset.csv",
    index=False
)

summary.to_csv(
    "cleaning_summary_report.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")
print("Summary report saved successfully.")

print("\nData Cleaning and Reporting Completed Successfully.")
