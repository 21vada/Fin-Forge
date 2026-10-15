from pathlib import Path
import pandas as pd

file_path = Path("C:\\Users\\pc\\Documents\\ieee-fraud-detection\\train_transaction.csv")

if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {file_path.resolve()}"
    )

# Read only 10,000 rows for initial exploration.
df = pd.read_csv(file_path, nrows=10_000)

print("Rows loaded:", len(df))
print("Columns:", len(df.columns))

print("\nFirst five transactions:")
columns = [
    c for c in [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "isFraud"
    ]
    if c in df.columns
]
print(df[columns].head())

if "isFraud" in df.columns:
    print("\nFraud distribution in this sample:")
    print(df["isFraud"].value_counts())

print("\nColumns with the most missing values:")
print(
    (df.isna().mean() * 100)
    .sort_values(ascending=False)
    .head(15)
    .round(2)
    .astype(str) + "%"
)