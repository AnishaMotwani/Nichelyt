import pandas as pd
from pathlib import Path

# Dataset ka folder
data_folder = Path("data/raw")

# CSV file select karna
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found in data/raw")

else:
    file_path = csv_files[0]

    print("Loading file:", file_path)

    # Dataset load karna
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")

    # 1. Duplicate rows
    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # 2. Statistical summary of numerical columns
    print("\nNumerical summary:")
    print(df.describe())

    # 3. Unique values in categorical columns
    print("\nAccount types:")
    print(df["account_type"].value_counts())

    print("\nMedia types:")
    print(df["media_type"].value_counts())

    print("\nContent categories:")
    print(df["content_category"].value_counts())

    print("\nPerformance labels:")
    print(df["performance_bucket_label"].value_counts())

    # 4. Unique CTA values
    print("\nCTA values:")
    print(df["has_call_to_action"].value_counts())