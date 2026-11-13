
import pandas as pd
from pathlib import Path

# Dataset ka folder
data_folder = Path("data/raw")

# Raw folder ke andar CSV files dhoondhna
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found in data/raw")

else:
    # Pehli CSV file select karna
    file_path = csv_files[0]

    print("Loading file:", file_path)

    # CSV file read karna
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)