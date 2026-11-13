
import pandas as pd
from pathlib import Path

# Dataset 2 ka file path
file_path = Path("data/raw/Marketing Analytics Competition 2026 Dataset.xlsx")

# Excel workbook ke sheet names check karna
excel_file = pd.ExcelFile(file_path)

print("========== SHEET NAMES ==========")
print(excel_file.sheet_names)

# Har sheet ko inspect karna
for sheet in excel_file.sheet_names:
    print(f"\n========== SHEET: {sheet} ==========")

    df = pd.read_excel(file_path, sheet_name=sheet)

    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

print("\nDataset 2 inspection completed!")