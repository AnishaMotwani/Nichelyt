from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_metadata.csv"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "caption_embeddings.npy"
)


# --------------------------------------------------
# 2. Load metadata and embeddings
# --------------------------------------------------

print("\nLoading metadata and embeddings...")

metadata = pd.read_csv(METADATA_PATH)
embeddings = np.load(EMBEDDINGS_PATH)

print(f"Metadata shape: {metadata.shape}")
print(f"Embeddings shape: {embeddings.shape}")


# --------------------------------------------------
# 3. Check alignment
# --------------------------------------------------

if len(metadata) != len(embeddings):
    raise ValueError(
        "ERROR: Metadata rows and embedding rows are not aligned."
    )

print("SUCCESS: Metadata and embeddings are aligned.")


# --------------------------------------------------
# 4. Convert date column
# --------------------------------------------------

required_columns = [
    "postcreateddate",
    "clean_text",
    "engagement_rate",
    "username",
]

for column in required_columns:
    if column not in metadata.columns:
        raise ValueError(
            f"ERROR: Required column missing: {column}"
        )


metadata["analysis_date"] = pd.to_datetime(
    metadata["postcreateddate"],
    errors="coerce"
)

metadata = metadata.dropna(
    subset=["analysis_date"]
).copy()


# --------------------------------------------------
# 5. Basic date analysis
# --------------------------------------------------

print("\nDATE INFORMATION")
print("----------------")

print(
    f"Earliest date: "
    f"{metadata['analysis_date'].min().date()}"
)

print(
    f"Latest date: "
    f"{metadata['analysis_date'].max().date()}"
)

date_range_days = (
    metadata["analysis_date"].max()
    - metadata["analysis_date"].min()
).days

print(f"Date range: {date_range_days} days")


# --------------------------------------------------
# 6. Show yearly distribution
# --------------------------------------------------

metadata["year"] = metadata["analysis_date"].dt.year

print("\nPOSTS BY YEAR")
print("-------------")

year_counts = (
    metadata["year"]
    .value_counts()
    .sort_index()
)

print(year_counts)


# --------------------------------------------------
# 7. Define relevance using similarity
# --------------------------------------------------

print("\nTREND DIAGNOSTIC")
print("----------------")

print(
    "This diagnostic uses the same relevance threshold "
    "as the current opportunity engine."
)

print(
    "\nIMPORTANT:"
    "\nThe similarity column is not present in the saved "
    "metadata file."
)

print(
    "Therefore, this file checks date coverage and "
    "metadata quality only."
)


# --------------------------------------------------
# 8. Check missing values
# --------------------------------------------------

print("\nMISSING VALUES")
print("--------------")

print(
    metadata[
        [
            "postcreateddate",
            "analysis_date",
            "clean_text",
            "engagement_rate",
            "username",
        ]
    ].isna().sum()
)


# --------------------------------------------------
# 9. Check relevant metadata quality
# --------------------------------------------------

print("\nTEXT QUALITY")
print("------------")

print(
    f"Total metadata rows: {len(metadata)}"
)

print(
    "Empty text rows:",
    (
        metadata["clean_text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )
)

print(
    "Unique creators:",
    metadata["username"].nunique()
)

print(
    "Unique dates:",
    metadata["analysis_date"].dt.date.nunique()
)


# --------------------------------------------------
# 10. Date-wise post count
# --------------------------------------------------

print("\nDATE-WISE POST COUNTS")
print("---------------------")

daily_counts = (
    metadata
    .groupby(metadata["analysis_date"].dt.date)
    .size()
    .sort_index()
)

print(daily_counts.head(10))

print("\nLast 10 dates:")
print(daily_counts.tail(10))


# --------------------------------------------------
# 11. Save diagnostic output
# --------------------------------------------------

output_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trend_date_diagnostic.csv"
)

daily_counts_df = daily_counts.reset_index()

daily_counts_df.columns = [
    "date",
    "post_count",
]

daily_counts_df.to_csv(
    output_path,
    index=False
)

print(
    f"\nDiagnostic file saved at:\n{output_path}"
)

print("\nDIAGNOSTIC COMPLETED SUCCESSFULLY.")