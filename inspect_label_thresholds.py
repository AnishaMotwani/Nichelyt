
import pandas as pd
from pathlib import Path


# 1. Dataset load karna
data_folder = Path("data/raw")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found in data/raw")

else:
    file_path = csv_files[0]
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")

    # 2. Engagement rate ke according data sort karna
    sorted_df = df.sort_values(
        by="engagement_rate"
    )

    # 3. Har label ka engagement rate range
    print("\n========== LABEL RANGES ==========")

    label_ranges = df.groupby(
        "performance_bucket_label"
    )["engagement_rate"].agg(
        ["min", "max", "mean", "median", "count"]
    )

    print(label_ranges)

    # 4. Label transitions inspect karna
    print("\n========== SORTED LABEL TRANSITIONS ==========")

    selected_columns = [
        "engagement_rate",
        "performance_bucket_label"
    ]

    print(
        sorted_df[selected_columns].head(20).to_string(
            index=False
        )
    )

    print("\n---------- Highest Engagement Rates ----------")

    print(
        sorted_df[selected_columns].tail(20).to_string(
            index=False
        )
    )

    # 5. Engagement rate quantiles
    print("\n========== ENGAGEMENT RATE QUANTILES ==========")

    quantiles = df["engagement_rate"].quantile(
        [0.25, 0.50, 0.75]
    )

    print(quantiles)

    print("\nLabel threshold inspection completed!")