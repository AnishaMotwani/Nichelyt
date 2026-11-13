
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

    # 2. Average engagement rate by performance label
    print("\nAverage Engagement Rate by Performance Label:")

    label_engagement = df.groupby(
        "performance_bucket_label"
    )["engagement_rate"].agg(
        ["count", "min", "max", "mean", "median"]
    )

    print(label_engagement)

    # 3. Engagement rate range of each label
    print("\nEngagement Rate Range by Performance Label:")

    for label in ["low", "medium", "high", "viral"]:

        label_data = df[
            df["performance_bucket_label"] == label
        ]["engagement_rate"]

        print(f"\n{label.upper()}:")

        print("Minimum:", label_data.min())
        print("Maximum:", label_data.max())
        print("Average:", label_data.mean())

    # 4. Check whether labels are ordered according to engagement rate
    print("\nAverage Metrics by Performance Label:")

    summary = df.groupby(
        "performance_bucket_label"
    )[[
        "likes",
        "comments",
        "shares",
        "saves",
        "reach",
        "impressions",
        "engagement_rate",
        "followers_gained"
    ]].mean()

    print(summary)

    # 5. Check average engagement rate by posting hour
    print("\nAverage Engagement Rate by Posting Hour:")

    hourly_engagement = df.groupby(
        "post_hour"
    )["engagement_rate"].mean().sort_values(
        ascending=False
    )

    print(hourly_engagement)

    # 6. Check average engagement rate by day
    print("\nAverage Engagement Rate by Day:")

    daily_engagement = df.groupby(
        "day_of_week"
    )["engagement_rate"].mean().sort_values(
        ascending=False
    )

    print(daily_engagement)