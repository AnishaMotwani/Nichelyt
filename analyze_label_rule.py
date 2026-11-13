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

    target_column = "performance_bucket_label"

    # 2. Label-wise engagement rate summary
    print("\nEngagement Rate Summary by Label:")

    engagement_summary = (
        df.groupby(target_column)["engagement_rate"]
        .agg(["count", "min", "max", "mean", "median"])
        .round(5)
    )

    print(engagement_summary)

    # 3. Label-wise likes summary
    print("\nLikes Summary by Label:")

    likes_summary = (
        df.groupby(target_column)["likes"]
        .agg(["count", "min", "max", "mean", "median"])
        .round(2)
    )

    print(likes_summary)

    # 4. Label-wise comments summary
    print("\nComments Summary by Label:")

    comments_summary = (
        df.groupby(target_column)["comments"]
        .agg(["count", "min", "max", "mean", "median"])
        .round(2)
    )

    print(comments_summary)

    # 5. Label-wise shares summary
    print("\nShares Summary by Label:")

    shares_summary = (
        df.groupby(target_column)["shares"]
        .agg(["count", "min", "max", "mean", "median"])
        .round(2)
    )

    print(shares_summary)

    # 6. Label-wise reach and impressions
    print("\nReach Summary by Label:")

    reach_summary = (
        df.groupby(target_column)["reach"]
        .agg(["count", "min", "max", "mean", "median"])
        .round(2)
    )

    print(reach_summary)

    print("\nImpressions Summary by Label:")

    impressions_summary = (
        df.groupby(target_column)["impressions"]
        .agg(["count", "min", "max", "mean", "median"])
        .round(2)
    )

    print(impressions_summary)

    # 7. Label-wise performance metrics correlation
    print("\nPerformance Metrics Correlation:")

    performance_columns = [
        "likes",
        "comments",
        "shares",
        "saves",
        "reach",
        "impressions",
        "engagement_rate",
        "followers_gained"
    ]

    print(
        df[performance_columns]
        .corr()
        .round(3)
    )

    print("\nLabel rule analysis completed successfully!")