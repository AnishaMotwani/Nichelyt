
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

    # 2. Average metrics by performance label
    print("\nAverage Metrics by Performance Label:")

    performance_summary = df.groupby(
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

    print(performance_summary)

    # 3. Media type and performance
    print("\nMedia Type and Performance Label:")

    media_performance = pd.crosstab(
        df["media_type"],
        df["performance_bucket_label"]
    )

    print(media_performance)

    # 4. Content category and performance
    print("\nContent Category and Performance Label:")

    category_performance = pd.crosstab(
        df["content_category"],
        df["performance_bucket_label"]
    )

    print(category_performance)

    # 5. Performance label distribution
    print("\nPerformance Label Distribution:")

    print(df["performance_bucket_label"].value_counts())

    # 6. Engagement rate summary
    print("\nEngagement Rate Summary:")

    print(df["engagement_rate"].describe())

    # 7. Average engagement rate by media type
    print("\nAverage Engagement Rate by Media Type:")

    media_engagement = df.groupby(
        "media_type"
    )["engagement_rate"].mean()

    print(media_engagement)

    # 8. Average engagement rate by content category
    print("\nAverage Engagement Rate by Content Category:")

    category_engagement = df.groupby(
        "content_category"
    )["engagement_rate"].mean()

    print(category_engagement)