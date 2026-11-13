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

    # 2. Target distribution
    target_column = "performance_bucket_label"

    print("\nTarget Distribution:")
    print(df[target_column].value_counts())

    # 3. Performance-related numerical columns
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

    print("\nAverage Performance Metrics by Target Label:")
    print(
        df.groupby(target_column)[performance_columns]
        .mean()
        .round(4)
    )

    # 4. Standard deviation by target label
    print("\nPerformance Metric Standard Deviation by Target:")
    print(
        df.groupby(target_column)[performance_columns]
        .std()
        .round(4)
    )

    # 5. Average pre-publication features by target
    pre_publication_columns = [
        "follower_count",
        "has_call_to_action",
        "post_hour",
        "caption_length",
        "hashtags_count"
    ]

    print("\nAverage Input Features by Target Label:")
    print(
        df.groupby(target_column)[pre_publication_columns]
        .mean()
        .round(4)
    )

    # 6. Target-label mapping check
    print("\nUnique Target Labels:")
    print(df[target_column].unique())

    # 7. Overall numerical correlation
    print("\nCorrelation with Numerical Features:")
    numerical_columns = [
        "follower_count",
        "has_call_to_action",
        "post_hour",
        "caption_length",
        "hashtags_count",
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
        df[numerical_columns]
        .corr()
        .round(3)
    )

    print("\nTarget diagnosis completed successfully!")