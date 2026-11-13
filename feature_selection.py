
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

    # 2. Target column define karna
    target_column = "performance_bucket_label"

    print("\nTarget Column:")
    print(target_column)

    # 3. Pre-publication features define karna
    # Ye features post publish hone se pehle available ho sakte hain
    pre_publication_features = [
        "account_type",
        "follower_count",
        "media_type",
        "content_category",
        "traffic_source",
        "has_call_to_action",
        "post_hour",
        "day_of_week",
        "caption_length",
        "hashtags_count"
    ]

    print("\nPotential Pre-Publication Features:")

    for feature in pre_publication_features:
        print("-", feature)

    # 4. Post-publication outcome features define karna
    # Ye metrics post publish hone ke baad generally available hote hain
    outcome_features = [
        "likes",
        "comments",
        "shares",
        "saves",
        "reach",
        "impressions",
        "engagement_rate",
        "followers_gained"
    ]

    print("\nPost-Publication Outcome Features:")

    for feature in outcome_features:
        print("-", feature)

    # 5. Feature availability check karna
    print("\nChecking Pre-Publication Features:")

    missing_input_features = []

    for feature in pre_publication_features:
        if feature not in df.columns:
            missing_input_features.append(feature)

    if not missing_input_features:
        print("All pre-publication features are available!")

    else:
        print("Missing pre-publication features:")

        for feature in missing_input_features:
            print("-", feature)

    # 6. Target column check karna
    print("\nChecking Target Column:")

    if target_column in df.columns:
        print("Target column is available!")

    else:
        print("Target column is missing!")

    # 7. Final selected features create karna
    selected_features = [
        feature
        for feature in pre_publication_features
        if feature in df.columns
    ]

    print("\nFinal Selected Features:")

    for feature in selected_features:
        print("-", feature)

    # 8. Selected features ka data check karna
    print("\nSelected Features Data Types:")

    print(df[selected_features].dtypes)

    # 9. Missing values check karna
    print("\nMissing Values in Selected Features:")

    print(df[selected_features].isnull().sum())

    # 10. Selected feature categories identify karna
    categorical_features = [
        "account_type",
        "media_type",
        "content_category",
        "traffic_source",
        "day_of_week"
    ]

    numerical_features = [
        "follower_count",
        "has_call_to_action",
        "post_hour",
        "caption_length",
        "hashtags_count"
    ]

    # Sirf available selected features ko retain karna
    selected_categorical_features = [
        feature
        for feature in categorical_features
        if feature in selected_features
    ]

    selected_numerical_features = [
        feature
        for feature in numerical_features
        if feature in selected_features
    ]

    print("\nCategorical Selected Features:")

    for feature in selected_categorical_features:
        print("-", feature)

    print("\nNumerical Selected Features:")

    for feature in selected_numerical_features:
        print("-", feature)

    # 11. Feature selection summary
    print("\nFeature Selection Summary:")

    print("Total Dataset Columns:", len(df.columns))
    print("Total Selected Features:", len(selected_features))
    print("Total Categorical Features:",
          len(selected_categorical_features))
    print("Total Numerical Features:",
          len(selected_numerical_features))
    print("Target Column:", target_column)

    print("\nFeature selection completed successfully!")