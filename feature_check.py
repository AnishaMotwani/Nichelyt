
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

    # 2. Dataset ke saare columns display karna
    print("\nAll Dataset Columns:")

    for column in df.columns:
        print("-", column)

    # 3. Target variable define karna
    target_column = "performance_bucket_label"

    print("\nTarget Column:")
    print(target_column)

    # 4. Potential input features define karna
    input_features = [
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

    print("\nPotential Input Features:")

    for feature in input_features:
        print("-", feature)

    # 5. Check karna ki required features dataset mein hain ya nahi
    print("\nChecking Required Features:")

    missing_features = []

    for feature in input_features:
        if feature not in df.columns:
            missing_features.append(feature)

    if not missing_features:
        print("All input features are available!")

    else:
        print("Missing features:")

        for feature in missing_features:
            print("-", feature)

    # 6. Target column available hai ya nahi
    print("\nChecking Target Column:")

    if target_column in df.columns:
        print("Target column is available!")

    else:
        print("Target column is missing!")

    # 7. Input features ka data type check karna
    print("\nInput Features Data Types:")

    print(df[input_features].dtypes)

    # 8. Input features mein missing values check karna
    print("\nMissing Values in Input Features:")

    print(df[input_features].isnull().sum())

    # 9. Target classes check karna
    print("\nTarget Classes:")

    print(df[target_column].value_counts())

    # 10. Final feature summary
    print("\nFinal Feature Summary:")

    print("Total Rows:", len(df))
    print("Total Input Features:", len(input_features))
    print("Target Column:", target_column)

    print("\nFeature checking completed successfully!")