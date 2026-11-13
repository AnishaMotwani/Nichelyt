
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


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

    # 3. Input features define karna
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

    # 4. Input features (X) aur target (y) separate karna
    X = df[input_features]
    y = df[target_column]

    print("\nInput Features Shape:")
    print(X.shape)

    print("\nTarget Shape:")
    print(y.shape)

    # 5. Dataset ko training aur testing mein split karna
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 6. Train-test split ka result display karna
    print("\nTrain-Test Split Completed Successfully!")

    print("\nTraining Input Shape:")
    print(X_train.shape)

    print("\nTesting Input Shape:")
    print(X_test.shape)

    print("\nTraining Target Shape:")
    print(y_train.shape)

    print("\nTesting Target Shape:")
    print(y_test.shape)

    # 7. Training target classes check karna
    print("\nTraining Target Class Distribution:")
    print(y_train.value_counts())

    # 8. Testing target classes check karna
    print("\nTesting Target Class Distribution:")
    print(y_test.value_counts())

    print("\nTrain-test split completed successfully!")