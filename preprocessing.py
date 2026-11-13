
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    # 4. Input features aur target separate karna
    X = df[input_features]
    y = df[target_column]

    print("\nInput Features Shape:")
    print(X.shape)

    print("\nTarget Shape:")
    print(y.shape)

    # 5. Train-test split karna
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain-Test Split Completed!")

    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)

    # 6. Categorical features define karna
    categorical_features = [
        "account_type",
        "media_type",
        "content_category",
        "traffic_source",
        "day_of_week"
    ]

    # 7. Numerical features define karna
    numerical_features = [
        "follower_count",
        "has_call_to_action",
        "post_hour",
        "caption_length",
        "hashtags_count"
    ]

    # 8. Numerical preprocessing pipeline
    numerical_transformer = StandardScaler()

    # 9. Categorical preprocessing pipeline
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    # 10. Combined preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                numerical_features
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    # 11. Training data par preprocessing fit karna
    X_train_processed = preprocessor.fit_transform(X_train)

    # 12. Testing data ko transform karna
    X_test_processed = preprocessor.transform(X_test)

    # 13. Processed data shapes display karna
    print("\nPreprocessing Completed Successfully!")

    print("Processed Training Data Shape:")
    print(X_train_processed.shape)

    print("\nProcessed Testing Data Shape:")
    print(X_test_processed.shape)

    # 14. Feature names obtain karna
    processed_feature_names = preprocessor.get_feature_names_out()

    print("\nTotal Processed Features:")
    print(len(processed_feature_names))

    print("\nFirst 20 Processed Feature Names:")

    for feature in processed_feature_names[:20]:
        print("-", feature)

    # 15. Final summary
    print("\nPreprocessing Summary:")
    print("Original Features:", len(input_features))
    print("Categorical Features:", len(categorical_features))
    print("Numerical Features:", len(numerical_features))
    print("Processed Features:", len(processed_feature_names))

    print("\nPreprocessing script completed successfully!")