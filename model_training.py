
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


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

    # 5. Categorical features define karna
    categorical_features = [
        "account_type",
        "media_type",
        "content_category",
        "traffic_source",
        "day_of_week"
    ]

    # 6. Numerical features define karna
    numerical_features = [
        "follower_count",
        "has_call_to_action",
        "post_hour",
        "caption_length",
        "hashtags_count"
    ]

    # 7. Train-test split karna
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

    # 8. Numerical preprocessing
    numerical_transformer = StandardScaler()

    # 9. Categorical preprocessing
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    # 10. Combined preprocessing
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

    # 11. Random Forest model define karna
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # 12. Preprocessing aur model ko pipeline mein combine karna
    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    print("\nModel Pipeline Created Successfully!")

    # 13. Model training
    print("\nModel Training Started...")

    pipeline.fit(X_train, y_train)

    print("Model Training Completed Successfully!")

    # 14. Training data par predictions
    y_train_pred = pipeline.predict(X_train)

    # 15. Testing data par predictions
    y_test_pred = pipeline.predict(X_test)

    # 16. Training accuracy
    train_accuracy = accuracy_score(
        y_train,
        y_train_pred
    )

    # 17. Testing accuracy
    test_accuracy = accuracy_score(
        y_test,
        y_test_pred
    )

    print("\nTraining Accuracy:")
    print(round(train_accuracy, 4))

    print("\nTesting Accuracy:")
    print(round(test_accuracy, 4))

    # 18. Classification report
    print("\nClassification Report:")
    print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)
    print("\nModel training and basic evaluation completed!")