
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# 1. Dataset load karna
data_folder = Path("data/raw")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found in data/raw")

else:
    file_path = csv_files[0]
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")

    # 2. Target column
    target_column = "performance_bucket_label"

    # 3. Input features
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

    # 4. Separate input and target
    X = df[input_features]
    y = df[target_column]

    # 5. Feature types
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

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 7. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_features
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    # 8. Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # 9. Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    # 10. Training
    print("\nModel training started...")
    pipeline.fit(X_train, y_train)
    print("Model training completed!")

    # 11. Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # 12. Accuracy
    train_accuracy = accuracy_score(
        y_train,
        y_train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        y_test_pred
    )

    print("\n========== ACCURACY RESULTS ==========")

    print("Training Accuracy:", round(train_accuracy, 4))
    print("Testing Accuracy:", round(test_accuracy, 4))

    # 13. Classification report
    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            y_test_pred,
            zero_division=0
        )
    )

    # 14. Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=["low", "medium", "high", "viral"]
    )

    print("\n========== CONFUSION MATRIX ==========")

    print("Labels: low, medium, high, viral")
    print(cm)

    # 15. Interpretation
    print("\n========== INTERPRETATION ==========")

    print(
        "The model was evaluated using pre-publication features only."
    )

    print(
        "Post-publication engagement metrics were not used as inputs."
    )