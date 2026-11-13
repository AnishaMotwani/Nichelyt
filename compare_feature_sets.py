import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# 1. Load dataset
data_folder = Path("data/raw")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found in data/raw")

else:
    file_path = csv_files[0]
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")

    # 2. Target
    target_column = "performance_bucket_label"
    y = df[target_column]

    # -------------------------------------------------
    # EXPERIMENT A: PRE-PUBLICATION FEATURES
    # -------------------------------------------------

    pre_features = [
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

    pre_categorical_features = [
        "account_type",
        "media_type",
        "content_category",
        "traffic_source",
        "day_of_week"
    ]

    pre_numerical_features = [
        "follower_count",
        "has_call_to_action",
        "post_hour",
        "caption_length",
        "hashtags_count"
    ]

    X_pre = df[pre_features]

    X_pre_train, X_pre_test, y_pre_train, y_pre_test = train_test_split(
        X_pre,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                pre_numerical_features
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                pre_categorical_features
            )
        ]
    )

    pre_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    pre_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", pre_model)
        ]
    )

    print("\n===================================")
    print("EXPERIMENT A: PRE-PUBLICATION")
    print("===================================")

    pre_pipeline.fit(X_pre_train, y_pre_train)

    pre_predictions = pre_pipeline.predict(X_pre_test)

    pre_accuracy = accuracy_score(
        y_pre_test,
        pre_predictions
    )

    pre_f1 = f1_score(
        y_pre_test,
        pre_predictions,
        average="macro"
    )

    print("Testing Accuracy:", round(pre_accuracy, 4))
    print("Macro F1-Score:", round(pre_f1, 4))

    # -------------------------------------------------
    # EXPERIMENT B: POST-PUBLICATION FEATURES
    # -------------------------------------------------

    post_features = [
        "likes",
        "comments",
        "shares",
        "saves",
        "reach",
        "impressions",
        "engagement_rate",
        "followers_gained"
    ]

    X_post = df[post_features]

    X_post_train, X_post_test, y_post_train, y_post_test = train_test_split(
        X_post,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    post_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    print("\n===================================")
    print("EXPERIMENT B: POST-PUBLICATION")
    print("===================================")

    post_model.fit(X_post_train, y_post_train)

    post_predictions = post_model.predict(X_post_test)

    post_accuracy = accuracy_score(
        y_post_test,
        post_predictions
    )

    post_f1 = f1_score(
        y_post_test,
        post_predictions,
        average="macro"
    )

    print("Testing Accuracy:", round(post_accuracy, 4))
    print("Macro F1-Score:", round(post_f1, 4))

    # -------------------------------------------------
    # COMPARISON
    # -------------------------------------------------

    print("\n===================================")
    print("EXPERIMENT COMPARISON")
    print("===================================")

    print(
        "Pre-publication Accuracy:",
        round(pre_accuracy, 4)
    )

    print(
        "Post-publication Accuracy:",
        round(post_accuracy, 4)
    )

    print(
        "Pre-publication Macro F1:",
        round(pre_f1, 4)
    )

    print(
        "Post-publication Macro F1:",
        round(post_f1, 4)
    )

    print("\nFeature-set comparison completed successfully!")