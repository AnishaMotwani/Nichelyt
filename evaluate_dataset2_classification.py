
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset2_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading Dataset 2...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# 3. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "followersatposting",
    "likes",
    "comments",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 4. CLEAN REQUIRED VALUES
# ============================================================

df = df.dropna(
    subset=[
        "followersatposting",
        "likes",
        "comments",
    ]
)

df = df[
    df["followersatposting"] > 0
].copy()


# ============================================================
# 5. CREATE ENGAGEMENT RATE
# ============================================================

df["engagement_rate"] = (
    df["likes"] + df["comments"]
) / df["followersatposting"]


# ============================================================
# 6. TEXT FEATURE ENGINEERING
# ============================================================

df["postscraped"] = (
    df["postscraped"]
    .fillna("")
    .astype(str)
)

df["caption_length"] = (
    df["postscraped"]
    .str.len()
)

df["word_count"] = (
    df["postscraped"]
    .str.split()
    .str.len()
)

df["hashtag_count"] = (
    df["postscraped"]
    .str.count("#")
)

df["mention_count"] = (
    df["postscraped"]
    .str.count("@")
)

df["is_caption_empty"] = (
    df["postscraped"]
    .str.strip()
    .eq("")
    .astype(int)
)


# ============================================================
# 7. DATE AND TIME FEATURE ENGINEERING
# ============================================================

df["post_datetime"] = pd.to_datetime(
    df["postcreateddate"].astype(str)
    + " "
    + df["postcreatedtime"].astype(str),
    errors="coerce"
)

df["post_year"] = (
    df["post_datetime"].dt.year
)

df["post_month"] = (
    df["post_datetime"].dt.month
)

df["post_day"] = (
    df["post_datetime"].dt.day
)

df["post_day_of_week"] = (
    df["post_datetime"].dt.dayofweek
)

df["post_hour"] = (
    df["post_datetime"].dt.hour
)


# ============================================================
# 8. SELECT FEATURES
# ============================================================

feature_columns = [
    "followersatposting",
    "type",
    "post_year",
    "post_month",
    "post_day",
    "post_day_of_week",
    "post_hour",
    "caption_length",
    "word_count",
    "hashtag_count",
    "mention_count",
    "is_caption_empty",
]

df = df.dropna(
    subset=["engagement_rate"]
).copy()

X = df[feature_columns].copy()

y_continuous = df["engagement_rate"].copy()


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train_continuous, y_test_continuous = (
    train_test_split(
        X,
        y_continuous,
        test_size=0.20,
        random_state=42,
    )
)


# ============================================================
# 10. CREATE ENGAGEMENT CATEGORIES
# ============================================================
# Thresholds are calculated ONLY from training data.
# This helps avoid test-data leakage.

low_threshold = y_train_continuous.quantile(1 / 3)
high_threshold = y_train_continuous.quantile(2 / 3)

print("\nEngagement category thresholds:")
print("Low threshold:", low_threshold)
print("High threshold:", high_threshold)


def create_engagement_category(values):
    return pd.cut(
        values,
        bins=[
            -np.inf,
            low_threshold,
            high_threshold,
            np.inf,
        ],
        labels=[
            "Low",
            "Medium",
            "High",
        ],
        include_lowest=True,
    ).astype(str)


y_train = create_engagement_category(
    y_train_continuous
)

y_test = create_engagement_category(
    y_test_continuous
)


print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# 11. PREPROCESSING
# ============================================================

numeric_features = [
    "followersatposting",
    "post_year",
    "post_month",
    "post_day",
    "post_day_of_week",
    "post_hour",
    "caption_length",
    "word_count",
    "hashtag_count",
    "mention_count",
    "is_caption_empty",
]

categorical_features = [
    "type",
]


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features,
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# ============================================================
# 12. DEFINE CLASSIFICATION MODELS
# ============================================================

models = {
    "Dummy Baseline": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# 13. TRAIN AND EVALUATE MODELS
# ============================================================

evaluation_results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "Low",
            "Medium",
            "High",
        ],
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Accuracy percentage:", round(accuracy * 100, 2), "%")

    print("Weighted Precision:", round(precision, 4))
    print("Weighted Recall:", round(recall, 4))
    print("Weighted F1-score:", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[
                "Low",
                "Medium",
                "High",
            ],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion)

    evaluation_results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "accuracy_percentage": accuracy * 100,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1_score": f1,
        }
    )


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    evaluation_results
)

output_path = (
    OUTPUT_DIR
    / "dataset2_classification_evaluation.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


print("\n" + "=" * 60)
print("FINAL CLASSIFICATION RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nResults saved at:")
print(output_path)