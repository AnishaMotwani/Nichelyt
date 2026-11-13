
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
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
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
# 3. CHECK CREATOR COLUMN
# ============================================================

if "username" not in df.columns:
    raise ValueError(
        "The 'username' column was not found in the dataset."
    )

df["username"] = (
    df["username"]
    .fillna("unknown_creator")
    .astype(str)
    .str.strip()
)

print("Unique creators:", df["username"].nunique())


# ============================================================
# 4. CONVERT NUMERIC COLUMNS
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
# 5. CLEAN DATA
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
# 6. CREATE ENGAGEMENT RATE
# ============================================================

df["engagement_rate"] = (
    df["likes"] + df["comments"]
) / df["followersatposting"]


# ============================================================
# 7. TEXT FEATURES
# ============================================================

df["postscraped"] = (
    df["postscraped"]
    .fillna("")
    .astype(str)
)

df["caption_length"] = (
    df["postscraped"].str.len()
)

df["word_count"] = (
    df["postscraped"]
    .str.split()
    .str.len()
)

df["hashtag_count"] = (
    df["postscraped"].str.count("#")
)

df["mention_count"] = (
    df["postscraped"].str.count("@")
)

df["is_caption_empty"] = (
    df["postscraped"]
    .str.strip()
    .eq("")
    .astype(int)
)


# ============================================================
# 8. DATE AND TIME FEATURES
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
# 9. SELECT FEATURES
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

groups = df["username"].copy()


# ============================================================
# 10. FEATURE GROUPS
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


# ============================================================
# 11. PREPROCESSOR FUNCTION
# ============================================================

def create_preprocessor():

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
                SimpleImputer(
                    strategy="most_frequent"
                ),
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

    return preprocessor


# ============================================================
# 12. CATEGORY CREATION FUNCTION
# ============================================================

def create_categories(
    values,
    low_threshold,
    high_threshold,
):

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


# ============================================================
# 13. MODELS
# ============================================================

def create_models():

    return {
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
# 14. CREATOR-AWARE CROSS-VALIDATION
# ============================================================

group_kfold = GroupKFold(n_splits=5)

all_results = []

fold_number = 1

for train_indices, test_indices in group_kfold.split(
    X,
    y_continuous,
    groups=groups,
):

    print("\n" + "=" * 70)
    print("FOLD:", fold_number)
    print("=" * 70)

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    y_train_continuous = (
        y_continuous.iloc[train_indices]
    )

    y_test_continuous = (
        y_continuous.iloc[test_indices]
    )

    train_groups = groups.iloc[train_indices]
    test_groups = groups.iloc[test_indices]

    print(
        "Training creators:",
        train_groups.nunique()
    )

    print(
        "Testing creators:",
        test_groups.nunique()
    )

    overlap = set(train_groups).intersection(
        set(test_groups)
    )

    print(
        "Creator overlap:",
        len(overlap)
    )

    if len(overlap) != 0:
        raise ValueError(
            "Creator leakage detected!"
        )

    # Thresholds are calculated only on the
    # training portion of this fold.
    low_threshold = (
        y_train_continuous.quantile(1 / 3)
    )

    high_threshold = (
        y_train_continuous.quantile(2 / 3)
    )

    y_train = create_categories(
        y_train_continuous,
        low_threshold,
        high_threshold,
    )

    y_test = create_categories(
        y_test_continuous,
        low_threshold,
        high_threshold,
    )

    models = create_models()

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(),
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

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        macro_precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        print("\nModel:", model_name)
        print(
            "Accuracy:",
            round(accuracy, 4)
        )
        print(
            "Macro Precision:",
            round(macro_precision, 4)
        )
        print(
            "Macro Recall:",
            round(macro_recall, 4)
        )
        print(
            "Macro F1-score:",
            round(macro_f1, 4)
        )
        print(
            "Weighted F1-score:",
            round(weighted_f1, 4)
        )

        all_results.append(
            {
                "fold": fold_number,
                "model": model_name,
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1_score": macro_f1,
                "weighted_f1_score": weighted_f1,
            }
        )

    fold_number += 1


# ============================================================
# 15. SAVE FOLD-WISE RESULTS
# ============================================================

fold_results_df = pd.DataFrame(
    all_results
)

fold_output_path = (
    OUTPUT_DIR
    / "dataset2_creator_aware_classification_folds.csv"
)

fold_results_df.to_csv(
    fold_output_path,
    index=False,
)


# ============================================================
# 16. CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

metric_columns = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1_score",
    "weighted_f1_score",
]

summary_rows = []

for model_name in (
    fold_results_df["model"].unique()
):

    model_results = fold_results_df[
        fold_results_df["model"] == model_name
    ]

    summary_row = {
        "model": model_name,
    }

    for metric in metric_columns:

        summary_row[
            metric + "_mean"
        ] = model_results[metric].mean()

        summary_row[
            metric + "_std"
        ] = model_results[metric].std()

    summary_rows.append(summary_row)


summary_df = pd.DataFrame(
    summary_rows
)

summary_output_path = (
    OUTPUT_DIR
    / "dataset2_creator_aware_classification_summary.csv"
)

summary_df.to_csv(
    summary_output_path,
    index=False,
)


# ============================================================
# 17. DISPLAY FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CREATOR-AWARE CLASSIFICATION SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nFold-wise results saved at:")
print(fold_output_path)

print("\nSummary results saved at:")
print(summary_output_path)

print("\nEvaluation completed successfully!")