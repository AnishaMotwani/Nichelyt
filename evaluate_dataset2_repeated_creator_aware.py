
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "dataset2_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. BASIC CLEANING
# ============================================================

numeric_columns = [
    "followersatposting",
    "likes",
    "comments"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "followersatposting",
        "likes",
        "comments",
        "username"
    ]
)


df = df[
    df["followersatposting"] > 0
].copy()


# ============================================================
# 4. CREATE ENGAGEMENT RATE
# ============================================================

df["engagement_rate"] = (
    df["likes"] + df["comments"]
) / df["followersatposting"]


# ============================================================
# 5. DATE AND TEXT FEATURES
# ============================================================

df["postcreateddate"] = pd.to_datetime(
    df["postcreateddate"],
    errors="coerce"
)


df["postcreatedtime"] = df[
    "postcreatedtime"
].fillna("").astype(str)


df["post_year"] = df[
    "postcreateddate"
].dt.year


df["post_month"] = df[
    "postcreateddate"
].dt.month


df["post_day"] = df[
    "postcreateddate"
].dt.day


df["post_day_of_week"] = df[
    "postcreateddate"
].dt.dayofweek


df["post_hour"] = pd.to_datetime(
    df["postcreatedtime"],
    errors="coerce"
).dt.hour


df["postscraped"] = df[
    "postscraped"
].fillna("").astype(str)


df["caption_length"] = df[
    "postscraped"
].str.len()


df["word_count"] = df[
    "postscraped"
].str.split().str.len()


df["hashtag_count"] = df[
    "postscraped"
].str.count("#")


df["mention_count"] = df[
    "postscraped"
].str.count("@")


df["is_caption_empty"] = (
    df["postscraped"].str.strip() == ""
).astype(int)


# ============================================================
# 6. FEATURE SELECTION
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
    "is_caption_empty"
]


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
    "is_caption_empty"
]


categorical_features = [
    "type"
]


X = df[feature_columns].copy()

groups = df["username"].copy()


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 8. MODELS
# ============================================================

models = {
    "Dummy Classifier": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 9. REPEATED CREATOR-AWARE SPLITS
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=10,
    test_size=0.20,
    random_state=42
)


all_results = []


# ============================================================
# 10. TRAIN AND EVALUATE
# ============================================================

for split_number, (train_indices, test_indices) in enumerate(
    splitter.split(X, groups=groups),
    start=1
):

    print("\n" + "=" * 60)
    print(f"Split {split_number}")
    print("=" * 60)


    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()


    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()


    y_train_continuous = train_df[
        "engagement_rate"
    ]


    y_test_continuous = test_df[
        "engagement_rate"
    ]


    train_creators = set(
        train_df["username"]
    )

    test_creators = set(
        test_df["username"]
    )


    overlap = train_creators.intersection(
        test_creators
    )


    print("Training creators:", len(train_creators))
    print("Testing creators:", len(test_creators))
    print("Creator overlap:", len(overlap))


    # --------------------------------------------------------
    # Create thresholds only from training data
    # --------------------------------------------------------

    low_threshold = y_train_continuous.quantile(
        0.333333
    )

    high_threshold = y_train_continuous.quantile(
        0.666666
    )


    def create_classes(values):
        return pd.cut(
            values,
            bins=[
                -np.inf,
                low_threshold,
                high_threshold,
                np.inf
            ],
            labels=[
                "Low",
                "Medium",
                "High"
            ],
            include_lowest=True
        ).astype(str)


    y_train = create_classes(
        y_train_continuous
    )

    y_test = create_classes(
        y_test_continuous
    )


    # --------------------------------------------------------
    # Evaluate every model
    # --------------------------------------------------------

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )


        pipeline.fit(
            X_train,
            y_train
        )


        predictions = pipeline.predict(
            X_test
        )


        accuracy = accuracy_score(
            y_test,
            predictions
        )


        macro_precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )


        macro_recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )


        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )


        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        result = {
            "split": split_number,
            "model": model_name,
            "training_creators": len(train_creators),
            "testing_creators": len(test_creators),
            "training_rows": len(train_indices),
            "testing_rows": len(test_indices),
            "creator_overlap": len(overlap),
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }


        all_results.append(result)


        print("\nModel:", model_name)
        print("Accuracy:", round(accuracy * 100, 2), "%")
        print(
            "Macro Precision:",
            round(macro_precision, 4)
        )
        print(
            "Macro Recall:",
            round(macro_recall, 4)
        )
        print(
            "Macro F1:",
            round(macro_f1, 4)
        )
        print(
            "Weighted F1:",
            round(weighted_f1, 4)
        )


# ============================================================
# 11. SAVE FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


fold_output_path = (
    OUTPUT_DIR /
    "dataset2_repeated_creator_aware_folds.csv"
)


results_df.to_csv(
    fold_output_path,
    index=False
)


# ============================================================
# 12. CREATE SUMMARY
# ============================================================

summary_df = results_df.groupby(
    "model"
).agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    mean_macro_precision=("macro_precision", "mean"),
    mean_macro_recall=("macro_recall", "mean"),
    mean_macro_f1=("macro_f1", "mean"),
    mean_weighted_f1=("weighted_f1", "mean")
).reset_index()


summary_df = summary_df.sort_values(
    by="mean_macro_f1",
    ascending=False
)


summary_output_path = (
    OUTPUT_DIR /
    "dataset2_repeated_creator_aware_summary.csv"
)


summary_df.to_csv(
    summary_output_path,
    index=False
)


# ============================================================
# 13. DISPLAY FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL REPEATED CREATOR-AWARE SUMMARY")
print("=" * 60)


print(
    summary_df.to_string(
        index=False
    )
)


print("\nFiles saved successfully:")
print(fold_output_path)
print(summary_output_path)