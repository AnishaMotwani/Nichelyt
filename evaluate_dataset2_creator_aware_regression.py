
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset2_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

FOLD_OUTPUT_FILE = (
    OUTPUT_DIR
    / "dataset2_creator_aware_regression_folds.csv"
)

SUMMARY_OUTPUT_FILE = (
    OUTPUT_DIR
    / "dataset2_creator_aware_regression_summary.csv"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("CREATOR-AWARE REGRESSION EVALUATION")
print("=" * 70)

print("Dataset shape:", df.shape)
print("Unique creators:", df["username"].nunique())


# ============================================================
# 3. DATA CLEANING
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
    subset=numeric_columns
).copy()

df = df[
    df["followersatposting"] > 0
].copy()


# ============================================================
# 4. TARGET VARIABLE
# ============================================================

df["engagement_rate"] = (
    df["likes"] + df["comments"]
) / df["followersatposting"]

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=["engagement_rate"]
).copy()

print("Rows after cleaning:", len(df))


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# Date features
df["postcreateddate"] = pd.to_datetime(
    df["postcreateddate"],
    errors="coerce"
)

df["post_year"] = (
    df["postcreateddate"].dt.year
)

df["post_month"] = (
    df["postcreateddate"].dt.month
)

df["post_day"] = (
    df["postcreateddate"].dt.day
)

df["post_day_of_week"] = (
    df["postcreateddate"].dt.dayofweek
)


# Time features
parsed_time = pd.to_datetime(
    df["postcreatedtime"].astype(str),
    errors="coerce"
)

df["post_hour"] = parsed_time.dt.hour


# Text features
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
# 6. FEATURES, TARGET AND GROUPS
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

X = df[feature_columns]

y = df["engagement_rate"]

groups = df["username"]


# ============================================================
# 7. PREPROCESSOR
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
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
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# 8. MODELS
# ============================================================

models = {
    "Dummy Baseline": DummyRegressor(
        strategy="mean"
    ),

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# 9. CREATOR-AWARE SPLITTER
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=10,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 10. TRAINING AND EVALUATION
# ============================================================

fold_results = []

for split_number, (
    train_indices,
    test_indices
) in enumerate(
    splitter.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_groups = groups.iloc[train_indices]
    test_groups = groups.iloc[test_indices]

    train_creators = set(
        train_groups.unique()
    )

    test_creators = set(
        test_groups.unique()
    )

    overlap = (
        train_creators
        .intersection(test_creators)
    )

    print("\n" + "-" * 70)

    print("Split:", split_number)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print(
        "Training creators:",
        len(train_creators)
    )
    print(
        "Testing creators:",
        len(test_creators)
    )
    print(
        "Creator overlap:",
        len(overlap)
    )

    if overlap:
        raise RuntimeError(
            "Creator leakage detected!"
        )

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor()
                ),
                (
                    "model",
                    deepcopy(model)
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

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        if y_test.nunique() > 1:

            r2 = r2_score(
                y_test,
                predictions
            )

        else:

            r2 = np.nan

        fold_results.append(
            {
                "split": split_number,
                "model": model_name,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "train_creators": len(
                    train_creators
                ),
                "test_creators": len(
                    test_creators
                ),
                "creator_overlap": len(
                    overlap
                ),
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            }
        )

        print(
            model_name,
            "| MAE:",
            round(mae, 6),
            "| RMSE:",
            round(rmse, 6),
            "| R2:",
            round(r2, 6)
        )


# ============================================================
# 11. SAVE FOLD RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fold_results_df = pd.DataFrame(
    fold_results
)

fold_results_df.to_csv(
    FOLD_OUTPUT_FILE,
    index=False
)


# ============================================================
# 12. SUMMARY
# ============================================================

summary_df = (
    fold_results_df
    .groupby("model")
    .agg(
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std")
    )
    .reset_index()
    .sort_values("mean_rmse")
)

summary_df.to_csv(
    SUMMARY_OUTPUT_FILE,
    index=False
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL CREATOR-AWARE REGRESSION SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nFiles saved successfully:")
print(FOLD_OUTPUT_FILE)
print(SUMMARY_OUTPUT_FILE)