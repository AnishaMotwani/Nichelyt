
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset2_cleaned.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "engagement_random_forest_pipeline.joblib"

METRICS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "engagement_model_metrics.csv"
)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. BASIC DATA CLEANING
# --------------------------------------------------

df["followersatposting"] = pd.to_numeric(
    df["followersatposting"],
    errors="coerce"
)

df["likes"] = pd.to_numeric(
    df["likes"],
    errors="coerce"
)

df["comments"] = pd.to_numeric(
    df["comments"],
    errors="coerce"
)

# Remove invalid follower values
df = df[
    (df["followersatposting"] > 0)
    & (df["likes"] >= 0)
    & (df["comments"] >= 0)
].copy()


# --------------------------------------------------
# 4. TARGET VARIABLE
# --------------------------------------------------

# Engagement rate:
# (likes + comments) / followers

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


# --------------------------------------------------
# 5. FEATURE ENGINEERING
# --------------------------------------------------

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


# Time feature
parsed_time = pd.to_datetime(
    df["postcreatedtime"],
    errors="coerce"
)

df["post_hour"] = (
    parsed_time.dt.hour
)


# Text features
caption = (
    df["postscraped"]
    .fillna("")
    .astype(str)
)

df["caption_length"] = (
    caption.str.len()
)

df["word_count"] = (
    caption.str.split().str.len()
)

df["hashtag_count"] = (
    caption.str.count("#")
)

df["mention_count"] = (
    caption.str.count("@")
)

df["is_caption_empty"] = (
    caption.str.strip().eq("").astype(int)
)


# --------------------------------------------------
# 6. SELECT FEATURES AND TARGET
# --------------------------------------------------

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

X = df[feature_columns]

y = df["engagement_rate"]


# --------------------------------------------------
# 7. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 8. PREPROCESSING
# --------------------------------------------------

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
    "type"
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


categorical_pipeline = Pipeline(
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
        ),
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
        ),
    ]
)


# --------------------------------------------------
# 9. RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        ),
    ]
)


# --------------------------------------------------
# 10. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 11. EVALUATE MODEL
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n------------------------------")
print("MODEL EVALUATION")
print("------------------------------")

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R2   : {r2:.6f}")


# --------------------------------------------------
# 12. SAVE METRICS
# --------------------------------------------------

metrics_df = pd.DataFrame(
    [
        {
            "model": "Random Forest Regression",
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "split_type": "Random train-test split",
        }
    ]
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# --------------------------------------------------
# 13. RETRAIN ON COMPLETE DATA
# --------------------------------------------------

print("\nTraining final pipeline on complete dataset...")

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        ),
    ]
)

final_pipeline.fit(
    X,
    y
)


# --------------------------------------------------
# 14. SAVE COMPLETE PIPELINE
# --------------------------------------------------

joblib.dump(
    {
        "pipeline": final_pipeline,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    },
    MODEL_PATH
)


print("\n------------------------------")
print("MODEL SAVED SUCCESSFULLY")
print("------------------------------")

print("Model path:")
print(MODEL_PATH)

print("\nMetrics path:")
print(METRICS_PATH)

print("\nPrototype training completed.")