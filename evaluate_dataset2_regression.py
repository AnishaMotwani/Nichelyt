
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "dataset2_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "dataset2_regression_evaluation.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET 2")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# 3. BASIC CLEANING
# ============================================================

print("\n" + "=" * 70)
print("BASIC CLEANING")
print("=" * 70)

# Convert important numeric columns to numeric format
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


# Remove rows with missing required numeric values
df = df.dropna(
    subset=[
        "followersatposting",
        "likes",
        "comments"
    ]
).copy()


# Remove rows with zero or negative followers
df = df[
    df["followersatposting"] > 0
].copy()


print("Shape after basic cleaning:", df.shape)


# ============================================================
# 4. CREATE TARGET VARIABLE
# ============================================================

print("\n" + "=" * 70)
print("CREATING TARGET VARIABLE")
print("=" * 70)

# Dataset-derived engagement ratio
df["engagement_rate"] = (
    df["likes"] + df["comments"]
) / df["followersatposting"]


# Remove invalid target values
df = df[
    np.isfinite(df["engagement_rate"])
].copy()


print("Target variable: engagement_rate")

print("\nTarget summary:")
print(df["engagement_rate"].describe())


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# Convert date and time columns
df["postcreateddate"] = pd.to_datetime(
    df["postcreateddate"],
    errors="coerce"
)

df["postcreatedtime"] = pd.to_datetime(
    df["postcreatedtime"],
    errors="coerce"
)


# Extract date-related features
df["post_year"] = df["postcreateddate"].dt.year
df["post_month"] = df["postcreateddate"].dt.month
df["post_day"] = df["postcreateddate"].dt.day
df["post_day_of_week"] = df["postcreateddate"].dt.dayofweek


# Extract hour from the time column
df["post_hour"] = df["postcreatedtime"].dt.hour


# Prepare caption text
df["postscraped"] = df["postscraped"].fillna("").astype(str)


# Caption length
df["caption_length"] = df["postscraped"].str.len()


# Word count
df["word_count"] = df["postscraped"].str.split().str.len()


# Hashtag count
df["hashtag_count"] = df["postscraped"].str.count("#")


# Mention count
df["mention_count"] = df["postscraped"].str.count("@")


# Whether caption is empty
df["is_caption_empty"] = (
    df["postscraped"].str.strip() == ""
).astype(int)


print("Feature engineering completed.")


# ============================================================
# 6. SELECT FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("SELECTING FEATURES")
print("=" * 70)

# IMPORTANT:
# We intentionally do not use likes, comments, or engagement_rate
# as input features because they are directly connected to the target.

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


X = df[feature_columns].copy()

y = df["engagement_rate"].copy()


print("Input feature columns:")
print(feature_columns)

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 8. DEFINE FEATURE TYPES
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
    "is_caption_empty"
]


categorical_features = [
    "type"
]


# ============================================================
# 9. PREPROCESSING PIPELINE
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
# 10. DEFINE MODELS
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
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)

evaluation_results = []

trained_models = {}


for model_name, model in models.items():

    print("\n" + "-" * 70)
    print("Training:", model_name)
    print("-" * 70)

    # Combine preprocessing and model
    model_pipeline = Pipeline(
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


    # Train the model
    model_pipeline.fit(
        X_train,
        y_train
    )


    # Generate predictions
    y_pred = model_pipeline.predict(
        X_test
    )


    # Calculate evaluation metrics
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


    # Store trained model
    trained_models[model_name] = model_pipeline


    # Store evaluation results
    evaluation_results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )


    print("MAE:", round(mae, 6))
    print("RMSE:", round(rmse, 6))
    print("R2:", round(r2, 6))


# ============================================================
# 12. COMPARE ALL MODELS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    evaluation_results
)


# Sort primarily by RMSE
results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(
    drop=True
)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE EVALUATION RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nEvaluation results saved at:")
print(OUTPUT_PATH)


# ============================================================
# 14. BEST MODEL SUMMARY
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL ACCORDING TO RMSE")
print("=" * 70)

print("Model:", best_model["Model"])
print("MAE:", round(best_model["MAE"], 6))
print("RMSE:", round(best_model["RMSE"], 6))
print("R2:", round(best_model["R2"], 6))


print("\n" + "=" * 70)
print("DATASET 2 REGRESSION EVALUATION COMPLETED")
print("=" * 70)