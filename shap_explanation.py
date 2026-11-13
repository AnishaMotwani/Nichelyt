
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "engagement_random_forest_pipeline.joblib"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset2_cleaned.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "shap_feature_importance.csv"
)


# ==========================================
# 2. LOAD MODEL AND DATA
# ==========================================

def load_model_and_data():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    model_bundle = joblib.load(MODEL_PATH)

    pipeline = model_bundle["pipeline"]

    dataset = pd.read_csv(DATA_PATH)

    return pipeline, dataset


# ==========================================
# 3. PREPARE FEATURES
# ==========================================

def prepare_features(dataset, feature_columns):

    data = dataset.copy()

    # Numeric conversion
    data["followersatposting"] = pd.to_numeric(
        data["followersatposting"],
        errors="coerce"
    )

    data["likes"] = pd.to_numeric(
        data["likes"],
        errors="coerce"
    )

    data["comments"] = pd.to_numeric(
        data["comments"],
        errors="coerce"
    )

    # Date processing
    parsed_date = pd.to_datetime(
        data["postcreateddate"],
        errors="coerce"
    )

    data["post_year"] = parsed_date.dt.year
    data["post_month"] = parsed_date.dt.month
    data["post_day"] = parsed_date.dt.day
    data["post_day_of_week"] = parsed_date.dt.dayofweek

    # Time processing
    parsed_time = pd.to_datetime(
        data["postcreatedtime"],
        errors="coerce"
    )

    data["post_hour"] = parsed_time.dt.hour

    # Caption processing
    caption = data["imagetext"].fillna("").astype(str)

    data["caption_length"] = caption.str.len()

    data["word_count"] = caption.str.split().str.len()

    data["hashtag_count"] = caption.str.count("#")

    data["mention_count"] = caption.str.count("@")

    data["is_caption_empty"] = (
        caption.str.strip().str.len() == 0
    ).astype(int)

    # Keep only the trained model features
    X = data[feature_columns].copy()

    return X


# ==========================================
# 4. CALCULATE SHAP VALUES
# ==========================================

def calculate_shap_values():

    print("\n" + "=" * 65)
    print("NICHELYT - SHAP FEATURE EXPLANATION")
    print("=" * 65)

    pipeline, dataset = load_model_and_data()

    # Retrieve feature information
    model_bundle = joblib.load(MODEL_PATH)

    feature_columns = model_bundle["feature_columns"]

    print("\nPreparing dataset features...")

    X = prepare_features(
        dataset,
        feature_columns
    )

    # Retrieve pipeline components
    preprocessor = pipeline.named_steps["preprocessor"]

    model = pipeline.named_steps["model"]

    # Transform features using the same preprocessing
    print("Applying preprocessing...")

    X_transformed = preprocessor.transform(X)

    # Convert sparse matrix to dense array if necessary
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed = np.asarray(X_transformed)

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    print("Calculating SHAP values...")

    # TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_transformed
    )

    # For regression, SHAP values are generally 2D:
    # samples x features
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    # Calculate mean absolute SHAP value
    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    # Create results DataFrame
    results = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": mean_absolute_shap
    })

    # Sort by SHAP importance
    results = (
        results
        .sort_values(
            by="mean_absolute_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # Save results
    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Display results
    print("\nSHAP FEATURE IMPORTANCE")
    print("-" * 65)

    for index, row in results.iterrows():

        print(
            f"{index + 1:2d}. "
            f"{row['feature']:40s} "
            f"Mean |SHAP|: "
            f"{row['mean_absolute_shap']:.6f}"
        )

    print("\n" + "-" * 65)

    print("SHAP analysis completed successfully.")

    print(f"Output path:\n{OUTPUT_PATH}")

    print("\nInterpretation:")

    print(
        "Mean absolute SHAP values show the average "
        "magnitude of feature contributions."
    )

    print(
        "They do not prove causation."
    )

    print(
        "This analysis explains the trained model, "
        "not Instagram's actual algorithm."
    )

    print("=" * 65)


# ==========================================
# 5. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    try:

        calculate_shap_values()

    except Exception as error:

        print(
            f"\nError during SHAP analysis:\n{error}"
        )