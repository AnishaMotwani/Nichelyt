
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap


# ============================================================
# NICHELYT - LOCAL SHAP EXPLANATION
# ============================================================

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


def prepare_features(data, feature_columns):
    """
    Prepare features in the same way as the training script.
    """

    data = data.copy()

    # Numeric conversion
    data["followersatposting"] = pd.to_numeric(
        data["followersatposting"],
        errors="coerce"
    )

    # Date features
    parsed_date = pd.to_datetime(
        data["postcreateddate"],
        errors="coerce"
    )

    data["post_year"] = parsed_date.dt.year
    data["post_month"] = parsed_date.dt.month
    data["post_day"] = parsed_date.dt.day
    data["post_day_of_week"] = parsed_date.dt.dayofweek

    # Time feature
    parsed_time = pd.to_datetime(
        data["postcreatedtime"].astype(str),
        errors="coerce"
    )

    data["post_hour"] = parsed_time.dt.hour

    # Caption features
    caption = data["imagetext"].fillna("").astype(str)

    data["caption_length"] = caption.str.len()
    data["word_count"] = caption.str.split().str.len()

    data["hashtag_count"] = caption.str.count(r"#\w+")
    data["mention_count"] = caption.str.count(r"@\w+")

    data["is_caption_empty"] = (
        caption.str.strip().eq("").astype(int)
    )

    return data[feature_columns]


def main():

    print("=" * 65)
    print("NICHELYT - LOCAL SHAP EXPLANATION")
    print("=" * 65)

    print("\nLoading model...")

    bundle = joblib.load(MODEL_PATH)

    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    print("Model loaded successfully.")

    print("\nLoading dataset...")

    dataset = pd.read_csv(DATA_PATH)

    # Use a single record for local explanation
    sample = dataset.iloc[[0]].copy()

    X_sample = prepare_features(
        sample,
        feature_columns
    )

    print("Preparing sample features...")

    # Apply preprocessing
    X_transformed = preprocessor.transform(X_sample)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    print("Calculating local SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_transformed
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    local_values = shap_values[0]

    prediction = model.predict(X_transformed)[0]

    base_value = explainer.expected_value

    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": local_values,
        "absolute_shap_value": np.abs(local_values)
    })

    explanation_df = explanation_df.sort_values(
        by="absolute_shap_value",
        ascending=False
    )

    print("\n" + "=" * 65)
    print("LOCAL SHAP EXPLANATION")
    print("=" * 65)

    print(f"\nPredicted engagement rate: {prediction:.6f}")
    print(f"Predicted engagement percentage: {prediction * 100:.2f}%")
    print(f"Base model value: {float(base_value):.6f}")

    print("\nTop feature contributions:")
    print("-" * 65)

    for _, row in explanation_df.head(10).iterrows():

        feature = row["feature"]
        shap_value = row["shap_value"]

        direction = (
            "positive contribution"
            if shap_value > 0
            else "negative contribution"
        )

        print(
            f"{feature:<45} "
            f"{shap_value:+.6f} "
            f"({direction})"
        )

    output_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "local_shap_explanation.csv"
    )

    explanation_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "-" * 65)
    print("Local SHAP explanation saved successfully.")
    print(f"Output path: {output_path}")

    print("\nImportant limitations:")
    print("- SHAP explains this trained model, not Instagram's actual algorithm.")
    print("- Contributions do not prove causation.")
    print("- This is a single historical dataset-based explanation.")
    print("- The model may perform poorly for unseen creators.")


if __name__ == "__main__":
    main()