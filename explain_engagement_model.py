
from pathlib import Path

import joblib
import pandas as pd


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "engagement_random_forest_pipeline.joblib"
)


# ==========================================
# 2. LOAD MODEL BUNDLE
# ==========================================

def load_model_bundle():
    """
    Load the saved model bundle.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model_bundle = joblib.load(MODEL_PATH)

    return model_bundle


# ==========================================
# 3. EXTRACT FEATURE IMPORTANCE
# ==========================================

def explain_model():

    print("\n" + "=" * 65)
    print("NICHELYT - FEATURE-BASED MODEL EXPLANATION")
    print("=" * 65)

    model_bundle = load_model_bundle()

    pipeline = model_bundle["pipeline"]

    # Retrieve preprocessing and model steps
    preprocessor = pipeline.named_steps["preprocessor"]

    model = pipeline.named_steps["model"]

    # Get the feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Get Random Forest feature importance values
    importances = model.feature_importances_

    # Create a DataFrame
    importance_dataframe = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Sort from highest to lowest importance
    importance_dataframe = (
        importance_dataframe
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # Calculate percentage
    importance_dataframe["importance_percentage"] = (
        importance_dataframe["importance"] * 100
    )

    # Display results
    print("\nFEATURE IMPORTANCE RANKING")
    print("-" * 65)

    for index, row in importance_dataframe.iterrows():

        feature_name = row["feature"]

        importance_value = row["importance"]

        importance_percentage = (
            row["importance_percentage"]
        )

        print(
            f"{index + 1:2d}. "
            f"{feature_name:40s} "
            f"Importance: {importance_value:.6f} "
            f"({importance_percentage:.2f}%)"
        )

    # Save results
    output_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "engagement_feature_importance.csv"
    )

    importance_dataframe.to_csv(
        output_path,
        index=False
    )

    print("\n" + "-" * 65)

    print("Feature importance results saved successfully.")

    print(f"Output path:\n{output_path}")

    print("\nInterpretation notes:")

    print(
        "1. Importance shows how useful a feature was "
        "to the Random Forest model."
    )

    print(
        "2. Importance does not prove causation."
    )

    print(
        "3. This is not the same as SHAP explanation."
    )

    print(
        "4. Results depend on the training dataset."
    )

    print("=" * 65)


# ==========================================
# 4. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    try:

        explain_model()

    except FileNotFoundError as error:

        print(f"\nFile Error: {error}")

    except Exception as error:

        print(f"\nUnexpected Error: {error}")