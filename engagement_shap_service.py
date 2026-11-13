
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "engagement_random_forest_pipeline.joblib"
)


# ============================================================
# 2. LOAD MODEL BUNDLE
# ============================================================

def load_model_bundle():
    """
    Loads the saved model bundle.

    The bundle contains:
    - pipeline
    - feature_columns
    - numeric_features
    - categorical_features
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model_bundle = joblib.load(MODEL_PATH)

    required_keys = [
        "pipeline",
        "feature_columns",
        "numeric_features",
        "categorical_features"
    ]

    for key in required_keys:
        if key not in model_bundle:
            raise KeyError(
                f"Required key '{key}' is missing "
                "from the model bundle."
            )

    return model_bundle


# ============================================================
# 3. FEATURE EXTRACTION HELPERS
# ============================================================

def extract_text_features(caption):
    """
    Extracts text-based features from a caption.

    Features:
    - caption_length
    - word_count
    - hashtag_count
    - mention_count
    - is_caption_empty
    """

    caption = str(caption).strip()

    caption_length = len(caption)

    word_count = len(caption.split())

    hashtag_count = caption.count("#")

    mention_count = caption.count("@")

    is_caption_empty = int(
        len(caption) == 0
    )

    return {
        "caption_length": caption_length,
        "word_count": word_count,
        "hashtag_count": hashtag_count,
        "mention_count": mention_count,
        "is_caption_empty": is_caption_empty
    }


def extract_date_time_features(
    post_date,
    post_time
):
    """
    Extracts date and time features.

    Expected date format:
    YYYY-MM-DD

    Expected time format:
    HH:MM
    """

    date_time_text = (
        f"{post_date} {post_time}"
    )

    date_time = pd.to_datetime(
        date_time_text,
        errors="coerce"
    )

    if pd.isna(date_time):
        raise ValueError(
            "Invalid date or time. "
            "Use date format YYYY-MM-DD "
            "and time format HH:MM."
        )

    return {
        "post_year": int(date_time.year),

        "post_month": int(date_time.month),

        "post_day": int(date_time.day),

        "post_day_of_week": int(
            date_time.dayofweek
        ),

        "post_hour": int(date_time.hour)
    }


# ============================================================
# 4. CREATE MODEL INPUT
# ============================================================

def create_input_dataframe(
    followers,
    post_type,
    caption,
    post_date,
    post_time
):
    """
    Creates a DataFrame using the same feature
    names expected by the trained model.
    """

    text_features = extract_text_features(
        caption
    )

    date_features = extract_date_time_features(
        post_date,
        post_time
    )

    input_features = {
        "followersatposting": float(
            followers
        ),

        "type": str(post_type),

        **date_features,

        **text_features
    }

    input_dataframe = pd.DataFrame(
        [input_features]
    )

    return input_dataframe


# ============================================================
# 5. GET TRANSFORMED FEATURE NAMES
# ============================================================

def get_transformed_feature_names(
    preprocessor
):
    """
    Gets feature names after preprocessing.

    This is important because categorical columns,
    such as 'type', may be converted into one-hot
    encoded columns.
    """

    try:
        feature_names = (
            preprocessor.get_feature_names_out()
        )

        return [
            str(name)
            for name in feature_names
        ]

    except Exception:
        warnings.warn(
            "Could not extract transformed "
            "feature names."
        )

        return []


# ============================================================
# 6. CONVERT SHAP VALUES TO NUMPY ARRAY
# ============================================================

def convert_shap_values_to_array(
    shap_values
):
    """
    Converts SHAP output into a 2D NumPy array.

    Different SHAP versions may return:
    - NumPy arrays
    - Lists of arrays
    - Arrays with extra dimensions
    """

    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            raise ValueError(
                "SHAP returned an empty list."
            )

        shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(
            1,
            -1
        )

    return shap_values


# ============================================================
# 7. CALCULATE SHAP EXPLANATION
# ============================================================

def calculate_shap_explanation(
    pipeline,
    input_dataframe
):
    """
    Calculates SHAP contributions for one input.

    Workflow:
    1. Extract preprocessing step.
    2. Extract trained model.
    3. Transform input using the same preprocessor.
    4. Create a TreeExplainer.
    5. Calculate SHAP values.
    6. Match contributions with feature names.
    """

    if "preprocessor" not in pipeline.named_steps:
        raise KeyError(
            "Preprocessor step was not found."
        )

    if "model" not in pipeline.named_steps:
        raise KeyError(
            "Model step was not found."
        )

    preprocessor = (
        pipeline.named_steps["preprocessor"]
    )

    model = (
        pipeline.named_steps["model"]
    )

    # Transform the input using the same
    # preprocessing used during training.
    transformed_input = (
        preprocessor.transform(
            input_dataframe
        )
    )

    # Convert sparse matrix to dense matrix
    # when required by SHAP.
    if hasattr(
        transformed_input,
        "toarray"
    ):
        transformed_input = (
            transformed_input.toarray()
        )

    transformed_input = np.asarray(
        transformed_input
    )

    # Get transformed feature names.
    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    if not feature_names:
        feature_names = [
            f"feature_{index}"
            for index in range(
                transformed_input.shape[1]
            )
        ]

    # Create SHAP explainer.
    explainer = shap.TreeExplainer(
        model
    )

    # Calculate SHAP values.
    shap_values = explainer.shap_values(
        transformed_input
    )

    shap_values = (
        convert_shap_values_to_array(
            shap_values
        )
    )

    if shap_values.shape[1] != len(
        feature_names
    ):
        raise ValueError(
            "The number of SHAP values does not "
            "match the number of transformed "
            "feature names."
        )

    # Get the base value.
    expected_value = (
        explainer.expected_value
    )

    if isinstance(
        expected_value,
        np.ndarray
    ):
        expected_value = (
            expected_value.flatten()[0]
        )

    if isinstance(
        expected_value,
        list
    ):
        expected_value = (
            expected_value[0]
        )

    expected_value = float(
        np.asarray(
            expected_value
        ).reshape(-1)[0]
    )

    # Extract SHAP values for the first row.
    individual_shap_values = (
        shap_values[0]
    )

    # Create feature contribution records.
    contributions = []

    for index, feature_name in enumerate(
        feature_names
    ):
        contribution = float(
            individual_shap_values[index]
        )

        contributions.append(
            {
                "feature": str(
                    feature_name
                ),

                "shap_value": round(
                    contribution,
                    6
                ),

                "impact": (
                    "positive"
                    if contribution > 0
                    else (
                        "negative"
                        if contribution < 0
                        else "neutral"
                    )
                )
            }
        )

    # Sort features by absolute SHAP impact.
    contributions.sort(
        key=lambda item: abs(
            item["shap_value"]
        ),
        reverse=True
    )

    positive_factors = [
        item
        for item in contributions
        if item["shap_value"] > 0
    ]

    negative_factors = [
        item
        for item in contributions
        if item["shap_value"] < 0
    ]

    return {
        "base_value": round(
            expected_value,
            6
        ),

        "feature_contributions": contributions,

        "positive_factors": (
            positive_factors[:5]
        ),

        "negative_factors": (
            negative_factors[:5]
        ),

        "explanation_limitations": [
            "SHAP explains model output contributions.",
            "SHAP values do not prove real-world causation.",
            "The explanation depends on the trained model.",
            "The model may perform poorly for unseen creators."
        ]
    }


# ============================================================
# 8. COMPLETE PREDICTION + SHAP SERVICE
# ============================================================

def predict_with_shap(
    followers,
    post_type,
    caption,
    post_date,
    post_time
):
    """
    Generates:
    - Engagement prediction
    - SHAP explanation
    - Input details
    - Limitations
    """

    try:
        model_bundle = (
            load_model_bundle()
        )

        pipeline = (
            model_bundle["pipeline"]
        )

        input_dataframe = (
            create_input_dataframe(
                followers=followers,
                post_type=post_type,
                caption=caption,
                post_date=post_date,
                post_time=post_time
            )
        )

        # Generate prediction.
        prediction = pipeline.predict(
            input_dataframe
        )

        predicted_rate = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

        # Calculate SHAP explanation.
        shap_explanation = (
            calculate_shap_explanation(
                pipeline=pipeline,
                input_dataframe=input_dataframe
            )
        )

        return {
            "status": "success",

            "estimated_engagement_rate": round(
                predicted_rate,
                6
            ),

            "estimated_engagement_percentage": round(
                predicted_rate * 100,
                4
            ),

            "input_features": {
                "followers": float(
                    followers
                ),

                "post_type": str(
                    post_type
                ),

                "caption": str(
                    caption
                ),

                "post_date": str(
                    post_date
                ),

                "post_time": str(
                    post_time
                )
            },

            "shap_explanation": (
                shap_explanation
            ),

            "limitations": [
                "This is an experimental historical estimate.",
                "The prediction is not guaranteed.",
                "The model was trained on the available dataset.",
                "The model may perform poorly for unseen creators.",
                "The model does not access Instagram's live algorithm."
            ]
        }

    except Exception as error:
        return {
            "status": "error",

            "error_type": type(
                error
            ).__name__,

            "error_message": str(
                error
            ),

            "limitations": [
                "SHAP explanation could not be generated.",
                "Check model compatibility and installed dependencies."
            ]
        }


# ============================================================
# 9. TEST THE SERVICE
# ============================================================

if __name__ == "__main__":

    result = predict_with_shap(
        followers=10000,

        post_type="Photo",

        caption=(
            "Learn machine learning step by step"
        ),

        post_date="2026-09-23",

        post_time="18:30"
    )

    print(
        "\nSHAP PREDICTION RESULT"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            result,
            indent=4,
            default=str
        )
    )