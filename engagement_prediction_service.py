from pathlib import Path

import joblib
import pandas as pd


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "engagement_random_forest_pipeline.joblib"
)


# ==================================================
# 2. TEXT FEATURE EXTRACTION
# ==================================================

def extract_text_features(caption):
    """
    Extract text-based features from a caption.
    """

    if caption is None:
        caption = ""

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


# ==================================================
# 3. DATE AND TIME FEATURE EXTRACTION
# ==================================================

def extract_date_time_features(
    post_date,
    post_time
):
    """
    Extract date and time features.
    """

    date_value = pd.to_datetime(
        post_date,
        format="%Y-%m-%d",
        errors="coerce"
    )

    time_value = pd.to_datetime(
        post_time,
        format="%H:%M",
        errors="coerce"
    )

    if pd.isna(date_value):
        raise ValueError(
            "Invalid date format. "
            "Use YYYY-MM-DD."
        )

    if pd.isna(time_value):
        raise ValueError(
            "Invalid time format. "
            "Use HH:MM."
        )

    return {
        "post_year": date_value.year,
        "post_month": date_value.month,
        "post_day": date_value.day,
        "post_day_of_week": date_value.dayofweek,
        "post_hour": time_value.hour
    }


# ==================================================
# 4. CREATE INPUT DATAFRAME
# ==================================================

def create_input_dataframe(
    followers,
    post_type,
    caption,
    post_date,
    post_time
):
    """
    Create one-row DataFrame using
    the features expected by the model.
    """

    if followers is None:
        raise ValueError(
            "Followers value cannot be empty."
        )

    followers = float(followers)

    if followers <= 0:
        raise ValueError(
            "Followers must be greater than zero."
        )

    post_type = str(post_type).strip().title()

    if post_type == "":
        raise ValueError(
            "Post type cannot be empty."
        )

    text_features = extract_text_features(
        caption
    )

    date_time_features = (
        extract_date_time_features(
            post_date,
            post_time
        )
    )

    input_data = {
        "followersatposting": followers,

        "type": post_type,

        "post_year": (
            date_time_features["post_year"]
        ),

        "post_month": (
            date_time_features["post_month"]
        ),

        "post_day": (
            date_time_features["post_day"]
        ),

        "post_day_of_week": (
            date_time_features[
                "post_day_of_week"
            ]
        ),

        "post_hour": (
            date_time_features["post_hour"]
        ),

        "caption_length": (
            text_features["caption_length"]
        ),

        "word_count": (
            text_features["word_count"]
        ),

        "hashtag_count": (
            text_features["hashtag_count"]
        ),

        "mention_count": (
            text_features["mention_count"]
        ),

        "is_caption_empty": (
            text_features["is_caption_empty"]
        )
    }

    return pd.DataFrame([input_data])


# ==================================================
# 5. LOAD MODEL
# ==================================================

def load_model():
    """
    Load the trained engagement model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(
        model_bundle,
        dict
    ):
        raise TypeError(
            "Expected model bundle to be a dictionary."
        )

    if "pipeline" not in model_bundle:
        raise KeyError(
            "The model bundle does not contain "
            "the 'pipeline' key."
        )

    return model_bundle["pipeline"]


# ==================================================
# 6. PREDICT ENGAGEMENT
# ==================================================

def predict_engagement(
    followers,
    post_type,
    caption,
    post_date,
    post_time
):
    """
    Generate an engagement-rate estimate.

    Returns a JSON-compatible dictionary.
    """

    input_dataframe = (
        create_input_dataframe(
            followers=followers,
            post_type=post_type,
            caption=caption,
            post_date=post_date,
            post_time=post_time
        )
    )

    pipeline = load_model()

    prediction = pipeline.predict(
        input_dataframe
    )[0]

    prediction = float(prediction)

    # Engagement rate cannot be negative.
    prediction = max(
        prediction,
        0.0
    )

    estimated_percentage = (
        prediction * 100
    )

    return {
        "status": "success",
        "estimated_engagement_rate": round(
            prediction,
            6
        ),
        "estimated_engagement_percentage": round(
            estimated_percentage,
            4
        ),
        "input_features": {
            "followers": float(followers),
            "post_type": str(post_type),
            "caption": str(caption or ""),
            "post_date": str(post_date),
            "post_time": str(post_time)
        },
        "limitations": [
            (
                "This is an experimental "
                "historical estimate."
            ),
            (
                "The prediction is not guaranteed."
            ),
            (
                "The model was trained on "
                "the available dataset."
            ),
            (
                "The model may perform poorly "
                "for unseen creators."
            ),
            (
                "The model does not access "
                "Instagram's live algorithm."
            )
        ]
    }


# ==================================================
# 7. TEST FUNCTION
# ==================================================

if __name__ == "__main__":

    try:

        result = predict_engagement(
            followers=10000,
            post_type="Photo",
            caption=(
                "Learn machine learning "
                "step by step"
            ),
            post_date="2026-09-23",
            post_time="18:30"
        )

        print("\nPREDICTION RESULT")
        print("=" * 50)

        for key, value in result.items():
            print(f"{key}: {value}")

    except Exception as error:

        print("\nPrediction Error:")
        print(
            type(error).__name__,
            ":",
            error
        )