
from pathlib import Path
from datetime import datetime

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
# 2. TEXT FEATURE EXTRACTION
# ==========================================

def extract_text_features(caption):
    """
    Extract text-based features from the caption.
    """

    if not caption:
        caption = ""

    caption = str(caption).strip()

    caption_length = len(caption)

    word_count = len(caption.split())

    hashtag_count = caption.count("#")

    mention_count = caption.count("@")

    is_caption_empty = int(len(caption) == 0)

    return {
        "caption_length": caption_length,
        "word_count": word_count,
        "hashtag_count": hashtag_count,
        "mention_count": mention_count,
        "is_caption_empty": is_caption_empty
    }


# ==========================================
# 3. DATE AND TIME FEATURE EXTRACTION
# ==========================================

def extract_date_time_features(post_date, post_time):
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
            "Invalid date format. Please use YYYY-MM-DD."
        )

    if pd.isna(time_value):
        raise ValueError(
            "Invalid time format. Please use HH:MM."
        )

    return {
        "post_year": date_value.year,
        "post_month": date_value.month,
        "post_day": date_value.day,
        "post_day_of_week": date_value.dayofweek,
        "post_hour": time_value.hour
    }


# ==========================================
# 4. CREATE INPUT DATAFRAME
# ==========================================

def create_input_dataframe(
    followers,
    post_type,
    caption,
    post_date,
    post_time
):
    """
    Create a single-row DataFrame
    with the same features used during training.
    """

    text_features = extract_text_features(caption)

    date_time_features = extract_date_time_features(
        post_date,
        post_time
    )

    input_data = {
        "followersatposting": followers,

        "type": post_type,

        "post_year": date_time_features["post_year"],

        "post_month": date_time_features["post_month"],

        "post_day": date_time_features["post_day"],

        "post_day_of_week": (
            date_time_features["post_day_of_week"]
        ),

        "post_hour": date_time_features["post_hour"],

        "caption_length": text_features["caption_length"],

        "word_count": text_features["word_count"],

        "hashtag_count": text_features["hashtag_count"],

        "mention_count": text_features["mention_count"],

        "is_caption_empty": text_features["is_caption_empty"]
    }

    input_dataframe = pd.DataFrame([input_data])

    return input_dataframe


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

def load_model():
    """
    Load the saved model bundle.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at:\n{MODEL_PATH}"
        )

    model_bundle = joblib.load(MODEL_PATH)

    pipeline = model_bundle["pipeline"]

    return pipeline


# ==========================================
# 6. GET USER INPUT
# ==========================================

def get_user_input():
    """
    Collect input from the user.
    """

    print("\n" + "=" * 55)
    print("NICHELYT - ENGAGEMENT PREDICTION")
    print("=" * 55)

    # Followers
    followers_input = input(
        "\nEnter followers at posting "
        "(default: 10000): "
    ).strip()

    if followers_input == "":
        followers = 10000
    else:
        followers = float(followers_input)

    if followers <= 0:
        raise ValueError(
            "Followers must be greater than zero."
        )

    # Post type
    post_type = input(
        "Enter post type "
        "(Photo/Album/Video, default: Photo): "
    ).strip()

    if post_type == "":
        post_type = "Photo"

    post_type = post_type.title()

    allowed_types = ["Photo", "Album", "Video"]

    if post_type not in allowed_types:
        print(
            "\nWarning: This post type was not present "
            "in the expected categories."
        )

        print(
            "The model will process it as an unknown category."
        )

    # Caption
    caption = input(
        "\nEnter your caption "
        "(press Enter if there is no caption): "
    ).strip()

    # Date
    current_date = datetime.now().strftime("%Y-%m-%d")

    post_date = input(
        f"Enter post date YYYY-MM-DD "
        f"(default: {current_date}): "
    ).strip()

    if post_date == "":
        post_date = current_date

    # Time
    current_time = datetime.now().strftime("%H:%M")

    post_time = input(
        f"Enter post time HH:MM "
        f"(default: {current_time}): "
    ).strip()

    if post_time == "":
        post_time = current_time

    return (
        followers,
        post_type,
        caption,
        post_date,
        post_time
    )


# ==========================================
# 7. PREDICT ENGAGEMENT
# ==========================================

def predict_engagement():
    """
    Load the model, prepare the input,
    and generate an engagement prediction.
    """

    (
        followers,
        post_type,
        caption,
        post_date,
        post_time
    ) = get_user_input()

    input_dataframe = create_input_dataframe(
        followers=followers,
        post_type=post_type,
        caption=caption,
        post_date=post_date,
        post_time=post_time
    )

    pipeline = load_model()

    prediction = pipeline.predict(input_dataframe)[0]

    prediction = float(prediction)

    # Engagement rate should not be negative.
    prediction = max(prediction, 0.0)

    estimated_percentage = prediction * 100

    print("\n" + "=" * 55)
    print("PREDICTION RESULT")
    print("=" * 55)

    print(
        f"\nEstimated engagement rate: "
        f"{prediction:.6f}"
    )

    print(
        f"Estimated engagement percentage: "
        f"{estimated_percentage:.2f}%"
    )

    print("\nImportant limitations:")
    print(
        "- This is an experimental historical estimate."
    )

    print(
        "- It is not a guaranteed engagement prediction."
    )

    print(
        "- The model was trained on the available dataset."
    )

    print(
        "- It may perform poorly for unseen creators."
    )

    print(
        "- It does not access Instagram's live algorithm."
    )

    print("=" * 55)


# ==========================================
# 8. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    try:
        predict_engagement()

    except ValueError as error:
        print(f"\nInput Error: {error}")

    except FileNotFoundError as error:
        print(f"\nFile Error: {error}")

    except Exception as error:
        print(
            f"\nUnexpected Error: {error}"
        )