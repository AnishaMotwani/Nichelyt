
from pathlib import Path
from datetime import datetime
import re
import json

import joblib
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "engagement_random_forest_pipeline.joblib"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "counterfactual_comparison.csv"


# ============================================================
# 2. MODEL LOADING
# ============================================================

_model_bundle = None


def load_model():
    """
    Loads the trained engagement prediction model.

    The model is loaded only once and reused for
    subsequent predictions.
    """

    global _model_bundle

    if _model_bundle is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        _model_bundle = joblib.load(MODEL_PATH)

    return _model_bundle


# ============================================================
# 3. TEXT PROCESSING FUNCTIONS
# ============================================================

def count_words(text):
    """
    Counts words in a caption.

    Example:
    'Learn Python step by step'
    Output: 5
    """

    if not text:
        return 0

    return len(str(text).split())


def count_hashtags(text):
    """
    Counts hashtags in the caption.

    Example:
    'Learn AI #Python #MachineLearning'
    Output: 2
    """

    if not text:
        return 0

    hashtags = re.findall(r"#\w+", str(text))

    return len(hashtags)


def count_mentions(text):
    """
    Counts mentions in the caption.

    Example:
    'Hello @creator @student'
    Output: 2
    """

    if not text:
        return 0

    mentions = re.findall(r"@\w+", str(text))

    return len(mentions)


def create_caption_with_target_length(
    original_caption,
    target_word_count,
    hashtag_count
):
    """
    Creates a hypothetical caption with a target length.

    The original caption words are repeated until
    the target word count is reached.

    Hashtags are added separately.

    This is used only for model-based comparison.
    """

    original_caption = str(original_caption).strip()

    original_words = original_caption.split()

    if not original_words:
        original_words = ["Create", "valuable", "content"]

    generated_words = []

    while len(generated_words) < target_word_count:

        remaining_words = (
            target_word_count - len(generated_words)
        )

        generated_words.extend(
            original_words[:remaining_words]
        )

    generated_words = generated_words[:target_word_count]

    caption = " ".join(generated_words)

    hashtags = []

    for index in range(hashtag_count):
        hashtags.append(f"#Topic{index + 1}")

    if hashtags:
        caption = caption + " " + " ".join(hashtags)

    return caption


# ============================================================
# 4. INPUT VALIDATION
# ============================================================

def validate_inputs(
    followers,
    original_caption,
    post_date,
    post_time
):
    """
    Validates the user inputs before prediction.
    """

    if followers is None:
        raise ValueError(
            "Followers count cannot be empty."
        )

    if float(followers) <= 0:
        raise ValueError(
            "Followers count must be greater than zero."
        )

    if not original_caption or not str(original_caption).strip():
        raise ValueError(
            "Caption cannot be empty."
        )

    try:
        datetime.strptime(
            str(post_date),
            "%Y-%m-%d"
        )

    except ValueError:
        raise ValueError(
            "post_date must use YYYY-MM-DD format."
        )

    try:
        datetime.strptime(
            str(post_time),
            "%H:%M"
        )

    except ValueError:
        raise ValueError(
            "post_time must use HH:MM format."
        )


# ============================================================
# 5. FEATURE PREPARATION
# ============================================================

def prepare_features(
    followers,
    post_type,
    caption,
    post_date,
    post_time,
    feature_columns
):
    """
    Converts input values into the exact feature format
    expected by the trained machine learning model.
    """

    caption = str(caption)

    data = pd.DataFrame(
        [
            {
                "followersatposting": float(followers),

                "type": post_type,

                "postcreateddate": post_date,

                "postcreatedtime": post_time,

                "imagetext": caption,

                "caption_length": len(caption),

                "word_count": count_words(caption),

                "hashtag_count": count_hashtags(caption),

                "mention_count": count_mentions(caption),

                "is_caption_empty": int(
                    len(caption.strip()) == 0
                )
            }
        ]
    )

    # --------------------------------------------------------
    # Date-based features
    # --------------------------------------------------------

    data["postcreateddate"] = pd.to_datetime(
        data["postcreateddate"],
        errors="coerce"
    )

    if data["postcreateddate"].isna().any():
        raise ValueError(
            "Invalid post date provided."
        )

    data["post_year"] = (
        data["postcreateddate"].dt.year
    )

    data["post_month"] = (
        data["postcreateddate"].dt.month
    )

    data["post_day"] = (
        data["postcreateddate"].dt.day
    )

    data["post_day_of_week"] = (
        data["postcreateddate"].dt.dayofweek
    )

    # --------------------------------------------------------
    # Time-based features
    # --------------------------------------------------------

    parsed_time = pd.to_datetime(
        data["postcreatedtime"],
        format="%H:%M",
        errors="coerce"
    )

    if parsed_time.isna().any():
        raise ValueError(
            "Invalid post time provided."
        )

    data["post_hour"] = parsed_time.dt.hour

    # --------------------------------------------------------
    # Check all required model features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            + str(missing_features)
        )

    # Return only the feature columns expected
    # by the trained model.

    return data[feature_columns]


# ============================================================
# 6. SCENARIO CREATION
# ============================================================

def create_scenario(
    followers,
    post_type,
    caption,
    post_date,
    post_time,
    feature_columns,
    caption_length_label,
    hashtag_count
):
    """
    Creates one hypothetical posting scenario.
    """

    features = prepare_features(
        followers=followers,
        post_type=post_type,
        caption=caption,
        post_date=post_date,
        post_time=post_time,
        feature_columns=feature_columns
    )

    return {
        "scenario": (
            f"Post type - {post_type} | "
            f"Caption - {caption_length_label} | "
            f"Hashtags - {hashtag_count}"
        ),

        "post_type": post_type,

        "caption_length_label": caption_length_label,

        "hashtag_count": hashtag_count,

        "caption": caption,

        "features": features
    }


# ============================================================
# 7. PREDICT ONE SCENARIO
# ============================================================

def predict_scenario(
    model,
    scenario
):
    """
    Predicts engagement rate for one scenario.
    """

    features = scenario["features"]

    prediction = model.predict(features)

    predicted_engagement_rate = float(
        prediction[0]
    )

    predicted_engagement_percentage = (
        predicted_engagement_rate * 100
    )

    result = {
        "scenario": scenario["scenario"],

        "post_type": scenario["post_type"],

        "caption_length_label": (
            scenario["caption_length_label"]
        ),

        "hashtag_count": scenario["hashtag_count"],

        "caption": scenario["caption"],

        "predicted_engagement_rate": round(
            predicted_engagement_rate,
            6
        ),

        "predicted_engagement_percentage": round(
            predicted_engagement_percentage,
            4
        )
    }

    return result


# ============================================================
# 8. MAIN COUNTERFACTUAL SERVICE
# ============================================================

def run_counterfactual_comparison(
    followers,
    original_caption,
    post_date,
    post_time,
    save_csv=True
):
    """
    Runs counterfactual comparisons for different:

    1. Post types
       - Photo
       - Album
       - Video

    2. Caption lengths
       - Short
       - Medium
       - Long

    3. Hashtag counts
       - 3
       - 8

    Returns a JSON-compatible dictionary.
    """

    try:

        # ----------------------------------------------------
        # Validate user inputs
        # ----------------------------------------------------

        validate_inputs(
            followers=followers,
            original_caption=original_caption,
            post_date=post_date,
            post_time=post_time
        )

        # ----------------------------------------------------
        # Load model bundle
        # ----------------------------------------------------

        model_bundle = load_model()

        model = model_bundle["pipeline"]

        feature_columns = model_bundle[
            "feature_columns"
        ]

        # ----------------------------------------------------
        # Define hypothetical scenarios
        # ----------------------------------------------------

        post_types = [
            "Photo",
            "Album",
            "Video"
        ]

        caption_lengths = {
            "Short": 15,
            "Medium": 35,
            "Long": 70
        }

        hashtag_counts = [
            3,
            8
        ]

        results = []

        # ----------------------------------------------------
        # Generate and predict every scenario
        # ----------------------------------------------------

        for post_type in post_types:

            for caption_length_label, target_length in (
                caption_lengths.items()
            ):

                for hashtag_count in hashtag_counts:

                    generated_caption = (
                        create_caption_with_target_length(
                            original_caption=original_caption,

                            target_word_count=target_length,

                            hashtag_count=hashtag_count
                        )
                    )

                    scenario = create_scenario(
                        followers=followers,

                        post_type=post_type,

                        caption=generated_caption,

                        post_date=post_date,

                        post_time=post_time,

                        feature_columns=feature_columns,

                        caption_length_label=(
                            caption_length_label
                        ),

                        hashtag_count=hashtag_count
                    )

                    prediction_result = (
                        predict_scenario(
                            model=model,

                            scenario=scenario
                        )
                    )

                    results.append(
                        prediction_result
                    )

        # ----------------------------------------------------
        # Convert results into DataFrame
        # ----------------------------------------------------

        results_df = pd.DataFrame(results)

        if results_df.empty:
            raise ValueError(
                "No counterfactual results were generated."
            )

        # ----------------------------------------------------
        # Define baseline scenario
        # ----------------------------------------------------
        #
        # Baseline:
        # Photo + Short caption + 3 hashtags
        #

        baseline_condition = (
            (results_df["post_type"] == "Photo")
            &
            (
                results_df["caption_length_label"]
                == "Short"
            )
            &
            (
                results_df["hashtag_count"] == 3
            )
        )

        baseline_rows = results_df[
            baseline_condition
        ]

        if baseline_rows.empty:
            raise ValueError(
                "Baseline scenario could not be found."
            )

        baseline_engagement_rate = float(
            baseline_rows.iloc[0][
                "predicted_engagement_rate"
            ]
        )

        baseline_scenario = (
            "Post type - Photo | "
            "Caption - Short | "
            "Hashtags - 3"
        )

        # ----------------------------------------------------
        # Calculate differences from baseline
        # ----------------------------------------------------

        results_df[
            "difference_from_baseline"
        ] = (
            results_df[
                "predicted_engagement_rate"
            ]
            - baseline_engagement_rate
        )

        results_df[
            "percentage_point_difference"
        ] = (
            results_df[
                "predicted_engagement_percentage"
            ]
            - (
                baseline_engagement_rate * 100
            )
        )

        results_df[
            "difference_from_baseline"
        ] = results_df[
            "difference_from_baseline"
        ].round(6)

        results_df[
            "percentage_point_difference"
        ] = results_df[
            "percentage_point_difference"
        ].round(4)

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        output_path = None

        if save_csv:

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            csv_df = results_df.drop(
                columns=["caption"],
                errors="ignore"
            )

            csv_df.to_csv(
                OUTPUT_PATH,
                index=False
            )

            output_path = str(
                OUTPUT_PATH
            )

        # ----------------------------------------------------
        # Convert DataFrame to JSON-compatible records
        # ----------------------------------------------------

        output_results = results_df.to_dict(
            orient="records"
        )

        # Remove caption from response if required
        # to keep the API response smaller.

        for result in output_results:

            if "caption" in result:
                del result["caption"]

        # ----------------------------------------------------
        # Limitations
        # ----------------------------------------------------

        limitations = [
            (
                "These are hypothetical model-based "
                "comparisons, not real Instagram experiments."
            ),

            (
                "The predictions are based on historical "
                "training data."
            ),

            (
                "The results do not guarantee actual "
                "Instagram engagement."
            ),

            (
                "The model does not use live Instagram "
                "algorithm signals."
            ),

            (
                "Performance may vary for unseen creators "
                "and different audiences."
            ),

            (
                "SHAP or model predictions should not be "
                "interpreted as causal proof."
            )
        ]

        return {
            "status": "success",

            "input": {
                "followers": followers,

                "original_caption": original_caption,

                "post_date": post_date,

                "post_time": post_time
            },

            "baseline_scenario": baseline_scenario,

            "baseline_engagement_rate": round(
                baseline_engagement_rate,
                6
            ),

            "total_scenarios": len(
                output_results
            ),

            "results": output_results,

            "output_path": output_path,

            "limitations": limitations
        }

    except Exception as error:

        return {
            "status": "error",

            "error": str(error)
        }


# ============================================================
# 9. TESTING THE SERVICE
# ============================================================

if __name__ == "__main__":

    response = run_counterfactual_comparison(

        followers=10000,

        original_caption=(
            "Learn machine learning step by step"
        ),

        post_date="2026-09-23",

        post_time="18:30",

        save_csv=True
    )

    print(
        json.dumps(
            response,
            indent=4,
            default=str
        )
    )