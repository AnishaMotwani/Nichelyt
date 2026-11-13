
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "engagement_random_forest_pipeline.joblib"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "counterfactual_comparison.csv"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading trained engagement model...")

bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]

print("Model loaded successfully.")


# ============================================================
# 3. USER INPUT FUNCTIONS
# ============================================================

def get_followers():
    while True:
        value = input(
            "\nEnter followers count "
            "(press Enter for 10000): "
        ).strip()

        if value == "":
            return 10000

        try:
            followers = int(value)

            if followers <= 0:
                print("Followers must be greater than zero.")
                continue

            return followers

        except ValueError:
            print("Enter a valid whole number.")


def get_caption():
    caption = input(
        "\nEnter your original caption:\n"
    ).strip()

    if caption == "":
        return (
            "Discover simple practical ideas to improve "
            "your daily life and achieve your goals."
        )

    return caption


def get_date():
    while True:
        value = input(
            "\nEnter date YYYY-MM-DD "
            "(press Enter for today's date): "
        ).strip()

        if value == "":
            return datetime.now().strftime("%Y-%m-%d")

        try:
            parsed_date = pd.to_datetime(
                value,
                format="%Y-%m-%d",
                errors="raise"
            )

            return parsed_date.strftime("%Y-%m-%d")

        except ValueError:
            print("Invalid date. Example: 2026-09-23")


def get_time():
    while True:
        value = input(
            "\nEnter time HH:MM "
            "(press Enter for current time): "
        ).strip()

        if value == "":
            return datetime.now().strftime("%H:%M")

        try:
            parsed_time = pd.to_datetime(
                value,
                format="%H:%M",
                errors="raise"
            )

            return parsed_time.strftime("%H:%M")

        except ValueError:
            print("Invalid time. Example: 14:30")


# ============================================================
# 4. CAPTION FEATURE HELPERS
# ============================================================

def count_words(text):
    return len(text.split())


def create_caption_with_target_length(
    original_caption,
    target_length,
    hashtag_count
):
    """
    Creates a caption variant.

    The target length is approximate.
    Hashtags are added separately.
    """

    base_words = original_caption.split()

    if len(base_words) >= target_length:
        selected_words = base_words[:target_length]

    else:
        selected_words = base_words.copy()

        extra_words = [
            "Learn",
            "explore",
            "improve",
            "discover",
            "simple",
            "practical",
            "ideas",
            "tips",
            "today",
            "step",
            "by",
            "step",
            "with",
            "confidence",
            "and",
            "consistency"
        ]

        index = 0

        while len(selected_words) < target_length:
            selected_words.append(
                extra_words[index % len(extra_words)]
            )

            index += 1

    hashtag_list = [
        "#AI",
        "#Technology",
        "#Innovation",
        "#Learning",
        "#Growth",
        "#Tips",
        "#Digital",
        "#Creator",
        "#Content",
        "#Strategy"
    ]

    selected_hashtags = hashtag_list[:hashtag_count]

    final_caption = (
        " ".join(selected_words)
        + " "
        + " ".join(selected_hashtags)
    )

    return final_caption.strip()


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def prepare_features(data):
    """
    Creates the same features used during model training.
    """

    data = data.copy()

    data["followersatposting"] = pd.to_numeric(
        data["followersatposting"],
        errors="coerce"
    )

    data["postcreateddate"] = pd.to_datetime(
        data["postcreateddate"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    data["post_year"] = data["postcreateddate"].dt.year
    data["post_month"] = data["postcreateddate"].dt.month
    data["post_day"] = data["postcreateddate"].dt.day

    data["post_day_of_week"] = (
        data["postcreateddate"].dt.dayofweek
    )

    data["postcreatedtime"] = pd.to_datetime(
        data["postcreatedtime"],
        format="mixed",
        errors="coerce"
    )

    data["post_hour"] = data["postcreatedtime"].dt.hour

    data["imagetext"] = (
        data["imagetext"]
        .fillna("")
        .astype(str)
    )

    data["caption_length"] = data["imagetext"].str.len()

    data["word_count"] = (
        data["imagetext"]
        .str.split()
        .str.len()
    )

    data["hashtag_count"] = (
        data["imagetext"]
        .str.count(r"#\w+")
    )

    data["mention_count"] = (
        data["imagetext"]
        .str.count(r"@\w+")
    )

    data["is_caption_empty"] = (
        data["imagetext"]
        .str.strip()
        .eq("")
        .astype(int)
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    return data[feature_columns]


# ============================================================
# 6. CREATE ONE SCENARIO
# ============================================================

def create_scenario(
    followers,
    post_type,
    caption,
    post_date,
    post_time,
    scenario_name
):
    return {
        "scenario": scenario_name,
        "followersatposting": followers,
        "type": post_type,
        "postcreateddate": post_date,
        "postcreatedtime": post_time,
        "imagetext": caption,
        "username": "custom_user"
    }


# ============================================================
# 7. PREDICT ONE SCENARIO
# ============================================================

def predict_scenario(scenario):
    scenario_dataframe = pd.DataFrame([scenario])

    X_scenario = prepare_features(
        scenario_dataframe
    )

    prediction = pipeline.predict(
        X_scenario
    )[0]

    return float(prediction)


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

print("\n" + "=" * 70)
print("NICHELYT - COUNTERFACTUAL COMPARISON")
print("=" * 70)

followers = get_followers()
original_caption = get_caption()
post_date = get_date()
post_time = get_time()


# ============================================================
# 9. CREATE COUNTERFACTUAL SCENARIOS
# ============================================================

scenarios = []

# ------------------------------------------------------------
# A. Post type comparison
# ------------------------------------------------------------

post_types = [
    "Photo",
    "Album",
    "Video"
]

for post_type in post_types:

    scenario = create_scenario(
        followers=followers,
        post_type=post_type,
        caption=original_caption,
        post_date=post_date,
        post_time=post_time,
        scenario_name=f"Post type - {post_type}"
    )

    scenarios.append(scenario)


# ------------------------------------------------------------
# B. Caption length comparison
# ------------------------------------------------------------

caption_length_variants = [
    ("Short caption", 15),
    ("Medium caption", 35),
    ("Long caption", 70)
]

for scenario_name, target_length in caption_length_variants:

    caption_variant = create_caption_with_target_length(
        original_caption=original_caption,
        target_length=target_length,
        hashtag_count=3
    )

    scenario = create_scenario(
        followers=followers,
        post_type="Photo",
        caption=caption_variant,
        post_date=post_date,
        post_time=post_time,
        scenario_name=scenario_name
    )

    scenarios.append(scenario)


# ------------------------------------------------------------
# C. Hashtag count comparison
# ------------------------------------------------------------

hashtag_variants = [
    ("3 hashtags", 3),
    ("8 hashtags", 8)
]

for scenario_name, hashtag_count in hashtag_variants:

    caption_variant = create_caption_with_target_length(
        original_caption=original_caption,
        target_length=35,
        hashtag_count=hashtag_count
    )

    scenario = create_scenario(
        followers=followers,
        post_type="Photo",
        caption=caption_variant,
        post_date=post_date,
        post_time=post_time,
        scenario_name=scenario_name
    )

    scenarios.append(scenario)


# ============================================================
# 10. RUN PREDICTIONS
# ============================================================

results = []

print("\nCalculating scenario predictions...")

for scenario in scenarios:

    prediction = predict_scenario(scenario)

    results.append(
        {
            "scenario": scenario["scenario"],
            "followers": scenario["followersatposting"],
            "post_type": scenario["type"],
            "caption_length": len(scenario["imagetext"]),
            "word_count": count_words(
                scenario["imagetext"]
            ),
            "hashtag_count": scenario["imagetext"].count("#"),
            "predicted_engagement_rate": prediction,
            "predicted_engagement_percentage": (
                prediction * 100
            )
        }
    )


# ============================================================
# 11. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

baseline_prediction = (
    results_df.iloc[0]["predicted_engagement_rate"]
)

results_df["difference_from_first_scenario"] = (
    results_df["predicted_engagement_rate"]
    - baseline_prediction
)

results_df["percentage_point_difference"] = (
    results_df["predicted_engagement_percentage"]
    - (
        baseline_prediction * 100
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("COUNTERFACTUAL COMPARISON RESULTS")
print("=" * 70)

display_columns = [
    "scenario",
    "post_type",
    "caption_length",
    "word_count",
    "hashtag_count",
    "predicted_engagement_percentage",
    "difference_from_first_scenario"
]

display_table = results_df[display_columns].copy()

display_table[
    "predicted_engagement_percentage"
] = display_table[
    "predicted_engagement_percentage"
].round(2)

display_table[
    "difference_from_first_scenario"
] = display_table[
    "difference_from_first_scenario"
].round(6)

print("\n")
print(display_table.to_string(index=False))


# ============================================================
# 14. DISPLAY LIMITATIONS
# ============================================================

print("\n" + "-" * 70)

print("Output path:")
print(OUTPUT_PATH)

print("\nImportant limitations:")

print(
    "- These are hypothetical model-based comparisons."
)

print(
    "- The results do not guarantee actual Instagram performance."
)

print(
    "- Changing one feature may indirectly change other "
    "caption-related features."
)

print(
    "- The model is trained on historical dataset patterns."
)

print(
    "- Creator-aware validation showed weak generalization "
    "to unseen creators."
)

print(
    "- The model does not use live Instagram ranking signals."
)

print(
    "- A higher model estimate does not prove that a scenario "
    "will perform better in real life."
)

print("=" * 70)