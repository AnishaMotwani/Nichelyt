
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap


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
    "custom_local_shap_explanation.csv"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained engagement model...")

bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]

feature_columns = bundle["feature_columns"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

print("Model loaded successfully.")


# ============================================================
# 3. USER INPUT FUNCTIONS
# ============================================================

def get_followers():
    while True:
        value = input(
            "\nEnter number of followers "
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
            print("Please enter a valid whole number.")


def get_post_type():
    valid_types = ["Photo", "Album", "Video"]

    while True:
        value = input(
            "\nEnter post type "
            "(Photo / Album / Video, default Photo): "
        ).strip()

        if value == "":
            return "Photo"

        normalized_value = value.lower()

        if normalized_value in ["photo", "photos"]:
            return "Photo"

        if normalized_value in ["album", "albums"]:
            return "Album"

        if normalized_value in ["video", "videos"]:
            return "Video"

        print("Invalid post type.")
        print("Please choose Photo, Album, or Video.")


def get_caption():
    caption = input(
        "\nEnter your caption "
        "(press Enter for an empty caption):\n"
    )

    return caption.strip()


def get_date():
    while True:
        value = input(
            "\nEnter post date in YYYY-MM-DD format "
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
            print("Invalid date.")
            print("Example: 2026-09-23")


def get_time():
    while True:
        value = input(
            "\nEnter post time in HH:MM format "
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
            print("Invalid time.")
            print("Example: 14:30")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def prepare_features(data):
    """
    Creates the same features used during model training.
    """

    data = data.copy()

    # --------------------------------------------------------
    # Convert numeric fields
    # --------------------------------------------------------

    data["followersatposting"] = pd.to_numeric(
        data["followersatposting"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    data["postcreatedtime"] = pd.to_datetime(
        data["postcreatedtime"],
        format="mixed",
        errors="coerce"
    )

    data["post_hour"] = data["postcreatedtime"].dt.hour

    # --------------------------------------------------------
    # Caption features
    # --------------------------------------------------------

    data["imagetext"] = data["imagetext"].fillna("").astype(str)

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
        data["imagetext"].str.strip().eq("").astype(int)
    )

    # --------------------------------------------------------
    # Select training features
    # --------------------------------------------------------

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
# 5. COLLECT CUSTOM USER INPUT
# ============================================================

print("\n" + "=" * 65)
print("NICHELYT - CUSTOM ENGAGEMENT PREDICTION + LOCAL SHAP")
print("=" * 65)

followers = get_followers()
post_type = get_post_type()
caption = get_caption()
post_date = get_date()
post_time = get_time()


# ============================================================
# 6. CREATE INPUT DATAFRAME
# ============================================================

custom_data = pd.DataFrame(
    [
        {
            "followersatposting": followers,
            "type": post_type,
            "postcreateddate": post_date,
            "postcreatedtime": post_time,
            "imagetext": caption
        }
    ]
)

# Add unused columns if necessary for compatibility
custom_data["username"] = "custom_user"


# ============================================================
# 7. PREPARE FEATURES
# ============================================================

X_custom = prepare_features(custom_data)


# ============================================================
# 8. PREDICT ENGAGEMENT
# ============================================================

prediction = pipeline.predict(X_custom)[0]

prediction_percentage = prediction * 100


# ============================================================
# 9. TRANSFORM INPUT FOR SHAP
# ============================================================

X_transformed = preprocessor.transform(X_custom)

feature_names = preprocessor.get_feature_names_out()

# Convert sparse matrix to dense matrix if necessary
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()


# ============================================================
# 10. CALCULATE LOCAL SHAP VALUES
# ============================================================

print("\nCalculating local SHAP explanation...")

explainer = shap.TreeExplainer(model)

raw_shap_values = explainer.shap_values(
    X_transformed
)

raw_shap_values = np.asarray(raw_shap_values)

# Handle different SHAP output formats
if raw_shap_values.ndim == 3:
    if raw_shap_values.shape[0] == 1:
        raw_shap_values = raw_shap_values[0]

    elif raw_shap_values.shape[-1] == 1:
        raw_shap_values = raw_shap_values[:, :, 0]

local_shap_values = raw_shap_values[0]


# ============================================================
# 11. GET BASE MODEL VALUE
# ============================================================

base_value = explainer.expected_value

base_value = np.asarray(base_value).flatten()[0]


# ============================================================
# 12. CREATE SHAP EXPLANATION DATAFRAME
# ============================================================

explanation_df = pd.DataFrame(
    {
        "feature": feature_names,
        "shap_value": local_shap_values
    }
)

explanation_df["absolute_shap_value"] = (
    explanation_df["shap_value"].abs()
)

explanation_df["contribution"] = np.where(
    explanation_df["shap_value"] >= 0,
    "positive",
    "negative"
)

explanation_df = explanation_df.sort_values(
    by="absolute_shap_value",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 13. SAVE EXPLANATION
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

explanation_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("CUSTOM LOCAL SHAP EXPLANATION")
print("=" * 65)

print(f"\nFollowers: {followers}")
print(f"Post type: {post_type}")
print(f"Post date: {post_date}")
print(f"Post time: {post_time}")

print("\nCaption:")
print(caption if caption else "[Empty caption]")

print("\n" + "-" * 65)

print(
    f"Predicted engagement rate: "
    f"{prediction:.6f}"
)

print(
    f"Predicted engagement percentage: "
    f"{prediction_percentage:.2f}%"
)

print(
    f"Base model value: "
    f"{base_value:.6f}"
)

print("\n" + "-" * 65)
print("TOP FEATURE CONTRIBUTIONS")
print("-" * 65)

top_features = explanation_df.head(10)

for _, row in top_features.iterrows():

    feature = row["feature"]
    shap_value = row["shap_value"]
    contribution = row["contribution"]

    sign = "+" if shap_value >= 0 else ""

    print(
        f"{feature:<45} "
        f"{sign}{shap_value:.6f} "
        f"({contribution} contribution)"
    )


# ============================================================
# 15. DISPLAY SEPARATE POSITIVE AND NEGATIVE FEATURES
# ============================================================

print("\n" + "-" * 65)
print("TOP POSITIVE CONTRIBUTORS")
print("-" * 65)

positive_features = explanation_df[
    explanation_df["shap_value"] > 0
].head(5)

if positive_features.empty:
    print("No positive contributors found.")

else:
    for _, row in positive_features.iterrows():
        print(
            f"{row['feature']}: "
            f"+{row['shap_value']:.6f}"
        )


print("\n" + "-" * 65)
print("TOP NEGATIVE CONTRIBUTORS")
print("-" * 65)

negative_features = explanation_df[
    explanation_df["shap_value"] < 0
].head(5)

if negative_features.empty:
    print("No negative contributors found.")

else:
    for _, row in negative_features.iterrows():
        print(
            f"{row['feature']}: "
            f"{row['shap_value']:.6f}"
        )


# ============================================================
# 16. OUTPUT AND LIMITATIONS
# ============================================================

print("\n" + "-" * 65)

print("Output path:")
print(OUTPUT_PATH)

print("\nImportant limitations:")
print(
    "- SHAP explains the trained model, "
    "not Instagram's actual algorithm."
)

print(
    "- SHAP contributions do not prove causation."
)

print(
    "- The prediction is based on historical dataset patterns."
)

print(
    "- The model may perform poorly for unseen creators."
)

print(
    "- The engagement estimate is experimental "
    "and is not a guarantee."
)

print(
    "- The model does not use live Instagram trends "
    "or Instagram's private ranking signals."
)

print("=" * 65)