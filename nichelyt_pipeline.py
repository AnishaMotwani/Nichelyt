
from pathlib import Path
from datetime import datetime
import json
import re

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


# ============================================================
# 2. IMPORT EXISTING SERVICES
# ============================================================

from opportunity_engine_v2 import OpportunityEngineV2

from counterfactual_service import (
    run_counterfactual_comparison
)

from gemini_content_service import (
    generate_content
)


# ============================================================
# 3. MODEL LOADING
# ============================================================

_model_bundle = None


def load_engagement_model():
    """
    Loads the trained engagement model only once.
    """

    global _model_bundle

    if _model_bundle is None:

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        _model_bundle = joblib.load(
            MODEL_PATH
        )

    return _model_bundle


# ============================================================
# 4. TEXT FEATURE FUNCTIONS
# ============================================================

def count_words(text):
    """
    Counts the words in a caption.
    """

    if not text:

        return 0

    return len(
        str(text).split()
    )


def count_hashtags(text):
    """
    Counts hashtags using a regular expression.
    """

    if not text:

        return 0

    hashtags = re.findall(
        r"#\w+",
        str(text)
    )

    return len(hashtags)


def count_mentions(text):
    """
    Counts mentions in the caption.
    """

    if not text:

        return 0

    mentions = re.findall(
        r"@\w+",
        str(text)
    )

    return len(mentions)


# ============================================================
# 5. INPUT VALIDATION
# ============================================================

def validate_inputs(
    followers,
    post_type,
    caption,
    post_date,
    post_time
):
    """
    Validates the main user inputs.
    """

    if followers is None:

        raise ValueError(
            "Followers count is required."
        )

    try:

        followers_value = float(
            followers
        )

    except (TypeError, ValueError):

        raise ValueError(
            "Followers count must be numeric."
        )

    if followers_value <= 0:

        raise ValueError(
            "Followers count must be greater than zero."
        )

    allowed_post_types = [
        "Photo",
        "Album",
        "Video"
    ]

    if post_type not in allowed_post_types:

        raise ValueError(
            f"Post type must be one of: "
            f"{allowed_post_types}"
        )

    if not caption or not str(caption).strip():

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
# 6. FEATURE PREPARATION
# ============================================================

def prepare_engagement_features(
    followers,
    post_type,
    caption,
    post_date,
    post_time,
    feature_columns
):
    """
    Creates the features expected by the trained model.

    The returned DataFrame follows the feature order
    saved in the model bundle.
    """

    caption = str(
        caption
    )

    data = pd.DataFrame(
        [
            {
                "followersatposting": float(
                    followers
                ),

                "type": post_type,

                "postcreateddate": post_date,

                "postcreatedtime": post_time,

                "imagetext": caption,

                "caption_length": len(
                    caption
                ),

                "word_count": count_words(
                    caption
                ),

                "hashtag_count": count_hashtags(
                    caption
                ),

                "mention_count": count_mentions(
                    caption
                ),

                "is_caption_empty": int(
                    len(
                        caption.strip()
                    ) == 0
                )
            }
        ]
    )

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    data["postcreateddate"] = pd.to_datetime(
        data["postcreateddate"],
        errors="coerce"
    )

    if data["postcreateddate"].isna().any():

        raise ValueError(
            "Invalid post date."
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
    # Time features
    # --------------------------------------------------------

    parsed_time = pd.to_datetime(
        data["postcreatedtime"],
        format="%H:%M",
        errors="coerce"
    )

    if parsed_time.isna().any():

        raise ValueError(
            "Invalid post time."
        )

    data["post_hour"] = (
        parsed_time.dt.hour
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model features: "
            + str(
                missing_features
            )
        )

    return data[
        feature_columns
    ]


# ============================================================
# 7. ENGAGEMENT PREDICTION
# ============================================================

def predict_engagement(
    followers,
    post_type,
    caption,
    post_date,
    post_time
):
    """
    Predicts the engagement rate for the supplied post.
    """

    model_bundle = load_engagement_model()

    model = model_bundle[
        "pipeline"
    ]

    feature_columns = model_bundle[
        "feature_columns"
    ]

    features = prepare_engagement_features(
        followers=followers,

        post_type=post_type,

        caption=caption,

        post_date=post_date,

        post_time=post_time,

        feature_columns=feature_columns
    )

    prediction = model.predict(
        features
    )

    engagement_rate = float(
        prediction[0]
    )

    return {
        "status": "success",

        "estimated_engagement_rate": round(
            engagement_rate,
            6
        ),

        "estimated_engagement_percentage": round(
            engagement_rate * 100,
            4
        ),

        "features_used": list(
            feature_columns
        )
    }


# ============================================================
# 8. SHAP EXPLANATION
# ============================================================

def generate_shap_explanation(
    followers,
    post_type,
    caption,
    post_date,
    post_time
):
    """
    Generates SHAP explanations for the prediction.

    SHAP explains model output contributions.
    It does not prove causation.
    """

    try:

        import shap

    except ImportError:

        return {
            "status": "unavailable",

            "reason": (
                "SHAP is not installed. "
                "Run: pip install shap"
            )
        }

    try:

        model_bundle = load_engagement_model()

        pipeline = model_bundle[
            "pipeline"
        ]

        feature_columns = model_bundle[
            "feature_columns"
        ]

        features = prepare_engagement_features(
            followers=followers,

            post_type=post_type,

            caption=caption,

            post_date=post_date,

            post_time=post_time,

            feature_columns=feature_columns
        )

        preprocessor = pipeline.named_steps[
            "preprocessor"
        ]

        trained_model = pipeline.named_steps[
            "model"
        ]

        transformed_features = (
            preprocessor.transform(
                features
            )
        )

        # Convert sparse matrix if necessary

        if hasattr(
            transformed_features,
            "toarray"
        ):

            transformed_features = (
                transformed_features.toarray()
            )

        # Get transformed feature names

        try:

            transformed_feature_names = (
                preprocessor.get_feature_names_out()
            )

            transformed_feature_names = [
                str(name)
                for name in transformed_feature_names
            ]

        except Exception:

            transformed_feature_names = [
                f"feature_{index}"
                for index in range(
                    transformed_features.shape[1]
                )
            ]

        explainer = shap.TreeExplainer(
            trained_model
        )

        shap_values = explainer.shap_values(
            transformed_features
        )

        # Handle common SHAP output formats

        if isinstance(
            shap_values,
            list
        ):

            shap_values = shap_values[0]

        if len(
            shap_values.shape
        ) == 2:

            shap_values = shap_values[0]

        base_value = explainer.expected_value

        if hasattr(
            base_value,
            "__len__"
        ):

            base_value = base_value[0]

        contributions = []

        for feature_name, contribution in zip(
            transformed_feature_names,
            shap_values
        ):

            contributions.append(
                {
                    "feature": feature_name,

                    "contribution": round(
                        float(
                            contribution
                        ),
                        6
                    )
                }
            )

        contributions.sort(
            key=lambda item: abs(
                item["contribution"]
            ),

            reverse=True
        )

        positive_factors = [
            item
            for item in contributions
            if item["contribution"] > 0
        ]

        negative_factors = [
            item
            for item in contributions
            if item["contribution"] < 0
        ]

        return {
            "status": "success",

            "base_value": round(
                float(
                    base_value
                ),
                6
            ),

            "feature_contributions": contributions,

            "positive_factors": positive_factors,

            "negative_factors": negative_factors,

            "interpretation_note": (
                "SHAP values explain model output "
                "contributions and do not establish "
                "causal relationships."
            )
        }

    except Exception as error:

        return {
            "status": "error",

            "error": str(
                error
            )
        }


# ============================================================
# 9. OPPORTUNITY ANALYSIS
# ============================================================

_opportunity_engine = None


def load_opportunity_engine():
    """
    Loads Opportunity Engine V2 only once.
    """

    global _opportunity_engine

    if _opportunity_engine is None:

        _opportunity_engine = (
            OpportunityEngineV2()
        )

    return _opportunity_engine


def analyze_opportunity(
    niche,
    goal,
    post_idea
):
    """
    Performs niche opportunity analysis.
    """

    try:

        engine = load_opportunity_engine()

        result = engine.analyze(
            niche=niche,

            goal=goal,

            post_idea=post_idea,

            top_k=10,

            relevance_threshold=0.35
        )

        return result

    except Exception as error:

        return {
            "status": "error",

            "error": str(
                error
            )
        }
    # ============================================================
# HIDDEN OPPORTUNITY DISCOVERY
# ============================================================

def discover_hidden_opportunities(niche):

    try:

        engine = load_opportunity_engine()

        result = engine.discover_hidden_opportunities(
            niche=niche,
            top_k=100,
            max_suggestions=5
        )

        return result

    except Exception as error:

        return {
            "status": "error",
            "error": str(error)
        }


# ============================================================
# 10. GEMINI CONTENT GENERATION
# ============================================================

def generate_gemini_content(
    niche,
    goal,
    post_idea,
    caption,
    tone="Educational",
    target_audience="General audience",
    language="English"
):
    """
    Generates AI-assisted content using Gemini.

    The existing Gemini service handles:

    1. Historical post retrieval
    2. Opportunity analysis
    3. Gemini API request
    4. Structured JSON generation

    This wrapper adds pipeline-level metadata and
    handles errors without stopping the complete pipeline.
    """

    try:

        result = generate_content(
            niche=niche,

            goal=goal,

            post_idea=post_idea,

            tone=tone,

            target_audience=target_audience,

            language=language
        )

        if not isinstance(
            result,
            dict
        ):

            return {
                "status": "error",

                "error": (
                    "Gemini service returned "
                    "an invalid response format."
                )
            }

        # Preserve the original user caption

        result["input_caption"] = caption

        # Add a human-review note

        result["pipeline_note"] = (
            "Gemini-generated content is AI-assisted "
            "and should be reviewed before publication."
        )

        return result

    except Exception as error:

        return {
            "status": "error",

            "error": str(
                error
            ),

            "fallback_message": (
                "Gemini content generation failed, "
                "but the remaining Nichelyt components "
                "can continue."
            )
        }


# ============================================================
# 11. MAIN NICHELYT PIPELINE
# ============================================================

def run_nichelyt_pipeline(
    niche,
    goal,
    post_idea,
    followers,
    post_type,
    caption,
    post_date,
    post_time,
    run_counterfactual=True,
    run_shap=True,
    run_gemini=True,
    tone="Educational",
    target_audience="General audience",
    language="English"
):
    """
    Runs the Nichelyt integrated workflow.

    Components:

    1. Input validation
    2. Engagement prediction
    3. Opportunity analysis
    4. SHAP explanation
    5. Counterfactual comparison
    6. Gemini content generation
    """

    try:

        # ----------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------

        validate_inputs(
            followers=followers,

            post_type=post_type,

            caption=caption,

            post_date=post_date,

            post_time=post_time
        )

        if not niche or not str(
            niche
        ).strip():

            raise ValueError(
                "Niche cannot be empty."
            )

        if not goal or not str(
            goal
        ).strip():

            raise ValueError(
                "Goal cannot be empty."
            )

        if not post_idea or not str(
            post_idea
        ).strip():

            raise ValueError(
                "Post idea cannot be empty."
            )

        # ----------------------------------------------------
        # Engagement prediction
        # ----------------------------------------------------

        engagement_result = predict_engagement(
            followers=followers,

            post_type=post_type,

            caption=caption,

            post_date=post_date,

            post_time=post_time
        )

        # ----------------------------------------------------
        # Opportunity analysis
        # ----------------------------------------------------

        opportunity_result = analyze_opportunity(
            niche=niche,

            goal=goal,

            post_idea=post_idea
        )

        # ----------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------

        if run_shap:

            shap_result = generate_shap_explanation(
                followers=followers,

                post_type=post_type,

                caption=caption,

                post_date=post_date,

                post_time=post_time
            )

        else:

            shap_result = {
                "status": "skipped"
            }

        # ----------------------------------------------------
        # Counterfactual analysis
        # ----------------------------------------------------

        if run_counterfactual:

            counterfactual_result = (
                run_counterfactual_comparison(
                    followers=followers,

                    original_caption=caption,

                    post_date=post_date,

                    post_time=post_time,

                    save_csv=True
                )
            )

        else:

            counterfactual_result = {
                "status": "skipped"
            }

        # ----------------------------------------------------
        # Gemini content generation
        # ----------------------------------------------------

        if run_gemini:

            gemini_result = generate_gemini_content(
                niche=niche,

                goal=goal,

                post_idea=post_idea,

                caption=caption,

                tone=tone,

                target_audience=target_audience,

                language=language
            )

        else:

            gemini_result = {
                "status": "skipped",

                "reason": (
                    "Gemini content generation "
                    "was disabled."
                )
            }
                    # ----------------------------------------------------
        # Hidden Opportunity Discovery
        # ----------------------------------------------------

        hidden_opportunity_result = (
            discover_hidden_opportunities(
                niche=niche
            )
        )

        # ----------------------------------------------------
        # Limitations
        # ----------------------------------------------------

        limitations = [

            (
                "Engagement predictions are estimates "
                "from historical training data."
            ),

            (
                "Opportunity analysis uses the available "
                "dataset and is not live Instagram market data."
            ),

            (
                "Counterfactual scenarios are hypothetical "
                "model-based comparisons."
            ),

            (
                "SHAP values explain model contributions "
                "and do not prove causation."
            ),

            (
                "Gemini-generated content is AI-assisted "
                "and should be reviewed for accuracy, "
                "originality, and suitability."
            ),

            (
                "Retrieved historical posts may not be "
                "perfectly relevant to every user query."
            ),

            (
                "The pipeline does not guarantee actual "
                "social media engagement."
            )
        ]

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {

            "status": "success",

            "project": "Nichelyt",

            "input": {

                "niche": niche,

                "goal": goal,

                "post_idea": post_idea,

                "followers": followers,

                "post_type": post_type,

                "caption": caption,

                "post_date": post_date,

                "post_time": post_time,

                "tone": tone,

                "target_audience": target_audience,

                "language": language
            },

            "opportunity_analysis": (
                opportunity_result
            ),

            "engagement_prediction": (
                engagement_result
            ),

            "shap_explanation": (
                shap_result
            ),

                        "counterfactual_comparison": (
                counterfactual_result
            ),

            "hidden_opportunity_discovery": (
                hidden_opportunity_result
            ),

            "gemini_content": (
                gemini_result
            ),

            "limitations": limitations

        }

    except Exception as error:

        return {

            "status": "error",

            "error": str(
                error
            )

        }


# ============================================================
# 12. TEST RUN
# ============================================================

if __name__ == "__main__":

    result = run_nichelyt_pipeline(

        niche="Machine Learning Education",

        goal="Increase Engagement",

        post_idea=(
            "Beginner-friendly machine learning concepts"
        ),

        followers=10000,

        post_type="Photo",

        caption=(
            "Learn machine learning step by step "
            "#Python #MachineLearning #AI"
        ),

        post_date="2026-09-23",

        post_time="18:30",

        run_counterfactual=True,

        run_shap=True,

        run_gemini=True,

        tone="Educational",

        target_audience=(
            "BCA students and machine learning beginners"
        ),

        language="English"
    )

    print(
        json.dumps(
            result,

            indent=4,

            default=str
        )
    )