from pathlib import Path
import joblib


# --------------------------------------------------
# 1. Model path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "engagement_random_forest_pipeline.joblib"
)


# --------------------------------------------------
# 2. Check model file
# --------------------------------------------------

print("=" * 60)
print("NICHELYT - MODEL STRUCTURE CHECK")
print("=" * 60)

print("\nChecking model path:")
print(MODEL_PATH)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )


# --------------------------------------------------
# 3. Load model bundle
# --------------------------------------------------

print("\nLoading model bundle...")

model_bundle = joblib.load(MODEL_PATH)

print("Model bundle loaded successfully.")


# --------------------------------------------------
# 4. Display bundle information
# --------------------------------------------------

print("\nModel bundle type:")
print(type(model_bundle))


if isinstance(model_bundle, dict):

    print("\nModel bundle keys:")

    for key in model_bundle.keys():
        print("-", key)

else:

    print(
        "\nWarning: Model bundle is not a dictionary."
    )


# --------------------------------------------------
# 5. Retrieve pipeline
# --------------------------------------------------

if not isinstance(model_bundle, dict):
    raise TypeError(
        "Expected model bundle to be a dictionary."
    )


if "pipeline" not in model_bundle:
    raise KeyError(
        "The model bundle does not contain "
        "the 'pipeline' key."
    )


pipeline = model_bundle["pipeline"]


# --------------------------------------------------
# 6. Display pipeline information
# --------------------------------------------------

print("\nPipeline type:")
print(type(pipeline))


if hasattr(pipeline, "named_steps"):

    print("\nPipeline step names:")

    for step_name in pipeline.named_steps.keys():
        print("-", step_name)

else:

    print(
        "\nWarning: Pipeline does not contain "
        "'named_steps'."
    )


# --------------------------------------------------
# 7. Display feature columns
# --------------------------------------------------

print("\nFeature columns:")

if "feature_columns" in model_bundle:

    for column in model_bundle["feature_columns"]:
        print("-", column)

else:

    print(
        "Feature columns are not available "
        "in the model bundle."
    )


# --------------------------------------------------
# 8. Check preprocessing component
# --------------------------------------------------

print("\nChecking preprocessing component...")

if hasattr(pipeline, "named_steps"):

    possible_names = [
        "preprocessor",
        "preprocessing"
    ]

    found = False

    for name in possible_names:

        if name in pipeline.named_steps:

            print(
                f"Found preprocessing step: {name}"
            )

            found = True

    if not found:

        print(
            "No expected preprocessing step found."
        )


# --------------------------------------------------
# 9. Check model component
# --------------------------------------------------

print("\nChecking model component...")

if hasattr(pipeline, "named_steps"):

    possible_model_names = [
        "model",
        "classifier",
        "regressor"
    ]

    found_model = False

    for name in possible_model_names:

        if name in pipeline.named_steps:

            print(
                f"Found model step: {name}"
            )

            found_model = True

    if not found_model:

        print(
            "No expected model step found."
        )


print("\n" + "=" * 60)
print("MODEL STRUCTURE CHECK COMPLETED")
print("=" * 60)