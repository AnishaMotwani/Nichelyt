
from nichelyt_pipeline import (
    load_engagement_model,
    prepare_engagement_features
)


def main():

    model_bundle = load_engagement_model()

    feature_columns = model_bundle[
        "feature_columns"
    ]

    followers = 10000

    post_type = "Photo"

    caption = (
        "Learn machine learning step by step "
        "#Python #MachineLearning #AI"
    )

    post_date = "2026-09-23"

    post_time = "18:30"

    features = prepare_engagement_features(
        followers=followers,

        post_type=post_type,

        caption=caption,

        post_date=post_date,

        post_time=post_time,

        feature_columns=feature_columns
    )

    print("\nGenerated Features:\n")

    print(
        features.to_string(
            index=False
        )
    )

    print("\nFeature Data Types:\n")

    print(
        features.dtypes
    )


if __name__ == "__main__":

    main()