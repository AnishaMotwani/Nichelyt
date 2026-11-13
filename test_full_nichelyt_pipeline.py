import json

from nichelyt_pipeline import run_nichelyt_pipeline


def main():
    print("=" * 70)
    print("NICHELYT FULL PIPELINE SMOKE TEST")
    print("=" * 70)

    result = run_nichelyt_pipeline(
        niche="Machine Learning Education",
        goal="Increase Engagement",
        post_idea="Beginner-friendly machine learning concepts",
        followers=10000,
        post_type="Photo",
        caption=(
            "Learn machine learning step by step "
            "#Python #MachineLearning #AI"
        ),
        post_date="2026-09-23",
        post_time="18:30",
        run_counterfactual=True,
        run_shap=True
    )

    print("\nPIPELINE STATUS:")
    print(result.get("status"))

    print("\nENGAGEMENT PREDICTION STATUS:")
    print(
        result.get("engagement_prediction", {}).get("status")
    )

    print("\nOPPORTUNITY ANALYSIS STATUS:")
    print(
        result.get("opportunity_analysis", {}).get("status")
    )

    print("\nSHAP STATUS:")
    print(
        result.get("shap_explanation", {}).get("status")
    )

    print("\nCOUNTERFACTUAL STATUS:")
    print(
        result.get("counterfactual_comparison", {}).get("status")
    )

    print("\nFULL COMPONENT VALIDATION:")

    expected_components = {
        "engagement_prediction": result.get(
            "engagement_prediction", {}
        ).get("status"),
        "opportunity_analysis": result.get(
            "opportunity_analysis", {}
        ).get("status"),
        "shap_explanation": result.get(
            "shap_explanation", {}
        ).get("status"),
        "counterfactual_comparison": result.get(
            "counterfactual_comparison", {}
        ).get("status")
    }

    all_successful = True

    for component, status in expected_components.items():
        if status == "success":
            print(f"[PASSED] {component}")
        else:
            print(f"[FAILED] {component}")
            all_successful = False

    print("\n" + "=" * 70)

    if result.get("status") == "success" and all_successful:
        print("FULL PIPELINE SMOKE TEST: PASSED")
    else:
        print("FULL PIPELINE SMOKE TEST: FAILED")

    print("=" * 70)

    print("\nSelected Output:")
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "engagement_prediction": result.get(
                    "engagement_prediction"
                ),
                "shap_status": result.get(
                    "shap_explanation", {}
                ).get("status"),
                "counterfactual_status": result.get(
                    "counterfactual_comparison", {}
                ).get("status")
            },
            indent=4,
            default=str
        )
    )


if __name__ == "__main__":
    main()