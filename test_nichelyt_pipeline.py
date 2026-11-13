
import json
from nichelyt_pipeline import run_nichelyt_pipeline


# ============================================================
# 1. TEST CASES
# ============================================================

test_cases = [
    {
        "name": "Normal Machine Learning Input",

        "niche": "Machine Learning Education",

        "goal": "Increase Engagement",

        "post_idea": "Beginner-friendly machine learning concepts",

        "followers": 10000,

        "post_type": "Photo",

        "caption": (
            "Learn machine learning step by step "
            "#Python #MachineLearning #AI"
        ),

        "post_date": "2026-09-23",

        "post_time": "18:30"
    },

    {
        "name": "Fitness Video Input",

        "niche": "Fitness",

        "goal": "Increase Engagement",

        "post_idea": "Beginner home workout",

        "followers": 5000,

        "post_type": "Video",

        "caption": (
            "Beginner home workout routine "
            "#Fitness #Workout"
        ),

        "post_date": "2026-09-23",

        "post_time": "10:00"
    },

    {
        "name": "Album Post Input",

        "niche": "Technology Education",

        "goal": "Increase Engagement",

        "post_idea": "Introduction to artificial intelligence",

        "followers": 15000,

        "post_type": "Album",

        "caption": (
            "Five important AI concepts "
            "#AI #Technology"
        ),

        "post_date": "2026-09-24",

        "post_time": "14:30"
    },

    {
        "name": "Invalid Followers",

        "niche": "Machine Learning",

        "goal": "Increase Engagement",

        "post_idea": "Introduction to neural networks",

        "followers": -100,

        "post_type": "Photo",

        "caption": "Learn neural networks #AI",

        "post_date": "2026-09-23",

        "post_time": "18:30"
    },

    {
        "name": "Empty Caption",

        "niche": "Machine Learning",

        "goal": "Increase Engagement",

        "post_idea": "Introduction to Python",

        "followers": 10000,

        "post_type": "Photo",

        "caption": "",

        "post_date": "2026-09-23",

        "post_time": "18:30"
    }
]


# ============================================================
# 2. RUN TESTS
# ============================================================

def run_tests():

    total_tests = len(test_cases)

    passed_tests = 0

    failed_tests = 0

    print("=" * 70)

    print("NICHELYT END-TO-END TESTING")

    print("=" * 70)

    for index, test_case in enumerate(test_cases, start=1):

        test_name = test_case["name"]

        print(f"\nTest {index}: {test_name}")

        print("-" * 70)

        try:

            result = run_nichelyt_pipeline(

                niche=test_case["niche"],

                goal=test_case["goal"],

                post_idea=test_case["post_idea"],

                followers=test_case["followers"],

                post_type=test_case["post_type"],

                caption=test_case["caption"],

                post_date=test_case["post_date"],

                post_time=test_case["post_time"],

                run_counterfactual=False,

                run_shap=False
            )

            if (
                test_case["followers"] <= 0
                or not test_case["caption"].strip()
            ):

                expected_error = True

            else:

                expected_error = False

            actual_status = result.get("status")

            if expected_error:

                if actual_status == "error":

                    print("Result: PASSED")

                    print("Expected validation error received.")

                    passed_tests += 1

                else:

                    print("Result: FAILED")

                    print(
                        "Expected an error, "
                        f"but received: {actual_status}"
                    )

                    failed_tests += 1

            else:

                if actual_status == "success":

                    print("Result: PASSED")

                    print(
                        "Engagement prediction status:",
                        result.get(
                            "engagement_prediction",
                            {}
                        ).get("status")
                    )

                    print(
                        "Opportunity analysis status:",
                        result.get(
                            "opportunity_analysis",
                            {}
                        ).get("status")
                    )

                    passed_tests += 1

                else:

                    print("Result: FAILED")

                    print(
                        "Pipeline returned:",
                        json.dumps(
                            result,
                            indent=2,
                            default=str
                        )
                    )

                    failed_tests += 1

        except Exception as error:

            print("Result: FAILED")

            print("Unexpected exception:", str(error))

            failed_tests += 1

    print("\n" + "=" * 70)

    print("TEST SUMMARY")

    print("=" * 70)

    print("Total tests:", total_tests)

    print("Passed:", passed_tests)

    print("Failed:", failed_tests)

    if failed_tests == 0:

        print("Overall result: ALL TESTS PASSED")

    else:

        print("Overall result: SOME TESTS FAILED")


if __name__ == "__main__":

    run_tests()