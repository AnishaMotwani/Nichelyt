import json

from gemini_content_service import generate_content


def main():

    print("\n" + "=" * 70)
    print("NICHELYT GEMINI CONTENT SERVICE TEST")
    print("=" * 70)

    try:

        result = generate_content(
            niche="Machine Learning Education",
            goal="Increase Engagement",
            post_idea=(
                "Beginner-friendly machine "
                "learning concepts"
            ),
            tone="Educational",
            target_audience=(
                "BCA students and beginners"
            ),
            language="English"
        )

        if not isinstance(result, dict):

            raise TypeError(
                "Generated result is not a Python dictionary."
            )

        print("\n========== TEST RESULT ==========\n")

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False,
                default=str
            )
        )

        print("\n" + "=" * 70)
        print("GEMINI CONTENT SERVICE TEST PASSED")
        print("=" * 70)

    except Exception as error:

        print("\n" + "=" * 70)
        print("GEMINI CONTENT SERVICE TEST FAILED")
        print("=" * 70)

        print("\nError type:")
        print(type(error).__name__)

        print("\nError message:")
        print(str(error))

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()