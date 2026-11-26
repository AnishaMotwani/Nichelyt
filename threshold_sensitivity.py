
import pandas as pd

from opportunity_engine_v2 import OpportunityEngineV2


def main():

    print("\nNICHELYT - THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 55)

    # Load engine only once
    engine = OpportunityEngineV2()

    # User input
    niche = input("\nEnter niche: ").strip()
    goal = input("Enter goal: ").strip()
    post_idea = input("Enter post idea: ").strip()

    if not niche or not goal:
        print("Niche and goal are required.")
        return

    # Thresholds to compare
    thresholds = [0.25, 0.30, 0.35, 0.40]

    comparison = []

    for threshold in thresholds:

        print(f"\nTesting threshold: {threshold}")
        output = engine.analyze(
            niche=niche,
            goal=goal,
            post_idea=post_idea,
            top_k=10,
            relevance_threshold=threshold
        )
        if threshold == 0.30:
          print("\nTREND DETAILS AT 0.30:")
          print(output["trend_momentum"])

        if output.get("status") != "success":
            print("Analysis failed.")
            continue

        saturation = output["saturation"]
        trend = output["trend_momentum"]

        row = {
            "threshold": threshold,
            "relevant_posts": saturation.get(
                "relevant_post_count"
            ),
            "total_posts": saturation.get(
                "total_post_count"
            ),
            "saturation_percentage": saturation.get(
                "saturation_percentage"
            ),
            "opportunity_from_saturation": saturation.get(
                "opportunity_from_saturation"
            ),
            "trend_status": trend.get("status"),
            "trend_relevant_dated_posts": trend.get(
                "relevant_dated_post_count"
            )
        }

        comparison.append(row)

    # Create comparison table
    df = pd.DataFrame(comparison)

    print("\n\nTHRESHOLD COMPARISON RESULTS")
    print("=" * 75)

    print(df.to_string(index=False))

    # Save report
    output_path = (
        engine.project_root
        / "data"
        / "processed"
        / "threshold_sensitivity_report.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nReport saved at:")
    print(output_path)


if __name__ == "__main__":
    main()
