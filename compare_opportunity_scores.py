
import pandas as pd
import numpy as np

from opportunity_engine_v2 import OpportunityEngineV2


def main():

    print("\nNICHELYT - OPPORTUNITY SCORE COMPARISON")
    print("=" * 55)

    # 1. Load the existing engine
    engine = OpportunityEngineV2()

    # 2. Take sample user inputs
    niche = input("\nEnter niche: ").strip()
    goal = input("Enter goal: ").strip()
    post_idea = input("Enter post idea: ").strip()

    if not niche or not goal:
        print("Niche and goal are required.")
        return

    # 3. Run the existing analysis
    output = engine.analyze(
        niche=niche,
        goal=goal,
        post_idea=post_idea,
        top_k=10
    )

    if output.get("status") != "success":
        print("Analysis failed:", output)
        return

    # 4. Convert top results into DataFrame
    results = pd.DataFrame(output["top_results"])

    if results.empty:
        print("No results found.")
        return

    # 5. Read existing normalized inputs
    similarity_normalized = engine._normalize_vector(
        results["similarity"].to_numpy()
    )

    engagement_percentiles = results[
        "engagement_percentile"
    ]

    if engagement_percentiles.notna().any():
        engagement_normalized = (
            engagement_percentiles.fillna(0).to_numpy() / 100
        )
    else:
        engagement_normalized = np.zeros(len(results))

    # 6. Existing formula: 60% similarity + 40% engagement
    current_score = (
        0.60 * similarity_normalized
        + 0.40 * engagement_normalized
    ) * 100

    # 7. Extract query-level saturation
    saturation = output["saturation"]
    print("\nFULL SATURATION DETAILS:")
    print(saturation)

    print("\nFULL TREND MOMENTUM DETAILS:")
    print(output["trend_momentum"])
    saturation_value = (
        saturation.get("saturation_score")
        if isinstance(saturation, dict)
        else None
    )

    # 8. Extract query-level trend momentum
    trend = output["trend_momentum"]

    trend_value = (
        trend.get("momentum_score")
        if isinstance(trend, dict)
        and trend.get("status") == "available"
        else None
    )

    # 9. Normalize saturation and trend to 0-1
    # Missing signals are not silently treated as zero.
    if saturation_value is None:
        print("\nSaturation unavailable. Cannot compare four-signal formula.")
        return

    if trend_value is None:
        print("\nTrend momentum unavailable. Cannot compare four-signal formula.")
        print("Current formula results will still be saved.")

    saturation_normalized = saturation_value / 100

    # 10. Calculate experimental score only if both signals exist
    results["current_score"] = np.round(current_score, 2)

    if trend_value is not None:

        trend_normalized = trend_value / 100

        experimental_score = (
            0.40 * similarity_normalized
            + 0.30 * engagement_normalized
            + 0.15 * (1 - saturation_normalized)
            + 0.15 * trend_normalized
        ) * 100

        results["experimental_score"] = np.round(
            experimental_score, 2
        )

        results["current_rank"] = (
            results["current_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        results["experimental_rank"] = (
            results["experimental_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        results["rank_changed"] = (
            results["current_rank"]
            != results["experimental_rank"]
        )

    # 11. Save comparison report
    output_path = (
        engine.project_root
        / "data"
        / "processed"
        / "opportunity_score_comparison.csv"
    )

    results.to_csv(output_path, index=False)

    print("\nQUERY-LEVEL INDICATORS")
    print("-" * 40)
    print("Saturation:", saturation_value)
    print("Trend momentum:", trend_value)

    print("\nSCORE COMPARISON")
    print("-" * 40)

    columns = ["current_score"]

    if trend_value is not None:
        columns += [
            "experimental_score",
            "current_rank",
            "experimental_rank",
            "rank_changed"
        ]

    print(results[columns].to_string(index=False))

    print("\nReport saved at:")
    print(output_path)

    if trend_value is not None:
        print(
            "\nRanking unchanged:",
            not results["rank_changed"].any()
        )


if __name__ == "__main__":
    main()
