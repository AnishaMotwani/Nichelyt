
import pandas as pd
from opportunity_engine_v2 import OpportunityEngineV2


def main():

    print("\nNICHELYT - RELEVANT POST INSPECTION")
    print("=" * 55)

    # Load the existing opportunity engine
    engine = OpportunityEngineV2()

    # Same query used in threshold sensitivity experiment
    niche = "Fitness and Exercise"
    goal = "Increase engagement"
    post_idea = (
        "Workout routines, nutrition tips and "
        "home fitness exercises"
    )

    threshold = 0.30

    # Construct query exactly like the main engine
    query = (
        f"Niche: {niche}. "
        f"Goal: {goal}. "
        f"Post idea: {post_idea}"
    )

    # Generate query embedding
    query_embedding = engine._get_query_embedding(query)

    # Calculate similarity with all dataset posts
    similarities = engine._calculate_similarity(
        query_embedding
    )

    # Select posts matching threshold
    relevant_mask = similarities >= threshold

    # Copy relevant metadata
    relevant_posts = engine.metadata.loc[
        relevant_mask
    ].copy()

    # Add similarity scores
    relevant_posts["similarity"] = similarities[
        relevant_mask
    ]

    # Select columns for manual inspection
    columns = [
        "clean_text",
        "similarity",
        "type",
        "engagement_rate",
        "postcreateddate"
    ]

    inspection_data = relevant_posts[columns].copy()

    # Sort by similarity, highest first
    inspection_data = inspection_data.sort_values(
        by="similarity",
        ascending=False
    )

    # Save inspection report
    output_path = (
        engine.project_root
        / "data"
        / "processed"
        / "relevant_posts_threshold_030.csv"
    )

    inspection_data.to_csv(
        output_path,
        index=False
    )

    print("\nINSPECTION SUMMARY")
    print("-" * 40)
    print("Threshold:", threshold)
    print("Relevant posts:", len(inspection_data))
    print("Total dataset posts:", len(engine.metadata))

    print("\nTop 5 matching posts:")
    print(
        inspection_data[
            ["clean_text", "similarity"]
        ].head().to_string(index=False)
    )

    print("\nReport saved at:")
    print(output_path)


if __name__ == "__main__":
    main()
