
"""
Nichelyt Engine
Phase 1:
1. Load embeddings and metadata
2. Semantic retrieval
3. Engagement percentile
4. Exploratory opportunity score
5. Basic explanations
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class NichelytEngine:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 10
    ):
        self.base_dir = Path(__file__).resolve().parent
        self.processed_dir = self.base_dir / "data" / "processed"

        self.embeddings_path = (
            self.processed_dir / "caption_embeddings.npy"
        )

        self.metadata_path = (
            self.processed_dir / "retrieval_metadata.csv"
        )

        self.text_ready_path = (
            self.processed_dir / "dataset2_text_ready.csv"
        )

        self.cleaned_path = (
            self.processed_dir / "dataset2_cleaned.csv"
        )

        self.model_name = model_name
        self.top_k = max(1, min(int(top_k), 20))

        self.model = None
        self.embeddings = None
        self.metadata = None
        self.cleaned_data = None
        self.engagement_reference = None

        self.is_loaded = False

    # =====================================================
    # LOAD RESOURCES
    # =====================================================

    def load_resources(self):

        if self.is_loaded:
            return

        required_files = [
            self.embeddings_path,
            self.cleaned_path
        ]

        for file_path in required_files:

            if not file_path.exists():

                raise FileNotFoundError(
                    f"Required file not found:\n{file_path}"
                )

        if self.metadata_path.exists():

            metadata = pd.read_csv(
                self.metadata_path
            )

        elif self.text_ready_path.exists():

            metadata = pd.read_csv(
                self.text_ready_path
            )

        else:

            raise FileNotFoundError(
                "Metadata file not found."
            )

        self.embeddings = np.load(
            self.embeddings_path
        )

        self.cleaned_data = pd.read_csv(
            self.cleaned_path
        )

        if len(metadata) != len(self.embeddings):

            raise ValueError(
                "Metadata and embeddings are not aligned.\n"
                f"Metadata rows: {len(metadata)}\n"
                f"Embedding rows: {len(self.embeddings)}\n"
                "Run build_embeddings.py again."
            )

        self.metadata = metadata.reset_index(
            drop=True
        )

        # Normalize embeddings
        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1

        self.embeddings = (
            self.embeddings / norms
        )

        print("Loading SentenceTransformer model...")

        self.model = SentenceTransformer(
            self.model_name
        )

        self.engagement_reference = (
            self._build_engagement_reference()
        )

        self.is_loaded = True

        print("All resources loaded successfully.")

    # =====================================================
    # ENGAGEMENT REFERENCE
    # =====================================================

    def _build_engagement_reference(self):

        data = self.cleaned_data.copy()

        if "engagement_rate" in data.columns:

            values = pd.to_numeric(
                data["engagement_rate"],
                errors="coerce"
            )

        elif {
            "likes",
            "comments",
            "followersatposting"
        }.issubset(data.columns):

            likes = pd.to_numeric(
                data["likes"],
                errors="coerce"
            ).fillna(0)

            comments = pd.to_numeric(
                data["comments"],
                errors="coerce"
            ).fillna(0)

            followers = pd.to_numeric(
                data["followersatposting"],
                errors="coerce"
            )

            values = (
                (likes + comments) / followers
            )

        else:

            raise ValueError(
                "Engagement-rate columns not found."
            )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        values = values[values >= 0]

        if values.empty:

            raise ValueError(
                "No valid engagement values found."
            )

        return values.to_numpy(
            dtype=float
        )

    # =====================================================
    # HELPER FUNCTIONS
    # =====================================================

    @staticmethod
    def _find_column(
        data: pd.DataFrame,
        possible_names: List[str]
    ) -> Optional[str]:

        column_map = {
            str(column).lower(): column
            for column in data.columns
        }

        for name in possible_names:

            if name.lower() in column_map:

                return column_map[name.lower()]

        return None

    @staticmethod
    def _safe_value(value):

        if pd.isna(value):

            return None

        if isinstance(value, np.integer):

            return int(value)

        if isinstance(value, np.floating):

            return float(value)

        return value

    @staticmethod
    def _min_max_normalize(values):

        values = np.asarray(
            values,
            dtype=float
        )

        minimum = np.min(values)
        maximum = np.max(values)

        if maximum == minimum:

            return np.full_like(
                values,
                0.5,
                dtype=float
            )

        return (
            (values - minimum) /
            (maximum - minimum)
        )

    def _get_metadata_value(
        self,
        row,
        possible_names,
        default=None
    ):

        column = self._find_column(
            self.metadata,
            possible_names
        )

        if column is None:

            return default

        return self._safe_value(
            row.get(column, default)
        )

    def _get_caption(self, row):

        caption_column = self._find_column(
            self.metadata,
            [
                "postscraped",
                "caption",
                "text",
                "clean_text",
                "caption_text",
                "content"
            ]
        )

        if caption_column is None:

            return ""

        value = row.get(
            caption_column,
            ""
        )

        if value is None or pd.isna(value):

            return ""

        return str(value)

    def _get_engagement_rate(self, row):

        column = self._find_column(
            self.metadata,
            [
                "engagement_rate",
                "engagementrate"
            ]
        )

        if column is not None:

            value = pd.to_numeric(
                row.get(column),
                errors="coerce"
            )

            if pd.notna(value):

                return float(value)

        likes_column = self._find_column(
            self.metadata,
            ["likes"]
        )

        comments_column = self._find_column(
            self.metadata,
            ["comments"]
        )

        followers_column = self._find_column(
            self.metadata,
            [
                "followersatposting",
                "followers",
                "follower_count"
            ]
        )

        if (
            likes_column is not None
            and comments_column is not None
            and followers_column is not None
        ):

            likes = pd.to_numeric(
                row.get(likes_column),
                errors="coerce"
            )

            comments = pd.to_numeric(
                row.get(comments_column),
                errors="coerce"
            )

            followers = pd.to_numeric(
                row.get(followers_column),
                errors="coerce"
            )

            if (
                pd.notna(likes)
                and pd.notna(comments)
                and pd.notna(followers)
                and followers > 0
            ):

                return float(
                    (likes + comments) / followers
                )

        return None

    def _calculate_percentile(
        self,
        engagement_rate
    ):

        if engagement_rate is None:

            return None

        percentile = (
            np.sum(
                self.engagement_reference
                <= engagement_rate
            )
            / len(self.engagement_reference)
        ) * 100

        return round(
            float(percentile),
            2
        )

    # =====================================================
    # QUERY CREATION
    # =====================================================

    def _build_query(
        self,
        niche: str,
        goal: str,
        post_idea: str = ""
    ):

        query_parts = [
            f"Niche: {niche.strip()}",
            f"Goal: {goal.strip()}"
        ]

        if post_idea.strip():

            query_parts.append(
                f"Post idea: {post_idea.strip()}"
            )

        return ". ".join(
            query_parts
        )

    # =====================================================
    # SEMANTIC RETRIEVAL
    # =====================================================

    def retrieve(
        self,
        niche: str,
        goal: str,
        post_idea: str = "",
        top_k: Optional[int] = None
    ):

        self.load_resources()

        if not niche or not niche.strip():

            raise ValueError(
                "Niche cannot be empty."
            )

        if not goal or not goal.strip():

            raise ValueError(
                "Goal cannot be empty."
            )

        query = self._build_query(
            niche,
            goal,
            post_idea
        )

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]

        similarities = np.dot(
            self.embeddings,
            query_embedding
        )

        requested_k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        requested_k = max(
            1,
            min(
                int(requested_k),
                len(similarities)
            )
        )

        top_indices = np.argsort(
            similarities
        )[::-1][:requested_k]

        selected_similarities = (
            similarities[top_indices]
        )

        similarity_normalized = (
            self._min_max_normalize(
                selected_similarities
            )
        )

        engagement_values = []
        result_rows = []

        for index, similarity, similarity_norm in zip(
            top_indices,
            selected_similarities,
            similarity_normalized
        ):

            row = self.metadata.iloc[
                int(index)
            ]

            engagement_rate = (
                self._get_engagement_rate(
                    row
                )
            )

            engagement_values.append(
                engagement_rate
                if engagement_rate is not None
                else 0.0
            )

            result_rows.append({

                "rank": len(result_rows) + 1,

                "source_index": int(index),

                "similarity": round(
                    float(similarity),
                    4
                ),

                "similarity_normalized": round(
                    float(similarity_norm),
                    4
                ),

                "caption": self._get_caption(
                    row
                ),

                "creator": self._get_metadata_value(
                    row,
                    [
                        "username",
                        "account",
                        "creator"
                    ],
                    "Unknown"
                ),

                "post_type": self._get_metadata_value(
                    row,
                    [
                        "type",
                        "media_type"
                    ],
                    "Unknown"
                ),

                "engagement_rate": (
                    round(
                        float(engagement_rate),
                        6
                    )
                    if engagement_rate is not None
                    else None
                ),

                "engagement_percentile": (
                    self._calculate_percentile(
                        engagement_rate
                    )
                )

            })

        engagement_array = np.asarray(
            engagement_values,
            dtype=float
        )

        engagement_normalized = (
            self._min_max_normalize(
                engagement_array
            )
        )

        for result, engagement_norm in zip(
            result_rows,
            engagement_normalized
        ):

            # Exploratory heuristic:
            # 60% similarity + 40% engagement
            opportunity_score = (
                0.60
                * result["similarity_normalized"]
                + 0.40
                * float(engagement_norm)
            ) * 100

            result["engagement_normalized"] = round(
                float(engagement_norm),
                4
            )

            result["opportunity_score"] = round(
                float(opportunity_score),
                2
            )

            result["explanation"] = (
                self._build_explanation(
                    result
                )
            )

        return {

            "query": query,

            "niche": niche,

            "goal": goal,

            "post_idea": post_idea,

            "top_k": requested_k,

            "results": result_rows

        }

    # =====================================================
    # EXPLANATION
    # =====================================================

    def _build_explanation(self, result):

        similarity = result[
            "similarity"
        ]

        percentile = result[
            "engagement_percentile"
        ]

        opportunity_score = result[
            "opportunity_score"
        ]

        explanation_parts = [

            f"Semantic similarity is "
            f"{similarity:.2f}."

        ]

        if percentile is not None:

            explanation_parts.append(

                f"Historical engagement is around "
                f"the {percentile:.1f}th percentile "
                f"in Dataset 2."

            )

        else:

            explanation_parts.append(

                "Historical engagement percentile "
                "is unavailable."

            )

        explanation_parts.append(

            f"Exploratory opportunity indicator: "
            f"{opportunity_score:.2f}/100."

        )

        explanation_parts.append(

            "This is a heuristic comparison, "
            "not a guaranteed future-performance "
            "prediction."

        )

        return " ".join(
            explanation_parts
        )

    # =====================================================
    # MAIN ANALYSIS FUNCTION
    # =====================================================

    def analyze_niche(
        self,
        niche: str,
        goal: str,
        post_idea: str = "",
        top_k: Optional[int] = None
    ):

        retrieval_output = self.retrieve(
            niche=niche,
            goal=goal,
            post_idea=post_idea,
            top_k=top_k
        )

        results = retrieval_output[
            "results"
        ]

        opportunity_scores = [

            item["opportunity_score"]
            for item in results
            if item.get(
                "opportunity_score"
            ) is not None

        ]

        percentiles = [

            item["engagement_percentile"]
            for item in results
            if item.get(
                "engagement_percentile"
            ) is not None

        ]

        summary = {

            "average_similarity": round(
                float(
                    np.mean(
                        [
                            item["similarity"]
                            for item in results
                        ]
                    )
                ),
                4
            ) if results else None,

            "average_engagement_percentile": round(
                float(
                    np.mean(percentiles)
                ),
                2
            ) if percentiles else None,

            "average_opportunity_score": round(
                float(
                    np.mean(opportunity_scores)
                ),
                2
            ) if opportunity_scores else None

        }

        return {

            "status": "success",

            "summary": summary,

            "data": retrieval_output,

            "limitations": [

                "Opportunity score is an exploratory heuristic.",

                "Engagement percentile is descriptive.",

                "Instagram live ranking algorithm is not accessed.",

                "Results do not guarantee future engagement.",

                "Semantic similarity is not model accuracy."

            ]

        }


# =========================================================
# TERMINAL TEST
# =========================================================

def print_demo_result(result):

    print("\n" + "=" * 70)

    print("NICHELYT ANALYSIS RESULT")

    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":

    print("Nichelyt Engine Test")
    print("-" * 30)

    niche = input(
        "Enter niche: "
    ).strip()

    goal = input(
        "Enter goal: "
    ).strip()

    post_idea = input(
        "Enter post idea (optional): "
    ).strip()

    engine = NichelytEngine(
        top_k=10
    )

    try:

        output = engine.analyze_niche(
            niche=niche,
            goal=goal,
            post_idea=post_idea
        )

        print_demo_result(
            output
        )

    except Exception as error:

        print("\nAn error occurred:")

        print(
            type(error).__name__,
            "-",
            error
        )