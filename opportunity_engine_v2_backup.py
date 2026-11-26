from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class OpportunityEngineV2:
    """
    Nichelyt Opportunity Engine V2

    Features:
    1. Semantic relevance
    2. Historical engagement percentile
    3. Content saturation indicator
    4. Trend momentum indicator
    5. Exploratory opportunity score

    Important:
    - Saturation and trend momentum are dataset-based indicators.
    - They are not live Instagram metrics.
    - Opportunity score is an exploratory heuristic.
    - Historical engagement percentile is descriptive.
    - Trend momentum is not a future prediction.
    """

    def __init__(self, project_root=None):

        if project_root is None:
            project_root = Path(__file__).resolve().parent

        self.project_root = Path(project_root)

        self.processed_dir = (
            self.project_root / "data" / "processed"
        )

        self.embedding_path = (
            self.processed_dir / "caption_embeddings.npy"
        )

        self.metadata_path = (
            self.processed_dir / "retrieval_metadata.csv"
        )

        self.cleaned_data_path = (
            self.processed_dir / "dataset2_cleaned.csv"
        )

        print("Loading SentenceTransformer model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Loading embeddings...")

        self.embeddings = np.load(
            self.embedding_path
        ).astype("float32")

        print("Loading metadata...")

        self.metadata = pd.read_csv(
            self.metadata_path
        )

        print("Loading cleaned dataset...")

        self.cleaned_data = pd.read_csv(
            self.cleaned_data_path
        )

        self._validate_data()

        self._prepare_date_features()

        print(
            "Opportunity Engine V2 loaded successfully."
        )

    # ---------------------------------------------------------
    # DATA VALIDATION
    # ---------------------------------------------------------

    def _validate_data(self):

        if len(self.embeddings) != len(self.metadata):

            raise ValueError(
                "Embedding and metadata row count do not match. "
                f"Embeddings: {len(self.embeddings)}, "
                f"Metadata: {len(self.metadata)}"
            )

        if len(self.embeddings) == 0:

            raise ValueError(
                "No embeddings were found."
            )

        if self.cleaned_data.empty:

            raise ValueError(
                "Cleaned dataset is empty."
            )

        print(
            f"Embedding shape: {self.embeddings.shape}"
        )

        print(
            f"Metadata shape: {self.metadata.shape}"
        )

        print(
            f"Cleaned dataset shape: "
            f"{self.cleaned_data.shape}"
        )

        print(
            "Metadata columns:",
            list(self.metadata.columns)
        )

        print(
            "Cleaned dataset columns:",
            list(self.cleaned_data.columns)
        )

    # ---------------------------------------------------------
    # DATE PREPARATION AND ALIGNMENT
    # ---------------------------------------------------------

    def _prepare_date_features(self):

        possible_date_columns = [
            "postcreateddate",
            "post_created_date",
            "post_date",
            "post_datetime",
            "date",
            "created_at",
            "datetime"
        ]

        self.date_column = None

        # Case 1: Date exists in metadata

        for column in possible_date_columns:

            if column in self.metadata.columns:

                self.date_column = column

                self.metadata["analysis_date"] = (
                    pd.to_datetime(
                        self.metadata[column],
                        errors="coerce",
                        format="mixed"
                    )
                )

                valid_dates = int(
                    self.metadata[
                        "analysis_date"
                    ].notna().sum()
                )

                print(
                    "Date source: metadata"
                )

                print(
                    f"Date column: {column}"
                )

                print(
                    "Valid dates available for "
                    f"trend analysis: {valid_dates}"
                )

                return

        # Case 2: Date exists only in cleaned dataset

        cleaned_date_column = None

        for column in possible_date_columns:

            if column in self.cleaned_data.columns:

                cleaned_date_column = column

                break

        if cleaned_date_column is None:

            self.metadata["analysis_date"] = pd.NaT

            print(
                "No usable date column found."
            )

            print(
                "Trend momentum will be unavailable."
            )

            return

        # Try to find a common alignment key

        possible_key_columns = [
            "source_index",
            "original_index",
            "dataset_index",
            "row_index",
            "post_id",
            "id",
            "index"
        ]

        common_key = None

        for key in possible_key_columns:

            if (
                key in self.metadata.columns
                and key in self.cleaned_data.columns
            ):

                common_key = key

                break

        if common_key is not None:

            print(
                "Attempting date alignment using key:",
                common_key
            )

            date_mapping = (
                self.cleaned_data[
                    [common_key, cleaned_date_column]
                ]
                .drop_duplicates(
                    subset=[common_key]
                )
            )

            date_mapping = date_mapping.rename(
                columns={
                    cleaned_date_column: "analysis_date"
                }
            )

            self.metadata = self.metadata.merge(
                date_mapping,
                on=common_key,
                how="left",
                sort=False
            )

            self.date_column = cleaned_date_column

            self.metadata["analysis_date"] = (
                pd.to_datetime(
                    self.metadata["analysis_date"],
                    errors="coerce",
                    format="mixed"
                )
            )

            valid_dates = int(
                self.metadata[
                    "analysis_date"
                ].notna().sum()
            )

            print(
                "Date source: cleaned dataset"
            )

            print(
                "Alignment key:",
                common_key
            )

            print(
                f"Valid dates available for "
                f"trend analysis: {valid_dates}"
            )

            return

        # Case 3: Same number of rows

        if len(self.metadata) == len(self.cleaned_data):

            print(
                "Metadata and cleaned dataset have "
                "the same number of rows."
            )

            print(
                "Using positional date alignment."
            )

            self.metadata["analysis_date"] = (
                pd.to_datetime(
                    self.cleaned_data[
                        cleaned_date_column
                    ].reset_index(drop=True),
                    errors="coerce",
                    format="mixed"
                ).to_numpy()
            )

            self.date_column = cleaned_date_column

            valid_dates = int(
                self.metadata[
                    "analysis_date"
                ].notna().sum()
            )

            print(
                f"Valid dates available for "
                f"trend analysis: {valid_dates}"
            )

            return

        # Case 4: No safe alignment

        self.metadata["analysis_date"] = pd.NaT

        self.date_column = None

        print(
            "Date column exists in cleaned dataset, "
            "but safe alignment with metadata was not found."
        )

        print(
            f"Metadata rows: {len(self.metadata)}"
        )

        print(
            f"Cleaned dataset rows: {len(self.cleaned_data)}"
        )

        print(
            "Trend momentum will be unavailable "
            "until a reliable alignment key is added."
        )

    # ---------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_vector(values):

        values = np.asarray(
            values,
            dtype=float
        )

        if len(values) == 0:

            return values

        finite_mask = np.isfinite(values)

        if not finite_mask.any():

            return np.zeros_like(values)

        minimum = np.nanmin(values)

        maximum = np.nanmax(values)

        if maximum == minimum:

            return np.zeros_like(values)

        normalized_values = (
            values - minimum
        ) / (
            maximum - minimum
        )

        normalized_values = np.nan_to_num(
            normalized_values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return normalized_values

    # ---------------------------------------------------------
    # QUERY EMBEDDING
    # ---------------------------------------------------------

    def _get_query_embedding(self, query):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        return query_embedding.astype(
            "float32"
        )

    # ---------------------------------------------------------
    # SIMILARITY
    # ---------------------------------------------------------

    def _calculate_similarity(self, query_embedding):

        similarities = np.dot(
            self.embeddings,
            query_embedding
        )

        return similarities

    # ---------------------------------------------------------
    # HISTORICAL ENGAGEMENT PERCENTILE
    # ---------------------------------------------------------

    def _calculate_historical_percentile(
        self,
        engagement_values
    ):

        if (
            "engagement_rate"
            not in self.cleaned_data.columns
        ):

            return np.full(
                len(engagement_values),
                np.nan
            )

        reference_values = pd.to_numeric(
            self.cleaned_data[
                "engagement_rate"
            ],
            errors="coerce"
        ).dropna()

        if len(reference_values) == 0:

            return np.full(
                len(engagement_values),
                np.nan
            )

        percentiles = []

        for value in engagement_values:

            if not np.isfinite(value):

                percentiles.append(np.nan)

                continue

            percentile = (
                reference_values <= value
            ).mean() * 100

            percentiles.append(
                percentile
            )

        return np.array(
            percentiles,
            dtype=float
        )

    # ---------------------------------------------------------
    # CONTENT SATURATION
    # ---------------------------------------------------------

    def _calculate_content_saturation(
        self,
        similarities,
        relevance_threshold=0.35
    ):

        similarities = np.asarray(
            similarities,
            dtype=float
        )

        relevant_mask = (
            similarities >= relevance_threshold
        )

        relevant_count = int(
            relevant_mask.sum()
        )

        total_count = len(
            similarities
        )

        if total_count == 0:

            saturation_percentage = 0.0

        else:

            saturation_percentage = (
                relevant_count / total_count
            ) * 100

        saturation_score = min(
            saturation_percentage,
            100.0
        )

        opportunity_from_saturation = (
            100.0 - saturation_score
        )

        return {
            "relevant_post_count": relevant_count,

            "total_post_count": total_count,

            "saturation_percentage": round(
                saturation_percentage,
                2
            ),

            "saturation_score": round(
                saturation_score,
                2
            ),

            "opportunity_from_saturation": round(
                opportunity_from_saturation,
                2
            ),

            "relevance_threshold": relevance_threshold
        }

    # ---------------------------------------------------------
    # TREND MOMENTUM
    # ---------------------------------------------------------

    def _calculate_trend_momentum(
        self,
        relevant_mask
    ):
        """
        Calculate dataset-based trend momentum.

        Important:
        This is based on historical dataset frequency.
        It is not live Instagram trend data.

        Warnings are added when the relevant sample
        size is small.
        """

        if (
            "analysis_date"
            not in self.metadata.columns
        ):

            return {
                "status": "unavailable",

                "reason": (
                    "Analysis date column is unavailable."
                ),

                "warnings": [
                    (
                        "Trend momentum cannot be calculated "
                        "because the date column is unavailable."
                    )
                ]
            }

        dates = self.metadata[
            "analysis_date"
        ]

        relevant_series = pd.Series(
            relevant_mask,
            index=self.metadata.index
        )

        valid_mask = (
            dates.notna()
            & relevant_series
        )

        relevant_dates = dates[
            valid_mask
        ]

        relevant_dated_post_count = int(
            len(relevant_dates)
        )

        warnings = []

        if relevant_dated_post_count < 30:

            warnings.append(
                (
                    "Small sample warning: fewer than 30 "
                    "relevant dated posts were found. "
                    "Trend momentum may be unstable."
                )
            )

        if relevant_dated_post_count < 10:

            warnings.append(
                (
                    "Trend momentum requires at least 10 "
                    "dated relevant posts."
                )
            )

            return {
                "status": "unavailable",

                "reason": (
                    "Not enough dated relevant posts "
                    "for trend analysis."
                ),

                "relevant_dated_post_count": (
                    relevant_dated_post_count
                ),

                "warnings": warnings
            }

        earliest_date = relevant_dates.min()

        latest_date = relevant_dates.max()

        total_days = (
            latest_date - earliest_date
        ).days

        if total_days < 14:

            warnings.append(
                (
                    "The relevant post date range is "
                    "shorter than 14 days."
                )
            )

            return {
                "status": "unavailable",

                "reason": (
                    "Date range is too short "
                    "for trend momentum."
                ),

                "relevant_dated_post_count": (
                    relevant_dated_post_count
                ),

                "date_range_days": int(
                    total_days
                ),

                "warnings": warnings
            }

        midpoint = (
            earliest_date
            + (
                latest_date - earliest_date
            ) / 2
        )

        older_count = int(
            (
                relevant_dates <= midpoint
            ).sum()
        )

        recent_count = int(
            (
                relevant_dates > midpoint
            ).sum()
        )

        older_days = max(
            (
                midpoint - earliest_date
            ).days,
            1
        )

        recent_days = max(
            (
                latest_date - midpoint
            ).days,
            1
        )

        older_daily_rate = (
            older_count / older_days
        )

        recent_daily_rate = (
            recent_count / recent_days
        )

        if older_count < 5:

            warnings.append(
                (
                    "Older-period sample is small "
                    "(fewer than 5 relevant posts)."
                )
            )

        if recent_count < 5:

            warnings.append(
                (
                    "Recent-period sample is small "
                    "(fewer than 5 relevant posts)."
                )
            )

        if older_daily_rate == 0:

            if recent_daily_rate > 0:

                growth_percentage = 100.0

            else:

                growth_percentage = 0.0

        else:

            growth_percentage = (
                (
                    recent_daily_rate
                    - older_daily_rate
                )
                / older_daily_rate
            ) * 100

        capped_growth = max(
            min(
                growth_percentage,
                100.0
            ),
            -100.0
        )

        momentum_score = (
            capped_growth + 100
        ) / 2

        warnings.append(
            (
                "This trend indicator measures historical "
                "content frequency in the dataset only. "
                "It does not measure live Instagram trends."
            )
        )

        return {
            "status": "available",

            "earliest_date": str(
                earliest_date.date()
            ),

            "latest_date": str(
                latest_date.date()
            ),

            "date_range_days": int(
                total_days
            ),

            "relevant_dated_post_count": (
                relevant_dated_post_count
            ),

            "older_relevant_count": older_count,

            "recent_relevant_count": recent_count,

            "older_period_days": int(
                older_days
            ),

            "recent_period_days": int(
                recent_days
            ),

            "older_daily_rate": round(
                older_daily_rate,
                4
            ),

            "recent_daily_rate": round(
                recent_daily_rate,
                4
            ),

            "growth_percentage": round(
                growth_percentage,
                2
            ),

            "capped_growth_percentage": round(
                capped_growth,
                2
            ),

            "momentum_score": round(
                momentum_score,
                2
            ),

            "warnings": warnings
        }

    # ---------------------------------------------------------
    # MAIN ANALYSIS
    # ---------------------------------------------------------

    def analyze(
        self,
        niche,
        goal,
        post_idea="",
        top_k=10,
        relevance_threshold=0.35
    ):

        if not isinstance(
            niche,
            str
        ) or not niche.strip():

            raise ValueError(
                "Niche cannot be empty."
            )

        if not isinstance(
            goal,
            str
        ) or not goal.strip():

            raise ValueError(
                "Goal cannot be empty."
            )

        if top_k < 1:

            raise ValueError(
                "top_k must be at least 1."
            )

        top_k = min(
            int(top_k),
            len(self.embeddings)
        )

        query = (
            f"Niche: {niche}. "
            f"Goal: {goal}. "
            f"Post idea: {post_idea}"
        )

        print(
            "\nGenerating query embedding..."
        )

        query_embedding = (
            self._get_query_embedding(
                query
            )
        )

        similarities = (
            self._calculate_similarity(
                query_embedding
            )
        )

        ranked_indices = np.argsort(
            similarities
        )[::-1]

        top_indices = (
            ranked_indices[:top_k]
        )

        top_similarities = (
            similarities[top_indices]
        )

        results = self.metadata.iloc[
            top_indices
        ].copy()

        results["similarity"] = (
            top_similarities
        )

        # -------------------------------------------------
        # ENGAGEMENT COLUMN
        # -------------------------------------------------

        engagement_column = None

        possible_engagement_columns = [
            "engagement_rate",
            "engagement",
            "engagementrate"
        ]

        for column in possible_engagement_columns:

            if column in results.columns:

                engagement_column = column

                break

        if engagement_column is not None:

            engagement_values = pd.to_numeric(
                results[
                    engagement_column
                ],
                errors="coerce"
            ).to_numpy()

            percentiles = (
                self._calculate_historical_percentile(
                    engagement_values
                )
            )

            results[
                "engagement_percentile"
            ] = percentiles

        else:

            results[
                "engagement_percentile"
            ] = np.nan

        # -------------------------------------------------
        # SATURATION
        # -------------------------------------------------

        saturation = (
            self._calculate_content_saturation(
                similarities=similarities,
                relevance_threshold=relevance_threshold
            )
        )

        relevant_mask = (
            similarities >= relevance_threshold
        )

        # -------------------------------------------------
        # TREND MOMENTUM
        # -------------------------------------------------

        trend = (
            self._calculate_trend_momentum(
                relevant_mask
            )
        )

        # -------------------------------------------------
        # NORMALIZE SIMILARITY
        # -------------------------------------------------

        similarity_normalized = (
            self._normalize_vector(
                results[
                    "similarity"
                ].to_numpy()
            )
        )

        # -------------------------------------------------
        # NORMALIZE ENGAGEMENT
        # -------------------------------------------------

        engagement_percentiles = (
            results[
                "engagement_percentile"
            ]
        )

        if engagement_percentiles.notna().any():

            engagement_normalized = (
                engagement_percentiles
                .fillna(0)
                .to_numpy()
                / 100
            )

        else:

            engagement_normalized = (
                np.zeros(
                    len(results)
                )
            )

        # -------------------------------------------------
        # OPPORTUNITY SCORE
        # -------------------------------------------------

        results[
            "opportunity_score"
        ] = (
            (
                0.60
                * similarity_normalized
            )
            + (
                0.40
                * engagement_normalized
            )
        ) * 100

        results[
            "opportunity_score"
        ] = (
            results[
                "opportunity_score"
            ].round(2)
        )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        average_similarity = float(
            results[
                "similarity"
            ].mean()
        )

        average_engagement_percentile = (
            results[
                "engagement_percentile"
            ].mean()
        )

        average_opportunity_score = float(
            results[
                "opportunity_score"
            ].mean()
        )

        if pd.isna(
            average_engagement_percentile
        ):

            average_engagement_percentile = None

        else:

            average_engagement_percentile = round(
                float(
                    average_engagement_percentile
                ),
                2
            )

        return {
            "status": "success",

            "query": query,

            "summary": {
                "average_similarity": round(
                    average_similarity,
                    4
                ),

                "average_engagement_percentile": (
                    average_engagement_percentile
                ),

                "average_opportunity_score": round(
                    average_opportunity_score,
                    2
                )
            },

            "saturation": saturation,

            "trend_momentum": trend,

            "top_results": results.to_dict(
                orient="records"
            ),

            "limitations": [
                (
                    "Saturation is estimated "
                    "from this dataset only."
                ),

                (
                    "Trend momentum is based "
                    "on relevant content volume."
                ),

                (
                    "Trend momentum may be unstable "
                    "when the number of relevant posts is small."
                ),

                (
                    "This is not live Instagram "
                    "trend data."
                ),

                (
                    "Historical engagement percentile "
                    "is descriptive."
                ),

                (
                    "Opportunity score is an "
                    "exploratory heuristic."
                ),

                (
                    "No future engagement guarantee "
                    "is provided."
                ),

                (
                    "Semantic similarity is not "
                    "model accuracy."
                )
            ]
        }


# ---------------------------------------------------------
# TERMINAL APPLICATION
# ---------------------------------------------------------

def main():

    print(
        "\nNichelyt Opportunity Engine V2"
    )

    print(
        "--------------------------------"
    )

    niche = input(
        "Enter niche: "
    ).strip()

    goal = input(
        "Enter goal: "
    ).strip()

    post_idea = input(
        "Enter post idea (optional): "
    ).strip()

    try:

        engine = OpportunityEngineV2()

        output = engine.analyze(
            niche=niche,
            goal=goal,
            post_idea=post_idea,
            top_k=10,
            relevance_threshold=0.35
        )

        print(
            "\n\nSUMMARY"
        )

        print(
            "-------"
        )

        print(
            "Average similarity:",
            output[
                "summary"
            ][
                "average_similarity"
            ]
        )

        print(
            "Average engagement percentile:",
            output[
                "summary"
            ][
                "average_engagement_percentile"
            ]
        )

        print(
            "Average opportunity score:",
            output[
                "summary"
            ][
                "average_opportunity_score"
            ]
        )

        print(
            "\nSATURATION"
        )

        print(
            "----------"
        )

        for key, value in output[
            "saturation"
        ].items():

            print(
                f"{key}: {value}"
            )

        print(
            "\nTREND MOMENTUM"
        )

        print(
            "--------------"
        )

        trend_output = output[
            "trend_momentum"
        ]

        for key, value in trend_output.items():

            if key != "warnings":

                print(
                    f"{key}: {value}"
                )

        warnings = trend_output.get(
            "warnings",
            []
        )

        if warnings:

            print(
                "\nWARNINGS"
            )

            print(
                "--------"
            )

            for number, warning in enumerate(
                warnings,
                start=1
            ):

                print(
                    f"{number}. {warning}"
                )

        print(
            "\nTOP RESULTS"
        )

        print(
            "-----------"
        )

        for index, result in enumerate(
            output[
                "top_results"
            ],
            start=1
        ):

            similarity = result.get(
                "similarity",
                0
            )

            opportunity_score = result.get(
                "opportunity_score",
                0
            )

            print(
                f"{index}. "
                f"Similarity: "
                f"{similarity:.4f}, "
                f"Opportunity Score: "
                f"{opportunity_score:.2f}"
            )

        print(
            "\nLIMITATIONS"
        )

        print(
            "-----------"
        )

        for number, limitation in enumerate(
            output[
                "limitations"
            ],
            start=1
        ):

            print(
                f"{number}. {limitation}"
            )

    except Exception as error:

        print(
            "\nAn error occurred:"
        )

        print(
            type(error).__name__,
            ":",
            error
        )


if __name__ == "__main__":

    main()