from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


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
    # HIDDEN OPPORTUNITY DISCOVERY
    # ---------------------------------------------------------

    def discover_hidden_opportunities(
        self,
        niche,
        top_k=100,
        max_suggestions=5
    ):

        if not isinstance(niche, str) or not niche.strip():
            raise ValueError("Niche cannot be empty.")

        if len(self.metadata) == 0:
            return {
                "status": "unavailable",
                "message": "No historical metadata is available.",
                "opportunities": []
            }

        if "clean_text" not in self.metadata.columns:
            return {
                "status": "unavailable",
                "message": "Caption text is missing from metadata.",
                "opportunities": []
            }

        # Create a niche-only semantic query
        query = f"Niche: {niche.strip()}"

        query_embedding = self._get_query_embedding(query)

        similarities = self._calculate_similarity(
            query_embedding
        )

        # Retrieve the most relevant historical posts
        top_indices = np.argsort(similarities)[::-1][
            :min(top_k, len(similarities))
        ]

        relevant_posts = self.metadata.iloc[
            top_indices
        ].copy()

        relevant_posts["similarity"] = similarities[
            top_indices
        ]
                # Keep only semantically relevant historical posts
        relevant_posts = relevant_posts[
            relevant_posts["similarity"] >= 0.35
        ].copy()

        # Clean caption text
        relevant_posts["clean_text"] = (
            relevant_posts["clean_text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        relevant_posts = relevant_posts[
            relevant_posts["clean_text"] != ""
        ].reset_index(drop=True)

        if len(relevant_posts) < 2:
            return {
                "status": "unavailable",
                "message": "Not enough relevant captions were found.",
                "opportunities": []
            }

        captions = relevant_posts["clean_text"].tolist()

        # Extract recurring two-word and three-word themes
        vectorizer = TfidfVectorizer(
            ngram_range=(2, 3),
            stop_words="english",
            min_df=2,
            max_df=0.90,
            max_features=1000,
            sublinear_tf=True
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(
                captions
            )

        except ValueError:
            return {
                "status": "unavailable",
                "message": "No recurring content themes were found.",
                "opportunities": []
            }

        feature_names = vectorizer.get_feature_names_out()

        candidate_themes = []

        # Evaluate every extracted theme
        for feature_index, theme in enumerate(feature_names):

            matching_indices = tfidf_matrix[
                :, feature_index
            ].nonzero()[0]

            frequency = len(matching_indices)

            if frequency < 2:
                continue

            matching_posts = relevant_posts.iloc[
                matching_indices
            ]

            theme_similarities = matching_posts[
                "similarity"
            ].to_numpy(dtype=float)

            average_similarity = float(
                np.mean(theme_similarities)
            )

            engagement_percentile = 0.0

            if "engagement_rate" in matching_posts.columns:

                engagement_values = pd.to_numeric(
                    matching_posts["engagement_rate"],
                    errors="coerce"
                ).to_numpy(dtype=float)

                percentiles = self._calculate_historical_percentile(
                    engagement_values
                )

                valid_percentiles = percentiles[
                    np.isfinite(percentiles)
                ]

                if len(valid_percentiles) > 0:
                    engagement_percentile = float(
                        np.mean(valid_percentiles)
                    )

            candidate_themes.append({
                "theme": theme,
                "frequency": frequency,
                "average_similarity": average_similarity,
                "engagement_percentile": engagement_percentile,
                "matching_indices": matching_indices
            })

        if not candidate_themes:
            return {
                "status": "unavailable",
                "message": "No recurring content themes were found.",
                "opportunities": []
            }

        # Normalize frequency for ranking
        maximum_frequency = max(
            item["frequency"]
            for item in candidate_themes
        )

        opportunities = []

        for item in candidate_themes:

            relevance_score = max(
                0.0,
                min(item["average_similarity"] * 100, 100.0)
            )

            frequency_score = (
                item["frequency"] / maximum_frequency
            ) * 100

            engagement_score = item[
                "engagement_percentile"
            ]

            # Exploratory ranking score
            discovery_score = (
                0.40 * relevance_score
                + 0.40 * engagement_score
                + 0.20 * frequency_score
            )

            matching_posts = relevant_posts.iloc[
                item["matching_indices"]
            ].sort_values(
                "similarity",
                ascending=False
            )

            supporting_posts = []

            for _, post in matching_posts.head(2).iterrows():

                supporting_posts.append({
                    "caption": str(
                        post["clean_text"]
                    )[:180],
                    "similarity": round(
                        float(post["similarity"]), 4
                    )
                })

            opportunities.append({
                "theme": item["theme"],
                "discovery_score": round(
                    discovery_score, 2
                ),
                "frequency": item["frequency"],
                "average_similarity": round(
                    item["average_similarity"], 4
                ),
                "engagement_percentile": round(
                    engagement_score, 2
                ),
                "explanation": (
                    f"This theme appeared in "
                    f"{item['frequency']} of the retrieved "
                    f"historical posts, with an average "
                    f"engagement percentile of "
                    f"{engagement_score:.2f}."
                ),
                "supporting_posts": supporting_posts
            })

        opportunities.sort(
            key=lambda item: item["discovery_score"],
            reverse=True
        )
        print("\n========== HIDDEN OPPORTUNITY DEBUG ==========")
        print("Selected Niche:", niche)
        print("Retrieved Posts:", len(relevant_posts))
        print("\nTOP 10 RANKED HISTORICAL THEMES:")

        for item in opportunities[:10]:
            print("\nTheme:", item["theme"])
            print("Discovery Score:", item["discovery_score"])
            print("Average Similarity:", item["average_similarity"])
            print("Supporting Captions:")

            for post in item["supporting_posts"]:
                print("-", post["caption"])

        print("\n==============================================")
        # Convert ranked historical themes into actionable ideas
        ranked_themes = opportunities[:max_suggestions]

        try:
            from gemini_content_service import generate_hidden_opportunities
            generated_ideas = generate_hidden_opportunities(
                niche=niche,
                candidate_themes=ranked_themes,
                max_suggestions=max_suggestions
            )

            polished_opportunities = []

            for idea in generated_ideas:

                source_indices = idea[
                    "source_theme_indices"
                ]

                supporting_themes = [
                    ranked_themes[index]
                    for index in source_indices
                ]

                best_evidence = max(
                    supporting_themes,
                    key=lambda item: item["discovery_score"]
                )

                polished_opportunities.append({
                    "theme": idea["title"],
                    "explanation": idea["description"],
                    "discovery_score": best_evidence[
                        "discovery_score"
                    ],
                    "frequency": sum(
                        item["frequency"]
                        for item in supporting_themes
                    ),
                    "average_similarity": best_evidence[
                        "average_similarity"
                    ],
                    "engagement_percentile": best_evidence[
                        "engagement_percentile"
                    ],
                    "supporting_posts": [
                        post
                        for item in supporting_themes
                        for post in item["supporting_posts"]
                    ][:2]
                })

            opportunities = polished_opportunities

        except Exception as error:

            print(
                "Hidden opportunity Gemini generation failed:",
                str(error)
            )

            return {
                "status": "unavailable",
                "message": (
                    "Actionable opportunities could not "
                    "be generated from the available evidence."
                ),
                "opportunities": []
            }


        return {
            "status": "success",
            "niche": niche.strip(),
            "retrieved_post_count": len(relevant_posts),
            "analyzed_theme_count": len(candidate_themes),
            "ranking_method": (
                "40% semantic relevance, "
                "40% historical engagement percentile, "
                "20% relative frequency."
            ),
            "opportunities": opportunities[:max_suggestions],
            "limitations": [
                "Themes are extracted from historical captions.",
                "Engagement percentiles are descriptive, not predictive.",
                "Discovery scores are exploratory ranking scores.",
                "Results do not represent live Instagram trends."
            ]
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

        similarities = self._calculate_similarity(
            query_embedding
        )

        print("DEBUG - Total similarities:", len(similarities))

        print(
            "DEBUG - Relevant posts:",
            int(np.sum(similarities >= relevance_threshold))
        )

        print("DEBUG - Threshold:", relevance_threshold)

        print("DEBUG - Min similarity:", similarities.min())

        print("DEBUG - Max similarity:", similarities.max())

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