import os
import json
import re

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from opportunity_engine_v2 import OpportunityEngineV2


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "caption_embeddings.npy"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "retrieval_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "integrated_generated_content.json"
)


# =====================================================
# 2. LOAD GEMINI API KEY
# =====================================================

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Check your .env file."
    )

client = genai.Client(
    api_key=api_key
)

MODEL_NAME = "gemini-3.5-flash-lite"


# =====================================================
# 3. LOAD RETRIEVAL DATA
# =====================================================

if not os.path.exists(
    EMBEDDINGS_PATH
):
    raise FileNotFoundError(
        f"Embeddings file not found: "
        f"{EMBEDDINGS_PATH}"
    )

if not os.path.exists(
    METADATA_PATH
):
    raise FileNotFoundError(
        f"Metadata file not found: "
        f"{METADATA_PATH}"
    )

embeddings = np.load(
    EMBEDDINGS_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)

if len(embeddings) != len(metadata):
    raise ValueError(
        "Embeddings and metadata row counts "
        "do not match."
    )

print(
    f"Loaded embeddings: {embeddings.shape}"
)

print(
    f"Loaded metadata: {metadata.shape}"
)


# =====================================================
# 4. FIND TEXT COLUMN
# =====================================================

possible_text_columns = [
    "clean_text",
    "caption",
    "imagetext",
    "text",
    "post_text",
    "description"
]

text_column = None

for column in possible_text_columns:
    if column in metadata.columns:
        text_column = column
        break

if text_column is None:
    raise ValueError(
        "No suitable text column found in "
        "retrieval_metadata.csv. "
        f"Available columns: "
        f"{list(metadata.columns)}"
    )

print(
    f"Using text column: {text_column}"
)


# =====================================================
# 5. NORMALIZE STORED EMBEDDINGS
# =====================================================

stored_embeddings = embeddings.astype(
    np.float32,
    copy=False
)

embedding_norms = np.linalg.norm(
    stored_embeddings,
    axis=1,
    keepdims=True
)

normalized_embeddings = (
    stored_embeddings
    /
    np.maximum(
        embedding_norms,
        1e-12
    )
)


# =====================================================
# 6. LOAD SENTENCE TRANSFORMER
# =====================================================

print(
    "\nLoading Sentence Transformer model..."
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Sentence Transformer loaded successfully."
)


# =====================================================
# 7. HELPER FUNCTIONS
# =====================================================

def safe_value(
    row,
    column,
    default="Not available"
):
    """
    Safely fetch a value from a metadata row.
    """
    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def clean_gemini_json(
    raw_output
):
    """
    Cleans Gemini response and extracts JSON.

    Handles:
    1. Normal JSON
    2. Markdown JSON code fences
    3. Additional text before or after JSON
    """
    if not raw_output:
        raise ValueError(
            "Gemini returned an empty response."
        )

    cleaned_output = raw_output.strip()

    # Remove Markdown JSON code fences
    cleaned_output = re.sub(
        r"^```json\s*",
        "",
        cleaned_output,
        flags=re.IGNORECASE
    )

    cleaned_output = re.sub(
        r"^```\s*",
        "",
        cleaned_output
    )

    cleaned_output = re.sub(
        r"\s*```$",
        "",
        cleaned_output
    )

    cleaned_output = cleaned_output.strip()

    # First attempt: Parse complete response
    try:
        return json.loads(
            cleaned_output
        )
    except json.JSONDecodeError:
        pass

    # Second attempt: Extract JSON object
    start_index = cleaned_output.find(
        "{"
    )

    end_index = cleaned_output.rfind(
        "}"
    )

    if (
        start_index == -1
        or
        end_index == -1
    ):
        raise json.JSONDecodeError(
            "No JSON object found in Gemini response.",
            cleaned_output,
            0
        )

    json_text = cleaned_output[
        start_index:
        end_index + 1
    ]

    return json.loads(
        json_text
    )


def retrieve_similar_posts(
    query,
    top_k=5
):
    """
    Retrieves historically similar posts
    using cosine similarity.
    """
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    similarities = (
        normalized_embeddings
        @
        query_embedding
    )

    # =====================================================
    # TWO-STAGE RETRIEVAL
    # Stage 1: Retrieve a larger semantic candidate pool
    # Stage 2: Rerank candidates using keyword overlap
    # =====================================================

    candidate_k = min(
        30,
        len(similarities)
    )

    candidate_indices = np.argsort(
        similarities
    )[::-1][:candidate_k]

    # Basic stopwords that do not add much topical meaning
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "how",
        "what",
        "why",
        "to",
        "of",
        "and",
        "in",
        "for",
        "on",
        "with",
        "can",
        "be",
        "from",
        "as"
    }

    # Convert the query into meaningful keywords
    query_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            query.lower()
        )
    ) - stopwords

    reranked_candidates = []

    for index in candidate_indices:
        row = metadata.iloc[
            index
        ]

        text_value = safe_value(
            row,
            text_column,
            ""
        )

        text = str(
            text_value
        ).lower()

        text_words = set(
            re.findall(
                r"\b[a-zA-Z]{3,}\b",
                text
            )
        )

        keyword_overlap = len(
            query_words.intersection(
                text_words
            )
        )

        keyword_score = (
            keyword_overlap /
            max(len(query_words), 1)
        )

        semantic_score = float(
            similarities[index]
        )

        # Semantic similarity remains the main factor.
        # Keyword overlap provides a small relevance bonus.
        combined_score = (
            0.85 * semantic_score
        ) + (
            0.15 * keyword_score
        )

        reranked_candidates.append(
            (
                index,
                combined_score
            )
        )

    reranked_candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    top_indices = [
        item[0]
        for item in reranked_candidates[:top_k]
    ]

    retrieved_posts = []

    for index in top_indices:
        row = metadata.iloc[
            index
        ]

        text_value = safe_value(
            row,
            text_column,
            ""
        )

        text = str(
            text_value
        ).strip()

        if (
            not text
            or
            text.lower() == "nan"
        ):
            continue

        engagement_value = safe_value(
            row,
            "engagement_rate",
            0
        )

        try:
            engagement_rate = round(
                float(
                    engagement_value
                ),
                6
            )
        except (
            ValueError,
            TypeError
        ):
            engagement_rate = (
                "Not available"
            )

        retrieved_post = {
            "similarity": round(
                float(
                    similarities[index]
                ),
                4
            ),
            "text": text[:1000],
            "username": str(
                safe_value(
                    row,
                    "username"
                )
            ),
            "type": str(
                safe_value(
                    row,
                    "type"
                )
            ),
            "engagement_rate": (
                engagement_rate
            ),
            "post_date": str(
                safe_value(
                    row,
                    "postcreateddate"
                )
            )
        }

        retrieved_posts.append(
            retrieved_post
        )

    return retrieved_posts


# =====================================================
# 8. USER INPUT
# =====================================================

print(
    "\n========== "
    "NICHELYT INTEGRATED AI GENERATOR "
    "==========\n"
)

niche = input(
    "Enter your niche: "
).strip()

goal = input(
    "Enter your goal "
    "(Education / Engagement / "
    "Brand Awareness / Lead Generation): "
).strip()

post_idea = input(
    "Enter your post idea: "
).strip()

tone = input(
    "Enter tone "
    "(Professional / Friendly / "
    "Educational / Bold): "
).strip()

target_audience = input(
    "Enter target audience (optional): "
).strip()

language = input(
    "Enter language (default: English): "
).strip()

if not language:
    language = "English"


# =====================================================
# 9. VALIDATE INPUT
# =====================================================

if not niche:
    raise ValueError(
        "Niche cannot be empty."
    )

if not goal:
    raise ValueError(
        "Goal cannot be empty."
    )

if not post_idea:
    raise ValueError(
        "Post idea cannot be empty."
    )

if not tone:
    raise ValueError(
        "Tone cannot be empty."
    )

if not target_audience:
    target_audience = (
        "General audience"
    )


# =====================================================
# 10. OPPORTUNITY ENGINE ANALYSIS
# =====================================================

print(
    "\nInitializing Opportunity Engine V2..."
)

try:
    opportunity_engine = (
        OpportunityEngineV2(
            project_root=BASE_DIR
        )
    )

    print(
        "\nAnalyzing niche opportunity..."
    )

    opportunity_analysis = (
        opportunity_engine.analyze(
            niche=niche,
            goal=goal,
            post_idea=post_idea,
            top_k=10,
            relevance_threshold=0.35
        )
    )

    if (
        opportunity_analysis.get(
            "status"
        )
        !=
        "success"
    ):
        print(
            "\nWarning: Opportunity analysis "
            "did not complete successfully."
        )

        opportunity_analysis = {
            "status": "unavailable",
            "message": (
                "Opportunity analysis "
                "was not available."
            )
        }
    else:
        print(
            "\nOpportunity analysis "
            "completed successfully."
        )

        opportunity_summary = (
            opportunity_analysis.get(
                "summary",
                {}
            )
        )

        print(
            "\nAverage similarity:",
            opportunity_summary.get(
                "average_similarity",
                "Not available"
            )
        )

        print(
            "Average engagement percentile:",
            opportunity_summary.get(
                "average_engagement_percentile",
                "Not available"
            )
        )

        print(
            "Average opportunity score:",
            opportunity_summary.get(
                "average_opportunity_score",
                "Not available"
            )
        )

except Exception as error:
    print(
        "\nWarning: Opportunity Engine "
        "could not be initialized."
    )

    print(
        type(error).__name__,
        ":",
        error
    )

    opportunity_analysis = {
        "status": "unavailable",
        "message": (
            "Opportunity analysis failed "
            "during execution."
        ),
        "error_type": (
            type(error).__name__
        )
    }


# =====================================================
# 11. SEMANTIC RETRIEVAL
# =====================================================

retrieval_query = f"""
Niche: {niche}
Goal: {goal}
Post idea: {post_idea}
Tone: {tone}
Target audience: {target_audience}
"""

print(
    "\nSearching similar historical posts..."
)

retrieved_posts = retrieve_similar_posts(
    retrieval_query,
    top_k=5
)

print(
    f"Retrieved "
    f"{len(retrieved_posts)} "
    f"similar posts."
)

for number, post in enumerate(
    retrieved_posts,
    start=1
):
    print(
        f"\n{number}. "
        f"Similarity: "
        f"{post['similarity']}"
    )

    print(
        f"Type: "
        f"{post['type']}"
    )

    print(
        f"Engagement rate: "
        f"{post['engagement_rate']}"
    )

    print(
        f"Text: "
        f"{post['text'][:250]}"
    )


# =====================================================
# 12. PREPARE RETRIEVAL CONTEXT
# =====================================================

retrieval_context = []

for number, post in enumerate(
    retrieved_posts,
    start=1
):
    retrieval_context.append(
        f"""
Historical example {number}:

Similarity: {post['similarity']}
Post type: {post['type']}
Historical engagement rate: {post['engagement_rate']}
Post date: {post['post_date']}
Text: {post['text']}
"""
    )

retrieval_context_text = "\n".join(
    retrieval_context
)

if not retrieval_context_text:
    retrieval_context_text = (
        "No suitable historical "
        "examples were found."
    )


# =====================================================
# 13. PREPARE OPPORTUNITY CONTEXT
# =====================================================

opportunity_context = {
    "status": opportunity_analysis.get(
        "status",
        "unavailable"
    ),
    "summary": opportunity_analysis.get(
        "summary",
        {}
    ),
    "saturation": opportunity_analysis.get(
        "saturation",
        {}
    ),
    "trend_momentum": opportunity_analysis.get(
        "trend_momentum",
        {}
    ),
    "limitations": opportunity_analysis.get(
        "limitations",
        []
    )
}

opportunity_context_text = json.dumps(
    opportunity_context,
    indent=4,
    ensure_ascii=False,
    default=str
)


# =====================================================
# 14. CREATE GEMINI PROMPT
# =====================================================

prompt = f"""
You are the AI content generation module
of Nichelyt.

Nichelyt is an experimental AI-assisted
social media content and niche opportunity
analysis system.

Generate original social media content
using the user inputs, retrieved historical
posts, and opportunity analysis as
contextual references.

USER INPUTS
-----------

Niche: {niche}
Goal: {goal}
Post idea: {post_idea}
Tone: {tone}
Target audience: {target_audience}
Language: {language}


RETRIEVED HISTORICAL EXAMPLES
-----------------------------

{retrieval_context_text}


OPPORTUNITY ANALYSIS CONTEXT
-----------------------------

{opportunity_context_text}


IMPORTANT INSTRUCTIONS
----------------------

1. Generate completely original content.

2. Do not copy historical examples
word for word.

3. Use historical examples only as
contextual references.

4. Do not claim that historical examples
prove future engagement.

5. Do not invent statistics or
historical results.

6. Do not promise viral performance.

7. Do not present historical engagement
rates as predictions.

8. Follow the requested niche, goal,
tone, and language.

9. Keep the content useful and relevant.

10. Return ONLY valid JSON.

11. Do not use Markdown code fences.

12. Use a JSON array for hashtags.

13. Keep warnings as a JSON array.

14. Opportunity score is an exploratory
heuristic, not a guaranteed performance
prediction.

15. Do not describe dataset-based trend
momentum as a live Instagram trend.

16. Do not claim that a high opportunity
score guarantees reach, engagement,
or virality.

17. Use opportunity analysis only to
improve the content angle and
strategic explanation.

18. Mention relevant limitations when
discussing opportunity analysis.

19. Do not present the opportunity score
as a scientific probability.

20. Clearly distinguish historical
descriptive information from future
performance predictions.


REQUIRED JSON STRUCTURE
-----------------------

{{
    "niche": "User niche",
    "goal": "User goal",
    "post_idea": "User post idea",
    "tone": "Requested tone",
    "content_angle": "Main content approach",
    "hook": "One clear opening line",
    "caption": "A useful original caption",
    "cta": "A relevant call to action",
    "hashtags": [
        "#RelevantHashtag1",
        "#RelevantHashtag2",
        "#RelevantHashtag3",
        "#RelevantHashtag4"
    ],
    "target_audience": "Intended audience",
    "retrieval_summary": "Briefly describe how the retrieved context was used",
    "rationale": "Explain the content approach",
    "warnings": [
        "Retrieved posts are historical references.",
        "Hashtags are suggestions, not live popularity measurements.",
        "Content performance is not guaranteed."
    ]
}}
"""


# =====================================================
# 15. CALL GEMINI
# =====================================================

try:
    print(
        "\nGenerating content using "
        "Gemini AI...\n"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    raw_output = response.text.strip()

    content = clean_gemini_json(
        raw_output
    )

    if not isinstance(
        content,
        dict
    ):
        raise ValueError(
            "Gemini response JSON "
            "is not an object."
        )

    # =================================================
    # 16. ADD SYSTEM INFORMATION
    # =================================================

    content[
        "retrieved_posts"
    ] = retrieved_posts

    content[
        "opportunity_analysis"
    ] = opportunity_analysis

    content[
        "generation_model"
    ] = MODEL_NAME

    content[
        "retrieval_method"
    ] = (
        "Sentence Transformer "
        "cosine similarity"
    )

    content[
        "opportunity_analysis_method"
    ] = (
        "Opportunity Engine V2 using "
        "historical dataset analysis"
    )

    content[
        "experimental_notice"
    ] = (
        "Generated content is experimental. "
        "Historical similarity does not "
        "guarantee future social media "
        "performance. "
        "Opportunity score is an exploratory "
        "heuristic based on historical "
        "dataset information."
    )

    # =================================================
    # 17. DISPLAY OPPORTUNITY ANALYSIS
    # =================================================

    print(
        "\n========== "
        "OPPORTUNITY ANALYSIS "
        "==========\n"
    )

    displayed_opportunity_summary = (
        opportunity_analysis.get(
            "summary",
            {}
        )
    )

    print(
        "Analysis status:",
        opportunity_analysis.get(
            "status",
            "unavailable"
        )
    )

    print(
        "Average Similarity:",
        displayed_opportunity_summary.get(
            "average_similarity",
            "Not available"
        )
    )

    print(
        "Average Engagement Percentile:",
        displayed_opportunity_summary.get(
            "average_engagement_percentile",
            "Not available"
        )
    )

    print(
        "Average Opportunity Score:",
        displayed_opportunity_summary.get(
            "average_opportunity_score",
            "Not available"
        )
    )

    print(
        "\nSaturation:"
    )

    print(
        json.dumps(
            opportunity_analysis.get(
                "saturation",
                {}
            ),
            indent=4,
            ensure_ascii=False,
            default=str
        )
    )

    print(
        "\nTrend Momentum:"
    )

    print(
        json.dumps(
            opportunity_analysis.get(
                "trend_momentum",
                {}
            ),
            indent=4,
            ensure_ascii=False,
            default=str
        )
    )

    # =================================================
    # 18. DISPLAY GENERATED CONTENT
    # =================================================

    print(
        "\n========== "
        "GENERATED CONTENT "
        "==========\n"
    )

    print(
        "CONTENT ANGLE:"
    )

    print(
        content.get(
            "content_angle",
            "Not available"
        )
    )

    print(
        "\nHOOK:"
    )

    print(
        content.get(
            "hook",
            "Not available"
        )
    )

    print(
        "\nCAPTION:"
    )

    print(
        content.get(
            "caption",
            "Not available"
        )
    )

    print(
        "\nCTA:"
    )

    print(
        content.get(
            "cta",
            "Not available"
        )
    )

    print(
        "\nHASHTAGS:"
    )

    hashtags = content.get(
        "hashtags",
        []
    )

    if isinstance(
        hashtags,
        list
    ):
        for hashtag in hashtags:
            print(
                hashtag
            )
    else:
        print(
            hashtags
        )

    print(
        "\nRETRIEVAL SUMMARY:"
    )

    print(
        content.get(
            "retrieval_summary",
            "Not available"
        )
    )

    print(
        "\nRATIONALE:"
    )

    print(
        content.get(
            "rationale",
            "Not available"
        )
    )

    print(
        "\nWARNINGS:"
    )

    warnings = content.get(
        "warnings",
        []
    )

    if isinstance(
        warnings,
        list
    ):
        for warning in warnings:
            print(
                "-",
                warning
            )
    else:
        print(
            warnings
        )

    # =================================================
    # 19. SAVE OUTPUT
    # =================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            content,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    print(
        "\nIntegrated content saved at:"
    )

    print(
        OUTPUT_PATH
    )

# =====================================================
# 20. ERROR HANDLING
# =====================================================

except json.JSONDecodeError:
    print(
        "\nERROR: Gemini did not "
        "return valid JSON."
    )

    print(
        "\nRaw response:"
    )

    print(
        raw_output
    )

except Exception as error:
    print(
        "\nAn error occurred:"
    )

    print(
        type(error).__name__,
        ":",
        error)