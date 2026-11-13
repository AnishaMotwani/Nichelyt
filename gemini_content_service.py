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


# =====================================================
# 2. GEMINI CONFIGURATION
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


# =====================================================
# 5. NORMALIZE EMBEDDINGS
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
    "Loading Sentence Transformer model..."
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
    Extract and parse JSON from Gemini output.
    """

    if not raw_output:
        raise ValueError(
            "Gemini returned an empty response."
        )

    cleaned_output = raw_output.strip()

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

    try:
        return json.loads(
            cleaned_output
        )

    except json.JSONDecodeError:
        pass

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
        raise ValueError(
            "No valid JSON object found "
            "in Gemini response."
        )

    json_text = cleaned_output[
        start_index:
        end_index + 1
    ]

    return json.loads(
        json_text
    )
# =====================================================
# 8. RETRIEVAL FUNCTION
# =====================================================

def retrieve_similar_posts(
    query,
    top_k=5
):
    """
    Retrieve similar historical posts using:

    1. Semantic similarity
    2. Keyword-overlap reranking

    Returns:
        list: Similar historical posts
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(
            "top_k must be a positive integer."
        )

    # -------------------------------------------------
    # 1. Generate query embedding
    # -------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype(
        np.float32
    )

    query_norm = np.linalg.norm(
        query_embedding
    )

    query_embedding = (
        query_embedding
        /
        max(query_norm, 1e-12)
    )

    # -------------------------------------------------
    # 2. Calculate semantic similarity
    # -------------------------------------------------

    semantic_scores = (
        normalized_embeddings
        @
        query_embedding
    )

    # -------------------------------------------------
    # 3. Select candidate posts
    # -------------------------------------------------

    candidate_count = min(
        30,
        len(metadata)
    )

    candidate_indices = np.argsort(
        semantic_scores
    )[::-1][:candidate_count]

    # -------------------------------------------------
    # 4. Prepare query keywords
    # -------------------------------------------------

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower()
        )
    )

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "into",
        "using",
        "about",
        "post",
        "content"
    }

    query_words = (
        query_words
        -
        stop_words
    )

    # -------------------------------------------------
    # 5. Keyword-overlap reranking
    # -------------------------------------------------

    reranked_results = []

    for index in candidate_indices:

        row = metadata.iloc[index]

        text_value = safe_value(
            row,
            text_column,
            default=""
        )

        text_value = str(
            text_value
        )

        text_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text_value.lower()
            )
        )

        if query_words:

            keyword_overlap = (
                len(
                    query_words
                    &
                    text_words
                )
                /
                len(query_words)
            )

        else:

            keyword_overlap = 0.0

        semantic_score = float(
            semantic_scores[index]
        )

        combined_score = (
            0.85 * semantic_score
            +
            0.15 * keyword_overlap
        )

        reranked_results.append(
            {
                "embedding_row_id": int(index),
                "similarity": round(
                    combined_score,
                    4
                ),
                "semantic_similarity": round(
                    semantic_score,
                    4
                ),
                "keyword_overlap": round(
                    keyword_overlap,
                    4
                ),
                "text": text_value,
                "type": safe_value(
                    row,
                    "type",
                    default="Not available"
                ),
                "engagement_rate": safe_value(
                    row,
                    "engagement_rate",
                    default="Not available"
                ),
                "post_date": safe_value(
                    row,
                    "postcreateddate",
                    default=safe_value(
                        row,
                        "post_date",
                        default="Not available"
                    )
                )
            }
        )

    # -------------------------------------------------
    # 6. Sort and return top results
    # -------------------------------------------------

    reranked_results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return reranked_results[:top_k]
# =====================================================
# 9. REUSABLE GEMINI GENERATION FUNCTION
# =====================================================

def generate_content(
    niche,
    goal,
    post_idea,
    tone,
    target_audience="General audience",
    language="English"
):
    """
    Generate original content using Gemini,
    historical retrieval, and Opportunity Engine V2.

    Returns:
        dict: Structured generated content and
        supporting analysis.
    """

    # =================================================
    # 1. VALIDATE INPUT
    # =================================================

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

    if not isinstance(
        post_idea,
        str
    ) or not post_idea.strip():

        raise ValueError(
            "Post idea cannot be empty."
        )

    if not isinstance(
        tone,
        str
    ) or not tone.strip():

        raise ValueError(
            "Tone cannot be empty."
        )

    if not target_audience:
        target_audience = "General audience"

    if not language:
        language = "English"

    # =================================================
    # 2. OPPORTUNITY ENGINE
    # =================================================

    try:

        opportunity_engine = (
            OpportunityEngineV2(
                project_root=BASE_DIR
            )
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
            != "success"
        ):

            opportunity_analysis = {
                "status": "unavailable",
                "message": (
                    "Opportunity analysis "
                    "was not available."
                )
            }

    except Exception as error:

        opportunity_analysis = {
            "status": "unavailable",
            "message": (
                "Opportunity analysis failed."
            ),
            "error_type": type(
                error
            ).__name__
        }

    # =================================================
    # 3. RETRIEVE SIMILAR POSTS
    # =================================================

    retrieval_query = f"""
Niche: {niche}
Goal: {goal}
Post idea: {post_idea}
Tone: {tone}
Target audience: {target_audience}
"""

    retrieved_posts = retrieve_similar_posts(
        retrieval_query,
        top_k=5
    )

    # =================================================
    # 4. PREPARE RETRIEVAL CONTEXT
    # =================================================

    retrieval_context = []

    for number, post in enumerate(
        retrieved_posts,
        start=1
    ):

        retrieval_context.append(
            f"""
Historical example {number}:

Similarity: {post.get("similarity")}
Post type: {post.get("type")}
Historical engagement rate: {
    post.get("engagement_rate")
}
Post date: {post.get("post_date")}
Text: {post.get("text")}
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

    # =================================================
    # 5. PREPARE OPPORTUNITY CONTEXT
    # =================================================

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

    # =================================================
    # 6. CREATE GEMINI PROMPT
    # =================================================

    prompt = f"""
You are the AI content generation module
of Nichelyt.

Generate original social media content
using the provided user inputs, retrieved
historical posts, and opportunity analysis.

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
2. Do not copy historical examples word for word.
3. Do not claim that historical examples prove future engagement.
4. Do not invent statistics or historical results.
5. Do not promise viral performance.
6. Follow the requested niche, goal, tone, and language.
7. Return only valid JSON.
8. Do not use Markdown code fences.
9. Use a JSON array for hashtags.
10. Keep warnings as a JSON array.
11. Treat opportunity score as an exploratory heuristic.
12. Do not describe historical dataset trends as live trends.
13. Do not present opportunity score as a probability.
14. Clearly distinguish historical information from predictions.

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
        "#RelevantHashtag3"
    ],
    "target_audience": "Intended audience",
    "retrieval_summary": "How retrieval was used",
    "rationale": "Explain the content approach",
    "warnings": [
        "Content performance is not guaranteed."
    ]
}}
"""

    # =================================================
    # 7. CALL GEMINI
    # =================================================

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    raw_output = response.text

    content = clean_gemini_json(
        raw_output
    )

    if not isinstance(
        content,
        dict
    ):

        raise ValueError(
            "Gemini response JSON "
            "must be an object."
        )

    # =================================================
    # 8. ADD SYSTEM INFORMATION
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
        "cosine similarity with "
        "keyword-overlap reranking"
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
        "guarantee future performance. "
        "Opportunity score is an exploratory "
        "heuristic based on historical data."
    )

    return content