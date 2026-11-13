import os
import json
import re

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    os.path.join(BASE_DIR, ".env")
)

api_key = os.getenv("GEMINI_API_KEY")

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

if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(
        f"Embeddings file not found: {EMBEDDINGS_PATH}"
    )

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_PATH}"
    )

embeddings = np.load(
    EMBEDDINGS_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)

if len(embeddings) != len(metadata):
    raise ValueError(
        "Embeddings and metadata row counts do not match."
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
        f"Available columns: {list(metadata.columns)}"
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
    stored_embeddings /
    np.maximum(embedding_norms, 1e-12)
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

def safe_value(row, column, default="Not available"):
    """
    Safely fetch a value from a metadata row.
    """

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def clean_gemini_json(raw_output):
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

    # Remove Markdown code fences
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

    # First attempt: parse complete response
    try:
        return json.loads(cleaned_output)

    except json.JSONDecodeError:
        pass

    # Second attempt: extract JSON object
    start_index = cleaned_output.find("{")
    end_index = cleaned_output.rfind("}")

    if start_index == -1 or end_index == -1:
        raise json.JSONDecodeError(
            "No JSON object found in Gemini response.",
            cleaned_output,
            0
        )

    json_text = cleaned_output[
        start_index:end_index + 1
    ]

    return json.loads(json_text)


def retrieve_similar_posts(query, top_k=5):
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
        normalized_embeddings @ query_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    retrieved_posts = []

    for index in top_indices:

        row = metadata.iloc[index]

        text_value = safe_value(
            row,
            text_column,
            ""
        )

        text = str(text_value).strip()

        if not text or text.lower() == "nan":
            continue

        engagement_value = safe_value(
            row,
            "engagement_rate",
            0
        )

        try:
            engagement_rate = round(
                float(engagement_value),
                6
            )

        except (ValueError, TypeError):
            engagement_rate = "Not available"

        retrieved_post = {
            "similarity": round(
                float(similarities[index]),
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

            "engagement_rate": engagement_rate,

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
    "\n========== NICHELYT INTEGRATED AI GENERATOR ==========\n"
)

niche = input(
    "Enter your niche: "
).strip()

goal = input(
    "Enter your goal "
    "(Education / Engagement / Brand Awareness / Lead Generation): "
).strip()

post_idea = input(
    "Enter your post idea: "
).strip()

tone = input(
    "Enter tone "
    "(Professional / Friendly / Educational / Bold): "
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
    target_audience = "General audience"


# =====================================================
# 10. SEMANTIC RETRIEVAL
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
    f"Retrieved {len(retrieved_posts)} similar posts."
)


for number, post in enumerate(
    retrieved_posts,
    start=1
):

    print(
        f"\n{number}. "
        f"Similarity: {post['similarity']}"
    )

    print(
        f"Type: {post['type']}"
    )

    print(
        f"Engagement rate: "
        f"{post['engagement_rate']}"
    )

    print(
        f"Text: {post['text'][:250]}"
    )


# =====================================================
# 11. PREPARE RETRIEVAL CONTEXT
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
        "No suitable historical examples were found."
    )


# =====================================================
# 12. CREATE GEMINI PROMPT
# =====================================================

prompt = f"""
You are the AI content generation module of Nichelyt.

Nichelyt is an experimental AI-assisted social media
content and niche opportunity analysis system.

Generate original social media content using the
user inputs and retrieved historical posts as
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

IMPORTANT INSTRUCTIONS
----------------------

1. Generate completely original content.
2. Do not copy historical examples word for word.
3. Use historical examples only as contextual references.
4. Do not claim that historical examples prove future engagement.
5. Do not invent statistics or historical results.
6. Do not promise viral performance.
7. Do not present historical engagement rates as predictions.
8. Follow the requested niche, goal, tone, and language.
9. Keep the content useful and relevant.
10. Return ONLY valid JSON.
11. Do not use Markdown code fences.
12. Use a JSON array for hashtags.
13. Keep warnings as a JSON array.

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
# 13. CALL GEMINI
# =====================================================

try:

    print(
        "\nGenerating content using Gemini AI...\n"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    raw_output = response.text.strip()

    content = clean_gemini_json(
        raw_output
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Gemini response JSON is not an object."
        )

    # Add retrieval information
    content["retrieved_posts"] = retrieved_posts

    # Add system information
    content["generation_model"] = MODEL_NAME

    content["retrieval_method"] = (
        "Sentence Transformer cosine similarity"
    )

    content["experimental_notice"] = (
        "Generated content is experimental. "
        "Historical similarity does not guarantee "
        "future social media performance."
    )


    # =================================================
    # 14. DISPLAY RESULT
    # =================================================

    print(
        "\n========== GENERATED CONTENT ==========\n"
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

    if isinstance(hashtags, list):

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

    if isinstance(warnings, list):

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
    # 15. SAVE OUTPUT
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
            ensure_ascii=False
        )

    print(
        "\nIntegrated content saved at:"
    )

    print(
        OUTPUT_PATH
    )


# =====================================================
# 16. ERROR HANDLING
# =====================================================

except json.JSONDecodeError:

    print(
        "\nERROR: Gemini did not return valid JSON."
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
        error
    )