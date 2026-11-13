
import os
import json

from dotenv import load_dotenv
from google import genai


# ==========================================
# 1. LOAD API KEY FROM .ENV
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please check your .env file."
    )


# ==========================================
# 2. CREATE GEMINI CLIENT
# ==========================================

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash-lite"


# ==========================================
# 3. TAKE USER INPUT
# ==========================================

print("\n========== NICHELYT AI CONTENT GENERATOR ==========\n")

niche = input("Enter your niche: ").strip()

goal = input(
    "Enter your goal "
    "(Education / Engagement / Brand Awareness / Lead Generation): "
).strip()

post_idea = input("Enter your post idea: ").strip()

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


# ==========================================
# 4. CREATE NICHELYT PROMPT
# ==========================================

prompt = f"""
You are the AI content generation module of Nichelyt.

Niche: {niche}
Goal: {goal}
Post Idea: {post_idea}
Tone: {tone}
Target Audience: {target_audience or "General audience"}
Language: {language}

Your task is to generate useful and relevant Instagram content.

Return ONLY a valid JSON object.
Do not add Markdown code fences.
Do not add any explanation outside the JSON.

The JSON must contain these fields:

{{
    "niche": "The provided niche",
    "goal": "The provided goal",
    "post_idea": "The provided post idea",
    "tone": "The provided tone",
    "content_angle": "The main approach used for the content",
    "hook": "One engaging opening line",
    "caption": "A useful and relevant Instagram caption",
    "cta": "A clear call to action",
    "hashtags": [
        "#RelevantHashtag1",
        "#RelevantHashtag2",
        "#RelevantHashtag3",
        "#RelevantHashtag4"
    ],
    "target_audience": "The intended audience",
    "rationale": "Briefly explain why this content approach was selected",
    "warnings": [
        "Hashtags are suggestions and are not live popularity measurements.",
        "Content performance is not guaranteed."
    ]
}}

Rules:
1. Keep the content relevant to the niche and post idea.
2. Follow the requested tone.
3. Do not invent research statistics, facts, or historical engagement numbers.
4. Do not claim that the content will definitely go viral.
5. Do not guarantee engagement.
6. Use natural and readable language.
7. Make the CTA relevant to the selected goal.
8. Return hashtags as a JSON array of strings.
"""


# ==========================================
# 5. SEND REQUEST TO GEMINI
# ==========================================

try:
    print("\nGenerating content using Gemini AI...\n")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    raw_output = response.text.strip()

    # ==========================================
    # 6. CONVERT RESPONSE INTO JSON
    # ==========================================

    content = json.loads(raw_output)

    # ==========================================
    # 7. DISPLAY GENERATED CONTENT
    # ==========================================

    print("========== GENERATED CONTENT ==========\n")

    print("CONTENT ANGLE:")
    print(content.get("content_angle", "Not available"))

    print("\nHOOK:")
    print(content.get("hook", "Not available"))

    print("\nCAPTION:")
    print(content.get("caption", "Not available"))

    print("\nCTA:")
    print(content.get("cta", "Not available"))

    print("\nHASHTAGS:")
    for hashtag in content.get("hashtags", []):
        print(hashtag)

    print("\nTARGET AUDIENCE:")
    print(content.get("target_audience", "Not available"))

    print("\nRATIONALE:")
    print(content.get("rationale", "Not available"))

    print("\nWARNINGS:")
    for warning in content.get("warnings", []):
        print("-", warning)

    # ==========================================
    # 8. SAVE JSON OUTPUT
    # ==========================================

    output_path = os.path.join(
        "data",
        "processed",
        "generated_content.json"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nGenerated content saved at:")
    print(output_path)

except json.JSONDecodeError:
    print("\nERROR: Gemini did not return valid JSON.")
    print("Raw response:")
    print(raw_output)

except Exception as error:
    print("\nAn error occurred:")
    print(type(error).__name__, ":", error)