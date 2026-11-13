
from datetime import datetime


# ============================================================
# NICHELYT - TEMPLATE-BASED CONTENT GENERATOR
# ============================================================


def clean_input(value, default_value=""):
    """
    Cleans user input.

    If the user enters nothing, the default value is returned.
    """
    value = str(value).strip()

    if not value:
        return default_value

    return value


def normalize_text(value):
    """
    Converts text into a simple normalized format.
    """
    return " ".join(str(value).strip().split())


def create_hook(niche, goal, post_idea, tone):
    """
    Generates a hook using predefined templates.
    """

    if post_idea:
        subject = post_idea
    else:
        subject = f"{niche}"

    if goal == "education":
        if tone == "professional":
            return (
                f"Want to understand {subject} more clearly? "
                f"Here are the key concepts you should know."
            )

        return (
            f"Confused about {subject}? "
            f"Let's understand it in a simple and practical way."
        )

    if goal == "engagement":
        return (
            f"What is your experience with {subject}? "
            f"Let's discuss it!"
        )

    if goal == "brand awareness":
        return (
            f"Discover a smarter way to explore {subject} "
            f"with practical insights and useful ideas."
        )

    if goal == "lead generation":
        return (
            f"Looking for practical solutions related to {subject}? "
            f"Explore these useful insights."
        )

    return (
        f"Here is something interesting you should know about {subject}."
    )


def create_caption(niche, goal, post_idea, tone):
    """
    Generates a caption based on niche, goal, idea, and tone.
    """

    subject = post_idea if post_idea else niche

    if goal == "education":
        caption = (
            f"Learning about {subject} can become easier when we "
            f"break complex concepts into simple steps.\n\n"
            f"In this post, we explore important ideas, practical "
            f"examples, and useful takeaways related to {niche}.\n\n"
            f"Keep learning, keep experimenting, and apply your "
            f"knowledge through practical projects."
        )

    elif goal == "engagement":
        caption = (
            f"Let's talk about {subject}!\n\n"
            f"Different people may have different experiences, "
            f"ideas, and opinions about {niche}.\n\n"
            f"Share your thoughts in the comments and let us know "
            f"what you think."
        )

    elif goal == "brand awareness":
        caption = (
            f"Exploring {subject} opens new possibilities in the "
            f"world of {niche}.\n\n"
            f"Our focus is on sharing useful ideas, practical "
            f"knowledge, and meaningful insights.\n\n"
            f"Follow for more content related to {niche}."
        )

    elif goal == "lead generation":
        caption = (
            f"Are you exploring {subject}?\n\n"
            f"Understanding the right approach can help you make "
            f"better decisions related to {niche}.\n\n"
            f"Connect with us to learn more and explore possible "
            f"solutions."
        )

    else:
        caption = (
            f"Exploring {subject and subject or niche} "
            f"through useful ideas and practical insights.\n\n"
            f"Stay curious, keep learning, and continue exploring "
            f"new possibilities in {niche}."
        )

    if tone == "professional":
        caption += (
            "\n\n"
            "Focus on continuous learning, practical implementation, "
            "and measurable improvement."
        )

    elif tone == "friendly":
        caption += (
            "\n\n"
            "Hope this helps you learn something new today! 😊"
        )

    elif tone == "bold":
        caption += (
            "\n\n"
            "Start experimenting. Learn from results. Keep improving. 🚀"
        )

    return caption


def create_cta(goal):
    """
    Generates a call-to-action according to the selected goal.
    """

    if goal == "education":
        return "Save this post for future reference and share it with someone who may find it useful."

    if goal == "engagement":
        return "What do you think? Share your opinion in the comments."

    if goal == "brand awareness":
        return "Follow for more practical insights and updates."

    if goal == "lead generation":
        return "Interested in learning more? Send us a message to continue the discussion."

    return "Like, share, and follow for more useful content."


def create_hashtags(niche, goal):
    """
    Creates basic hashtags from the niche and goal.

    This is a rule-based hashtag generator, not a live trending
    hashtag recommendation system.
    """

    niche_words = niche.lower().replace("-", " ").split()

    hashtags = []

    for word in niche_words:
        cleaned_word = "".join(
            character for character in word
            if character.isalnum()
        )

        if cleaned_word:
            hashtags.append(f"#{cleaned_word}")

    common_hashtags = [
        "#ContentCreation",
        "#DigitalMarketing",
        "#Learning",
        "#Innovation"
    ]

    goal_hashtags = {
        "education": [
            "#Education",
            "#LearnSomethingNew",
            "#KnowledgeSharing"
        ],
        "engagement": [
            "#Community",
            "#Discussion",
            "#YourOpinion"
        ],
        "brand awareness": [
            "#BrandBuilding",
            "#PersonalBranding",
            "#CreatorEconomy"
        ],
        "lead generation": [
            "#BusinessGrowth",
            "#MarketingStrategy",
            "#GrowthStrategy"
        ]
    }

    hashtags.extend(goal_hashtags.get(goal, common_hashtags))

    # Remove duplicates while maintaining order.
    unique_hashtags = []

    for hashtag in hashtags:
        if hashtag not in unique_hashtags:
            unique_hashtags.append(hashtag)

    return unique_hashtags[:12]


def generate_content(niche, goal, post_idea, tone):
    """
    Generates the complete content package.
    """

    hook = create_hook(
        niche=niche,
        goal=goal,
        post_idea=post_idea,
        tone=tone
    )

    caption = create_caption(
        niche=niche,
        goal=goal,
        post_idea=post_idea,
        tone=tone
    )

    cta = create_cta(goal)

    hashtags = create_hashtags(
        niche=niche,
        goal=goal
    )

    return {
        "niche": niche,
        "goal": goal,
        "post_idea": post_idea,
        "tone": tone,
        "hook": hook,
        "caption": caption,
        "cta": cta,
        "hashtags": hashtags,
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        )
    }


def print_content_result(content):
    """
    Displays the generated content in a readable format.
    """

    print("\n" + "=" * 70)
    print("NICHELYT - GENERATED CONTENT")
    print("=" * 70)

    print(f"\nNiche       : {content['niche']}")
    print(f"Goal        : {content['goal']}")
    print(f"Post idea   : {content['post_idea'] or 'Not provided'}")
    print(f"Tone        : {content['tone']}")

    print("\n" + "-" * 70)
    print("CONTENT HOOK")
    print("-" * 70)
    print(content["hook"])

    print("\n" + "-" * 70)
    print("CAPTION")
    print("-" * 70)
    print(content["caption"])

    print("\n" + "-" * 70)
    print("CALL-TO-ACTION")
    print("-" * 70)
    print(content["cta"])

    print("\n" + "-" * 70)
    print("HASHTAGS")
    print("-" * 70)
    print(" ".join(content["hashtags"]))

    print("\n" + "-" * 70)
    print("LIMITATIONS")
    print("-" * 70)
    print("- This is a template-based content generator.")
    print("- It does not use a live trend database.")
    print("- Hashtags are rule-based, not live popularity recommendations.")
    print("- Generated content is not guaranteed to increase engagement.")
    print("- External LLM integration can be added in a future version.")

    print("\n" + "=" * 70)


def main():
    """
    Main function for user interaction.
    """

    print("\n" + "=" * 70)
    print("NICHELYT - CONTENT GENERATOR")
    print("=" * 70)

    niche = clean_input(
        input("\nEnter your niche: "),
        "Technology"
    )

    print("\nAvailable goals:")
    print("1. Education")
    print("2. Engagement")
    print("3. Brand Awareness")
    print("4. Lead Generation")

    goal_input = clean_input(
        input("\nEnter your goal: "),
        "education"
    ).lower()

    goal_mapping = {
        "1": "education",
        "2": "engagement",
        "3": "brand awareness",
        "4": "lead generation",
        "education": "education",
        "engagement": "engagement",
        "brand awareness": "brand awareness",
        "lead generation": "lead generation"
    }

    goal = goal_mapping.get(
        goal_input,
        "education"
    )

    post_idea = clean_input(
        input(
            "\nEnter your post idea "
            "(press Enter to skip): "
        )
    )

    print("\nAvailable tones:")
    print("1. Professional")
    print("2. Friendly")
    print("3. Educational")
    print("4. Bold")

    tone_input = clean_input(
        input("\nEnter your tone: "),
        "friendly"
    ).lower()

    tone_mapping = {
        "1": "professional",
        "2": "friendly",
        "3": "educational",
        "4": "bold",
        "professional": "professional",
        "friendly": "friendly",
        "educational": "educational",
        "bold": "bold"
    }

    tone = tone_mapping.get(
        tone_input,
        "friendly"
    )

    content = generate_content(
        niche=niche,
        goal=goal,
        post_idea=post_idea,
        tone=tone
    )

    print_content_result(content)


if __name__ == "__main__":
    main()