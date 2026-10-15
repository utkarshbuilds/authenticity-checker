
"""Analyze text using simple, explainable language signals."""

import re


# These words and phrases can indicate sensational wording.
# They do NOT prove that a claim is false.
SENSATIONAL_PHRASES = (
    "shocking",
    "secret",
    "they don't want you to know",
    "breaking",
    "urgent",
    "guaranteed",
    "100% true",
)


def analyze_text(text: str) -> dict:
    """
    Extract basic text features and possible sensational-language signals.

    Args:
        text: Caption, claim, or other text to analyze.

    Returns:
        A dictionary containing normalized text, tokens, counts,
        and detected sensational phrases.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    # Convert to lowercase and remove extra spaces.
    normalized_text = " ".join(text.lower().split())

    # Extract words and numbers. For example, "Breaking news!" becomes
    # ["breaking", "news"].
    tokens = re.findall(r"\b\w+\b", normalized_text)

    # A set keeps only unique tokens.
    unique_tokens = set(tokens)

    # Find phrases that appear in the normalized text.
    detected_phrases = [
        phrase
        for phrase in SENSATIONAL_PHRASES
        if phrase in normalized_text
    ]

    return {
        "original_text": text,
        "normalized_text": normalized_text,
        "tokens": tokens,
        "word_count": len(tokens),
        "unique_word_count": len(unique_tokens),
        "repeated_word_count": len(tokens) - len(unique_tokens),
        "text_length": len(text),
        "detected_sensational_phrases": detected_phrases,
        "has_sensational_language": bool(detected_phrases),
    }
