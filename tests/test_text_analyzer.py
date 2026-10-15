
import pytest

from src.analyzers.text_analyzer import analyze_text


def test_analyze_normal_text():
    result = analyze_text("A real photograph shows a busy street.")

    assert result["normalized_text"] == (
        "a real photograph shows a busy street."
    )
    assert result["tokens"] == [
        "a", "real", "photograph", "shows", "a", "busy", "street"
    ]
    assert result["word_count"] == 7
    assert result["unique_word_count"] == 6
    assert result["repeated_word_count"] == 1
    assert result["has_sensational_language"] is False


def test_normalizes_case_and_extra_spaces():
    result = analyze_text("  BREAKING   News  ")

    assert result["normalized_text"] == "breaking news"
    assert result["tokens"] == ["breaking", "news"]


def test_detects_sensational_phrase():
    result = analyze_text("This is shocking news!")

    assert result["detected_sensational_phrases"] == ["shocking"]
    assert result["has_sensational_language"] is True


def test_detects_multiword_phrase():
    result = analyze_text("They don't want you to know this fact.")

    assert "they don't want you to know" in (
        result["detected_sensational_phrases"]
    )


def test_empty_text():
    result = analyze_text("")

    assert result["tokens"] == []
    assert result["word_count"] == 0
    assert result["unique_word_count"] == 0
    assert result["repeated_word_count"] == 0
    assert result["has_sensational_language"] is False


def test_text_containing_only_spaces():
    result = analyze_text("   \n\t  ")

    assert result["normalized_text"] == ""
    assert result["word_count"] == 0


def test_repeated_words_are_counted():
    result = analyze_text("alert alert alert")

    assert result["word_count"] == 3
    assert result["unique_word_count"] == 1
    assert result["repeated_word_count"] == 2


def test_non_string_input_raises_error():
    with pytest.raises(TypeError, match="must be a string"):
        analyze_text(None)
