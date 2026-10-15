import pytest

from src.dsa.inverted_index import InvertedIndex


def test_search_finds_documents_containing_word():
    index = InvertedIndex()
    index.add_document("claim_1", "Flood reported in Delhi")
    index.add_document("claim_2", "Earthquake reported in Nepal")

    assert index.search("flood") == {"claim_1"}


def test_search_is_case_insensitive():
    index = InvertedIndex()
    index.add_document("claim_1", "Synthetic Image")

    assert index.search("SYNTHETIC") == {"claim_1"}


def test_repeated_word_does_not_duplicate_document():
    index = InvertedIndex()
    index.add_document("claim_1", "flood flood flood")

    assert index.search("flood") == {"claim_1"}


def test_search_all_returns_documents_with_every_word():
    index = InvertedIndex()
    index.add_document("claim_1", "flood in Delhi")
    index.add_document("claim_2", "flood in Nepal")
    index.add_document("claim_3", "earthquake in Delhi")

    assert index.search_all(["flood", "delhi"]) == {"claim_1"}


def test_search_missing_word_returns_empty_set():
    index = InvertedIndex()
    index.add_document("claim_1", "authentic photograph")

    assert index.search("synthetic") == set()


def test_readding_document_updates_old_words():
    index = InvertedIndex()
    index.add_document("claim_1", "flood in Delhi")
    index.add_document("claim_1", "earthquake in Nepal")

    assert index.search("flood") == set()
    assert index.search("earthquake") == {"claim_1"}


def test_remove_document():
    index = InvertedIndex()
    index.add_document("claim_1", "flood in Delhi")

    assert index.remove_document("claim_1") is True
    assert index.search("flood") == set()
    assert index.remove_document("claim_1") is False


def test_empty_document_id_is_rejected():
    index = InvertedIndex()

    with pytest.raises(ValueError):
        index.add_document("   ", "some text")


def test_non_string_text_is_rejected():
    index = InvertedIndex()

    with pytest.raises(TypeError):
        index.add_document("claim_1", 123)


def test_empty_search_all_returns_empty_set():
    index = InvertedIndex()

    assert index.search_all([]) == set()
