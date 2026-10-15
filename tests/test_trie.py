import pytest

from src.dsa.trie import Trie


def test_search_inserted_word():
    trie = Trie()
    trie.insert("fake news")

    assert trie.search("fake news") is True


def test_search_is_case_insensitive():
    trie = Trie()
    trie.insert("Synthetic Image")

    assert trie.search("synthetic image") is True
    assert trie.search("SYNTHETIC IMAGE") is True


def test_search_rejects_prefix_as_complete_word():
    trie = Trie()
    trie.insert("fake news")

    assert trie.search("fake") is False


def test_starts_with_existing_prefix():
    trie = Trie()
    trie.insert("fake news")

    assert trie.starts_with("fake") is True


def test_starts_with_missing_prefix():
    trie = Trie()
    trie.insert("authentic image")

    assert trie.starts_with("synthetic") is False


def test_empty_word_is_rejected():
    trie = Trie()

    with pytest.raises(ValueError):
        trie.insert("   ")


def test_empty_search_returns_false():
    trie = Trie()

    assert trie.search("") is False


def test_duplicate_insertion_is_safe():
    trie = Trie()
    trie.insert("fake news")
    trie.insert("fake news")

    assert trie.search("fake news") is True