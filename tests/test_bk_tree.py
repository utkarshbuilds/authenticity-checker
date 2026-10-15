
import pytest

from src.dsa.bk_tree import BKTree, hamming_distance


def test_hamming_distance_counts_different_bits():
    assert hamming_distance("1010", "1001") == 2


def test_hamming_distance_rejects_different_lengths():
    with pytest.raises(ValueError):
        hamming_distance("1010", "101")


def test_add_and_find_exact_hash():
    tree = BKTree()
    tree.add("10101010")

    assert tree.search("10101010", 0) == [("10101010", 0)]


def test_search_finds_nearby_hashes():
    tree = BKTree()
    tree.add("0000")
    tree.add("0001")
    tree.add("0011")
    tree.add("1111")

    results = tree.search("0000", max_distance=1)

    assert results == [("0000", 0), ("0001", 1)]


def test_search_respects_maximum_distance():
    tree = BKTree()
    tree.add("0000")
    tree.add("1111")

    assert tree.search("0000", max_distance=2) == [("0000", 0)]


def test_duplicate_hash_is_not_added_twice():
    tree = BKTree()
    tree.add("1010")
    tree.add("1010")

    assert len(tree) == 1


def test_empty_tree_returns_no_matches():
    tree = BKTree()

    assert tree.search("1010", max_distance=2) == []


def test_negative_max_distance_is_rejected():
    tree = BKTree()

    with pytest.raises(ValueError):
        tree.search("1010", max_distance=-1)


def test_hashes_must_be_binary():
    with pytest.raises(ValueError):
        hamming_distance("10x0", "1000")
