
import pytest

from src.dsa.bloom_filter import BloomFilter


def test_added_item_might_be_present():
    bloom = BloomFilter(capacity=100)
    bloom.add("image-hash-123")

    assert bloom.might_contain("image-hash-123") is True


def test_added_item_supports_in_operator():
    bloom = BloomFilter(capacity=100)
    bloom.add("claim-001")

    assert "claim-001" in bloom


def test_new_filter_reports_item_absent():
    bloom = BloomFilter(capacity=100)

    assert bloom.might_contain("never-added") is False


def test_adding_duplicate_does_not_break_membership():
    bloom = BloomFilter(capacity=100)
    bloom.add("same-item")
    bloom.add("same-item")

    assert bloom.might_contain("same-item") is True


def test_invalid_capacity_raises_error():
    with pytest.raises(ValueError):
        BloomFilter(capacity=0)


def test_invalid_false_positive_rate_raises_error():
    with pytest.raises(ValueError):
        BloomFilter(capacity=100, false_positive_rate=1.0)


def test_non_string_item_raises_error():
    bloom = BloomFilter(capacity=100)

    with pytest.raises(TypeError):
        bloom.add(123)


def test_invalid_query_type_raises_error():
    bloom = BloomFilter(capacity=100)

    with pytest.raises(TypeError):
        bloom.might_contain(123)
