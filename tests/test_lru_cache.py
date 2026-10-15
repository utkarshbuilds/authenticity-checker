
import pytest

from src.dsa.lru_cache import LRUCache


def test_get_returns_value_for_existing_key():
    cache = LRUCache(2)
    cache.put("image1", "AUTHENTIC")

    assert cache.get("image1") == "AUTHENTIC"


def test_get_returns_none_for_missing_key():
    cache = LRUCache(2)

    assert cache.get("missing") is None


def test_cache_evicts_least_recently_used_item():
    cache = LRUCache(2)
    cache.put("A", 1)
    cache.put("B", 2)

    # Access A, making B the least recently used.
    assert cache.get("A") == 1

    cache.put("C", 3)

    assert cache.get("B") is None
    assert cache.get("A") == 1
    assert cache.get("C") == 3


def test_put_updates_existing_key():
    cache = LRUCache(2)
    cache.put("image1", "AUTHENTIC")
    cache.put("image1", "MANIPULATED")

    assert cache.get("image1") == "MANIPULATED"
    assert len(cache) == 1


def test_cache_never_exceeds_capacity():
    cache = LRUCache(2)
    cache.put("A", 1)
    cache.put("B", 2)
    cache.put("C", 3)

    assert len(cache) == 2


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        LRUCache(0)

    with pytest.raises(ValueError):
        LRUCache(-1)


def test_capacity_one():
    cache = LRUCache(1)
    cache.put("A", 1)
    cache.put("B", 2)

    assert cache.get("A") is None
    assert cache.get("B") == 2
