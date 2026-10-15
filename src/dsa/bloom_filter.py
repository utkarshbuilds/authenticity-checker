
"""A simple Bloom Filter for memory-efficient membership checks."""

import hashlib
import math


class BloomFilter:
    """A probabilistic set: no false negatives, but possible false positives."""

    def __init__(self, capacity: int, false_positive_rate: float = 0.01):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than zero.")

        if not 0 < false_positive_rate < 1:
            raise ValueError("False positive rate must be between 0 and 1.")

        # Estimate the number of bits needed for the requested error rate.
        self.size = max(
            1,
            math.ceil(
                -capacity * math.log(false_positive_rate) / (math.log(2) ** 2)
            ),
        )

        # Estimate the best number of hash positions per item.
        self.hash_count = max(
            1, round((self.size / capacity) * math.log(2))
        )

        # Each element represents one bit: 0 = unset, 1 = set.
        self.bits = bytearray(self.size)
        self.count = 0

    def _positions(self, item: str):
        """Generate deterministic bit positions for an item."""
        if not isinstance(item, str):
            raise TypeError("BloomFilter items must be strings.")

        # SHA-256 gives us a repeatable digest for the same input.
        digest = hashlib.sha256(item.encode("utf-8")).digest()

        # Use digest chunks to derive multiple positions.
        for i in range(self.hash_count):
            start = (i * 4) % (len(digest) - 3)
            number = int.from_bytes(digest[start:start + 4], "big")
            yield number % self.size

    def add(self, item: str) -> None:
        """Add an item by setting all its bit positions."""
        for position in self._positions(item):
            self.bits[position] = 1
        self.count += 1

    def might_contain(self, item: str) -> bool:
        """Return False if definitely absent, otherwise possibly present."""
        return all(self.bits[position] for position in self._positions(item))

    def __contains__(self, item: str) -> bool:
        """Allow syntax such as: 'image-123' in bloom_filter."""
        return self.might_contain(item)

