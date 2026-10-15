
"""BK-tree for finding nearby fixed-width binary hashes using Hamming distance."""


def hamming_distance(first: str, second: str) -> int:
    """Count differing bit positions in two equal-length binary strings."""
    if len(first) != len(second):
        raise ValueError("Hashes must have equal lengths.")

    if any(bit not in "01" for bit in first + second):
        raise ValueError("Hashes must contain only '0' and '1'.")

    return sum(a != b for a, b in zip(first, second))


class _Node:
    """A BK-tree node containing a hash and children keyed by distance."""

    def __init__(self, value: str):
        self.value = value
        self.children = {}


class BKTree:
    """Store binary hashes and find those within a Hamming-distance limit."""

    def __init__(self):
        self.root = None
        self.size = 0

    def add(self, value: str) -> None:
        """Insert a hash, ignoring duplicates."""
        # Validate the value before inserting it.
        hamming_distance(value, value)

        if self.root is None:
            self.root = _Node(value)
            self.size = 1
            return

        current = self.root

        while True:
            distance = hamming_distance(value, current.value)

            if distance == 0:
                return  # This hash is already stored.

            if distance not in current.children:
                current.children[distance] = _Node(value)
                self.size += 1
                return

            current = current.children[distance]

    def search(self, query: str, max_distance: int) -> list[tuple[str, int]]:
        """Return (stored_hash, distance) pairs within the given limit."""
        hamming_distance(query, query)

        if max_distance < 0:
            raise ValueError("Maximum distance cannot be negative.")

        if self.root is None:
            return []

        matches = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            distance = hamming_distance(query, node.value)

            if distance <= max_distance:
                matches.append((node.value, distance))

            # By the triangle inequality, only these child distances can
            # contain a match within max_distance of the query.
            lower = distance - max_distance
            upper = distance + max_distance

            for child_distance, child in node.children.items():
                if lower <= child_distance <= upper:
                    stack.append(child)

        # Return the closest matches first, with a stable tie-break.
        matches.sort(key=lambda match: (match[1], match[0]))
        return matches

    def __len__(self) -> int:
        """Return the number of unique hashes stored."""
        return self.size
