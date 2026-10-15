
"""LRU (Least Recently Used) cache using a dictionary and doubly linked list."""


class _Node:
    """One item in the doubly linked list."""

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """A fixed-capacity cache that evicts the least recently used item."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than zero.")

        self.capacity = capacity
        self.items = {}  # key -> node, for average O(1) lookup

        # Dummy ends make adding and removing nodes simpler.
        self.head = _Node()
        self.tail = _Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Remove a node from its current position."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_most_recent(self, node: _Node) -> None:
        """Insert a node right before the tail (most recently used)."""
        last = self.tail.prev
        last.next = node
        node.prev = last
        node.next = self.tail
        self.tail.prev = node

    def get(self, key):
        """Return a cached value, or None if the key is missing."""
        node = self.items.get(key)

        if node is None:
            return None

        # Using an item makes it the most recently used.
        self._remove(node)
        self._add_to_most_recent(node)
        return node.value

    def put(self, key, value) -> None:
        """Insert or update an item, evicting the least recent if needed."""
        node = self.items.get(key)

        if node is not None:
            # Update the existing value and refresh its recency.
            node.value = value
            self._remove(node)
            self._add_to_most_recent(node)
            return

        node = _Node(key, value)
        self.items[key] = node
        self._add_to_most_recent(node)

        if len(self.items) > self.capacity:
            # The node after head is the least recently used item.
            least_recent = self.head.next
            self._remove(least_recent)
            del self.items[least_recent.key]

    def __len__(self) -> int:
        """Return the number of cached items."""
        return len(self.items)
