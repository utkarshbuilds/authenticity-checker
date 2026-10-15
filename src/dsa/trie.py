"""
Trie data structure for efficient word and prefix searching.

Each node stores:
- children: dictionary mapping a character to the next TrieNode
- is_end_of_word: whether a complete word ends at this node
"""


class TrieNode:
    """One node in the Trie."""

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    """Store words and search for complete words or prefixes."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word into the Trie."""
        if not isinstance(word, str):
            raise TypeError("word must be a string")

        word = word.strip().lower()

        if not word:
            raise ValueError("word must not be empty")

        current = self.root

        for character in word:
            # Create a node only if this character is not present.
            if character not in current.children:
                current.children[character] = TrieNode()

            current = current.children[character]

        # Mark the last node as the end of a complete word.
        current.is_end_of_word = True

    def search(self, word: str) -> bool:
        """Return True only if the complete word exists."""
        if not isinstance(word, str):
            raise TypeError("word must be a string")

        word = word.strip().lower()

        if not word:
            return False

        current = self.root

        for character in word:
            if character not in current.children:
                return False

            current = current.children[character]

        return current.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Return True if at least one word starts with the prefix."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")

        prefix = prefix.strip().lower()

        if not prefix:
            return True

        current = self.root

        for character in prefix:
            if character not in current.children:
                return False

            current = current.children[character]

        # The path exists, even if it isn't a complete word.
        return True
