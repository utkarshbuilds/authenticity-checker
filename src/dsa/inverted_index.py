"""
Inverted Index for searching reference documents by keyword.

Each word maps to a set of document IDs that contain that word.
"""

import re


class InvertedIndex:
    """Index documents so we can find them by their words."""

    def __init__(self):
        # Example: {"flood": {"claim_1", "claim_3"}}
        self.index = {}

        # Keep the original text for each document.
        self.documents = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Convert text into lowercase words."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return re.findall(r"\b\w+\b", text.lower())

    def add_document(self, document_id: str, text: str) -> None:
        """Store a document and add its words to the index."""
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        document_id = document_id.strip()

        # Remove the document's old index entries if it already exists.
        if document_id in self.documents:
            self.remove_document(document_id)

        self.documents[document_id] = text

        # A set avoids indexing the same word twice for one document.
        for word in set(self.tokenize(text)):
            if word not in self.index:
                self.index[word] = set()

            self.index[word].add(document_id)

    def search(self, word: str) -> set[str]:
        """Return document IDs containing the exact word."""
        if not isinstance(word, str):
            raise TypeError("word must be a string")

        tokens = self.tokenize(word.strip())

        # Search expects one word, not a phrase.
        if len(tokens) != 1:
            return set()

        return set(self.index.get(tokens[0], set()))

    def search_all(self, words: list[str]) -> set[str]:
        """Return documents containing every requested word."""
        if not isinstance(words, list):
            raise TypeError("words must be a list of strings")

        if not words:
            return set()

        # Start with documents containing the first word.
        result = self.search(words[0])

        # Keep only documents that contain each additional word.
        for word in words[1:]:
            result &= self.search(word)

        return result

    def remove_document(self, document_id: str) -> bool:
        """Remove a document and its word entries. Return success."""
        if document_id not in self.documents:
            return False

        text = self.documents.pop(document_id)

        for word in set(self.tokenize(text)):
            document_ids = self.index.get(word)

            if document_ids is not None:
                document_ids.discard(document_id)

                # Remove words that no longer occur in any document.
                if not document_ids:
                    del self.index[word]

        return True
