import json
import re
from pathlib import Path
from collections import Counter


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medlineplus_chunks.json"
)


# =====================================================
# SEARCH ENGINE
# =====================================================

class KeywordSearch:

    def __init__(self, chunks):

        self.chunks = chunks

        self.inverted_index = {}

        self._build_index()


    # =================================================
    # TOKENIZATION
    # =================================================

    @staticmethod
    def tokenize(text):

        text = text.lower()

        tokens = re.findall(
            r"\b\w+\b",
            text
        )

        return tokens


    # =================================================
    # BUILD INVERTED INDEX
    # =================================================

    def _build_index(self):

        print(
            "Building keyword index..."
        )

        for chunk_index, chunk in enumerate(
            self.chunks
        ):

            searchable_text = " ".join(
                [
                    chunk["title"],
                    chunk["content"],
                    chunk["category"],
                ]
            )

            tokens = self.tokenize(
                searchable_text
            )

            token_counts = Counter(
                tokens
            )

            for token, count in token_counts.items():

                if token not in self.inverted_index:

                    self.inverted_index[token] = {}

                self.inverted_index[token][
                    chunk_index
                ] = count

        print(
            f"Indexed {len(self.chunks)} chunks."
        )

        print(
            f"Unique terms: {len(self.inverted_index)}"
        )


    # =================================================
    # SEARCH
    # =================================================

    def search(
        self,
        query,
        top_k=5,
        language=None,
        category=None,
    ):

        query_tokens = self.tokenize(
            query
        )

        scores = Counter()

        for token in query_tokens:

            if token not in self.inverted_index:

                continue

            matching_chunks = (
                self.inverted_index[token]
            )

            for chunk_index, frequency in (
                matching_chunks.items()
            ):

                chunk = self.chunks[
                    chunk_index
                ]

                # -----------------------------
                # Language Filter
                # -----------------------------

                if language is not None:

                    if chunk["language"] != language:

                        continue

                # -----------------------------
                # Category Filter
                # -----------------------------

                if category is not None:

                    if (
                        chunk["category"].lower()
                        != category.lower()
                    ):

                        continue

                # -----------------------------
                # Scoring
                # -----------------------------

                score = frequency

                # Title Boost
                title_tokens = self.tokenize(
                    chunk["title"]
                )

                if token in title_tokens:

                    score *= 3

                # Category Boost
                category_tokens = self.tokenize(
                    chunk["category"]
                )

                if token in category_tokens:

                    score *= 2

                scores[
                    chunk_index
                ] += score

        # -----------------------------------------
        # Get Top Results
        # -----------------------------------------

        top_results = []

        for chunk_index, score in scores.most_common(
            top_k
        ):

            result = self.chunks[
                chunk_index
            ].copy()

            result["score"] = score

            top_results.append(
                result
            )

        return top_results


# =====================================================
# LOAD SEARCH ENGINE
# =====================================================

def load_search_engine():

    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(
            file
        )

    return KeywordSearch(
        chunks
    )


# =====================================================
# TEST SEARCH
# =====================================================

if __name__ == "__main__":

    search_engine = load_search_engine()

    query = (
        "What is an A1C test?"
    )

    results = search_engine.search(
        query=query,
        top_k=5,
        language="en",
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"QUERY: {query}"
    )

    print(
        "=" * 60
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {index}"
        )

        print(
            f"Score: {result['score']}"
        )

        print(
            f"Title: {result['title']}"
        )

        print(
            f"Category: {result['category']}"
        )

        print(
            f"Language: {result['language']}"
        )

        print(
            f"Content: {result['content'][:500]}..."
        )

        print(
            f"Source: {result['source_url']}"
        )