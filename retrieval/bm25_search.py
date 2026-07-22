import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


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
# TOKENIZATION
# =====================================================

def tokenize(text: str):
    """
    Convert text into normalized tokens.
    """

    text = text.lower()

    tokens = re.findall(
        r"\b\w+\b",
        text
    )

    return tokens


# =====================================================
# BM25 SEARCH ENGINE
# =====================================================

class BM25Search:

    def __init__(self, chunks):

        self.chunks = chunks

        # Searchable text for every chunk
        self.documents = [

            " ".join(
                [
                    chunk["title"],
                    chunk["content"],
                    chunk["category"],
                ]
            )

            for chunk in chunks

        ]

        # Tokenized documents
        self.tokenized_documents = [

            tokenize(document)

            for document in self.documents

        ]

        print(
            "Building BM25 index..."
        )

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

        print(
            f"Indexed {len(self.chunks)} chunks."
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

        query_tokens = tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        # Sort by score descending
        ranked_indices = sorted(

            range(
                len(scores)
            ),

            key=lambda index: scores[index],

            reverse=True

        )

        results = []

        for chunk_index in ranked_indices:

            chunk = self.chunks[
                chunk_index
            ]

            # -----------------------------------------
            # Language Filter
            # -----------------------------------------

            if language is not None:

                if chunk["language"] != language:

                    continue

            # -----------------------------------------
            # Category Filter
            # -----------------------------------------

            if category is not None:

                if (
                    chunk["category"].lower()
                    != category.lower()
                ):

                    continue

            # -----------------------------------------
            # Ignore zero-score results
            # -----------------------------------------

            if scores[chunk_index] <= 0:

                continue

            result = chunk.copy()

            result["score"] = float(
                scores[chunk_index]
            )

            results.append(
                result
            )

            if len(results) >= top_k:

                break

        return results


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

    return BM25Search(
        chunks
    )


# =====================================================
# TEST SEARCH
# =====================================================

if __name__ == "__main__":

    search_engine = (
        load_search_engine()
    )

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
            f"Score: {result['score']:.4f}"
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