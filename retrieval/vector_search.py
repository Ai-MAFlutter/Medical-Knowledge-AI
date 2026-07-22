import json
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


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
# EMBEDDING MODEL
# =====================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# =====================================================
# VECTOR SEARCH ENGINE
# =====================================================

class VectorSearch:

    def __init__(self, chunks):

        self.chunks = chunks

        print(
            f"Loading embedding model: {MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Creating document embeddings..."
        )

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

        self.embeddings = (

            self.model.encode(

                self.documents,

                show_progress_bar=True,

                convert_to_numpy=True,

                normalize_embeddings=True,

            )

        )

        # -----------------------------------------
        # FAISS Index
        # -----------------------------------------

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            self.embeddings.astype(
                "float32"
            )
        )

        print(
            f"Indexed {len(self.chunks)} chunks."
        )

        print(
            f"Embedding dimension: {dimension}"
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

        query_embedding = (

            self.model.encode(

                [

                    query

                ],

                convert_to_numpy=True,

                normalize_embeddings=True,

            )

        )

        scores, indices = (

            self.index.search(

                query_embedding.astype(

                    "float32"

                ),

                top_k * 5,

            )

        )

        results = []

        for score, chunk_index in zip(

            scores[0],

            indices[0],

        ):

            if chunk_index == -1:

                continue

            chunk = self.chunks[
                chunk_index
            ]

            # -----------------------------------------
            # Language Filter
            # -----------------------------------------

            if language is not None:

                if (

                    chunk["language"]

                    != language

                ):

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

            result = chunk.copy()

            result["score"] = float(
                score
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

            f"Chunks file not found: "

            f"{CHUNKS_PATH}"

        )

    with open(

        CHUNKS_PATH,

        "r",

        encoding="utf-8"

    ) as file:

        chunks = json.load(
            file
        )

    return VectorSearch(
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

            f"Score: "

            f"{result['score']:.4f}"

        )

        print(

            f"Title: "

            f"{result['title']}"

        )

        print(

            f"Category: "

            f"{result['category']}"

        )

        print(

            f"Language: "

            f"{result['language']}"

        )

        print(

            f"Content: "

            f"{result['content'][:500]}..."

        )

        print(

            f"Source: "

            f"{result['source_url']}"

        )