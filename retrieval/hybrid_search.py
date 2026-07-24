from retrieval.bm25_search import load_search_engine as load_bm25_search


class HybridSearch:

    def __init__(
        self,
        bm25_engine,
        vector_engine=None,
        rrf_k=60,
    ):

        self.bm25_engine = bm25_engine
        self.vector_engine = vector_engine
        self.rrf_k = rrf_k

    def search(
        self,
        query,
        top_k=5,
        language=None,
        category=None,
    ):

        # =================================================
        # BM25 RESULTS
        # =================================================

        bm25_results = self.bm25_engine.search(
            query=query,
            top_k=top_k * 2,
            language=language,
            category=category,
        )

        # =================================================
        # FALLBACK MODE
        # =================================================

        if self.vector_engine is None:

            print(
                "Vector search unavailable. "
                "Using BM25 fallback."
            )

            return bm25_results[:top_k]

        # =================================================
        # VECTOR RESULTS
        # =================================================

        try:

            vector_results = self.vector_engine.search(
                query=query,
                top_k=top_k * 2,
                language=language,
                category=category,
            )

        except Exception as error:

            print(
                f"Vector search failed: {error}"
            )

            print(
                "Using BM25 fallback."
            )

            return bm25_results[:top_k]

        # =================================================
        # RECIPROCAL RANK FUSION
        # =================================================

        fused_scores = {}

        documents = {}

        # -----------------------------
        # BM25
        # -----------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            doc_id = result["chunk_id"]

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0)
                +
                1 / (self.rrf_k + rank)
            )

            documents[doc_id] = result

        # -----------------------------
        # VECTOR
        # -----------------------------

        for rank, result in enumerate(
            vector_results,
            start=1
        ):

            doc_id = result["chunk_id"]

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0)
                +
                1 / (self.rrf_k + rank)
            )

            documents[doc_id] = result

        # =================================================
        # SORT RESULTS
        # =================================================

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        final_results = []

        for doc_id in ranked_ids[:top_k]:

            result = documents[doc_id].copy()

            result["hybrid_score"] = (
                fused_scores[doc_id]
            )

            final_results.append(result)

        return final_results


def load_search_engine():

    print(
        "Loading BM25 engine..."
    )

    bm25_engine = load_bm25_search()

    # =================================================
    # IMPORTANT
    # =================================================
    # Vector search is temporarily disabled because
    # SentenceTransformer crashes the Python process
    # on the current Windows/Python environment.
    #
    # The Hybrid Search remains functional through
    # BM25 fallback.
    # =================================================

    vector_engine = None

    print(
        "Vector search disabled safely."
    )

    print(
        "Hybrid Search will use BM25 fallback."
    )

    return HybridSearch(
        bm25_engine=bm25_engine,
        vector_engine=vector_engine,
    )


if __name__ == "__main__":

    search_engine = load_search_engine()

    query = "What is an A1C test?"

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
            f"Title: "
            f"{result['title']}"
        )

        print(
            f"Category: "
            f"{result['category']}"
        )

        print(
            f"Content: "
            f"{result['content'][:500]}..."
        )

        print(
            f"Source: "
            f"{result['source_url']}"
        )