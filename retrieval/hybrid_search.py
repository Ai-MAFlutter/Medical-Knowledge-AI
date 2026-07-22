import json
from pathlib import Path

from retrieval.bm25_search import load_search_engine as load_bm25_search
from retrieval.vector_search import load_search_engine as load_vector_search


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medlineplus_chunks.json"
)


class HybridSearch:

    def __init__(
        self,
        bm25_engine,
        vector_engine,
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

        # -----------------------------------------
        # BM25 Results
        # -----------------------------------------

        bm25_results = (

            self.bm25_engine.search(

                query=query,

                top_k=top_k * 2,

                language=language,

                category=category,

            )

        )


        # -----------------------------------------
        # Vector Results
        # -----------------------------------------

        vector_results = (

            self.vector_engine.search(

                query=query,

                top_k=top_k * 2,

                language=language,

                category=category,

            )

        )


        # -----------------------------------------
        # Reciprocal Rank Fusion
        # -----------------------------------------

        fused_scores = {}

        documents = {}


        for rank, result in enumerate(

            bm25_results,

            start=1

        ):

            doc_id = result["chunk_id"]


            fused_scores[doc_id] = (

                fused_scores.get(

                    doc_id,

                    0

                )

                +

                1

                /

                (

                    self.rrf_k

                    +

                    rank

                )

            )


            documents[doc_id] = result


        for rank, result in enumerate(

            vector_results,

            start=1

        ):

            doc_id = result["chunk_id"]


            fused_scores[doc_id] = (

                fused_scores.get(

                    doc_id,

                    0

                )

                +

                1

                /

                (

                    self.rrf_k

                    +

                    rank

                )

            )


            documents[doc_id] = result


        # -----------------------------------------
        # Sort Fused Results
        # -----------------------------------------

        ranked_ids = sorted(

            fused_scores,

            key=lambda doc_id:

            fused_scores[doc_id],

            reverse=True,

        )


        final_results = []


        for doc_id in ranked_ids[

            :top_k

        ]:

            result = documents[doc_id].copy()


            result["hybrid_score"] = (

                fused_scores[doc_id]

            )


            final_results.append(

                result

            )


        return final_results


def load_search_engine():

    print(

        "Loading BM25 engine..."

    )

    bm25_engine = (

        load_bm25_search()

    )


    print(

        "Loading Vector engine..."

    )

    vector_engine = (

        load_vector_search()

    )


    return HybridSearch(

        bm25_engine=bm25_engine,

        vector_engine=vector_engine,

    )


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

            f"Hybrid Score: "

            f"{result['hybrid_score']:.6f}"

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