import json
import sys
import time
from pathlib import Path


# =====================================================
# PROJECT PATH
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================
# IMPORT RETRIEVERS
# =====================================================

from retrieval.keyword_search import (
    load_search_engine as load_keyword_search
)

from retrieval.bm25_search import (
    load_search_engine as load_bm25_search
)

from retrieval.hybrid_search import (
    load_search_engine as load_hybrid_search
)


# =====================================================
# PATHS
# =====================================================

EVALUATION_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_queries.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_results.json"
)


# =====================================================
# METRICS
# =====================================================

def hit_rate_at_k(retrieved_ids, relevant_ids):

    return int(
        any(
            doc_id in relevant_ids
            for doc_id in retrieved_ids
        )
    )


def recall_at_k(retrieved_ids, relevant_ids):

    if not relevant_ids:
        return 0.0

    retrieved_relevant = len(
        set(retrieved_ids)
        &
        set(relevant_ids)
    )

    return (
        retrieved_relevant
        /
        len(relevant_ids)
    )


def reciprocal_rank(retrieved_ids, relevant_ids):

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if doc_id in relevant_ids:

            return 1.0 / rank

    return 0.0


# =====================================================
# EVALUATE ONE RETRIEVER
# =====================================================

def evaluate_retriever(
    search_engine,
    queries,
    top_k=5,
):

    hits = []
    recalls = []
    reciprocal_ranks = []

    start_time = time.time()

    for index, item in enumerate(
        queries,
        start=1
    ):

        query = item["query"]

        relevant_ids = set(
            item["relevant_document_ids"]
        )

        results = search_engine.search(
            query=query,
            top_k=top_k,
            language=item.get("language"),
        )

        retrieved_ids = [
            result["document_id"]
            for result in results
        ]

        hits.append(
            hit_rate_at_k(
                retrieved_ids,
                relevant_ids
            )
        )

        recalls.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved_ids,
                relevant_ids
            )
        )

        if index % 25 == 0:

            elapsed = time.time() - start_time

            print(
                f"Evaluated "
                f"{index}/"
                f"{len(queries)} "
                f"queries "
                f"({elapsed:.2f}s)",
                flush=True
            )

    total_time = time.time() - start_time

    return {

        "hit_rate": (
            sum(hits)
            /
            len(hits)
        ),

        "recall": (
            sum(recalls)
            /
            len(recalls)
        ),

        "mrr": (
            sum(reciprocal_ranks)
            /
            len(reciprocal_ranks)
        ),

        "total_queries": len(queries),

        "total_time_seconds": round(
            total_time,
            2
        ),

        "average_latency_seconds": round(
            total_time / len(queries),
            4
        ),

    }


# =====================================================
# PRINT RESULTS
# =====================================================

def print_results(name, results):

    print(f"\n{name}:")

    print(
        f"Hit Rate@5: "
        f"{results['hit_rate']:.4f}"
    )

    print(
        f"Recall@5: "
        f"{results['recall']:.4f}"
    )

    print(
        f"MRR@5: "
        f"{results['mrr']:.4f}"
    )

    print(
        f"Average Latency: "
        f"{results['average_latency_seconds']:.4f}s"
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "RETRIEVAL EVALUATION",
        flush=True
    )

    print("=" * 70)


    # =================================================
    # LOAD QUERIES
    # =================================================

    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        queries = json.load(file)


    print(
        f"\nEvaluation queries: "
        f"{len(queries)}",
        flush=True
    )


    all_results = {}


    # =================================================
    # KEYWORD SEARCH
    # =================================================

    print(
        "\nEvaluating Keyword Search...",
        flush=True
    )

    keyword_start = time.time()

    keyword_search = load_keyword_search()

    print(
        f"Keyword Search loaded in "
        f"{time.time() - keyword_start:.2f}s",
        flush=True
    )

    all_results["Keyword Search"] = evaluate_retriever(
        search_engine=keyword_search,
        queries=queries,
        top_k=5,
    )


    # =================================================
    # BM25 SEARCH
    # =================================================

    print(
        "\nEvaluating BM25...",
        flush=True
    )

    bm25_start = time.time()

    bm25_search = load_bm25_search()

    print(
        f"BM25 loaded in "
        f"{time.time() - bm25_start:.2f}s",
        flush=True
    )

    all_results["BM25"] = evaluate_retriever(
        search_engine=bm25_search,
        queries=queries,
        top_k=5,
    )


    # =================================================
    # HYBRID SEARCH
    # =================================================

    print(
        "\nEvaluating Hybrid Search...",
        flush=True
    )

    hybrid_start = time.time()

    hybrid_search = load_hybrid_search()

    print(
        f"Hybrid Search loaded in "
        f"{time.time() - hybrid_start:.2f}s",
        flush=True
    )

    all_results["Hybrid Search"] = evaluate_retriever(
        search_engine=hybrid_search,
        queries=queries,
        top_k=5,
    )


    # =================================================
    # FINAL RESULTS
    # =================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 70
    )


    for name, results in all_results.items():

        print_results(
            name,
            results
        )


    # =================================================
    # FIND WINNER
    # =================================================

    winner = max(
        all_results,
        key=lambda name:
        all_results[name]["mrr"]
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"WINNER BASED ON MRR@5: "
        f"{winner}"
    )

    print(
        "=" * 70
    )


    # =================================================
    # SAVE RESULTS
    # =================================================

    final_output = {

        "evaluation_queries": len(queries),

        "results": all_results,

        "winner": winner,

    }


    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_output,
            file,
            indent=2,
            ensure_ascii=False
        )


    print(
        f"\nResults saved to:\n"
        f"{RESULTS_PATH}"
    )