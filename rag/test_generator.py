from rag.retrieval_pipeline import MedicalRetrievalPipeline
from rag.generator import generate_answer


def main():

    print("=" * 70)
    print("MEDICAL KNOWLEDGE AI - RAG GENERATION TEST")
    print("=" * 70)

    # ============================================================
    # USER INPUT
    # ============================================================

    query = "What is an A1C test?"
    user_level = "beginner"

    print("\nUser Query:")
    print(query)

    print("\nUser Level:")
    print(user_level)

    # ============================================================
    # INITIALIZE RETRIEVAL PIPELINE
    # ============================================================

    print("\nInitializing retrieval pipeline...")

    pipeline = MedicalRetrievalPipeline()

    # ============================================================
    # RETRIEVE DOCUMENTS
    # ============================================================

    print("\nRetrieving medical documents...")

    retrieval_result = pipeline.retrieve(
        query=query,
        user_type="general",
        explanation_level=user_level,
        candidate_k=10,
        final_k=5
    )

    # ============================================================
    # VALIDATE RETRIEVAL RESULT
    # ============================================================

    if not isinstance(retrieval_result, dict):

        raise TypeError(
            "Expected retrieval pipeline to return a dictionary."
        )

    # ============================================================
    # EXTRACT DATA
    # ============================================================

    rewritten_query = retrieval_result.get(
        "rewritten_query",
        {}
    )

    candidates = retrieval_result.get(
        "candidates",
        []
    )

    documents = retrieval_result.get(
        "documents",
        []
    )

    # ============================================================
    # DISPLAY RETRIEVAL SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("RETRIEVAL SUMMARY")
    print("=" * 70)

    print(
        f"\nOriginal Query:\n"
        f"{query}"
    )

    if isinstance(rewritten_query, dict):

        print(
            f"\nRewritten Query:\n"
            f"{rewritten_query.get('rewritten_query', '')}"
        )

        print(
            f"\nSearch Intent:\n"
            f"{rewritten_query.get('search_intent', '')}"
        )

        print(
            f"\nMedical Entities:\n"
            f"{rewritten_query.get('medical_entities', [])}"
        )

    print(
        f"\nCandidate Documents Retrieved: "
        f"{len(candidates)}"
    )

    print(
        f"Final Documents Selected: "
        f"{len(documents)}"
    )

    # ============================================================
    # SAFETY CHECK
    # ============================================================

    if not documents:

        print(
            "\nNo relevant medical documents were found."
        )

        return

    # ============================================================
    # DISPLAY SOURCES
    # ============================================================

    print("\n" + "=" * 70)
    print("SELECTED MEDICAL SOURCES")
    print("=" * 70)

    for index, document in enumerate(
        documents,
        start=1
    ):

        print(f"\nSOURCE {index}")

        print(
            f"Title: "
            f"{document.get('title', '')}"
        )

        print(
            f"Category: "
            f"{document.get('category', '')}"
        )

        print(
            f"Rerank Score: "
            f"{document.get('rerank_score', 0):.4f}"
        )

        print(
            f"Source URL: "
            f"{document.get('source_url', '')}"
        )

    # ============================================================
    # GENERATE ANSWER
    # ============================================================

    print("\n" + "=" * 70)
    print("GENERATING ANSWER")
    print("=" * 70)

    answer = generate_answer(
        query=query,
        documents=documents,
        user_level=user_level
    )

    # ============================================================
    # DISPLAY ANSWER
    # ============================================================

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    print(answer)

    # ============================================================
    # COMPLETION
    # ============================================================

    print("\n" + "=" * 70)
    print("RAG GENERATION TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()