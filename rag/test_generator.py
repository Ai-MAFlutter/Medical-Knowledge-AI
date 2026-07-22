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

    print(f"\nUser Query:")
    print(query)

    print(f"\nUser Level:")
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

    retrieval_result = pipeline.retrieve(query)

    # ============================================================
    # EXTRACT DOCUMENTS CORRECTLY
    # ============================================================

    if isinstance(retrieval_result, dict):

        documents = retrieval_result.get("documents", [])

    elif isinstance(retrieval_result, list):

        documents = retrieval_result

    else:

        raise TypeError(
            f"Unexpected retrieval result type: "
            f"{type(retrieval_result)}"
        )

    print(f"\nRetrieved documents: {len(documents)}")

    # ============================================================
    # SAFETY CHECK
    # ============================================================

    if not documents:

        print("\nNo relevant medical documents found.")

        return

    # ============================================================
    # GENERATE ANSWER
    # ============================================================

    print("\nGenerating answer...")

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

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()