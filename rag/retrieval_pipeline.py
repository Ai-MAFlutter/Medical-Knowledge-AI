import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.query_rewriter import rewrite_query
from retrieval.vector_search import load_search_engine
from retrieval.reranker import MedicalReranker


class MedicalRetrievalPipeline:

    def __init__(self):

        print("=" * 70)
        print("INITIALIZING MEDICAL RETRIEVAL PIPELINE")
        print("=" * 70)

        print("\n[1/3] Loading Vector Search Engine...")

        self.vector_search = load_search_engine()

        print("\n[2/3] Loading Document Reranker...")

        self.reranker = MedicalReranker()

        print("\n[3/3] Query Rewriter Ready")

        print("\nPipeline initialized successfully!")

    def retrieve(
        self,
        query: str,
        user_type: str = "general",
        explanation_level: str = "beginner",
        candidate_k: int = 10,
        final_k: int = 5
    ):

        print("\n" + "=" * 70)
        print("MEDICAL RETRIEVAL PIPELINE")
        print("=" * 70)

        print(f"\nOriginal Query:")
        print(query)

        # ====================================================
        # STEP 1: QUERY REWRITING
        # ====================================================

        print("\n[1] Rewriting query...")

        rewritten_result = rewrite_query(
            query=query,
            user_type=user_type,
            explanation_level=explanation_level
        )

        rewritten_query = rewritten_result["rewritten_query"]

        print(f"Rewritten Query:")
        print(rewritten_query)

        # ====================================================
        # STEP 2: VECTOR SEARCH
        # ====================================================

        print("\n[2] Running vector search...")

        candidates = self.vector_search.search(
            query=rewritten_query,
            top_k=candidate_k
        )

        print(f"Retrieved {len(candidates)} candidates.")

        # ====================================================
        # STEP 3: RERANKING
        # ====================================================

        print("\n[3] Running document reranking...")

        final_documents = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=final_k
        )

        print(f"Selected {len(final_documents)} final documents.")

        return {
            "original_query": query,
            "rewritten_query": rewritten_result,
            "candidates": candidates,
            "documents": final_documents
        }


if __name__ == "__main__":

    pipeline = MedicalRetrievalPipeline()

    result = pipeline.retrieve(
        query="What is an A1C test?",
        user_type="general",
        explanation_level="beginner",
        candidate_k=10,
        final_k=5
    )

    print("\n" + "=" * 70)
    print("FINAL RERANKED DOCUMENTS")
    print("=" * 70)

    for index, document in enumerate(
        result["documents"],
        start=1
    ):

        print(f"\nRESULT {index}")
        print(f"Title: {document['title']}")
        print(f"Category: {document['category']}")
        print(
            f"Rerank Score: "
            f"{document['rerank_score']:.4f}"
        )
        print(f"Source: {document['source_url']}")
        print(
            f"Content: "
            f"{document['content'][:500]}..."
        )