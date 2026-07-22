import sys
import time
from pathlib import Path

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from rag.query_rewriter import rewrite_query
from retrieval.vector_search import load_search_engine
from retrieval.reranker import MedicalReranker

from monitoring.logger import RequestLogger


# ============================================================
# MEDICAL RETRIEVAL PIPELINE
# ============================================================

class MedicalRetrievalPipeline:

    def __init__(self):

        print("=" * 70)
        print("INITIALIZING MEDICAL RETRIEVAL PIPELINE")
        print("=" * 70)

        # ----------------------------------------------------
        # Monitoring
        # ----------------------------------------------------

        self.logger = RequestLogger()

        # ----------------------------------------------------
        # Load Vector Search
        # ----------------------------------------------------

        print("\n[1/3] Loading Vector Search Engine...")

        self.vector_search = load_search_engine()

        # ----------------------------------------------------
        # Load Reranker
        # ----------------------------------------------------

        print("\n[2/3] Loading Document Reranker...")

        self.reranker = MedicalReranker()

        # ----------------------------------------------------
        # Query Rewriter
        # ----------------------------------------------------

        print("\n[3/3] Query Rewriter Ready")

        print("\nPipeline initialized successfully!")

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        user_type: str = "general",
        explanation_level: str = "beginner",
        candidate_k: int = 10,
        final_k: int = 5
    ):

        # ----------------------------------------------------
        # START MONITORING REQUEST
        # ----------------------------------------------------

        request_data = self.logger.start_request(query)

        start_time = time.time()

        try:

            print("\n" + "=" * 70)
            print("MEDICAL RETRIEVAL PIPELINE")
            print("=" * 70)

            print("\nOriginal Query:")
            print(query)

            # =================================================
            # STEP 1: QUERY REWRITING
            # =================================================

            print("\n[1] Rewriting query...")

            rewritten_result = rewrite_query(
                query=query,
                user_type=user_type,
                explanation_level=explanation_level
            )

            rewritten_query = rewritten_result["rewritten_query"]

            print("\nRewritten Query:")
            print(rewritten_query)

            # =================================================
            # STEP 2: VECTOR SEARCH
            # =================================================

            print("\n[2] Running vector search...")

            candidates = self.vector_search.search(
                query=rewritten_query,
                top_k=candidate_k
            )

            print(
                f"Retrieved {len(candidates)} candidates."
            )

            # =================================================
            # STEP 3: RERANKING
            # =================================================

            print("\n[3] Running document reranking...")

            final_documents = self.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=final_k
            )

            print(
                f"Selected {len(final_documents)} "
                f"final documents."
            )

            # =================================================
            # FINISH MONITORING
            # =================================================

            log_entry = self.logger.finish_request(

                request_data=request_data,

                rewritten_query=rewritten_query,

                retrieved_documents=final_documents,

                answer=None,

                error=None
            )

            # =================================================
            # RETURN PIPELINE RESULT
            # =================================================

            return {

                "original_query": query,

                "rewritten_query": rewritten_result,

                "candidates": candidates,

                "documents": final_documents,

                "monitoring": {

                    "request_id":
                    log_entry["request_id"],

                    "latency_seconds":
                    log_entry["latency_seconds"],

                    "success":
                    log_entry["success"]
                }

            }

        except Exception as error:

            # =================================================
            # LOG FAILED REQUEST
            # =================================================

            self.logger.finish_request(

                request_data=request_data,

                rewritten_query=None,

                retrieved_documents=[],

                answer=None,

                error=str(error)
            )

            print("\nERROR IN RETRIEVAL PIPELINE:")

            print(str(error))

            raise error


# ============================================================
# TEST PIPELINE
# ============================================================

if __name__ == "__main__":

    pipeline = MedicalRetrievalPipeline()

    result = pipeline.retrieve(

        query="What is an A1C test?",

        user_type="general",

        explanation_level="beginner",

        candidate_k=10,

        final_k=5

    )

    # ========================================================
    # PRINT FINAL DOCUMENTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RERANKED DOCUMENTS")
    print("=" * 70)

    for index, document in enumerate(

        result["documents"],

        start=1

    ):

        print(f"\nRESULT {index}")

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
            f"Source: "
            f"{document.get('source_url', '')}"
        )

        print(
            f"Content: "
            f"{document.get('content', '')[:500]}..."
        )

    # ========================================================
    # PRINT MONITORING INFO
    # ========================================================

    print("\n" + "=" * 70)
    print("MONITORING INFORMATION")
    print("=" * 70)

    print(
        f"Request ID: "
        f"{result['monitoring']['request_id']}"
    )

    print(
        f"Latency: "
        f"{result['monitoring']['latency_seconds']} seconds"
    )

    print(
        f"Success: "
        f"{result['monitoring']['success']}"
    )