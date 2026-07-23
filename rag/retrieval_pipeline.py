import sys
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

        # ====================================================
        # MONITORING
        # ====================================================

        self.logger = RequestLogger()

        # ====================================================
        # LOAD VECTOR SEARCH ENGINE
        # ====================================================

        print("\n[1/3] Loading Vector Search Engine...")

        self.vector_search = load_search_engine()

        # ====================================================
        # LOAD DOCUMENT RERANKER
        # ====================================================

        print("\n[2/3] Loading Document Reranker...")

        self.reranker = MedicalReranker()

        # ====================================================
        # QUERY REWRITER
        # ====================================================

        print("\n[3/3] Query Rewriter Ready")

        print("\nPipeline initialized successfully!")

    # ========================================================
    # REMOVE DUPLICATE DOCUMENTS
    # ========================================================

    def remove_duplicate_documents(self, documents):

        unique_documents = []

        seen_documents = set()

        for document in documents:

            document_id = document.get(
                "document_id"
            )

            # ------------------------------------------------
            # If document_id does not exist
            # use chunk_id as fallback
            # ------------------------------------------------

            if not document_id:

                document_id = document.get(
                    "chunk_id"
                )

            # ------------------------------------------------
            # Skip duplicate documents
            # ------------------------------------------------

            if document_id in seen_documents:

                continue

            # ------------------------------------------------
            # Add document to seen set
            # ------------------------------------------------

            seen_documents.add(
                document_id
            )

            unique_documents.append(
                document
            )

        return unique_documents

    # ========================================================
    # RETRIEVE MEDICAL DOCUMENTS
    # ========================================================

    def retrieve(
        self,
        query: str,
        user_type: str = "general",
        explanation_level: str = "beginner",
        candidate_k: int = 20,
        final_k: int = 5
    ):

        # ====================================================
        # START MONITORING
        # ====================================================

        request_data = self.logger.start_request(
            query
        )

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

            rewritten_query = rewritten_result.get(

                "rewritten_query",

                query

            )

            print("\nRewritten Query:")

            print(
                rewritten_query
            )

            # =================================================
            # STEP 2: VECTOR SEARCH
            # =================================================

            print("\n[2] Running vector search...")

            candidates = self.vector_search.search(

                query=rewritten_query,

                top_k=candidate_k

            )

            print(

                f"Retrieved "
                f"{len(candidates)} "
                f"candidates."

            )

            # =================================================
            # STEP 2.5: REMOVE DUPLICATES
            # =================================================

            candidates = (

                self.remove_duplicate_documents(

                    candidates

                )

            )

            print(

                f"After deduplication: "

                f"{len(candidates)} "

                f"unique documents."

            )

            # =================================================
            # STEP 3: DOCUMENT RERANKING
            # =================================================

            print(

                "\n[3] Running "
                "document reranking..."

            )

            final_documents = (

                self.reranker.rerank(

                    query=query,

                    documents=candidates,

                    top_k=final_k

                )

            )

            print(

                f"Selected "
                f"{len(final_documents)} "

                f"final documents."

            )

            # =================================================
            # FINISH MONITORING
            # =================================================

            log_entry = (

                self.logger.finish_request(

                    request_data=request_data,

                    rewritten_query=rewritten_query,

                    retrieved_documents=final_documents,

                    answer=None,

                    error=None

                )

            )

            # =================================================
            # RETURN COMPLETE RESULT
            # =================================================

            return {

                "original_query": query,

                "rewritten_query": rewritten_result,

                "candidates": candidates,

                "documents": final_documents,

                "monitoring": {

                    "request_id":

                    log_entry.get(

                        "request_id"

                    ),

                    "latency_seconds":

                    log_entry.get(

                        "latency_seconds"

                    ),

                    "success":

                    log_entry.get(

                        "success"

                    )

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

            print(

                "\n" + "=" * 70

            )

            print(

                "ERROR IN RETRIEVAL PIPELINE"

            )

            print(

                "=" * 70

            )

            print(

                str(error)

            )

            raise


# ============================================================
# TEST PIPELINE
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # INITIALIZE PIPELINE
    # ========================================================

    pipeline = (

        MedicalRetrievalPipeline()

    )

    # ========================================================
    # RUN RETRIEVAL
    # ========================================================

    result = (

        pipeline.retrieve(

            query="What is an A1C test?",

            user_type="general",

            explanation_level="beginner",

            candidate_k=20,

            final_k=5

        )

    )

        # ========================================================
    # RETRIEVAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("RETRIEVAL SUMMARY")
    print("=" * 70)

    print(
        "\nOriginal Query:"
    )

    print(
        result["original_query"]
    )

    print(
        "\nRewritten Query:"
    )

    print(
        result["rewritten_query"].get(
            "rewritten_query",
            ""
        )
    )

    print(
        "\nSearch Intent:"
    )

    print(
        result["rewritten_query"].get(
            "search_intent",
            ""
        )
    )

    print(
        "\nMedical Entities:"
    )

    print(
        result["rewritten_query"].get(
            "medical_entities",
            []
        )
    )

    print(
        "\nCandidate Documents Retrieved:"
    )

    print(
        len(
            result["candidates"]
        )
    )

    print(
        "\nFinal Documents Selected:"
    )

    print(
        len(
            result["documents"]
        )
    )

    # ========================================================
    # PRINT SELECTED SOURCES
    # ========================================================

    print("\n" + "=" * 70)
    print("SELECTED MEDICAL SOURCES")
    print("=" * 70)

    for index, document in enumerate(
        result["documents"],
        start=1
    ):

        print(
            f"\nSOURCE {index}"
        )

        print(
            "\nTitle:"
        )

        print(
            document.get(
                "title",
                ""
            )
        )

        print(
            "\nCategory:"
        )

        print(
            document.get(
                "category",
                ""
            )
        )

        print(
            "\nRerank Score:"
        )

        print(
            f"{document.get('rerank_score', 0):.4f}"
        )

        print(
            "\nSource URL:"
        )

        print(
            document.get(
                "source_url",
                ""
            )
        )

    # ========================================================
    # MONITORING INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("MONITORING INFORMATION")
    print("=" * 70)

    monitoring = result.get(
        "monitoring",
        {}
    )

    print(
        "\nRequest ID:"
    )

    print(
        monitoring.get(
            "request_id"
        )
    )

    print(
        "\nLatency:"
    )

    print(
        monitoring.get(
            "latency_seconds"
        ),
        "seconds"
    )

    print(
        "\nSuccess:"
    )

    print(
        monitoring.get(
            "success"
        )
    )