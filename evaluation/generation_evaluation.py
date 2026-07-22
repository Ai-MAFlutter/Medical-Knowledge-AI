import json
import time
from pathlib import Path

from rag.retrieval_pipeline import MedicalRetrievalPipeline
from rag.generator import generate_answer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_results.json"
)


# ============================================================
# EVALUATION QUERIES
# ============================================================

EVALUATION_QUERIES = [
    {
        "query": "What is an A1C test?",
        "user_level": "beginner",
    },
    {
        "query": "What causes high blood pressure?",
        "user_level": "beginner",
    },
    {
        "query": "What are the symptoms of type 2 diabetes?",
        "user_level": "beginner",
    },
    {
        "query": "What is asthma?",
        "user_level": "beginner",
    },
    {
        "query": "How can I prevent heart disease?",
        "user_level": "beginner",
    },
    {
        "query": "What causes anemia?",
        "user_level": "beginner",
    },
    {
        "query": "What are the symptoms of depression?",
        "user_level": "beginner",
    },
    {
        "query": "What is cholesterol?",
        "user_level": "beginner",
    },
    {
        "query": "What are the symptoms of pneumonia?",
        "user_level": "beginner",
    },
    {
        "query": "What is a stroke?",
        "user_level": "beginner",
    },
]


# ============================================================
# QUALITY CHECKS
# ============================================================

def check_answer_quality(answer, documents):

    answer_lower = answer.lower()

    # --------------------------------------------------------
    # 1. Answer is not empty
    # --------------------------------------------------------

    has_answer = len(answer.strip()) > 50

    # --------------------------------------------------------
    # 2. Contains structured sections
    # --------------------------------------------------------

    has_direct_answer = (
        "direct answer" in answer_lower
    )

    has_explanation = (
        "explanation" in answer_lower
    )

    # --------------------------------------------------------
    # 3. Contains medical disclaimer
    # --------------------------------------------------------

    has_disclaimer = (
        "disclaimer" in answer_lower
        or "educational purposes" in answer_lower
        or "healthcare professional" in answer_lower
    )

    # --------------------------------------------------------
    # 4. Context grounding
    # --------------------------------------------------------

    context_text = " ".join(
        doc.get("content", "")
        for doc in documents
        if isinstance(doc, dict)
    ).lower()

    # Extract meaningful words from answer
    answer_words = set(
        word.strip(".,!?():;")
        for word in answer_lower.split()
        if len(word) > 5
    )

    context_words = set(
        word.strip(".,!?():;")
        for word in context_text.split()
        if len(word) > 5
    )

    overlapping_words = answer_words.intersection(
        context_words
    )

    grounded = len(overlapping_words) >= 5

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    checks = {
        "has_answer": has_answer,
        "has_direct_answer": has_direct_answer,
        "has_explanation": has_explanation,
        "has_disclaimer": has_disclaimer,
        "grounded_in_context": grounded,
    }

    score = sum(checks.values()) / len(checks)

    return {
        "checks": checks,
        "quality_score": round(score, 2),
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("MEDICAL KNOWLEDGE AI - GENERATION EVALUATION")
    print("=" * 70)

    print(
        f"\nEvaluation queries: "
        f"{len(EVALUATION_QUERIES)}"
    )

    # --------------------------------------------------------
    # Initialize retrieval pipeline once
    # --------------------------------------------------------

    print("\nInitializing retrieval pipeline...")

    pipeline = MedicalRetrievalPipeline()

    results = []

    # --------------------------------------------------------
    # Evaluate each query
    # --------------------------------------------------------

    for index, item in enumerate(
        EVALUATION_QUERIES,
        start=1
    ):

        query = item["query"]
        user_level = item["user_level"]

        print("\n" + "-" * 70)

        print(
            f"Evaluating "
            f"{index}/{len(EVALUATION_QUERIES)}"
        )

        print(f"Query: {query}")

        start_time = time.time()

        try:

            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            retrieval_result = pipeline.retrieve(
                query
            )

            if isinstance(
                retrieval_result,
                dict
            ):

                documents = retrieval_result.get(
                    "documents",
                    []
                )

            else:

                documents = retrieval_result

            # ------------------------------------------------
            # Generation
            # ------------------------------------------------

            answer = generate_answer(
                query=query,
                documents=documents,
                user_level=user_level
            )

            # ------------------------------------------------
            # Quality Evaluation
            # ------------------------------------------------

            quality = check_answer_quality(
                answer,
                documents
            )

            elapsed_time = (
                time.time() - start_time
            )

            result = {

                "query": query,

                "user_level": user_level,

                "retrieved_documents": len(
                    documents
                ),

                "answer": answer,

                "quality_score": quality[
                    "quality_score"
                ],

                "quality_checks": quality[
                    "checks"
                ],

                "latency_seconds": round(
                    elapsed_time,
                    2
                ),

                "status": "success",

            }

            results.append(result)

            print(
                f"Quality Score: "
                f"{quality['quality_score']}"
            )

            print(
                f"Latency: "
                f"{elapsed_time:.2f}s"
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            results.append({

                "query": query,

                "status": "failed",

                "error": str(error),

            })

    # ========================================================
    # SUMMARY
    # ========================================================

    successful_results = [

        result
        for result in results
        if result["status"] == "success"

    ]

    if successful_results:

        average_quality = (

            sum(
                result["quality_score"]
                for result in successful_results
            )
            / len(successful_results)

        )

        average_latency = (

            sum(
                result["latency_seconds"]
                for result in successful_results
            )
            / len(successful_results)

        )

    else:

        average_quality = 0

        average_latency = 0

    summary = {

        "total_queries": len(
            EVALUATION_QUERIES
        ),

        "successful_queries": len(
            successful_results
        ),

        "failed_queries": (

            len(results)
            - len(successful_results)

        ),

        "average_quality_score": round(
            average_quality,
            3
        ),

        "average_latency_seconds": round(
            average_latency,
            2
        ),

    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {

        "summary": summary,

        "results": results,

    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("GENERATION EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nTotal Queries: "
        f"{summary['total_queries']}"
    )

    print(
        f"Successful Queries: "
        f"{summary['successful_queries']}"
    )

    print(
        f"Failed Queries: "
        f"{summary['failed_queries']}"
    )

    print(
        f"Average Quality Score: "
        f"{summary['average_quality_score']}"
    )

    print(
        f"Average Latency: "
        f"{summary['average_latency_seconds']} seconds"
    )

    print("\n")
    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("\n")
    print("=" * 70)
    print("GENERATION EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()