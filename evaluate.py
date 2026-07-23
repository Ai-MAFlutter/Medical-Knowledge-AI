import sys
import json
import time
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from rag.retrieval_pipeline import MedicalRetrievalPipeline


# ============================================================
# EVALUATION DATASET
# ============================================================

EVALUATION_DATASET = [

    {
        "question": "What is an A1C test?",
        "expected_keywords": [
            "A1C",
            "blood sugar",
            "diabetes"
        ],
        "expected_source_keywords": [
            "A1C",
            "Diabetes"
        ]
    },

    {
        "question": "What are the symptoms of diabetes?",
        "expected_keywords": [
            "thirsty",
            "urinating",
            "tired",
            "weight"
        ],
        "expected_source_keywords": [
            "Diabetes"
        ]
    },

    {
        "question": "What is asthma?",
        "expected_keywords": [
            "airways",
            "breathing",
            "wheezing",
            "coughing"
        ],
        "expected_source_keywords": [
            "Asthma"
        ]
    },

    {
        "question": "What causes high blood pressure?",
        "expected_keywords": [
            "blood pressure",
            "heart",
            "risk"
        ],
        "expected_source_keywords": [
            "Blood Pressure",
            "Hypertension"
        ]
    },

    {
        "question": "What are the symptoms of anemia?",
        "expected_keywords": [
            "tired",
            "weak",
            "fatigue"
        ],
        "expected_source_keywords": [
            "Anemia"
        ]
    },

    {
        "question": "What are the symptoms of the flu?",
        "expected_keywords": [
            "fever",
            "cough",
            "tired"
        ],
        "expected_source_keywords": [
            "Flu",
            "Influenza"
        ]
    }

]


# ============================================================
# EVALUATION HELPERS
# ============================================================


def calculate_keyword_score(
    documents,
    expected_keywords
):

    if not documents:
        return 0.0

    combined_text = " ".join(

        document.get(
            "content",
            ""
        ).lower()

        for document in documents

    )

    matched_keywords = 0

    for keyword in expected_keywords:

        if keyword.lower() in combined_text:

            matched_keywords += 1

    return matched_keywords / len(
        expected_keywords
    )


def calculate_source_score(
    documents,
    expected_source_keywords
):

    if not documents:
        return 0.0

    for document in documents:

        title = document.get(
            "title",
            ""
        ).lower()

        for keyword in expected_source_keywords:

            if keyword.lower() in title:

                return 1.0

    return 0.0


# ============================================================
# MAIN EVALUATION
# ============================================================


def run_evaluation():

    print("=" * 80)
    print("MEDICAL KNOWLEDGE AI - RAG EVALUATION")
    print("=" * 80)

    print("\nInitializing pipeline...")

    pipeline = MedicalRetrievalPipeline()

    results = []

    total_latency = 0

    total_keyword_score = 0

    total_source_score = 0


    # ========================================================
    # RUN TESTS
    # ========================================================

    for index, test_case in enumerate(

        EVALUATION_DATASET,

        start=1

    ):

        question = test_case["question"]

        print("\n" + "=" * 80)

        print(
            f"TEST {index}/{len(EVALUATION_DATASET)}"
        )

        print("=" * 80)

        print(
            f"\nQuestion:\n{question}"
        )


        start_time = time.time()


        try:

            result = pipeline.retrieve(

                query=question,

                user_type="general",

                explanation_level="beginner",

                candidate_k=10,

                final_k=5

            )


            latency = time.time() - start_time


            documents = result.get(

                "documents",

                []

            )


            keyword_score = calculate_keyword_score(

                documents,

                test_case["expected_keywords"]

            )


            source_score = calculate_source_score(

                documents,

                test_case["expected_source_keywords"]

            )


            total_latency += latency

            total_keyword_score += keyword_score

            total_source_score += source_score


            evaluation_result = {

                "question": question,

                "latency_seconds": round(

                    latency,

                    4

                ),

                "documents_retrieved": len(

                    documents

                ),

                "keyword_score": round(

                    keyword_score,

                    4

                ),

                "source_score": round(

                    source_score,

                    4

                ),

                "success": True

            }


            results.append(

                evaluation_result

            )


            print(

                f"\nLatency: "

                f"{latency:.2f} seconds"

            )

            print(

                f"Documents: "

                f"{len(documents)}"

            )

            print(

                f"Keyword Score: "

                f"{keyword_score:.2f}"

            )

            print(

                f"Source Score: "

                f"{source_score:.2f}"

            )


        except Exception as error:


            print(

                "\nERROR:"

            )

            print(

                str(error)

            )


            results.append(

                {

                    "question": question,

                    "success": False,

                    "error": str(error)

                }

            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    successful_results = [

        result

        for result in results

        if result.get(

            "success",

            False

        )

    ]


    if successful_results:

        count = len(

            successful_results

        )


        average_latency = (

            total_latency / count

        )


        average_keyword_score = (

            total_keyword_score / count

        )


        average_source_score = (

            total_source_score / count

        )


    else:

        average_latency = 0

        average_keyword_score = 0

        average_source_score = 0


    summary = {

        "total_tests": len(

            EVALUATION_DATASET

        ),

        "successful_tests": len(

            successful_results

        ),

        "average_latency_seconds": round(

            average_latency,

            4

        ),

        "average_keyword_score": round(

            average_keyword_score,

            4

        ),

        "average_source_score": round(

            average_source_score,

            4

        )

    }


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    evaluation_dir = PROJECT_ROOT / "evaluation"

    evaluation_dir.mkdir(

        exist_ok=True

    )


    output_file = (

        evaluation_dir /

        "evaluation_results.json"

    )


    with open(

        output_file,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            {

                "summary": summary,

                "results": results

            },

            file,

            indent=4

        )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n\n")

    print("=" * 80)

    print(

        "FINAL EVALUATION SUMMARY"

    )

    print("=" * 80)


    print(

        f"\nTotal Tests: "

        f"{summary['total_tests']}"

    )

    print(

        f"Successful Tests: "

        f"{summary['successful_tests']}"

    )

    print(

        f"Average Latency: "

        f"{summary['average_latency_seconds']} seconds"

    )

    print(

        f"Average Keyword Score: "

        f"{summary['average_keyword_score'] * 100:.2f}%"

    )

    print(

        f"Average Source Score: "

        f"{summary['average_source_score'] * 100:.2f}%"

    )


    print(

        f"\nResults saved to: "

        f"{output_file}"

    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    run_evaluation()