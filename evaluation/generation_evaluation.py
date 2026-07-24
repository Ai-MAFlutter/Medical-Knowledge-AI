import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import time


from rag.retrieval_pipeline import MedicalRetrievalPipeline
from rag.generator import (
    client,
    MODEL_NAME,
    SYSTEM_PROMPT,
    generate_answer,
    build_context,
)


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
# QUALITY EVALUATION
# ============================================================

def check_answer_quality(
    answer,
    documents,
):
    """
    Evaluate answer quality using deterministic checks.

    The evaluation checks:

    1. Non-empty answer
    2. Structured response
    3. Medical disclaimer
    4. Context grounding
    5. Source usage
    """

    if not answer:

        return {
            "quality_score": 0.0,
            "checks": {
                "has_answer": False,
                "has_direct_answer": False,
                "has_explanation": False,
                "has_disclaimer": False,
                "grounded_in_context": False,
            },
        }

    answer_lower = answer.lower()

    # --------------------------------------------------------
    # 1. Answer exists
    # --------------------------------------------------------

    has_answer = len(
        answer.strip()
    ) > 50

    # --------------------------------------------------------
    # 2. Structured sections
    # --------------------------------------------------------

    has_direct_answer = (
        "direct answer"
        in answer_lower
    )

    has_explanation = (
        "explanation"
        in answer_lower
    )

    # --------------------------------------------------------
    # 3. Medical disclaimer
    # --------------------------------------------------------

    has_disclaimer = (

        "disclaimer"
        in answer_lower

        or

        "educational purposes"
        in answer_lower

        or

        "healthcare professional"
        in answer_lower

    )

    # --------------------------------------------------------
    # 4. Context grounding
    # --------------------------------------------------------

    context_text = " ".join(

        doc.get(
            "content",
            ""
        )

        for doc in documents

        if isinstance(
            doc,
            dict
        )

    ).lower()

    answer_words = set(

        word.strip(
            ".,!?():;\"'"
        )

        for word in answer_lower.split()

        if len(word) > 5

    )

    context_words = set(

        word.strip(
            ".,!?():;\"'"
        )

        for word in context_text.split()

        if len(word) > 5

    )

    overlapping_words = (

        answer_words
        .intersection(
            context_words
        )

    )

    grounded = (

        len(
            overlapping_words
        ) >= 5

    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    checks = {

        "has_answer":
        has_answer,

        "has_direct_answer":
        has_direct_answer,

        "has_explanation":
        has_explanation,

        "has_disclaimer":
        has_disclaimer,

        "grounded_in_context":
        grounded,

    }

    score = (

        sum(
            checks.values()
        )

        /

        len(
            checks
        )

    )

    return {

        "checks":
        checks,

        "quality_score":
        round(
            score,
            2
        ),

    }


# ============================================================
# BASELINE LLM
# ============================================================

def generate_baseline_answer(
    query,
    user_level="beginner",
):
    """
    Generate an answer without using
    the medical knowledge base.

    This is used as the baseline approach.
    """

    prompt = f"""

USER QUESTION:
{query}

USER LEVEL:
{user_level}

Answer the question clearly.

Use the following structure:

## Direct Answer

## Explanation

## Important Points

## Educational Disclaimer

Do not diagnose the user.

Do not prescribe medications.

Include a short educational disclaimer.

"""

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",

                "content": (
                    "You are a general "
                    "medical educational assistant."
                ),

            },

            {
                "role": "user",

                "content": prompt,

            },

        ],

        temperature=0.2,

        max_tokens=1000,

    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# IMPROVED RAG PROMPT
# ============================================================

def generate_improved_rag_answer(
    query,
    documents,
    user_level="beginner",
):
    """
    Generate an answer using an improved
    context-grounded RAG prompt.
    """

    if not documents:

        return (
            "I don't have enough information "
            "in my medical knowledge base "
            "to answer this question."
        )

    context = build_context(
        documents
    )

    improved_system_prompt = """

You are Medical Knowledge AI.

You are a strict retrieval-augmented
medical education assistant.

Your answer MUST be grounded ONLY
in the provided medical sources.

Rules:

1. Use only the provided context.
2. Never invent medical facts.
3. Never diagnose the user.
4. Never prescribe medication.
5. If the context is insufficient, say so clearly.
6. Prefer the most relevant sources.
7. Be concise and accurate.
8. Include an educational disclaimer.
9. Mention uncertainty when appropriate.
10. Do not use external medical knowledge.

"""

    prompt = f"""

QUESTION:
{query}

USER LEVEL:
{user_level}

RETRIEVED MEDICAL SOURCES:
{context}

ANSWER FORMAT:

## Direct Answer

Answer the question directly using
only the retrieved sources.

## Explanation

Explain the answer clearly.

## Important Points

List the most important points
supported by the sources.

## Educational Disclaimer

This information is for educational
purposes only and does not replace
advice from a qualified healthcare
professional.

"""

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",

                "content":
                improved_system_prompt,

            },

            {
                "role": "user",

                "content":
                prompt,

            },

        ],

        temperature=0.1,

        max_tokens=1000,

    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# EVALUATE ONE APPROACH
# ============================================================

def evaluate_approach(
    approach_name,
    query,
    user_level,
    documents,
):
    """
    Evaluate one generation approach.
    """

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        if approach_name == "baseline":

            answer = generate_baseline_answer(

                query=query,

                user_level=user_level,

            )

            evaluation_documents = []

        # ----------------------------------------------------
        # NORMAL RAG
        # ----------------------------------------------------

        elif approach_name == "rag":

            answer = generate_answer(

                query=query,

                documents=documents,

                user_level=user_level,

            )

            evaluation_documents = documents

        # ----------------------------------------------------
        # IMPROVED RAG
        # ----------------------------------------------------

        elif approach_name == "improved_rag":

            answer = generate_improved_rag_answer(

                query=query,

                documents=documents,

                user_level=user_level,

            )

            evaluation_documents = documents

        else:

            raise ValueError(
                f"Unknown approach: "
                f"{approach_name}"
            )

        quality = check_answer_quality(

            answer,

            evaluation_documents,

        )

        latency = (

            time.time()
            - start_time

        )

        return {

            "approach":
            approach_name,

            "answer":
            answer,

            "quality_score":
            quality[
                "quality_score"
            ],

            "quality_checks":
            quality[
                "checks"
            ],

            "latency_seconds":
            round(
                latency,
                2
            ),

            "status":
            "success",

        }

    except Exception as error:

        return {

            "approach":
            approach_name,

            "status":
            "failed",

            "error":
            str(
                error
            ),

        }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)

    print(
        "MEDICAL KNOWLEDGE AI - "
        "GENERATION EVALUATION"
    )

    print("=" * 70)

    print(
        f"\nEvaluation queries: "
        f"{len(EVALUATION_QUERIES)}"
    )

    print(
        "\nInitializing retrieval pipeline..."
    )

    pipeline = (
        MedicalRetrievalPipeline()
    )

    all_results = []

    approaches = [

        "baseline",

        "rag",

        "improved_rag",

    ]

    # --------------------------------------------------------
    # Evaluate every query
    # --------------------------------------------------------

    for index, item in enumerate(

        EVALUATION_QUERIES,

        start=1,

    ):

        query = item[
            "query"
        ]

        user_level = item[
            "user_level"
        ]

        print("\n")

        print(
            "-" * 70
        )

        print(

            f"QUERY "
            f"{index}/"
            f"{len(EVALUATION_QUERIES)}"

        )

        print(
            f"Question: "
            f"{query}"
        )

        # ----------------------------------------------------
        # Retrieve documents once
        # ----------------------------------------------------

        print(
            "\nRetrieving documents..."
        )

        retrieval_result = (

            pipeline.retrieve(
                query
            )

        )

        if isinstance(

            retrieval_result,

            dict,

        ):

            documents = (

                retrieval_result
                .get(
                    "documents",
                    []
                )

            )

        else:

            documents = (
                retrieval_result
            )

        print(

            f"Retrieved documents: "
            f"{len(documents)}"

        )

        # ----------------------------------------------------
        # Evaluate approaches
        # ----------------------------------------------------

        query_results = {

            "query":
            query,

            "user_level":
            user_level,

            "retrieved_documents":
            len(
                documents
            ),

            "approaches":
            {},

        }

        for approach in approaches:

            print("\n")

            print(

                f"Evaluating: "
                f"{approach}"

            )

            result = (

                evaluate_approach(

                    approach_name=
                    approach,

                    query=
                    query,

                    user_level=
                    user_level,

                    documents=
                    documents,

                )

            )

            query_results[
                "approaches"
            ][
                approach
            ] = result

            if (

                result[
                    "status"
                ]

                ==

                "success"

            ):

                print(

                    f"Quality Score: "
                    f"{result['quality_score']}"

                )

                print(

                    f"Latency: "
                    f"{result['latency_seconds']}s"

                )

            else:

                print(

                    f"ERROR: "
                    f"{result['error']}"

                )

        all_results.append(
            query_results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {}

    for approach in approaches:

        successful_results = [

            query_result[
                "approaches"
            ][
                approach
            ]

            for query_result
            in all_results

            if (

                query_result[
                    "approaches"
                ][
                    approach
                ][
                    "status"
                ]

                ==

                "success"

            )

        ]

        if successful_results:

            average_quality = (

                sum(

                    result[
                        "quality_score"
                    ]

                    for result
                    in successful_results

                )

                /

                len(
                    successful_results
                )

            )

            average_latency = (

                sum(

                    result[
                        "latency_seconds"
                    ]

                    for result
                    in successful_results

                )

                /

                len(
                    successful_results
                )

            )

        else:

            average_quality = 0

            average_latency = 0

        summary[
            approach
        ] = {

            "total_queries":
            len(
                EVALUATION_QUERIES
            ),

            "successful_queries":
            len(
                successful_results
            ),

            "average_quality_score":
            round(
                average_quality,
                3
            ),

            "average_latency_seconds":
            round(
                average_latency,
                2
            ),

        }

    # --------------------------------------------------------
    # Winner
    # --------------------------------------------------------

    winner = max(

        summary,

        key=lambda approach:

        summary[
            approach
        ][
            "average_quality_score"
        ],

    )

    summary[
        "winner"
    ] = winner

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {

        "summary":
        summary,

        "results":
        all_results,

    }

    OUTPUT_FILE.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    with open(

        OUTPUT_FILE,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            output,

            file,

            ensure_ascii=False,

            indent=2,

        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "GENERATION EVALUATION RESULTS"
    )

    print("=" * 70)

    for approach in approaches:

        print("\n")

        print(
            approach.upper()
        )

        print(

            f"Average Quality: "

            f"{summary[approach]['average_quality_score']}"

        )

        print(

            f"Average Latency: "

            f"{summary[approach]['average_latency_seconds']}s"

        )

    print("\n")

    print(
        "=" * 70
    )

    print(

        f"WINNER: "
        f"{winner.upper()}"

    )

    print(
        "=" * 70
    )

    print("\nResults saved to:")

    print(
        OUTPUT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()