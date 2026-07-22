import os
import json
from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# LLM CLIENT
# ============================================================

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)


# ============================================================
# QUERY REWRITING PROMPT
# ============================================================

QUERY_REWRITE_PROMPT = """
You are a medical information retrieval query optimizer.

Your task is to rewrite a user's medical question into a clear,
search-optimized query for retrieving relevant medical documents.

You MUST NOT answer the medical question.

You MUST preserve the original medical intent.

Rules:
1. Extract the main medical topic.
2. Add important medical terminology when appropriate.
3. Expand vague symptoms into medically relevant concepts.
4. Do not invent patient information.
5. Do not provide diagnosis or treatment.
6. Keep the rewritten query concise.
7. Return ONLY valid JSON.

Return exactly this format:

{
    "original_query": "...",
    "rewritten_query": "...",
    "medical_entities": [],
    "search_intent": "definition | symptoms | causes | diagnosis | treatment | prevention | risk_factors | comparison | other"
}

User question:
"""


# ============================================================
# QUERY REWRITER
# ============================================================

def rewrite_query(
    query: str,
    user_type: str = "general",
    explanation_level: str = "beginner"
) -> dict:

    prompt = f"""
{QUERY_REWRITE_PROMPT}

User Type:
{user_type}

Explanation Level:
{explanation_level}

Question:
{query}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You optimize medical search queries. "
                    "You do not provide medical advice."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_queries = [
        "What is an A1C test?",
        "Why am I always tired?",
        "What causes high blood pressure?",
        "How can diabetes affect the body?"
    ]

    for query in test_queries:

        print("\n" + "=" * 70)
        print("ORIGINAL QUERY:")
        print(query)

        result = rewrite_query(
            query=query,
            user_type="general",
            explanation_level="beginner"
        )

        print("\nREWRITTEN QUERY:")
        print(result["rewritten_query"])

        print("\nMEDICAL ENTITIES:")
        print(result["medical_entities"])

        print("\nSEARCH INTENT:")
        print(result["search_intent"])