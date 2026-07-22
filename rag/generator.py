import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GROQ CLIENT
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )


client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


MODEL_NAME = "llama-3.3-70b-versatile"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Medical Knowledge AI, an educational medical knowledge assistant.

Your job is to explain medical information clearly and safely.

IMPORTANT RULES:

1. Answer ONLY using the provided medical context.
2. Do not invent medical facts.
3. If the answer is not available in the context, say:
   "I don't have enough information in my medical knowledge base to answer this question."
4. Do not diagnose diseases.
5. Do not prescribe medications.
6. Do not replace a qualified healthcare professional.
7. For urgent or emergency symptoms, recommend seeking immediate medical care.
8. Always provide a short educational disclaimer.
"""


# ============================================================
# USER LEVEL INSTRUCTIONS
# ============================================================

LEVEL_INSTRUCTIONS = {
    "beginner": """
Explain the answer in very simple language.
Avoid complex medical terminology.
Use short paragraphs and simple examples when helpful.
""",

    "student": """
Explain the topic at a medical student level.
Include important medical terminology and mechanisms.
Organize the answer clearly.
""",

    "advanced": """
Provide a detailed and technically accurate explanation.
Use appropriate medical terminology.
Discuss mechanisms, clinical relevance, and important distinctions.
"""
}


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents):
    """
    Convert retrieved documents into context for the LLM.
    """

    context_parts = []

    for i, doc in enumerate(documents, start=1):

        context_parts.append(
            f"""
SOURCE {i}

Title:
{doc.get("title", "")}

Category:
{doc.get("category", "")}

Source:
{doc.get("source_url", "")}

Content:
{doc.get("content", "")}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    documents,
    user_level="beginner"
):
    """
    Generate an educational medical answer
    using retrieved medical documents.
    """

    context = build_context(documents)

    user_level = user_level.lower()

    level_instruction = LEVEL_INSTRUCTIONS.get(
        user_level,
        LEVEL_INSTRUCTIONS["beginner"]
    )

    user_prompt = f"""
USER QUESTION:
{query}

USER LEVEL:
{user_level}

EXPLANATION STYLE:
{level_instruction}

MEDICAL KNOWLEDGE BASE CONTEXT:
{context}

Answer the user's question based only on the medical knowledge base.

Use this structure:

## Direct Answer

## Explanation

## Important Points

## Educational Disclaimer

Remember:
- Do not diagnose the user.
- Do not prescribe medication.
- Do not invent information.
- Use only the provided medical context.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content