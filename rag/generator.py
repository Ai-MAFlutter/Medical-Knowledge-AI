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
        "GROQ_API_KEY not found. "
        "Please add GROQ_API_KEY to your .env file."
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

Your job is to explain medical information clearly, accurately, and safely.

IMPORTANT RULES:

1. Answer ONLY using the provided medical context.
2. Do not invent medical facts.
3. If the answer is not available in the context, say:

"I don't have enough information in my medical knowledge base
to answer this question."

4. Do not diagnose diseases.
5. Do not prescribe medications.
6. Do not replace a qualified healthcare professional.
7. For urgent or emergency symptoms, recommend seeking immediate medical care.
8. Always provide a short educational disclaimer.
9. Never claim certainty beyond the provided medical sources.
10. Prefer information from the most relevant retrieved sources.
"""


# ============================================================
# USER LEVEL INSTRUCTIONS
# ============================================================

LEVEL_INSTRUCTIONS = {

    "beginner": """
Explain the answer in very simple language.

Avoid unnecessary medical terminology.

Use short paragraphs and simple explanations.

If you use a medical term, explain it briefly.
""",

    "student": """
Explain the topic at a medical student level.

Include important medical terminology and mechanisms.

Organize the answer clearly.

Mention important clinical distinctions when supported by the context.
""",

    "advanced": """
Provide a detailed and technically accurate explanation.

Use appropriate medical terminology.

Discuss mechanisms, clinical relevance, and important distinctions.

Do not include information that is not supported by the provided context.
"""
}


# ============================================================
# NORMALIZE DOCUMENT
# ============================================================

def normalize_document(document):
    """
    Make sure every retrieved document is represented
    as a dictionary.

    This protects the generator from unexpected data formats.
    """

    if isinstance(document, dict):
        return document

    return {
        "title": "",
        "category": "",
        "source_url": "",
        "content": str(document)
    }


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents):
    """
    Convert retrieved documents into LLM context.
    """

    context_parts = []

    for index, document in enumerate(documents, start=1):

        doc = normalize_document(document)

        context_parts.append(
            f"""
==============================
SOURCE {index}
==============================

TITLE:
{doc.get("title", "")}

CATEGORY:
{doc.get("category", "")}

SOURCE URL:
{doc.get("source_url", "")}

CONTENT:
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
    using only retrieved medical documents.
    """

    # --------------------------------------------------------
    # Validate documents
    # --------------------------------------------------------

    if not documents:

        return """
## Direct Answer

I don't have enough information in my medical knowledge base
to answer this question.

## Educational Disclaimer

This information is for educational purposes only and does not
replace advice from a qualified healthcare professional.
"""

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(documents)

    # --------------------------------------------------------
    # Normalize user level
    # --------------------------------------------------------

    user_level = user_level.lower().strip()

    level_instruction = LEVEL_INSTRUCTIONS.get(
        user_level,
        LEVEL_INSTRUCTIONS["beginner"]
    )

    # --------------------------------------------------------
    # Build user prompt
    # --------------------------------------------------------

    user_prompt = f"""
USER QUESTION:
{query}

USER LEVEL:
{user_level}

EXPLANATION STYLE:
{level_instruction}

MEDICAL KNOWLEDGE BASE CONTEXT:
{context}

==================================================
ANSWERING INSTRUCTIONS
==================================================

Answer the user's question using ONLY the medical
knowledge base context above.

If the context does not contain enough information,
clearly say that you do not have enough information.

Use the following structure:

## Direct Answer

Give a direct answer to the question.

## Explanation

Explain the topic according to the user's level.

## Important Points

Use bullet points for the most important information.

## Educational Disclaimer

Include a short medical educational disclaimer.

IMPORTANT:

- Do not diagnose the user.
- Do not prescribe medications.
- Do not invent medical information.
- Do not use external knowledge.
- Do not make unsupported medical claims.
"""


    # --------------------------------------------------------
    # Call Groq LLM
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Extract answer
    # --------------------------------------------------------

    answer = response.choices[0].message.content

    return answer