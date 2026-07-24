import sys
from pathlib import Path
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from rag.retrieval_pipeline import MedicalRetrievalPipeline
from rag.generator import generate_answer


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Medical Knowledge AI API",
    description=(
        "AI-powered medical knowledge assistant "
        "using RAG, semantic search, reranking, "
        "and Groq LLM generation."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODELS
# ============================================================

class MedicalQuestion(BaseModel):

    query: str

    explanation_level: str = "beginner"


class UserFeedback(BaseModel):

    request_id: str

    feedback: str

    query: str = ""


# ============================================================
# GLOBAL PIPELINE
# ============================================================

pipeline = None


# ============================================================
# FEEDBACK STORAGE
# ============================================================

FEEDBACK_DIR = PROJECT_ROOT / "monitoring" / "logs"

FEEDBACK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "Medical Knowledge AI API"
    }


# ============================================================
# ASK MEDICAL QUESTION
# ============================================================

@app.post("/ask")
def ask_medical_question(
    request: MedicalQuestion
):

    global pipeline

    try:

        # ----------------------------------------------------
        # INITIALIZE PIPELINE ON FIRST REQUEST
        # ----------------------------------------------------

        if pipeline is None:

            pipeline = MedicalRetrievalPipeline()

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        retrieval_result = pipeline.retrieve(

            query=request.query,

            user_type="general",

            explanation_level=request.explanation_level,

            candidate_k=10,

            final_k=5

        )

        documents = retrieval_result.get(
            "documents",
            []
        )

        if not documents:

            return {

                "query": request.query,

                "answer": (
                    "I could not find relevant "
                    "medical information."
                ),

                "documents": [],

                "monitoring":
                retrieval_result.get(
                    "monitoring",
                    {}
                )

            }

        # ----------------------------------------------------
        # LLM GENERATION
        # ----------------------------------------------------

        answer = generate_answer(

            query=request.query,

            documents=documents,

            user_level=request.explanation_level

        )

        # ----------------------------------------------------
        # RETURN RESPONSE
        # ----------------------------------------------------

        return {

            "query": request.query,

            "answer": answer,

            "documents": documents,

            "monitoring":
            retrieval_result.get(
                "monitoring",
                {}
            )

        }

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)

        )


# ============================================================
# USER FEEDBACK
# ============================================================

@app.post("/feedback")
def save_feedback(
    feedback: UserFeedback
):

    try:

        # ----------------------------------------------------
        # VALIDATE FEEDBACK
        # ----------------------------------------------------

        allowed_feedback = [

            "positive",

            "negative"

        ]

        if feedback.feedback not in allowed_feedback:

            raise HTTPException(

                status_code=400,

                detail=(
                    "Feedback must be "
                    "'positive' or 'negative'."
                )

            )

        # ----------------------------------------------------
        # CREATE FEEDBACK RECORD
        # ----------------------------------------------------

        feedback_record = {

            "request_id":
            feedback.request_id,

            "feedback":
            feedback.feedback,

            "query":
            feedback.query,

            "timestamp":
            datetime.utcnow().isoformat()

        }

        # ----------------------------------------------------
        # SAVE AS JSONL
        # ----------------------------------------------------

        with open(

            FEEDBACK_FILE,

            "a",

            encoding="utf-8"

        ) as file:

            file.write(

                json.dumps(

                    feedback_record,

                    ensure_ascii=False

                )

                + "\n"

            )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "status": "success",

            "message":
            "Feedback saved successfully.",

            "feedback":
            feedback_record

        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)

        )