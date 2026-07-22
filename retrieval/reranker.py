import json
from pathlib import Path

from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medlineplus_chunks.json"
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# RERANKER
# ============================================================

class MedicalReranker:

    def __init__(self):

        print(f"Loading reranker model: {MODEL_NAME}")

        self.model = CrossEncoder(MODEL_NAME)

        print("Reranker loaded successfully.")

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int = 5
    ) -> list:

        if not documents:
            return []

        pairs = [
            [query, document["content"]]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked_documents = []

        for document, score in zip(documents, scores):

            document_copy = document.copy()

            document_copy["rerank_score"] = float(score)

            reranked_documents.append(document_copy)

        reranked_documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_documents[:top_k]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Loading medical chunks...")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:

        documents = json.load(file)

    print(f"Loaded {len(documents)} chunks.")

    reranker = MedicalReranker()

    query = "What is an A1C test?"

    # Simulated retrieved documents
    candidate_documents = documents[:10]

    results = reranker.rerank(
        query=query,
        documents=candidate_documents,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    for index, result in enumerate(results, start=1):

        print(f"\nRESULT {index}")
        print(f"Score: {result['rerank_score']:.4f}")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print(f"Content: {result['content'][:500]}...")