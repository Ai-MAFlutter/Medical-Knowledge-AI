import json
import time
import uuid
from datetime import datetime
from pathlib import Path


class RequestLogger:
    """
    Logger for tracking RAG pipeline requests.

    Stores:
    - User queries
    - Rewritten queries
    - Retrieved documents
    - Generated answers
    - Latency
    - Retrieval quality
    - Errors
    """

    def __init__(self, log_dir="monitoring/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "requests.jsonl"

    def start_request(self, query: str):
        """
        Start tracking a new request.
        """

        request_id = str(uuid.uuid4())

        return {
            "request_id": request_id,
            "query": query,
            "start_time": time.time(),
            "timestamp": datetime.utcnow().isoformat()
        }

    def finish_request(
        self,
        request_data: dict,
        rewritten_query: str = None,
        retrieved_documents: list = None,
        answer: str = None,
        error: str = None
    ):
        """
        Finish request and save monitoring data.
        """

        end_time = time.time()

        latency = end_time - request_data["start_time"]

        log_entry = {
            "request_id": request_data["request_id"],
            "timestamp": request_data["timestamp"],

            "query": request_data["query"],

            "rewritten_query": rewritten_query,

            "retrieved_documents": self._serialize_documents(
                retrieved_documents
            ),

            "num_documents": len(retrieved_documents)
            if retrieved_documents
            else 0,

            "answer": answer,

            "latency_seconds": round(latency, 4),

            "success": error is None,

            "error": error
        }

        self._write_log(log_entry)

        return log_entry

    def log_request(
        self,
        query: str,
        rewritten_query: str = None,
        retrieved_documents: list = None,
        answer: str = None,
        error: str = None
    ):
        """
        Convenience method for logging a complete request.
        """

        request = self.start_request(query)

        return self.finish_request(
            request_data=request,
            rewritten_query=rewritten_query,
            retrieved_documents=retrieved_documents,
            answer=answer,
            error=error
        )

    def _serialize_documents(self, documents):
        """
        Convert retrieved documents to JSON-safe format.
        """

        if not documents:
            return []

        serialized = []

        for doc in documents:

            if isinstance(doc, dict):

                serialized.append({
                    "title": doc.get("title", ""),
                    "category": doc.get("category", ""),
                    "language": doc.get("language", ""),
                    "source": doc.get("source", ""),
                    "content": doc.get("content", "")[:500]
                })

            else:

                serialized.append({
                    "document": str(doc)
                })

        return serialized

    def _write_log(self, log_entry: dict):
        """
        Append one JSON object per line.
        """

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    log_entry,
                    ensure_ascii=False
                )
                + "\n"
            )