import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEEDBACK_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
    / "feedback.jsonl"
)


def save_feedback(
    request_id,
    query,
    rating,
    comment=None,
):
    feedback = {
        "request_id": request_id,
        "query": query,
        "rating": rating,
        "comment": comment,
        "timestamp": datetime.utcnow().isoformat(),
    }

    FEEDBACK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        FEEDBACK_PATH,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                feedback,
                ensure_ascii=False
            )
            + "\n"
        )