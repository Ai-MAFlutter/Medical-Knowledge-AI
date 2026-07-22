import json
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medlineplus_documents.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_queries.json"
)


# =====================================================
# QUESTION TEMPLATES
# =====================================================

QUESTION_TEMPLATES = {

    "en": [
        "What is {title}?",
        "Can you explain {title}?",
        "What should I know about {title}?",
    ],

    "es": [
        "¿Qué es {title}?",
        "¿Puedes explicar {title}?",
        "¿Qué debo saber sobre {title}?",
    ]

}


# =====================================================
# CREATE EVALUATION DATASET
# =====================================================

def create_evaluation_dataset(
    documents,
    max_documents=50,
):

    queries = []

    selected_documents = [

        document

        for document in documents

        if document["language"] == "en"

    ][:max_documents]

    for document in selected_documents:

        language = document["language"]

        templates = QUESTION_TEMPLATES[
            language
        ]

        for template in templates:

            query = template.format(

                title=document["title"]

            )

            queries.append(

                {

                    "query": query,

                    "relevant_document_ids": [

                        document["id"]

                    ],

                    "language": language,

                }

            )

    return queries


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    with open(

        DOCUMENTS_PATH,

        "r",

        encoding="utf-8"

    ) as file:

        documents = json.load(
            file
        )


    queries = create_evaluation_dataset(

        documents,

        max_documents=50

    )


    with open(

        OUTPUT_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            queries,

            file,

            ensure_ascii=False,

            indent=2

        )


    print(

        f"Created {len(queries)} evaluation queries."

    )

    print(

        f"Saved to: {OUTPUT_PATH}"

    )