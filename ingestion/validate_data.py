import json
from collections import Counter
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medlineplus_documents.json"
)


# =====================================================
# LOAD DATA
# =====================================================

def load_documents():
    """
    Load processed medical documents from JSON.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found: {DATA_PATH}"
        )

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    return documents


# =====================================================
# VALIDATE REQUIRED FIELDS
# =====================================================

def validate_required_fields(documents):

    required_fields = [
        "id",
        "title",
        "content",
        "source",
        "source_url",
        "category",
        "difficulty",
        "audience",
        "language",
    ]

    invalid_documents = []

    for document in documents:

        missing_fields = [
            field
            for field in required_fields
            if field not in document
        ]

        empty_fields = [
            field
            for field in required_fields
            if field in document
            and not document[field]
        ]

        if missing_fields or empty_fields:

            invalid_documents.append(
                {
                    "id": document.get("id"),
                    "missing_fields": missing_fields,
                    "empty_fields": empty_fields,
                }
            )

    return invalid_documents


# =====================================================
# CHECK DUPLICATES
# =====================================================

def find_duplicates(documents):

    ids = [
        document["id"]
        for document in documents
    ]

    id_counts = Counter(ids)

    duplicate_ids = {
        document_id: count
        for document_id, count in id_counts.items()
        if count > 1
    }

    return duplicate_ids


# =====================================================
# CATEGORY DISTRIBUTION
# =====================================================

def get_category_distribution(documents):

    categories = [
        document["category"]
        for document in documents
    ]

    return Counter(categories)


# =====================================================
# DOCUMENT STATISTICS
# =====================================================

def get_document_statistics(documents):

    content_lengths = [
        len(document["content"])
        for document in documents
    ]

    word_counts = [
        len(document["content"].split())
        for document in documents
    ]

    return {
        "total_documents": len(documents),
        "min_characters": min(content_lengths),
        "max_characters": max(content_lengths),
        "average_characters": round(
            sum(content_lengths)
            / len(content_lengths),
            2,
        ),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "average_words": round(
            sum(word_counts)
            / len(word_counts),
            2,
        ),
    }


# =====================================================
# MAIN VALIDATION
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MEDICAL KNOWLEDGE DATA QUALITY VALIDATION")
    print("=" * 60)

    documents = load_documents()

    print(
        f"\nLoaded documents: {len(documents)}"
    )

    # ---------------------------------------------
    # Required Fields
    # ---------------------------------------------

    invalid_documents = validate_required_fields(
        documents
    )

    print(
        "\nInvalid documents:",
        len(invalid_documents)
    )

    if invalid_documents:

        print(
            "\nFirst invalid documents:"
        )

        for document in invalid_documents[:5]:

            print(document)

    else:

        print(
            "All documents contain valid required fields."
        )

    # ---------------------------------------------
    # Duplicate IDs
    # ---------------------------------------------

    duplicate_ids = find_duplicates(
        documents
    )

    print(
        "\nDuplicate IDs:",
        len(duplicate_ids)
    )

    if duplicate_ids:

        print(duplicate_ids)

    else:

        print(
            "No duplicate document IDs found."
        )

    # ---------------------------------------------
    # Statistics
    # ---------------------------------------------

    statistics = get_document_statistics(
        documents
    )

    print(
        "\nDocument Statistics:"
    )

    for key, value in statistics.items():

        print(
            f"{key}: {value}"
        )

    # ---------------------------------------------
    # Categories
    # ---------------------------------------------

    category_distribution = (
        get_category_distribution(
            documents
        )
    )

    print(
        "\nCategory Distribution:"
    )

    for category, count in (
        category_distribution.most_common()
    ):

        print(
            f"{category}: {count}"
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "DATA QUALITY VALIDATION COMPLETED"
    )

    print(
        "=" * 60
    )