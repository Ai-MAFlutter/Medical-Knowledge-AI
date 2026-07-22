import json
import re
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medlineplus_documents.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medlineplus_chunks.json"
)


# =====================================================
# CONFIGURATION
# =====================================================

MAX_WORDS = 500
OVERLAP_WORDS = 50


# =====================================================
# LOAD DOCUMENTS
# =====================================================

def load_documents():
    """
    Load processed medical documents.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =====================================================
# TEXT CLEANING
# =====================================================

def normalize_text(text: str) -> str:
    """
    Normalize whitespace.
    """

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =====================================================
# CHUNKING
# =====================================================

def create_chunks(
    text: str,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
):
    """
    Split text into overlapping word-based chunks.
    """

    text = normalize_text(text)

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + max_words,
            len(words)
        )

        chunk_words = words[start:end]

        chunk_text = " ".join(
            chunk_words
        )

        chunks.append(
            chunk_text
        )

        if end == len(words):
            break

        start = end - overlap_words

    return chunks


# =====================================================
# PROCESS DOCUMENTS
# =====================================================

def process_documents(documents):

    all_chunks = []

    for document in documents:

        chunks = create_chunks(
            document["content"]
        )

        for chunk_index, chunk_text in enumerate(
            chunks
        ):

            chunk = {

                "chunk_id": (
                    f"{document['id']}"
                    f"_chunk_{chunk_index:03d}"
                ),

                "document_id": document["id"],

                "title": document["title"],

                "content": chunk_text,

                "source": document["source"],

                "source_url": document["source_url"],

                "category": document["category"],

                "difficulty": document["difficulty"],

                "audience": document["audience"],

                "language": document["language"],

                "chunk_index": chunk_index,

                "total_chunks": len(chunks),

            }

            all_chunks.append(
                chunk
            )

    return all_chunks


# =====================================================
# SAVE CHUNKS
# =====================================================

def save_chunks(chunks):

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved {len(chunks)} chunks to:"
    )

    print(
        OUTPUT_PATH
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print(
        "Loading documents..."
    )

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents."
    )

    print(
        "Creating chunks..."
    )

    chunks = process_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    save_chunks(
        chunks
    )

    print(
        "\nFirst chunk:"
    )

    print(
        json.dumps(
            chunks[0],
            indent=2,
            ensure_ascii=False
        )
    )