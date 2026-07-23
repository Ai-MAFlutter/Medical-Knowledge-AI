import os
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medlineplus_chunks.json"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vector_cache"
)

EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
METADATA_PATH = CACHE_DIR / "metadata.json"


# ============================================================
# LOCAL EMBEDDING MODEL
# ============================================================

MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "all-MiniLM-L6-v2"
)


# ============================================================
# VECTOR SEARCH ENGINE
# ============================================================

class VectorSearchEngine:

    def __init__(
        self,
        data_path: str = None,
        model_name: str = "all-MiniLM-L6-v2"
    ):

        self.data_path = (
            Path(data_path)
            if data_path
            else DATA_PATH
        )

        self.model_name = model_name

        self.documents = []

        self.embeddings = None

        self.model = None

        # ----------------------------------------------------
        # INITIALIZE
        # ----------------------------------------------------

        self._load_documents()

        self._load_model()

        self._load_or_create_embeddings()

    # ========================================================
    # LOAD DOCUMENTS
    # ========================================================

    def _load_documents(self):

        print(
            f"Loading documents from: "
            f"{self.data_path}"
        )

        if not self.data_path.exists():

            raise FileNotFoundError(
                "Dataset not found:\n"
                f"{self.data_path}"
            )

        with open(
            self.data_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.documents = json.load(file)

        print(
            f"Loaded "
            f"{len(self.documents)} "
            f"documents."
        )

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    def _load_model(self):

        print(
            "\nLoading embedding model:"
        )

        print(
            MODEL_PATH
        )

        # ----------------------------------------------------
        # LOCAL MODEL PATH
        # ----------------------------------------------------

        model_path = Path(MODEL_PATH)

        if model_path.exists():

            self.model = SentenceTransformer(
                str(model_path),
                device="cpu"
            )

        # ----------------------------------------------------
        # HUGGINGFACE MODEL NAME
        # ----------------------------------------------------

        else:

            self.model = SentenceTransformer(
                MODEL_PATH,
                device="cpu"
            )

        print(
            "\nEmbedding model loaded successfully."
        )

    # ========================================================
    # BUILD DOCUMENT TEXT
    # ========================================================

    def _get_document_text(
        self,
        document
    ):

        title = document.get(
            "title",
            ""
        )

        content = document.get(
            "content",
            ""
        )

        category = document.get(
            "category",
            ""
        )

        language = document.get(
            "language",
            ""
        )

        return (

            f"Title: {title}\n"

            f"Category: {category}\n"

            f"Language: {language}\n"

            f"Content: {content}"

        )

    # ========================================================
    # LOAD OR CREATE EMBEDDINGS
    # ========================================================

    def _load_or_create_embeddings(self):

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # LOAD CACHE
        # ----------------------------------------------------

        if (

            EMBEDDINGS_PATH.exists()

            and

            METADATA_PATH.exists()

        ):

            print(
                "\nLoading cached embeddings..."
            )

            self.embeddings = np.load(
                EMBEDDINGS_PATH
            )

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(file)

            # ------------------------------------------------
            # VALIDATE CACHE
            # ------------------------------------------------

            cache_is_valid = (

                metadata.get(
                    "document_count"
                )

                == len(
                    self.documents
                )

                and

                metadata.get(
                    "embedding_dimension"
                )

                == self.embeddings.shape[1]

                and

                metadata.get(
                    "model_path"
                )

                == str(
                    MODEL_PATH
                )

            )

            if cache_is_valid:

                print(
                    "Loaded cached embeddings: "
                    f"{self.embeddings.shape}"
                )

                return

            print(
                "Cache does not match dataset "
                "or model."
            )

            print(
                "Rebuilding embeddings..."
            )

        # ----------------------------------------------------
        # CREATE EMBEDDINGS
        # ----------------------------------------------------

        print(
            "\nCreating document embeddings..."
        )

        texts = [

            self._get_document_text(
                document
            )

            for document in self.documents

        ]

        self.embeddings = self.model.encode(

            texts,

            batch_size=32,

            show_progress_bar=True,

            convert_to_numpy=True,

            normalize_embeddings=True

        )

        # ----------------------------------------------------
        # SAVE EMBEDDINGS
        # ----------------------------------------------------

        np.save(
            EMBEDDINGS_PATH,
            self.embeddings
        )

        metadata = {

            "model_name":
            self.model_name,

            "model_path":
            str(
                MODEL_PATH
            ),

            "document_count":
            len(
                self.documents
            ),

            "embedding_dimension":
            int(
                self.embeddings.shape[1]
            )

        }

        with open(

            METADATA_PATH,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                metadata,

                file,

                indent=2

            )

        print(
            "\nEmbeddings cached successfully."
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(

        self,

        query: str,

        top_k: int = 10,

        language: str = None,

        category: str = None,

        track: str = None,

        **kwargs

    ):

        """
        Semantic vector search.

        Supported optional filters:

        - language
        - category
        - track
        """

        if not query:

            return []

        # ----------------------------------------------------
        # EMBED QUERY
        # ----------------------------------------------------

        query_embedding = self.model.encode(

            query,

            convert_to_numpy=True,

            normalize_embeddings=True

        )

        # ----------------------------------------------------
        # COSINE SIMILARITY
        # ----------------------------------------------------

        scores = np.dot(

            self.embeddings,

            query_embedding

        )

        # ----------------------------------------------------
        # FILTER DOCUMENTS
        # ----------------------------------------------------

        valid_indices = []

        for index, document in enumerate(

            self.documents

        ):

            if (

                language is not None

                and

                document.get(
                    "language"
                )

                != language

            ):

                continue

            if (

                category is not None

                and

                document.get(
                    "category"
                )

                != category

            ):

                continue

            if (

                track is not None

                and

                document.get(
                    "track"
                )

                != track

            ):

                continue

            valid_indices.append(
                index
            )

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if not valid_indices:

            return []

        # ----------------------------------------------------
        # SORT VALID DOCUMENTS
        # ----------------------------------------------------

        valid_scores = [

            (

                index,

                scores[index]

            )

            for index in valid_indices

        ]

        valid_scores.sort(

            key=lambda item: item[1],

            reverse=True

        )

        top_results = valid_scores[:top_k]

        # ----------------------------------------------------
        # BUILD RESULTS
        # ----------------------------------------------------

        results = []

        for index, score in top_results:

            document = dict(

                self.documents[index]

            )

            document["score"] = float(
                score
            )

            results.append(
                document
            )

        return results


# ============================================================
# LOAD SEARCH ENGINE
# ============================================================

_search_engine = None


def load_search_engine():

    global _search_engine

    if _search_engine is None:

        _search_engine = VectorSearchEngine()

    return _search_engine


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "TESTING VECTOR SEARCH ENGINE"
    )

    print("=" * 70)

    engine = load_search_engine()

    results = engine.search(

        query=
        "What are the symptoms of diabetes?",

        top_k=5

    )

    print(
        "\nTop Results:"
    )

    for index, result in enumerate(

        results,

        start=1

    ):

        print(

            f"\n{index}. "
            f"{result.get('title', '')}"

        )

        print(

            f"Score: "
            f"{result.get('score', 0):.4f}"

        )