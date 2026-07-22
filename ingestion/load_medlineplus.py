import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from bs4 import BeautifulSoup

from schema import MedicalDocument


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XML_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mplus_topics_2026-07-21.xml"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "medlineplus_documents.json"
)


# =====================================================
# HELPERS
# =====================================================

def clean_html_text(raw_text: str) -> str:
    """
    Convert HTML-encoded content into clean plain text.
    """

    if not raw_text:
        return ""

    # Decode HTML entities such as &lt;p&gt;
    decoded_text = html.unescape(raw_text)

    # Remove HTML tags
    soup = BeautifulSoup(
        decoded_text,
        "html.parser"
    )

    text = soup.get_text(
        " ",
        strip=True
    )

    # Normalize multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =====================================================
# LOAD MEDLINEPLUS DOCUMENTS
# =====================================================

def load_medlineplus_documents():
    """
    Load MedlinePlus XML and convert every topic
    into a MedicalDocument object.
    """

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"MedlinePlus XML file not found: {XML_PATH}"
        )

    print(
        "Loading MedlinePlus XML..."
    )

    tree = ET.parse(
        XML_PATH
    )

    root = tree.getroot()

    documents = []

    for index, topic in enumerate(
        root.findall("health-topic")
    ):

        # ---------------------------------------------
        # Extract title
        # ---------------------------------------------

        title = topic.attrib.get(
            "title",
            ""
        ).strip()

        # ---------------------------------------------
        # Extract source URL
        # ---------------------------------------------

        url = topic.attrib.get(
            "url",
            ""
        ).strip()

        # ---------------------------------------------
        # Extract full summary
        # ---------------------------------------------

        summary_element = topic.find(
            "full-summary"
        )

        raw_summary = ""

        if summary_element is not None:

            raw_summary = "".join(
                summary_element.itertext()
            )

        content = clean_html_text(
            raw_summary
        )

        # ---------------------------------------------
        # Skip invalid documents
        # ---------------------------------------------

        if not title or not content:

            continue

        # ---------------------------------------------
        # Extract category
        # ---------------------------------------------

        group_element = topic.find(
            "group"
        )

        category = "General"

        if group_element is not None:

            category = (
                group_element.text
                or "General"
            )

        # ---------------------------------------------
        # Extract language
        # ---------------------------------------------

        language_code = {
            "English": "en",
            "Spanish": "es",
        }.get(
            topic.attrib.get(
                "language",
                ""
            ),
            "unknown"
        )

        # ---------------------------------------------
        # Create unified document
        # ---------------------------------------------

        document = MedicalDocument(

            id=f"medlineplus_{index:04d}",

            title=title,

            content=content,

            source="MedlinePlus",

            source_url=url,

            category=category.strip(),

            difficulty="beginner",

            audience=[
                "general",
                "student"
            ],

            language=language_code,
        )

        documents.append(
            document
        )

    print(
        f"Loaded {len(documents)} documents."
    )

    return documents


# =====================================================
# SAVE DOCUMENTS
# =====================================================

def save_documents(documents):
    """
    Save processed medical documents as JSON.
    """

    data = [

        document.to_dict()

        for document in documents

    ]

    with open(

        OUTPUT_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            data,

            file,

            ensure_ascii=False,

            indent=2

        )

    print(
        f"Saved {len(data)} documents to:"
    )

    print(
        OUTPUT_PATH
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    documents = (
        load_medlineplus_documents()
    )

    save_documents(
        documents
    )

    print(
        "\nFirst document:"
    )

    print(
        documents[0].to_dict()
    )