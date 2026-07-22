import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

XML_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mplus_topics_2026-07-21.xml"
)


tree = ET.parse(XML_PATH)
root = tree.getroot()

languages = []

for topic in root.findall("health-topic"):

    language = topic.attrib.get(
        "language",
        "UNKNOWN"
    )

    languages.append(language)


print(
    "Language distribution:"
)

for language, count in Counter(
    languages
).items():

    print(
        f"{language}: {count}"
    )