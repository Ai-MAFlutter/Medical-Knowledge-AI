from dataclasses import dataclass, field
from typing import List


@dataclass
class MedicalDocument:
    """
    Unified document schema for all medical knowledge sources.
    """

    id: str
    title: str
    content: str
    source: str
    source_url: str = ""
    category: str = "general"
    difficulty: str = "beginner"
    audience: List[str] = field(default_factory=list)
    language: str = "en"

    def to_dict(self) -> dict:
        """
        Convert the document into a dictionary.
        """
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "source_url": self.source_url,
            "category": self.category,
            "difficulty": self.difficulty,
            "audience": self.audience,
            "language": self.language,
        }