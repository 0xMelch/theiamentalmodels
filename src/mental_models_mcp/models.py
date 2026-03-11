"""Data loading, parsing, and indexing for mental models."""

import json
import re
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"

# Section headers found in the Description field
SECTION_HEADERS = ["Title", "Model", "Heart of the Model", "Math Intuition", "Examples", "Limitations"]
SECTION_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*$",
    re.MULTILINE,
)


def _slugify(discipline: str, name: str) -> str:
    """Create a URL-safe ID from discipline and name."""
    slug = f"{discipline}--{name}".lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


def parse_description(description: str) -> dict:
    """Parse the semi-structured Description field into sections."""
    sections: dict[str, str] = {}
    matches = list(SECTION_PATTERN.finditer(description))

    for i, match in enumerate(matches):
        header = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(description)
        content = description[start:end].strip()
        sections[header] = content

    return sections


def parse_raw_model(raw: dict) -> dict:
    """Parse a single raw model into structured format."""
    sections = parse_description(raw.get("Description", ""))
    name = raw.get("Name", sections.get("Title", "Unknown"))
    discipline = raw.get("Discipline", "Unknown")

    # Extract summary: first 2-3 sentences of the Model section
    model_text = sections.get("Model", "")
    sentences = re.split(r"(?<=[.!?])\s+", model_text)
    summary = " ".join(sentences[:3]).strip() if sentences else ""

    return {
        "id": _slugify(discipline, name),
        "name": name,
        "discipline": discipline,
        "chapter": raw.get("Chapter", "General"),
        "properties": raw.get("Properties", ""),
        "summary": summary,
        "heart": sections.get("Heart of the Model"),
        "math": sections.get("Math Intuition"),
        "examples": sections.get("Examples"),
        "limitations": sections.get("Limitations"),
        "full_description": raw.get("Description", ""),
    }


def load_raw_models(path: Optional[Path] = None) -> list[dict]:
    """Load the raw mental_models.json file."""
    path = path or DATA_DIR / "mental_models.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parsed_models(path: Optional[Path] = None) -> list[dict]:
    """Load pre-parsed models from models_parsed.json."""
    path = path or DATA_DIR / "models_parsed.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_all_models(raw_path: Optional[Path] = None) -> list[dict]:
    """Parse all raw models into structured format."""
    raw_models = load_raw_models(raw_path)
    return [parse_raw_model(m) for m in raw_models]


def save_parsed_models(models: list[dict], path: Optional[Path] = None) -> None:
    """Save parsed models to JSON."""
    path = path or DATA_DIR / "models_parsed.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2, ensure_ascii=False)


class ModelIndex:
    """In-memory index of all parsed mental models."""

    def __init__(self, models: Optional[list[dict]] = None):
        if models is None:
            models = load_parsed_models()
        self._models = models
        self._by_id: dict[str, dict] = {m["id"]: m for m in models}
        self._by_name: dict[str, dict] = {m["name"].lower(): m for m in models}
        self._by_discipline: dict[str, list[dict]] = {}
        for m in models:
            disc = m["discipline"]
            self._by_discipline.setdefault(disc, []).append(m)

    @property
    def all_models(self) -> list[dict]:
        return self._models

    def get_by_id(self, model_id: str) -> Optional[dict]:
        return self._by_id.get(model_id)

    def get_by_name(self, name: str) -> Optional[dict]:
        """Exact or fuzzy match by name."""
        lower = name.lower()
        # Exact match
        if lower in self._by_name:
            return self._by_name[lower]
        # Fuzzy: find the closest match by substring
        for key, model in self._by_name.items():
            if lower in key or key in lower:
                return model
        return None

    def get_disciplines(self) -> list[dict]:
        """Return all disciplines with model counts and chapter breakdowns."""
        result = []
        for disc, models in sorted(self._by_discipline.items()):
            chapters: dict[str, int] = {}
            for m in models:
                ch = m["chapter"]
                chapters[ch] = chapters.get(ch, 0) + 1
            result.append({
                "discipline": disc,
                "model_count": len(models),
                "chapters": chapters,
            })
        return result

    def get_index_listing(self) -> dict[str, list[str]]:
        """Return all model names grouped by discipline (for MCP resource)."""
        return {
            disc: [m["name"] for m in models]
            for disc, models in sorted(self._by_discipline.items())
        }
