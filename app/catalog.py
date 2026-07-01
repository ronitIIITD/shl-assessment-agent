import json
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field


TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


class Assessment(BaseModel):
    id: str
    name: str
    url: str
    description: str = ""
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration_minutes: int | None = None
    test_type: str = ""
    remote_testing: bool | None = None
    adaptive: bool | None = None
    aliases: list[str] = Field(default_factory=list)
    source: str = "shl_catalog"

    @property
    def test_type_label(self) -> str:
        return TEST_TYPE_LABELS.get(self.test_type, self.test_type)

    @property
    def search_text(self) -> str:
        parts: list[str] = [
            self.name,
            self.description,
            self.test_type,
            self.test_type_label,
            " ".join(self.job_levels),
            " ".join(self.languages),
            " ".join(self.aliases),
        ]
        return " ".join(p for p in parts if p)


def _default_catalog_path() -> Path:
    return Path(os.getenv("CATALOG_PATH", "data/catalog.json"))


def load_catalog(path: str | Path | None = None) -> list[Assessment]:
    catalog_path = Path(path) if path else _default_catalog_path()
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found at {catalog_path}. Run `python scripts/scrape_catalog.py` first "
            "or set CATALOG_PATH to a valid catalog JSON."
        )

    raw: list[dict[str, Any]] = json.loads(catalog_path.read_text(encoding="utf-8"))
    assessments: list[Assessment] = []
    seen_urls: set[str] = set()
    for item in raw:
        try:
            assessment = Assessment(**item)
        except Exception:
            continue
        if not assessment.name or not assessment.url:
            continue
        if "/products/product-catalog/view/" not in assessment.url:
            continue
        if assessment.url in seen_urls:
            continue
        seen_urls.add(assessment.url)
        assessments.append(assessment)
    if not assessments:
        raise ValueError("Catalog loaded, but no valid SHL assessment records were found.")
    return assessments


def allowed_urls(catalog: list[Assessment]) -> set[str]:
    return {item.url for item in catalog}
