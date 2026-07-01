from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx

DEFAULT_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

LABEL_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

TYPE_PRIORITY = ["K", "A", "P", "S", "B", "C", "E", "D"]


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def slug_from_url(url):
    path = urlsplit(url).path.rstrip("/")
    return path.split("/")[-1] if path else clean(url).lower().replace(" ", "-")


def parse_minutes(raw):
    if not raw:
        return None
    m = re.search(r"(\d+)", raw)
    return int(m.group(1)) if m else None


def yn_to_bool(value):
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"yes", "y", "true", "1"}:
        return True
    if v in {"no", "n", "false", "0"}:
        return False
    return None


def primary_type(keys):
    codes = []
    for key in keys or []:
        code = LABEL_TO_CODE.get(clean(key))
        if code and code not in codes:
            codes.append(code)
    if not codes:
        return ""
    for code in TYPE_PRIORITY:
        if code in codes:
            return code
    return codes[0]


def build_aliases(name, keys, description):
    aliases = set()
    lower = clean(name).lower()
    aliases.add(name)
    aliases.add(lower.replace("(new)", "").strip())
    aliases.add(lower.replace("shl", "").strip())

    for key in keys or []:
        aliases.add(clean(key).lower())

    blob = f"{lower} {description.lower()}"

    if "opq" in blob:
        aliases.update(["opq", "opq32", "opq32r", "occupational personality questionnaire"])
    if "general skills assessment" in blob or "global skills assessment" in blob or "gsa" in blob:
        aliases.update(["gsa", "general skills assessment", "global skills assessment"])
    if "verify" in blob or "general ability" in blob or "g+" in blob:
        aliases.update(["gsa", "g+", "verify g+", "general ability", "general ability screen"])
    if "java" in blob:
        aliases.update(["java", "core java", "java test", "java developer"])
    if "python" in blob:
        aliases.update(["python", "python test", "python developer"])
    if "sql" in blob:
        aliases.update(["sql", "sql test", "database test"])
    if "excel" in blob:
        aliases.update(["excel", "microsoft excel", "spreadsheet"])
    if "call center" in blob or "contact center" in blob:
        aliases.update(["customer service", "contact center", "call center"])

    return sorted(a for a in aliases if a)


def normalize_item(item):
    name = clean(item.get("name"))
    url = clean(item.get("link") or item.get("url"))

    if not name or not url:
        return None

    if "/products/product-catalog/view/" not in url:
        return None

    if clean(item.get("status", "ok")).lower() not in {"ok", ""}:
        return None

    keys = [clean(k) for k in item.get("keys", []) if clean(k)]
    desc = clean(item.get("description"))

    return {
        "id": clean(item.get("entity_id")) or slug_from_url(url),
        "name": name,
        "url": url.split("?")[0].split("#")[0].rstrip("/") + "/",
        "description": desc,
        "job_levels": [clean(x) for x in item.get("job_levels", []) if clean(x)],
        "languages": [clean(x) for x in item.get("languages", []) if clean(x)],
        "duration_minutes": parse_minutes(item.get("duration") or item.get("duration_raw")),
        "test_type": primary_type(keys),
        "remote_testing": yn_to_bool(item.get("remote")),
        "adaptive": yn_to_bool(item.get("adaptive")),
        "aliases": build_aliases(name, keys, desc),
        "source": "shl_catalog_json",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/catalog.json")
    args = parser.parse_args()

    response = httpx.get(args.url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()

    import json as jsonlib
    raw = jsonlib.loads(response.text, strict=False)
    if not isinstance(raw, list):
        raise ValueError("Expected catalog endpoint to return a JSON list.")

    seen = set()
    normalized = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        rec = normalize_item(item)

        if not rec or rec["url"] in seen:
            continue

        seen.add(rec["url"])
        normalized.append(rec)

    normalized.sort(key=lambda x: x["name"].lower())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"downloaded {len(raw)} raw products")
    print(f"wrote {len(normalized)} normalized assessments to {out}")


if __name__ == "__main__":
    main()
