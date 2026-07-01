"""Scrape SHL Individual Test Solutions into data/catalog.json.

Usage:
    python scripts/scrape_catalog.py --out data/catalog.json

The assignment restricts the system to the SHL Product Catalog and Individual Test
Solutions. The catalog pages have historically supported ?start=<offset>&type=1
for this filter. The scraper still defensively filters detail URLs and validates
that every output URL is a product-catalog/view URL.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

ROOT = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SHLAssessmentRecommender/1.0; +https://example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def canonical_url(href: str) -> str:
    full = urljoin(ROOT, href)
    return full.split("?")[0].split("#")[0].rstrip("/") + "/"


def extract_between(text: str, start: str, end: str | None = None) -> str:
    pattern = rf"{re.escape(start)}\s*(.*?)"
    if end:
        pattern += rf"\s*{re.escape(end)}"
    else:
        pattern += r"$"
    match = re.search(pattern, text, flags=re.I | re.S)
    return clean(match.group(1)) if match else ""


def parse_detail(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = clean(soup.find("h1").get_text(" ")) if soup.find("h1") else ""
    text = clean(soup.get_text("\n"))

    description = extract_between(text, "Description", "Job levels")
    job_levels_raw = extract_between(text, "Job levels", "Languages")
    languages_raw = extract_between(text, "Languages", "Assessment length")

    duration = None
    dur_match = re.search(r"Approximate Completion Time in minutes\s*=\s*(\d+)", text, flags=re.I)
    if not dur_match:
        dur_match = re.search(r"(\d+)\s*minutes", text, flags=re.I)
    if dur_match:
        duration = int(dur_match.group(1))

    test_type = ""
    type_match = re.search(r"Test Type:\s*([A-Z, ]+)", text)
    if type_match:
        codes = [c for c in re.findall(r"[A-Z]", type_match.group(1)) if c in TYPE_LABELS]
        test_type = codes[0] if codes else ""

    return {
        "id": slug_from_url(url),
        "name": title,
        "url": url,
        "description": description,
        "job_levels": [clean(x) for x in job_levels_raw.split(",") if clean(x)],
        "languages": [clean(x) for x in languages_raw.split(",") if clean(x)],
        "duration_minutes": duration,
        "test_type": test_type,
        "remote_testing": None,
        "adaptive": None,
        "aliases": build_aliases(title),
        "source": "shl_catalog",
    }


def build_aliases(name: str) -> list[str]:
    aliases: set[str] = set()
    lower = name.lower()
    aliases.add(name)
    aliases.add(lower.replace("(new)", "").strip())
    aliases.add(lower.replace("shl", "").strip())

    if "opq" in lower:
        aliases.update(["opq", "opq32", "opq32r", "occupational personality questionnaire"])
    if "verify" in lower or "general ability" in lower or "g+" in lower:
        aliases.update(["gsa", "g+", "verify g+", "general ability", "general ability screen"])
    if "java" in lower:
        aliases.update(["java", "core java", "java test"])
    if "sql" in lower:
        aliases.update(["sql", "sql test", "database test"])
    if "excel" in lower:
        aliases.update(["excel", "microsoft excel"])

    return sorted(a for a in aliases if a)


def parse_catalog_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    products: list[dict] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/solutions/products/product-catalog/view/" not in href:
            continue
        url = canonical_url(href)
        if url in seen:
            continue
        seen.add(url)
        name = clean(a.get_text(" "))
        if not name or name.lower() in {"view", "learn more"}:
            # Try nearby row text if the anchor itself is just a button.
            row = a.find_parent("tr") or a.find_parent("li") or a.find_parent("div")
            name = clean(row.get_text(" "))[:160] if row else slug_from_url(url)
        row_text = clean((a.find_parent("tr") or a.find_parent("li") or a).get_text(" "))
        code_candidates = [c for c in re.findall(r"\b[A-Z]\b", row_text) if c in TYPE_LABELS]
        products.append({
            "id": slug_from_url(url),
            "name": name,
            "url": url,
            "description": "",
            "job_levels": [],
            "languages": [],
            "duration_minutes": None,
            "test_type": code_candidates[0] if code_candidates else "",
            "remote_testing": None,
            "adaptive": None,
            "aliases": build_aliases(name),
            "source": "shl_catalog",
        })
    return products


def fetch(client: httpx.Client, url: str) -> str:
    resp = client.get(url, headers=HEADERS, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def scrape(max_pages: int = 60, page_size: int = 12, enrich: bool = True, sleep_sec: float = 0.15) -> list[dict]:
    found: dict[str, dict] = {}
    with httpx.Client(headers=HEADERS) as client:
        empty_pages = 0
        for page in range(max_pages):
            start = page * page_size
            url = f"{CATALOG_URL}?start={start}&type=1"
            try:
                html = fetch(client, url)
            except Exception as exc:
                print(f"WARN: failed catalog page {url}: {exc}")
                empty_pages += 1
                if empty_pages >= 3:
                    break
                continue
            products = parse_catalog_page(html)
            new_count = 0
            for p in products:
                if p["url"] not in found:
                    found[p["url"]] = p
                    new_count += 1
            print(f"page={page:02d} start={start:03d} products={len(products)} new={new_count} total={len(found)}")
            if new_count == 0:
                empty_pages += 1
                if empty_pages >= 3:
                    break
            else:
                empty_pages = 0
            time.sleep(sleep_sec)

        if enrich:
            urls = list(found.keys())
            for i, url in enumerate(urls, 1):
                try:
                    detail = parse_detail(fetch(client, url), url)
                    # Keep catalog-page fields if detail missed them.
                    merged = found[url] | {k: v for k, v in detail.items() if v not in ["", [], None]}
                    if detail.get("duration_minutes") is None:
                        merged["duration_minutes"] = found[url].get("duration_minutes")
                    found[url] = merged
                except Exception as exc:
                    print(f"WARN: failed detail {url}: {exc}")
                if i % 25 == 0:
                    print(f"enriched {i}/{len(urls)}")
                time.sleep(sleep_sec)

    return list(found.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/catalog.json")
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--no-enrich", action="store_true")
    args = parser.parse_args()

    data = scrape(max_pages=args.max_pages, enrich=not args.no_enrich)
    data = sorted(data, key=lambda x: x["name"].lower())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(data)} assessments to {out}")


if __name__ == "__main__":
    main()
