import re
from urllib.parse import urlparse

TOKEN_RE = re.compile(r"[a-zA-Z0-9+#.]+")


def normalize_text(text: str) -> str:
    text = text.replace("&", " and ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def slugify_name(name: str) -> str:
    slug = normalize_text(name)
    slug = slug.replace("+", " plus ").replace("#", " sharp ").replace(".", " ")
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or "assessment"


def canonical_url(url: str) -> str:
    # Keep only canonical SHL product URL, removing query strings and fragments.
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") + "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"
