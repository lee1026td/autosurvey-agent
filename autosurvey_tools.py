import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from langchain_community.tools import DuckDuckGoSearchResults


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

MAIN_CONTENT_HINT = re.compile(
    r"article|content|post|entry|story|main|markdown|blog|news|body|text",
    re.IGNORECASE,
)

BOILERPLATE_HINT = re.compile(
    r"nav|menu|header|footer|sidebar|breadcrumb|share|social|comment|ads|promo",
    re.IGNORECASE,
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class FetchedDocument:
    title: str
    url: str
    final_url: str
    domain: str
    text: str
    html: str
    content_type: str


def duckduckgo_search(query: str, num_results: int = 5) -> list[SearchResult]:
    search = DuckDuckGoSearchResults(output_format="list", num_results=num_results)
    raw = search.invoke(query)

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []

    results: list[SearchResult] = []
    for item in raw[:num_results]:
        url = item.get("link") or item.get("url") or ""
        if not url:
            continue
        results.append(
            SearchResult(
                title=item.get("title", "").strip(),
                url=url.strip(),
                snippet=item.get("snippet", "").strip(),
            )
        )
    return results


def fetch_webpage(url: str, timeout_sec: int = 15, max_chars: int = 25000) -> FetchedDocument:
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=timeout_sec, allow_redirects=True)
    response.raise_for_status()

    raw_html = response.text or ""
    soup = BeautifulSoup(raw_html, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    body = soup.body if soup.body else soup
    _strip_noise_tags(body)

    main_node = _select_main_content_node(body)
    text = _extract_meaningful_text(main_node, max_chars=max_chars)
    html = str(main_node)[:max_chars]

    final_url = response.url
    domain = urlparse(final_url).netloc.lower()
    return FetchedDocument(
        title=title,
        url=url,
        final_url=final_url,
        domain=domain,
        text=text,
        html=html,
        content_type=response.headers.get("Content-Type", ""),
    )


def _strip_noise_tags(root: Tag) -> None:
    for tag in root(["script", "style", "noscript", "iframe", "svg", "canvas", "form", "button"]):
        tag.decompose()
    for tag in root(["nav", "header", "footer", "aside"]):
        tag.decompose()


def _candidate_nodes(root: Tag) -> list[Tag]:
    candidates: list[Tag] = []
    for selector in (
        "article",
        "main",
        "[role='main']",
        "section",
        "div",
    ):
        candidates.extend(root.select(selector))
    if not candidates:
        return [root]
    return candidates


def _node_hint_text(node: Tag) -> str:
    attrs = [
        node.get("id", ""),
        " ".join(node.get("class", [])) if node.get("class") else "",
        node.name or "",
    ]
    return " ".join(attrs)


def _content_score(node: Tag) -> float:
    text = node.get_text(" ", strip=True)
    if not text:
        return -1.0

    text_len = len(text)
    if text_len < 120:
        return -1.0

    link_text_len = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
    link_density = (link_text_len / text_len) if text_len else 1.0

    p_count = len(node.find_all("p"))
    h_count = len(node.find_all(["h1", "h2", "h3"]))
    hint = _node_hint_text(node)

    score = float(text_len)
    score += p_count * 80.0
    score += h_count * 25.0
    if MAIN_CONTENT_HINT.search(hint):
        score += 1500.0
    if BOILERPLATE_HINT.search(hint):
        score -= 2000.0
    score -= link_density * 2500.0
    return score


def _select_main_content_node(root: Tag) -> Tag:
    candidates = _candidate_nodes(root)
    best = root
    best_score = _content_score(root)

    for node in candidates:
        score = _content_score(node)
        if score > best_score:
            best = node
            best_score = score
    return best


def _extract_meaningful_text(node: Tag, max_chars: int) -> str:
    parts: list[str] = []
    # Prefer semantically rich blocks first.
    for el in node.find_all(["h1", "h2", "h3", "p", "li", "blockquote", "pre"]):
        segment = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not segment:
            continue

        # Drop menu-like short labels unless they look like a meaningful sentence.
        if len(segment) < 25 and not re.search(r"[.!?]|[0-9]", segment):
            continue
        parts.append(segment)

        if sum(len(p) + 1 for p in parts) >= max_chars:
            break

    if not parts:
        fallback = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        return fallback[:max_chars]

    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(normalize_text(a).split())
    sb = set(normalize_text(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_duplicate_document(new_text: str, existing_texts: list[str], threshold: float = 0.82) -> tuple[bool, float]:
    best = 0.0
    for old_text in existing_texts:
        score = jaccard_similarity(new_text[:6000], old_text[:6000])
        best = max(best, score)
        if score >= threshold:
            return True, score
    return False, best


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
