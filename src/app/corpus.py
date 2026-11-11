"""Small, offline-testable building blocks for the HW2 corpus."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from bs4 import BeautifulSoup

USER_AGENT = "UniAssistResearch/0.1 (educational corpus; one request per second)"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_url(url: str) -> str:
    """Drop fragments, preserve query parameters for robots/scope rejection."""
    parts = urlsplit(url)
    path = quote(unquote(parts.path or "/"), safe="/%:@-._~!$&'()*+,;=")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def official_url(url: str) -> bool:
    parts = urlsplit(url)
    return (
        parts.scheme == "https"
        and parts.netloc == "abit.itmo.ru"
        and not parts.query
        and not parts.username
    )


def normalize_text(text: str) -> str:
    """Preserve line order and repeated values (important for tables)."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def html_text(content: bytes) -> tuple[str, str, int, int, list[tuple[str, str]]]:
    soup = BeautifulSoup(content.decode("utf-8", errors="replace"), "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    root = soup.find("main") or soup.find("article") or soup.body or soup
    links = [(str(a.get("href", "")), a.get_text(" ", strip=True)) for a in root.select("a[href]")]
    tables = len(root.select("table"))
    full_chars = len(soup.get_text(" ", strip=True))
    for node in root.select("script, style, noscript, nav, header, footer, svg, form"):
        node.decompose()
    return title, normalize_text(root.get_text("\n", strip=True)), tables, full_chars, links


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    text: str
    words: int


def chunks(text: str, size: int = 180, overlap: int = 30) -> list[Chunk]:
    """Word windows with exact character offsets; PDF callers pass one page at a time."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    words = list(re.finditer(r"\S+", text))
    result = []
    for i in range(0, len(words), size - overlap):
        window = words[i : i + size]
        if not window:
            break
        start, end = window[0].start(), window[-1].end()
        result.append(Chunk(start, end, text[start:end], len(window)))
        if i + size >= len(words):
            break
    return result


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е"))


def word_shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = tokens(text)
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}
