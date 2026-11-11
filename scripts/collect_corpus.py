"""Sequential, robots-aware collection. Never overwrite a previous snapshot.

Initial exploration: --discover (sitemap public Russian information pages).
Frozen reproduction: --registry data/sources/registry.csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from pypdf import PdfReader

from app.corpus import USER_AGENT, canonical_url, html_text, normalize_text, official_url, sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://abit.itmo.ru"
MAX_BYTES = 20 * 1024 * 1024


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


class Fetcher:
    def __init__(self, raw: Path) -> None:
        self.raw = raw
        self.client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=40)
        self.last_request = 0.0
        self.robots_rules: list[tuple[int, bool, re.Pattern[str]]] = []
        self.calls: list[dict[str, Any]] = []

    def allowed(self, url: str) -> bool:
        # Longest rule wins; equal-length Allow wins. This collector
        # supports the default (*) group used by this host, not arbitrary sites.
        path = urlsplit(url).path
        matches = [(length, allow) for length, allow, p in self.robots_rules if p.match(path)]
        return not matches or max(matches)[1]

    def get(self, url: str, *, check_robots: bool = True) -> httpx.Response:
        url = canonical_url(url)
        for _ in range(6):
            if not official_url(url):
                raise ValueError(f"URL outside public allowlist: {url}")
            if check_robots and not self.allowed(url):
                raise ValueError(f"robots.txt disallows {url}")
            time.sleep(max(0, 1.0 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            with self.client.stream("GET", url) as stream:
                body = bytearray()
                for block in stream.iter_bytes():
                    body.extend(block)
                    if len(body) > MAX_BYTES:
                        raise ValueError("Response exceeds 20 MiB")
                response = httpx.Response(
                    stream.status_code,
                    # iter_bytes already decoded gzip; do not decode a second time.
                    headers={
                        k: v
                        for k, v in stream.headers.items()
                        if k not in {"content-encoding", "content-length"}
                    },
                    content=bytes(body),
                    request=stream.request,
                )
            self.calls.append(
                {
                    "url": url,
                    "status": response.status_code,
                    "bytes": len(body),
                    "at": datetime.now(UTC).isoformat(),
                }
            )
            if response.is_redirect:
                url = canonical_url(urljoin(url, response.headers["location"]))
                continue
            response.raise_for_status()
            return response
        raise ValueError("Too many redirects")

    def setup(self) -> list[str]:
        response = self.get(BASE + "/robots.txt", check_robots=False)
        (self.raw / "robots.txt").write_bytes(response.content)
        self.parse_robots(response.text)
        response = self.get(BASE + "/sitemap.xml")
        (self.raw / "sitemap.xml").write_bytes(response.content)
        return [
            str(e.text) for e in ET.fromstring(response.content).iter() if e.tag.endswith("loc")
        ]

    def parse_robots(self, text: str) -> None:
        active = False
        self.robots_rules = []
        for line in text.splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            value = value.split("#", 1)[0].strip()
            if key.lower() == "user-agent":
                active = value == "*"
            elif active and key.lower() in {"allow", "disallow"} and value.startswith("/"):
                pattern = re.escape(value).replace(r"\*", ".*")
                if pattern.endswith(r"\$"):
                    pattern = pattern[:-2] + "$"
                self.robots_rules.append(
                    (len(value.encode()), key.lower() == "allow", re.compile("^" + pattern))
                )


def extract(response: httpx.Response, source: dict[str, str], raw: Path) -> dict[str, Any]:
    content = response.content
    digest = sha256(content)
    pdf = content.startswith(b"%PDF-")
    suffix = ".pdf" if pdf else ".html"
    raw_path = raw / (digest + suffix)
    if not raw_path.exists():
        raw_path.write_bytes(content)
    pages: list[str] = []
    pdf_links: list[dict[str, str]] = []
    tables, full_chars = 0, 0
    if pdf:
        reader = PdfReader(io.BytesIO(content))
        pages = [normalize_text(page.extract_text() or "") for page in reader.pages]
        title = source.get("title") or Path(urlsplit(str(response.url)).path).name
        text = "\n".join(pages)
        full_chars = len(text)
    else:
        title, text, tables, full_chars, links = html_text(content)
        for href, label in links:
            link = canonical_url(urljoin(str(response.url), href))
            if official_url(link) and urlsplit(link).path.lower().endswith(".pdf"):
                pdf_links.append({"url": link, "title": label, "parent_url": source["url"]})
    lines = text.splitlines()
    duplicate_line_chars = sum(
        len(line) for i, line in enumerate(lines) if len(line) >= 30 and line in lines[:i]
    )
    return {
        **source,
        "status": "ok",
        "title": title,
        "final_url": str(response.url),
        "http_status": response.status_code,
        "fetched_at": datetime.now(UTC).isoformat(),
        "format": "pdf" if pdf else "html",
        "raw_sha256": digest,
        "text_sha256": sha256(text.encode()),
        "raw_file": str(raw_path.relative_to(ROOT)),
        "bytes": len(content),
        "chars": len(text),
        "words": len(text.split()),
        "full_chars": full_chars,
        "html_tables": tables,
        "pdf_pages": len(pages),
        "empty_pdf_pages": sum(len(p) < 50 for p in pages),
        "replacement_chars": text.count("\ufffd"),
        "repeated_line_char_fraction": round(duplicate_line_chars / max(1, len(text)), 4),
        "year_mentions": sorted(set(re.findall(r"\b20[12][0-9]\b", text))),
        "text": text,
        "pages": pages,
        "pdf_links": list({link["url"]: link for link in pdf_links}.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if args.discover == bool(args.registry):
        parser.error("Choose exactly one of --discover / --registry")
    raw = ROOT / "data/raw" / args.run_id
    raw.mkdir(parents=True, exist_ok=False)
    fetcher = Fetcher(raw)
    sitemap = fetcher.setup()
    if args.discover:
        urls = [u for u in sitemap if urlsplit(u).path.startswith(("/page/", "/program/bachelor/"))]
        urls += [BASE + p for p in ["/bachelor", "/faq", "/contacts", "/exam/bachelor"]]
        sources = [
            {"url": u, "topic": "candidate", "scope": "pending"}
            for u in sorted(set(urls))
            if official_url(u) and not urlsplit(u).path.startswith("/en/")
        ]
    else:
        assert args.registry is not None
        with args.registry.open() as stream:
            sources = list(csv.DictReader(stream))
    records = []
    for i, source in enumerate(sources, 1):
        try:
            record = extract(fetcher.get(source["url"]), source, raw)
        except Exception as exc:  # Record each failure, do not hide a partial collection.
            record = {**source, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        print(
            f"{i}/{len(sources)} {record['status']} {source['url']} {record.get('title', '')[:65]}",
            flush=True,
        )
    with (raw / "documents.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(raw / "requests.json", fetcher.calls)
    write_json(
        raw / "run.json",
        {
            "run_id": args.run_id,
            "command": sys.argv,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
            "started_from": "hw_1 / 4b0d953",
            "source_count": len(sources),
            "success_count": sum(r["status"] == "ok" for r in records),
            "packages": {p: version(p) for p in ["httpx", "beautifulsoup4", "pypdf"]},
            "robots_sha256": sha256((raw / "robots.txt").read_bytes()),
            "registry_sha256": sha256(args.registry.read_bytes()) if args.registry else None,
            "collection_policy": (
                "one request per second; no authentication; no external API; no retries"
            ),
        },
    )
    print(f"Snapshot: {raw}")


if __name__ == "__main__":
    main()
