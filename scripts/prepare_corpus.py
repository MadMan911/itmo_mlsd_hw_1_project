"""Prepare local text/chunks and publish text-free corpus diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from app.corpus import chunks, sha256, word_shingles

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output-id")
    args = parser.parse_args()
    records = [
        json.loads(line) for line in (args.snapshot / "documents.jsonl").read_text().splitlines()
    ]
    output_id = args.output_id or args.snapshot.name
    reviews = {}
    if args.review:
        with args.review.open() as stream:
            reviews = {r["url"]: r for r in csv.DictReader(stream)}
    out = ROOT / "data/processed" / output_id
    out.mkdir(parents=True, exist_ok=False)
    diagnostics = ROOT / "reports/data" / output_id
    diagnostics.mkdir(parents=True, exist_ok=False)
    docs, fragment_metadata, fragments = [], [], []
    seen_hashes = {}
    for record in records:
        if record["status"] != "ok":
            continue
        review = reviews.get(record["url"])
        if review:
            if record["raw_sha256"] != review["raw_sha256"]:
                raise ValueError("Review refers to a different source version")
            record["freshness"] = review["freshness"]
            record["education_level"] = review["education_level"]
        doc = {k: v for k, v in record.items() if k not in {"text", "pages", "pdf_links"}}
        doc["source_id"] = sha256(record["url"].encode())[:16]
        doc["year_mentions"] = ";".join(doc["year_mentions"])
        doc["exact_duplicate_of"] = seen_hashes.get(doc["text_sha256"], "")
        seen_hashes.setdefault(doc["text_sha256"], doc["source_id"])
        # Contacts below are automated flags, not a privacy clearance.
        doc["email_count"] = len(set(re.findall(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", record["text"])))
        doc["phone_candidate_count"] = len(
            set(
                re.findall(
                    r"(?:\+7|8)\s*\(?\d{3}\)?[\s-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}", record["text"]
                )
            )
        )
        doc["pdf_link_count"] = len(record["pdf_links"])
        doc["low_text"] = doc["words"] < 80
        doc["retrieval_eligible"] = (
            doc.get("scope") == "include"
            and not doc["exact_duplicate_of"]
            and not doc["low_text"]
            and doc.get("freshness") not in {"historical", "mixed_campaign"}
        )
        docs.append(doc)
        pages = record["pages"] if doc["format"] == "pdf" else [record["text"]]
        for page_index, text in enumerate(pages, 1):
            for i, chunk in enumerate(chunks(text)):
                chunk_id = f"{doc['source_id']}:p{page_index}:c{i}"
                metadata = {
                    "chunk_id": chunk_id,
                    "source_id": doc["source_id"],
                    "url": doc["url"],
                    "title": doc["title"],
                    "format": doc["format"],
                    "topic": doc["topic"],
                    "scope": doc["scope"],
                    "retrieval_eligible": doc["retrieval_eligible"],
                    "page": page_index if doc["format"] == "pdf" else 0,
                    "start": chunk.start,
                    "end": chunk.end,
                    "words": chunk.words,
                    "chars": len(chunk.text),
                    "text_sha256": sha256(chunk.text.encode()),
                }
                fragment_metadata.append(metadata)
                fragments.append({**metadata, "text": chunk.text})
    write_csv(diagnostics / "documents.csv", docs)
    write_csv(diagnostics / "chunks.csv", fragment_metadata)
    write_csv(diagnostics / "failures.csv", [r for r in records if r["status"] != "ok"])
    pairs = []
    successful = [r for r in records if r["status"] == "ok"]
    shingles = [word_shingles(r["text"]) for r in successful]
    for i, left in enumerate(successful):
        for j in range(i + 1, len(successful)):
            right = successful[j]
            union = shingles[i] | shingles[j]
            similarity = len(shingles[i] & shingles[j]) / len(union) if union else 0
            if similarity >= 0.35:
                pairs.append(
                    {
                        "left_url": left["url"],
                        "right_url": right["url"],
                        "jaccard_5gram": round(similarity, 4),
                    }
                )
    write_csv(diagnostics / "near_duplicates.csv", pairs)
    # Compare fragmentation cost without tuning on evaluation questions.
    sizes = []
    for size, overlap in [(100, 20), (180, 30), (300, 50)]:
        sizes.append(
            {
                "size_words": size,
                "overlap_words": overlap,
                "chunks": sum(
                    len(chunks(p, size, overlap))
                    for r in successful
                    for p in (r["pages"] or [r["text"]])
                ),
            }
        )
    write_csv(diagnostics / "chunking_comparison.csv", sizes)
    with (out / "chunks.jsonl").open("w") as stream:
        for fragment in fragments:
            stream.write(json.dumps(fragment, ensure_ascii=False) + "\n")
    with (out / "documents.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (diagnostics / "run.json").write_text(
        json.dumps(
            {
                "snapshot": args.snapshot.name,
                "output_id": output_id,
                "review_sha256": sha256(args.review.read_bytes()) if args.review else None,
                "documents_sha256": sha256((args.snapshot / "documents.jsonl").read_bytes()),
                "chunk_size_words": 180,
                "overlap_words": 30,
                "document_count": len(docs),
                "chunk_count": len(fragments),
                "topic_counts": dict(Counter(d["topic"] for d in docs)),
                "privacy": "No full source text or contact values in public diagnostics",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "diagnostics": str(diagnostics),
                "documents": len(docs),
                "chunks": len(fragments),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
