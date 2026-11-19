"""Compare retrieval variants on development at the same context budget."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi
from scripts.evaluate_bm25 import load_jsonl, validate_questions

from app.corpus import chunks, sha256, tokens

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 900
SNAPSHOT = "20261111_itmo_bachelor_reviewed_v2"
SCORES_URL = "https://abit.itmo.ru/page/73"
SCORE_ROW = re.compile(
    r"^Общеобразовательный предмет\n([^\n]+)\nМинимальное количество баллов\n(\d+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Passage:
    url: str
    start: int
    end: int
    text: str


def windows(
    url: str, text: str, size: int, overlap: int, *, start: int = 0, title: str = ""
) -> list[Passage]:
    return [
        Passage(url, start + c.start, start + c.end, f"{title}\n{c.text}" if title else c.text)
        for c in chunks(text, size, overlap)
    ]


def program_sections(docs: dict[str, dict], path: Path) -> list[Passage]:
    """Only index the manually reviewed descriptive range of each passport."""
    result = []
    with path.open() as stream:
        for review in csv.DictReader(stream):
            url = review["url"]
            doc = docs[url]
            start, end = int(review["start"]), int(review["end"])
            section = doc["text"][start:end]
            if (
                doc["raw_sha256"] != review["raw_sha256"]
                or not 0 <= start < end <= len(doc["text"])
                or sha256(section.encode()) != review["text_sha256"]
            ):
                raise ValueError(f"Program section review no longer applies: {url}")
            result.extend(windows(url, section, 180, 30, start=start, title=doc["title"]))
    return result


def score_rows(doc: dict, source_sha256: str) -> list[Passage]:
    """Keep the two table scopes distinct and preserve the source row offsets."""
    text = doc["text"]
    exception = "\n*Минимальные баллы по предметам"
    _, sep, _ = text.partition(exception)
    if not sep:
        raise ValueError("Missing table scope separator")
    boundary = text.index(exception)
    rows = []
    for match in SCORE_ROW.finditer(text):
        scope = "Бакалавриат" if match.start() < boundary else "Компьютерные технологии, 01.03.02"
        subject, score = match.groups()
        card = f"Минимальные баллы ЕГЭ, 2026, {scope}. {subject}: {score} баллов."
        rows.append(Passage(doc["url"], match.start(), match.end(), card))
    if len(rows) != 15 or doc["raw_sha256"] != source_sha256:
        raise ValueError("The score table must be reviewed for this snapshot")
    return rows


def evidence_covered(passages: list[Passage], q: dict) -> bool:
    # Require the full span, possibly covered by several overlapping fragments.
    end = q["evidence_start"]
    for p in sorted((p for p in passages if p.url == q["relevant_url"]), key=lambda p: p.start):
        if p.start > end:
            break
        end = max(end, p.end)
    return end >= q["evidence_end"]


def evaluate(label: str, passages: list[Passage], questions: list[dict]) -> list[dict]:
    ranked = sorted(passages, key=lambda p: (p.url, p.start, p.end))
    model = BM25Okapi([tokens(p.text) for p in ranked], k1=1.5, b=0.75)
    output = []
    for q in questions:
        scores = model.get_scores(tokens(q["question"]))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        delivered = []
        used = 0
        for i in order:
            size = len(ranked[i].text.split())
            if used + size > BUDGET:
                break
            used += size
            delivered.append(ranked[i])
        q_id = q["question_id"]
        output.append(
            {
                "variant": label,
                "question_id": q_id,
                "topic": q["topic"],
                "indexed_passages": len(ranked),
                "context_words": used,
                "evidence_in_index": int(evidence_covered(ranked, q)),
                "evidence_in_single_indexed_passage": int(
                    any(
                        p.url == q["relevant_url"]
                        and p.start <= q["evidence_start"]
                        and p.end >= q["evidence_end"]
                        for p in ranked
                    )
                ),
                "evidence_in_context": int(evidence_covered(delivered, q)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = ROOT / "data/processed" / SNAPSHOT
    source = ROOT / "data/evaluation/development_questions.jsonl"
    records = load_jsonl(snapshot / "documents.jsonl")
    docs = {d["url"]: d for d in records if d["status"] == "ok"}
    indexed = {r["url"] for r in load_jsonl(snapshot / "chunks.jsonl") if r["retrieval_eligible"]}
    questions = load_jsonl(source)
    validate_questions(questions, docs)
    questions = [q for q in questions if q["expected_action"] == "answer_from_snapshot"]
    variants = {}
    for size, overlap in ((100, 20), (180, 30), (300, 50)):
        variants[f"words_{size}"] = [
            p for url in sorted(indexed) for p in windows(url, docs[url]["text"], size, overlap)
        ]
    base = variants["words_180"]
    variants["program_sections"] = base + program_sections(
        docs, ROOT / "data/sources/program_sections.csv"
    )
    # Version the narrow parser against the exact annotated source snapshot.
    score_doc = docs[SCORES_URL]
    with (ROOT / "reports/data" / SNAPSHOT / "documents.csv").open() as stream:
        source_hash = next(
            r["raw_sha256"] for r in csv.DictReader(stream) if r["url"] == SCORES_URL
        )
    table = score_rows(score_doc, source_hash)
    variants["score_table"] = [p for p in base if p.url != SCORES_URL] + table
    rows = [row for name, index in variants.items() for row in evaluate(name, index, questions)]
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
