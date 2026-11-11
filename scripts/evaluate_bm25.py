"""Fixed BM25 retrieval smoke evaluation, not an answer-generation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.corpus import sha256, tokens

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_questions(questions: list[dict], docs: dict[str, dict]) -> None:
    groups: dict[str, str] = {}
    ids = set()
    for q in questions:
        if q["question_id"] in ids:
            raise ValueError("Duplicate question ID")
        ids.add(q["question_id"])
        previous = groups.setdefault(q["group_id"], q["split"])
        if previous != q["split"]:
            raise ValueError("Question group leaks between splits")
        if q["expected_action"] != "answer_from_snapshot":
            continue
        doc = docs[q["relevant_url"]]
        if doc["raw_sha256"] != q["raw_sha256"]:
            raise ValueError("Labels refer to a different source snapshot")
        start, end = q["evidence_start"], q["evidence_end"]
        if not 0 <= start < end <= len(doc["text"]):
            raise ValueError("Invalid evidence offsets")
        if sha256(doc["text"][start:end].encode()) != q["evidence_sha256"]:
            raise ValueError("Evidence hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="20261111_itmo_bachelor_reviewed_v2")
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/evaluation/development_questions.jsonl"
    )
    parser.add_argument("--output-id", default="bm25_dev_v1")
    args = parser.parse_args()
    processed = ROOT / "data/processed" / args.snapshot
    docs = {r["url"]: r for r in load_jsonl(processed / "documents.jsonl") if r["status"] == "ok"}
    fragments = sorted(
        [r for r in load_jsonl(processed / "chunks.jsonl") if r["retrieval_eligible"]],
        key=lambda r: r["chunk_id"],
    )
    questions = load_jsonl(args.questions)
    validate_questions(questions, docs)
    # Frozen before this first run; there is no parameter search on these labels.
    model = BM25Okapi([tokens(r["text"]) for r in fragments], k1=1.5, b=0.75)
    rows, top_rows = [], []
    for q in questions:
        if q["expected_action"] != "answer_from_snapshot":
            continue  # Policy cases require a real routing module; do not invent metrics.
        start = time.perf_counter()
        scores = model.get_scores(tokens(q["question"]))
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], fragments[i]["chunk_id"]))
        latency_ms = (time.perf_counter() - start) * 1000
        evidence_rank = source_rank = None
        for rank, index in enumerate(ranking[:10], 1):
            fragment = fragments[index]
            same_source = fragment["url"] == q["relevant_url"]
            has_evidence = (
                same_source
                and fragment["start"] <= q["evidence_start"]
                and fragment["end"] >= q["evidence_end"]
            )
            if same_source and source_rank is None:
                source_rank = rank
            if has_evidence and evidence_rank is None:
                evidence_rank = rank
            top_rows.append(
                {
                    "question_id": q["question_id"],
                    "rank": rank,
                    "chunk_id": fragment["chunk_id"],
                    "url": fragment["url"],
                    "score": round(float(scores[index]), 6),
                    "has_labeled_evidence": has_evidence,
                }
            )
        rows.append(
            {
                "question_id": q["question_id"],
                "topic": q["topic"],
                "source_hit_at_5": int(source_rank is not None and source_rank <= 5),
                "evidence_hit_at_5": int(evidence_rank is not None and evidence_rank <= 5),
                "evidence_rr_at_10": 1 / evidence_rank if evidence_rank else 0,
                "evidence_rank_at_10": evidence_rank or "",
                "query_ms": round(latency_ms, 4),
            }
        )
    out = ROOT / "reports/baseline" / args.output_id
    out.mkdir(parents=True, exist_ok=False)
    for name, data in [("per_question.csv", rows), ("top10.csv", top_rows)]:
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "snapshot": args.snapshot,
        "split": "development",
        "labeled_questions": len(questions),
        "answerable_questions": len(rows),
        "policy_cases_not_evaluated": len(questions) - len(rows),
        "source_hit_at_5": float(np.mean([r["source_hit_at_5"] for r in rows])),
        "evidence_hit_at_5": float(np.mean([r["evidence_hit_at_5"] for r in rows])),
        "evidence_mrr_at_10": float(np.mean([r["evidence_rr_at_10"] for r in rows])),
        "query_ms_median": float(np.median([r["query_ms"] for r in rows])),
        "query_ms_p95": float(np.quantile([r["query_ms"] for r in rows], 0.95)),
        "index_chunks": len(fragments),
        "parameters": {"k1": 1.5, "b": 0.75},
        "tokenization": "lowercase, ё->е, alphanumeric regex; no stemming; no stopword removal",
        "questions_sha256": sha256(args.questions.read_bytes()),
        "chunks_sha256": sha256((processed / "chunks.jsonl").read_bytes()),
        "python": platform.python_version(),
        "rank_bm25": version("rank-bm25"),
        "limitations": [
            "model-authored development cases; no independent annotator",
            "one known supporting passage per question; qrels not exhaustive",
            "not held-out quality; no generation, escalation or privacy metric",
            "query-only timings from a single small CPU run, not service latency",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
