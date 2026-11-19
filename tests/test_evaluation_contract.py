import csv
import json
from pathlib import Path

import pytest
from scripts.evaluate_bm25 import validate_questions

from app.corpus import sha256

ROOT = Path(__file__).resolve().parents[1]


def test_reject_labels_after_source_changes() -> None:
    doc = {"raw_sha256": "new-version", "text": "evidence"}
    question = {
        "question_id": "q1",
        "group_id": "g1",
        "split": "development",
        "expected_action": "answer_from_snapshot",
        "relevant_url": "source",
        "raw_sha256": "old-version",
        "evidence_start": 0,
        "evidence_end": 8,
        "evidence_sha256": sha256(b"evidence"),
    }
    with pytest.raises(ValueError, match="different source"):
        validate_questions([question], {"source": doc})


def test_reject_paraphrases_across_splits() -> None:
    cases = [
        {
            "question_id": f"q{i}",
            "group_id": "same-intent",
            "split": split,
            "expected_action": "clarify",
        }
        for i, split in enumerate(["development", "test"])
    ]
    with pytest.raises(ValueError, match="leaks"):
        validate_questions(cases, {})


def test_published_qrels_match_manifest_and_eligible_evidence() -> None:
    questions = [
        json.loads(line)
        for line in (ROOT / "data/evaluation/development_questions.jsonl").read_text().splitlines()
    ]
    data = ROOT / "reports/data/20261111_itmo_bachelor_reviewed_v2"
    with (data / "documents.csv").open() as stream:
        docs = {r["url"]: r for r in csv.DictReader(stream)}
    with (data / "chunks.csv").open() as stream:
        fragments = list(csv.DictReader(stream))
    assert len({q["question_id"] for q in questions}) == len(questions)
    for q in questions:
        # These questions have been read and must never be sold as held-out test.
        assert q["split"] == "development"
        assert q["origin"].startswith("model_authored")
        if q["expected_action"] != "answer_from_snapshot":
            continue
        doc = docs[q["relevant_url"]]
        assert doc["raw_sha256"] == q["raw_sha256"]
        assert 0 <= q["evidence_start"] < q["evidence_end"] <= int(doc["chars"])
        assert any(
            r["url"] == q["relevant_url"]
            and r["retrieval_eligible"] == "True"
            and int(r["start"]) <= q["evidence_start"]
            and int(r["end"]) >= q["evidence_end"]
            for r in fragments
        )
