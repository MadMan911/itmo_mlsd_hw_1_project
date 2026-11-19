.PHONY: install lint format test collect prepare eda notebook baseline

install:
	uv sync --group eda --locked
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run --group eda mypy src

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run --group eda pytest

SNAPSHOT ?= 20261111_itmo_bachelor_reviewed_v2
BASELINE_RUN ?= bm25_dev_v2
RAW_SNAPSHOT ?= 20261111_itmo_bachelor_v1

collect:
	uv run --group eda python scripts/collect_corpus.py --registry data/sources/registry.csv

prepare:
	uv run --group eda python scripts/prepare_corpus.py --snapshot data/raw/$(RAW_SNAPSHOT) --review data/sources/review.csv --output-id $(SNAPSHOT)

eda:
	uv run --group eda python scripts/run_eda.py --snapshot $(SNAPSHOT)

notebook:
	uv run --group eda python scripts/execute_notebook.py

baseline:
	uv run --group eda python scripts/evaluate_bm25.py --snapshot $(SNAPSHOT) --output-id $(BASELINE_RUN)
