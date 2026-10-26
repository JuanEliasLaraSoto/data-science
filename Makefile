.PHONY: install data scorecard scorecard-german lint format typecheck test check

install:  ## Create the virtual environment and install pre-commit hooks
	uv sync
	uv run pre-commit install

data:  ## Download raw data (Lending Club + FRED) into data/raw
	uv run python scripts/download_data.py

scorecard:  ## Phase 1 on Lending Club
	uv run python scripts/run_scorecard.py

scorecard-german:  ## Phase 1 on the German Credit fallback
	uv run python scripts/download_data.py german-credit
	uv run python scripts/run_scorecard.py --dataset german-credit

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test  ## Everything CI runs
