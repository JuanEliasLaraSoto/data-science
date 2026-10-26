.PHONY: install data lint format typecheck test check

install:  ## Create the virtual environment and install pre-commit hooks
	uv sync
	uv run pre-commit install

data:  ## Download raw data (Lending Club + FRED) into data/raw
	uv run python scripts/download_data.py

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
