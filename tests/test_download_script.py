"""Tests for ``scripts/download_data.py`` (source selection and exit codes)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from credit_risk_lab import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def record(name: str) -> Any:
        def fake(*args: Any, **kwargs: Any) -> None:
            seen.append(name)

        return fake

    monkeypatch.setattr(data, "download_lending_club", record("lending-club"))
    monkeypatch.setattr(data, "download_fred", record("fred"))
    monkeypatch.setattr(data, "download_german_credit", record("german-credit"))
    return seen


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ["lending-club", "fred"]),
        (["german-credit"], ["german-credit"]),
        (["all"], ["lending-club", "fred", "german-credit"]),
    ],
)
def test_source_selection(
    calls: list[str], tmp_path: Path, argv: list[str], expected: list[str]
) -> None:
    assert load_script().main([*argv, "--raw-dir", str(tmp_path)]) == 0
    assert calls == expected


def test_failure_sets_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def failing(*args: Any, **kwargs: Any) -> None:
        raise data.DataDownloadError("blocked")

    monkeypatch.setattr(data, "download_fred", failing)
    assert load_script().main(["fred", "--raw-dir", str(tmp_path)]) == 1


def test_unknown_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as info:
        load_script().main(["kaggle", "--raw-dir", str(tmp_path)])
    assert info.value.code == 2
