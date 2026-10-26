import importlib

import pytest

import credit_risk_lab

MODULES = [
    "data",
    "woe",
    "scorecard",
    "benchmark",
    "metrics",
    "validation",
    "portfolio",
    "stress",
]


def test_version_is_exposed() -> None:
    assert credit_risk_lab.__version__ == "0.1.0"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_and_is_documented(name: str) -> None:
    module = importlib.import_module(f"credit_risk_lab.{name}")
    assert module.__doc__
