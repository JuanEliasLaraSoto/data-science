"""End-to-end test of ``scripts/run_scorecard.py`` on synthetic files in the real formats."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from credit_risk_lab import benchmark, data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_scorecard.py"
SMALL_GRIDS = {
    "Decision tree": {"model__max_depth": [3]},
    "Random forest": {"model__max_depth": [4], "model__n_estimators": [10]},
    "Gradient boosting": {"model__max_iter": [20]},
}


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_scorecard", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_lending_club(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.period_range("2012-01", "2015-12", freq="M")
    issue = rng.choice(months.astype(str), n)
    fico_low = rng.integers(66, 80, n) * 10.0
    dti = rng.uniform(0, 35, n)
    income = rng.lognormal(11, 0.5, n)
    amount = rng.integers(10, 350, n) * 100.0
    purpose = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "car"], n)
    log_odds = -1.7 - 0.012 * (fico_low - 700) + 0.04 * (dti - 17) + 0.3 * (amount / income)
    bad = rng.random(n) < 1 / (1 + np.exp(-log_odds))
    status = np.where(bad, "Charged Off", "Fully Paid")
    status[:30] = "Current"
    return pd.DataFrame(
        {
            "id": np.arange(n).astype(str),
            "loan_amnt": amount,
            "funded_amnt": amount,  # leaky: must be dropped on load
            "term": np.where(np.arange(n) % 20 == 0, " 60 months", " 36 months"),
            "int_rate": 12.5,  # excluded: incumbent pricing
            "grade": "B",
            "emp_length": rng.choice(["< 1 year", "3 years", "10+ years", "n/a"], n),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
            "annual_inc": income,
            "verification_status": rng.choice(["Verified", "Not Verified"], n),
            "issue_d": pd.PeriodIndex(issue, freq="M").strftime("%b-%Y"),
            "loan_status": status,
            "purpose": purpose,
            "dti": dti,
            "earliest_cr_line": rng.choice(["Jan-1995", "Jun-2005", "Mar-2010"], n),
            "fico_range_low": fico_low,
            "fico_range_high": fico_low + 4,
            "revol_util": np.round(rng.uniform(0, 100, n), 1),
            "inq_last_6mths": rng.poisson(1.0, n),
            "total_pymnt": np.where(bad, 0.5, 1.1) * amount,  # leaky
            "recoveries": np.where(bad, 100.0, 0.0),  # leaky
            "application_type": "Individual",
        }
    )


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    path = raw / "lending_club" / data.LENDING_CLUB_FILE
    path.parent.mkdir(parents=True)
    synthetic_lending_club().to_csv(path, index=False)
    return raw


def test_lending_club_pipeline(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()

    def small_challengers(X: pd.DataFrame, y: Any) -> dict[str, benchmark.ChallengerResult]:
        return benchmark.fit_challengers(X, y, param_grids=SMALL_GRIDS)

    monkeypatch.setattr(script, "fit_challengers", small_challengers)
    out, processed = tmp_path / "reports", tmp_path / "processed"
    argv = [
        "--raw-dir", str(raw_dir), "--out-dir", str(out), "--processed-dir", str(processed),
        "--dev-start", "2012-01", "--dev-end", "2014-12",
    ]  # fmt: skip
    assert script.main(argv) == 0

    report = out / "lending_club"
    performance = pd.read_csv(report / "tables" / "performance.csv")
    assert set(performance["sample"]) == {"train", "test", "oot"}
    assert performance["model"].nunique() == 4
    card_rows = performance[performance["model"] == "Scorecard (WoE logit)"]
    assert (card_rows["gini"] > 0.1).all()

    comparison = pd.read_csv(report / "tables" / "model_comparison.csv")
    assert len(comparison) == 6  # 3 challengers x (test, oot)
    assert comparison["p_value"].between(0, 1).all()
    assert (performance["gini_ci_low"] < performance["gini"]).all()

    iv = pd.read_csv(report / "tables" / "iv_table.csv")
    for leaky in ("funded_amnt", "total_pymnt", "recoveries", "int_rate", "grade"):
        assert leaky not in set(iv["feature"])
    assert {"fico_score", "dti", "loan_to_income"} <= set(iv["feature"])

    for figure in (
        "default_rate_by_vintage",
        "roc_by_model",
        "calibration_by_sample",
        "ks_curve",
        "scorecard_binnings",
    ):
        assert (report / "figures" / f"{figure}.png").exists()
    summary = (report / "summary.md").read_text()
    assert "## Performance" in summary
    assert "Gradient boosting" in summary

    with (processed / "lending_club" / "scorecard.pkl").open("rb") as fh:
        card = pickle.load(fh)
    scored = pd.read_parquet(processed / "lending_club" / "scored_samples.parquet")
    oot = scored[scored["sample"] == "oot"]
    assert oot["vintage"].str.startswith("2015").all()
    np.testing.assert_allclose(card.scaling.score_from_pd(oot["pd"]), oot["score"])


def test_german_credit_pipeline_without_benchmark(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 1000
    duration = rng.integers(6, 60, n)
    checking = rng.choice(["<0", "0<=X<200", "no checking"], n)
    log_odds = -1.2 + 0.03 * (duration - 20) + np.where(checking == "<0", 0.8, -0.3)
    bad = rng.random(n) < 1 / (1 + np.exp(-log_odds))
    raw = pd.DataFrame(
        {
            "checking_status": checking,
            "duration": duration,
            "credit_amount": rng.integers(500, 15000, n),
            "personal_status": "male single",
            "foreign_worker": "yes",
            "class": np.where(bad, "bad", "good"),
        }
    )
    path = tmp_path / "raw" / "german_credit" / "credit_g.csv"
    path.parent.mkdir(parents=True)
    raw.to_csv(path, index=False)
    script = load_script()
    argv = [
        "--dataset", "german-credit", "--raw-dir", str(tmp_path / "raw"),
        "--out-dir", str(tmp_path / "reports"), "--processed-dir", str(tmp_path / "processed"),
        "--skip-benchmark",
    ]  # fmt: skip
    assert script.main(argv) == 0
    performance = pd.read_csv(tmp_path / "reports" / "german_credit" / "tables" / "performance.csv")
    assert set(performance["sample"]) == {"train", "test"}
    summary = (tmp_path / "reports" / "german_credit" / "summary.md").read_text()
    assert "random split only" in summary
    assert "personal_status" not in summary


def test_missing_input_file_returns_error(tmp_path: Path) -> None:
    script = load_script()
    assert script.main(["--raw-dir", str(tmp_path), "--out-dir", str(tmp_path / "r")]) == 1
