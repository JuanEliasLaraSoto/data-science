from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.outliers_influence import variance_inflation_factor

from credit_risk_lab.selection import select_features, variance_inflation_factors


def correlated_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pd.DataFrame(
        {
            "a": a,
            "a_copy": a + 0.1 * rng.normal(size=n),  # |rho| ~ 0.99 with a
            "b": b,
            "c": 0.5 * a + 0.5 * b + 0.05 * rng.normal(size=n),  # collinear with a and b
            "d": rng.normal(size=n),
        }
    )


def test_vif_matches_statsmodels() -> None:
    X = correlated_frame()[["a", "b", "c", "d"]]
    design = np.column_stack([np.ones(len(X)), X.to_numpy()])
    expected = [variance_inflation_factor(design, i) for i in range(1, design.shape[1])]
    np.testing.assert_allclose(variance_inflation_factors(X).to_numpy(), expected, rtol=1e-6)


def test_vif_of_single_column_is_one() -> None:
    assert variance_inflation_factors(pd.DataFrame({"x": [1.0, 2.0, 4.0]})).tolist() == [1.0]


def test_selection_applies_iv_correlation_and_vif_filters() -> None:
    woe = correlated_frame()
    woe["weak"] = np.random.default_rng(1).normal(size=len(woe))
    woe["leaky"] = woe["d"]
    iv = pd.Series(
        {"a": 0.30, "a_copy": 0.25, "b": 0.20, "c": 0.15, "d": 0.10, "weak": 0.01, "leaky": 0.9}
    )
    result = select_features(woe, iv, max_vif=5.0)
    reasons = result.log.set_index("feature")
    assert reasons.loc["weak", "stage"] == "iv"
    assert reasons.loc["leaky", "stage"] == "iv"
    assert "suspicious" in str(reasons.loc["leaky", "reason"])
    assert reasons.loc["a_copy", "stage"] == "correlation"
    assert "with a" in str(reasons.loc["a_copy", "reason"])
    assert reasons.loc["c", "stage"] in {"correlation", "vif"}
    assert result.selected == ["a", "b", "d"]


def test_vif_stage_drops_the_worst_offender() -> None:
    woe = correlated_frame()[["a", "b", "c"]]
    iv = pd.Series({"a": 0.3, "b": 0.2, "c": 0.1})
    result = select_features(woe, iv, max_corr=0.99, max_vif=5.0)
    assert result.log["stage"].tolist() == ["vif"]
    assert result.log["feature"].tolist() == ["c"]
    assert result.selected == ["a", "b"]


@pytest.mark.parametrize("max_vif", [5.0, 1e9])
def test_selection_without_problems_keeps_everything(max_vif: float) -> None:
    woe = correlated_frame()[["a", "b", "d"]]
    iv = pd.Series({"a": 0.3, "b": 0.2, "d": 0.1})
    result = select_features(woe, iv, max_vif=max_vif)
    assert result.selected == ["a", "b", "d"]
    assert result.log.empty
    assert list(result.log.columns) == ["feature", "stage", "decision", "reason"]


def test_reviewed_high_iv_characteristic_is_kept_and_logged() -> None:
    woe = correlated_frame()[["a", "b", "d"]]
    iv = pd.Series({"a": 0.9, "b": 0.2, "d": 0.1})
    result = select_features(woe, iv, reviewed_high_iv={"a": "known at application"})
    assert result.selected == ["a", "b", "d"]
    row = result.log.set_index("feature").loc["a"]
    assert row["decision"] == "kept"
    assert "known at application" in str(row["reason"])
