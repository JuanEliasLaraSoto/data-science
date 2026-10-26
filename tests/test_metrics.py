from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from credit_risk_lab import metrics


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 5000
    signal = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(signal - 2))))
    p = 1 / (1 + np.exp(-(signal - 2)))
    return y, p


def test_ks_matches_scipy(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = scored
    expected = ks_2samp(p[y == 1], p[y == 0]).statistic
    assert metrics.ks_statistic(y, p) == pytest.approx(expected)


def test_ks_does_not_split_ties() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert metrics.ks_statistic(y, p) == pytest.approx(0.0)


def test_gini_is_twice_auc_minus_one(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = scored
    assert metrics.gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1)


def test_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.gini(y, p) == pytest.approx(1.0)
    assert metrics.ks_statistic(y, p) == pytest.approx(1.0)


def test_brier_score() -> None:
    assert metrics.brier_score([0, 1], [0.2, 0.6]) == pytest.approx((0.04 + 0.16) / 2)


def test_performance_table(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = scored
    table = metrics.performance_table({"train": (y, p), "test": (y[:2000], p[:2000])})
    assert list(table.index) == ["train", "test"]
    assert table.loc["train", "n"] == 5000
    assert table.loc["train", "gini"] == pytest.approx(2 * table.loc["train", "auc"] - 1)
    assert table.loc["train", "default_rate"] == pytest.approx(y.mean())


def test_calibration_table_of_calibrated_model(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = scored
    table = metrics.calibration_table(y, p, n_bins=10)
    assert table["n"].sum() == len(y)
    assert table["mean_pd"].is_monotonic_increasing
    assert np.abs(table["mean_pd"] - table["default_rate"]).max() < 0.05


@pytest.mark.parametrize(
    ("y", "p", "message"),
    [([0, 1], [0.1], "same shape"), ([1, 1], [0.1, 0.2], "both goods and bads")],
)
def test_invalid_inputs(y: list[int], p: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        metrics.auc(y, p)


def test_plots_draw_one_line_per_sample(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = scored
    samples = {"train": (y, p), "test": (y[:2000], p[:2000])}
    roc = metrics.plot_roc(samples)
    calib = metrics.plot_calibration(samples)
    assert len(roc.get_lines()) == 3  # diagonal + 2 curves
    assert len(calib.get_lines()) == 3
    assert "Gini" in roc.get_legend().get_texts()[1].get_text()
    plt.close("all")


def test_ks_curve_matches_ks_statistic(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = scored
    score = 600 - 30 * np.log(p / (1 - p))  # higher score = lower risk
    _, cum_bad, cum_good = metrics.ks_curve(y, score)
    assert cum_bad[-1] == pytest.approx(1.0)
    assert cum_good[-1] == pytest.approx(1.0)
    assert np.max(cum_bad - cum_good) == pytest.approx(metrics.ks_statistic(y, p))
    ax = metrics.plot_ks_curve(y, score)
    assert f"KS = {metrics.ks_statistic(y, p):.3f}" in ax.texts[0].get_text()
    plt.close("all")


def test_delong_auc_matches_sklearn_and_bootstrap(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = scored
    aucs, cov = metrics.delong_covariance(y, [p])
    assert aucs[0] == pytest.approx(roc_auc_score(y, p))
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(300):
        idx = rng.integers(0, len(y), len(y))
        boot.append(roc_auc_score(y[idx], p[idx]))
    assert np.sqrt(cov[0, 0]) == pytest.approx(np.std(boot), rel=0.2)


def test_delong_handles_ties() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.5, 0.5, 0.9, 0.2, 0.5])
    aucs, _ = metrics.delong_covariance(y, [p])
    assert aucs[0] == pytest.approx(roc_auc_score(y, p))


def test_auc_confidence_interval_contains_estimate(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = scored
    value, low, high = metrics.auc_confidence_interval(y, p)
    assert low < value < high
    summary = metrics.performance_summary(y, p)
    assert summary["gini_ci_low"] == pytest.approx(2 * low - 1)


def test_delong_test(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = scored
    rng = np.random.default_rng(1)
    noisy = p + rng.normal(scale=0.3, size=len(p))
    result = metrics.delong_test(y, p, noisy)
    assert result["auc_1"] > result["auc_2"]
    assert result["p_value"] < 0.01
    same = metrics.delong_test(y, p, p)
    assert same["p_value"] == 1.0
    assert same["z"] == 0.0
