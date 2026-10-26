from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_lab.scorecard import PointsScaling, Scorecard, backward_elimination, fit_logit
from credit_risk_lab.woe import WoEEncoder

FEATURES = ["fico", "dti", "purpose", "noise"]


@pytest.fixture
def fitted(credit_sample: pd.DataFrame) -> tuple[Scorecard, pd.DataFrame]:
    X = credit_sample[FEATURES]
    encoder = WoEEncoder().fit(X, credit_sample["target"])
    card = Scorecard.fit(X, credit_sample["target"], encoder, FEATURES)
    return card, X


# ------------------------------------------------------------------ scaling


def test_scaling_anchors() -> None:
    scaling = PointsScaling(base_score=600, base_odds=50, pdo=20)
    pd_at_base = 1 / 51  # odds good:bad = 50:1
    assert scaling.score_from_pd(pd_at_base) == pytest.approx(600)
    pd_double_odds = 1 / 101  # odds 100:1
    assert scaling.score_from_pd(pd_double_odds) == pytest.approx(620)
    assert scaling.factor == pytest.approx(20 / np.log(2))
    assert scaling.offset == pytest.approx(600 - 20 / np.log(2) * np.log(50))


def test_scaling_round_trip() -> None:
    scaling = PointsScaling()
    pds = np.array([0.001, 0.02, 0.1, 0.5, 0.9])
    np.testing.assert_allclose(scaling.pd_from_score(scaling.score_from_pd(pds)), pds)


# ------------------------------------------------------------------ elimination


def test_backward_elimination_drops_wrong_sign_then_insignificant() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    good_driver = rng.normal(size=n)
    wrong_sign = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1.0 - 1.0 * good_driver + 0.8 * wrong_sign))))
    X = pd.DataFrame({"good": good_driver, "wrong": wrong_sign, "noise": noise})
    result, kept, log = backward_elimination(X, y, max_pvalue=0.05)
    assert kept == ["good"]
    reasons = log.set_index("feature")["reason"]
    assert set(reasons.index) == {"wrong", "noise"}
    assert "inconsistent with WoE" in reasons["wrong"]
    assert result.params["good"] < 0


def test_backward_elimination_fails_when_nothing_survives() -> None:
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"bad_sign": rng.normal(size=2000)})
    y = rng.binomial(1, 1 / (1 + np.exp(-(X["bad_sign"] - 1))))
    with pytest.raises(ValueError, match="removed every characteristic"):
        backward_elimination(X, y)


# ------------------------------------------------------------------ scorecard


def test_scorecard_drops_noise_and_has_negative_coefficients(
    fitted: tuple[Scorecard, pd.DataFrame],
) -> None:
    card, _ = fitted
    assert "noise" not in card.features
    assert set(card.features) == {"fico", "dti", "purpose"}
    assert (card.params[card.features] < 0).all()
    assert card.summary.loc["fico", "p_value"] < 0.05
    assert {"aic", "bic", "pseudo_r2", "log_likelihood", "n_obs"} <= set(card.fit_stats)


def test_pd_matches_statsmodels(
    fitted: tuple[Scorecard, pd.DataFrame], credit_sample: pd.DataFrame
) -> None:
    card, X = fitted
    result = fit_logit(card.woe(X), credit_sample["target"])
    np.testing.assert_allclose(card.predict_pd(X), result.predict(), rtol=1e-6)


def test_score_is_consistent_with_pd(fitted: tuple[Scorecard, pd.DataFrame]) -> None:
    card, X = fitted
    np.testing.assert_allclose(card.score(X), card.scaling.score_from_pd(card.predict_pd(X)))


def test_points_add_up_to_score(fitted: tuple[Scorecard, pd.DataFrame]) -> None:
    card, X = fitted
    table = card.points_table()
    exact_total = np.zeros(len(X))
    for feature in card.features:
        pts = table.loc[table["feature"] == feature, "points_exact"].to_numpy()
        exact_total += pts[card.binnings[feature].assign(X[feature])]
    np.testing.assert_allclose(exact_total, card.score(X))
    rounding_error = np.abs(card.points(X) - card.score(X))
    assert rounding_error.max() <= 0.5 * len(card.features) + 1e-9


def test_points_increase_with_woe(fitted: tuple[Scorecard, pd.DataFrame]) -> None:
    card, _ = fitted
    for _, group in card.points_table().groupby("feature"):
        ordered = group.sort_values("woe")
        assert ordered["points_exact"].is_monotonic_increasing
