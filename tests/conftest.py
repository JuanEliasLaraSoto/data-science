"""Shared fixtures: synthetic credit data with known risk drivers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


def make_credit_sample(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Synthetic applications where ``fico`` lowers and ``dti`` raises the PD.

    ``purpose`` has a nominal effect, ``noise`` none at all. ``dti`` has 8%
    missing values that are riskier than average.
    """
    rng = np.random.default_rng(seed)
    fico = rng.normal(700, 35, n)
    dti = rng.uniform(0, 40, n)
    purpose = rng.choice(["debt", "card", "home", "car"], n, p=[0.45, 0.3, 0.15, 0.1])
    effect = pd.Series(purpose).map({"debt": 0.3, "card": 0.0, "home": -0.3, "car": -0.6})
    noise = rng.normal(size=n)
    dti_missing = rng.random(n) < 0.08
    log_odds = -1.8 - 0.03 * (fico - 700) + 0.05 * (dti - 20) + effect.to_numpy()
    log_odds = np.where(dti_missing, log_odds + 0.8, log_odds)
    y = rng.binomial(1, 1 / (1 + np.exp(-log_odds)))
    return pd.DataFrame(
        {
            "fico": fico,
            "dti": np.where(dti_missing, np.nan, dti),
            "purpose": purpose,
            "noise": noise,
            "target": y,
        }
    )


@pytest.fixture
def credit_sample() -> pd.DataFrame:
    return make_credit_sample()
