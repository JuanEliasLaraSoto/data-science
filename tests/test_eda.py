from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from credit_risk_lab import data, eda
from credit_risk_lab.plotting import SERIES, SURFACE, apply_style, save_figure


@pytest.fixture
def loans() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 3000
    issue = pd.to_datetime("2013-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), "D")
    late_feature = np.where(issue.year >= 2014, rng.normal(size=n), np.nan)
    return pd.DataFrame(
        {
            data.ISSUE_DATE: issue,
            data.VINTAGE: issue.to_period("Q"),
            data.TARGET: rng.binomial(1, 0.12, n),
            "fico_score": rng.normal(700, 30, n),
            "dti": rng.uniform(0, 40, n),
            "new_bureau_field": late_feature,
        }
    )


def test_default_rate_by_vintage(loans: pd.DataFrame) -> None:
    table = eda.default_rate_by_vintage(loans)
    assert len(table) == 12
    assert table["loans"].sum() == len(loans)
    assert table["default_rate"].between(0, 1).all()


def test_missing_rates_and_by_year(loans: pd.DataFrame) -> None:
    rates = eda.missing_rates(loans)
    assert rates.index[0] == "new_bureau_field"
    by_year = eda.missing_by_year(loans, ["new_bureau_field", "dti"])
    assert by_year.loc[2013, "new_bureau_field"] == 1.0
    assert by_year.loc[2014, "new_bureau_field"] == 0.0
    assert by_year["dti"].eq(0).all()


def test_plots(loans: pd.DataFrame, tmp_path: Path) -> None:
    apply_style()
    ax = eda.plot_default_rate_by_vintage(
        eda.default_rate_by_vintage(loans), {"Development": ("2013Q1", "2014Q4")}
    )
    assert len(ax.patches) == 1  # the window span
    fig = eda.plot_distributions(loans, ["fico_score", "dti"], ncols=3)
    assert sum(a.get_visible() for a in fig.axes) == 2
    heat = eda.plot_missing_by_year(eda.missing_by_year(loans, ["new_bureau_field", "dti"]))
    path = save_figure(heat, tmp_path / "figures" / "missing.png")
    assert path.exists()
    plt.close("all")


def test_style_uses_fixed_palette() -> None:
    apply_style()
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    assert colors[:3] == list(SERIES[:3])
    assert mpl.rcParams["figure.facecolor"] == SURFACE
