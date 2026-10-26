"""Exploratory summaries and figures: default rate by vintage, missing values, distributions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, PercentFormatter

from credit_risk_lab.data import ISSUE_DATE, TARGET, VINTAGE
from credit_risk_lab.plotting import GRID, INK_SECONDARY, SERIES

# Two quiet surface tints for the sample windows (never series colours).
WINDOW_TINTS = ("#f0efec", "#e6eefa")


def default_rate_by_vintage(df: pd.DataFrame) -> pd.DataFrame:
    """Count loans and bads and compute the default rate per (quarterly) vintage."""
    table = df.groupby(VINTAGE, observed=True).agg(loans=(TARGET, "size"), bads=(TARGET, "sum"))
    table["default_rate"] = table["bads"] / table["loans"]
    return table


def missing_rates(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.Series:
    """Share of missing values per column, sorted in decreasing order."""
    selected = list(df.columns) if columns is None else list(columns)
    return df[selected].isna().mean().sort_values(ascending=False)


def missing_by_year(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Share of missing values per issue year (rows) and column.

    A column that is missing in early years only was not collected yet: its
    missing bin would encode the vintage rather than the borrower.
    """
    year = df[ISSUE_DATE].dt.year.rename("year")
    return df[list(columns)].isna().groupby(year).mean()


def _decimal_year(period: pd.Period) -> float:
    """Start of a quarterly period as a decimal year (``2015Q3`` -> 2015.5)."""
    return period.year + (period.quarter - 1) / 4


def plot_default_rate_by_vintage(
    table: pd.DataFrame,
    windows: Mapping[str, tuple[str, str]] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Default rate per vintage with the loan count as a light context band.

    Parameters
    ----------
    table
        Output of :func:`default_rate_by_vintage`.
    windows
        Optional named sample windows (``{"Development": ("2007Q1", "2014Q4")}``)
        drawn as labelled spans.
    ax
        Axis to draw on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 3.8))
    x = [_decimal_year(p) for p in pd.PeriodIndex(table.index)]
    ax.plot(x, table["default_rate"], color=SERIES[0], marker="o", markersize=4)
    for color, (name, (start, end)) in zip(WINDOW_TINTS, (windows or {}).items(), strict=False):
        lo = _decimal_year(pd.Period(start, "Q"))
        hi = _decimal_year(pd.Period(end, "Q") + 1)
        ax.axvspan(lo, hi, color=color, zorder=0, linewidth=0)
        ax.text(
            lo,
            0.03,
            f" {name}",
            transform=ax.get_xaxis_transform(),
            va="bottom",
            color=INK_SECONDARY,
            fontsize=9,
        )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Default rate (lifetime)")
    return ax


def plot_distributions(
    df: pd.DataFrame, features: Sequence[str], *, bins: int = 40, ncols: int = 3
) -> Figure:
    """Small multiples: distribution of each numeric feature for goods and bads.

    Values are clipped to the 1st-99th percentile so that outliers do not
    flatten the histograms.
    """
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), squeeze=False)
    for ax, feature in zip(axes.flat, features, strict=False):
        values = df[feature]
        lo, hi = values.quantile([0.01, 0.99])
        edges = np.linspace(lo, hi, bins + 1) if hi > lo else bins
        for color, (label, flag) in zip(SERIES, (("Good", 0), ("Bad", 1)), strict=False):
            subset = values[df[TARGET] == flag].clip(lo, hi).dropna()
            ax.hist(
                subset,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=color,
                label=label,
            )
        ax.set_title(feature, fontsize=9)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in list(axes.flat)[len(features) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_missing_by_year(table: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Heatmap of missing shares by issue year (single-hue sequential scale)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 0.25 * len(table.columns) + 1.5))
    image = ax.imshow(table.T.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(table.index)), [str(i) for i in table.index])
    ax.set_yticks(range(len(table.columns)), list(table.columns), fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    plt.colorbar(image, ax=ax, label="Share missing", fraction=0.03)
    return ax
