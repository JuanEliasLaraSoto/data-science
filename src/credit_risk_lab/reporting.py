"""Helpers that turn result tables into Markdown and figures for the reports."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from credit_risk_lab.plotting import SERIES
from credit_risk_lab.woe import Binning


def _format_cell(value: object, float_format: str) -> str:
    if isinstance(value, float | np.floating):
        return "" if np.isnan(value) else format(value, float_format)
    if isinstance(value, int | np.integer):
        return f"{value:,}"
    return str(value).replace("|", "\\|")


def markdown_table(df: pd.DataFrame, *, float_format: str = ".3f", index: bool = True) -> str:
    """Render ``df`` as a GitHub-flavoured Markdown table."""
    frame = df.reset_index() if index else df
    header = [str(c) for c in frame.columns]
    numeric = [pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---:" if is_num else "---" for is_num in numeric) + "|",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_format_cell(v, float_format) for v in row) + " |")
    return "\n".join(lines)


def _shorten(label: str, width: int) -> str:
    return label if len(label) <= width else label[: width - 1].rstrip() + "…"


def plot_binnings(
    binnings: Mapping[str, Binning], *, ncols: int = 2, label_width: int = 34
) -> Figure:
    """Small multiples: bad rate per bin (horizontal bars, first bin on top)."""
    nrows = int(np.ceil(len(binnings) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 2.5 * nrows), squeeze=False)
    for ax, (feature, binning) in zip(axes.flat, binnings.items(), strict=False):
        table = binning.table
        positions = np.arange(len(table))[::-1]
        ax.barh(positions, table["bad_rate"], height=0.6, color=SERIES[0])
        labels = [_shorten(str(b), label_width) for b in table["bin"]]
        ax.set_yticks(positions, labels, fontsize=8)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(f"{feature} (IV {binning.iv:.3f})", fontsize=9)
        ax.grid(axis="y", visible=False)
    for ax in list(axes.flat)[len(binnings) :]:
        ax.set_visible(False)
    fig.supxlabel("Bad rate", fontsize=9)
    fig.tight_layout()
    return fig
