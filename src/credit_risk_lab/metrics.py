"""Discrimination and calibration metrics, and their standard plots.

* **AUC**: probability that a randomly chosen bad gets a higher PD than a
  randomly chosen good.
* **Gini** (accuracy ratio): ``2 * AUC - 1``. 0 means random, 1 means perfect.
* **KS**: maximum distance between the cumulative score distributions of
  goods and bads.
* **Brier score**: mean squared error of the PD. It mixes calibration and
  discrimination, and lower is better.

Sampling uncertainty of the AUC (and therefore of the Gini) comes from the
DeLong variance. The same machinery gives the paired test used to compare two
models scored on the same loans.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score, roc_curve

from credit_risk_lab.plotting import INK_MUTED, SERIES

Sample = tuple[npt.ArrayLike, npt.ArrayLike]  # (y, predicted PD)


def _arrays(
    y: npt.ArrayLike, p: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    y_arr, p_arr = np.asarray(y).astype(np.int64), np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"y and p must have the same shape, got {y_arr.shape} and {p_arr.shape}.")
    if y_arr.min() == y_arr.max():
        raise ValueError("y must contain both goods and bads.")
    return y_arr, p_arr


def auc(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Area under the ROC curve."""
    y_arr, p_arr = _arrays(y, p)
    return float(roc_auc_score(y_arr, p_arr))


def gini(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Gini coefficient (accuracy ratio), ``2 * AUC - 1``."""
    return 2.0 * auc(y, p) - 1.0


def ks_statistic(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Kolmogorov-Smirnov statistic between the PD distributions of bads and goods.

    Loans are sorted by decreasing PD. The cumulative shares of bads and goods
    are compared at every distinct PD value, so ties are never split.
    """
    y_arr, p_arr = _arrays(y, p)
    order = np.argsort(-p_arr, kind="stable")
    p_sorted, y_sorted = p_arr[order], y_arr[order]
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    end_of_tie = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    return float(np.max(np.abs(cum_bad - cum_good)[end_of_tie]))


def delong_covariance(
    y: npt.ArrayLike, predictions: Sequence[npt.ArrayLike]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """AUCs of several predictions of the same target and their DeLong covariance.

    Uses the fast midrank algorithm of Sun & Xu (2014). DeLong et al. (1988)
    write the AUC as a mean of placement values, a U-statistic. Its variance
    then follows from the variances of the placement values of bads and goods.

    Returns
    -------
    tuple
        ``(aucs, covariance)`` with shapes ``(k,)`` and ``(k, k)``.
    """
    y_arr = np.asarray(y).astype(np.int64)
    preds = np.vstack([_arrays(y_arr, p)[1] for p in predictions])
    bads, goods = preds[:, y_arr == 1], preds[:, y_arr == 0]
    m, n = bads.shape[1], goods.shape[1]
    aucs = np.empty(len(preds))
    v_bad = np.empty((len(preds), m))
    v_good = np.empty((len(preds), n))
    for r in range(len(preds)):
        rank_bad = rankdata(bads[r])
        rank_good = rankdata(goods[r])
        rank_all = rankdata(np.concatenate([bads[r], goods[r]]))
        aucs[r] = (rank_all[:m].sum() - m * (m + 1) / 2) / (m * n)
        v_bad[r] = (rank_all[:m] - rank_bad) / n  # share of goods ranked below each bad
        v_good[r] = 1.0 - (rank_all[m:] - rank_good) / m  # share of bads ranked above each good
    covariance = np.atleast_2d(np.cov(v_bad)) / m + np.atleast_2d(np.cov(v_good)) / n
    return aucs, covariance


def auc_confidence_interval(
    y: npt.ArrayLike, p: npt.ArrayLike, level: float = 0.95
) -> tuple[float, float, float]:
    """AUC with a normal-approximation confidence interval from the DeLong variance."""
    aucs, covariance = delong_covariance(y, [p])
    se = float(np.sqrt(covariance[0, 0]))
    z = float(norm.ppf(0.5 + level / 2))
    value = float(aucs[0])
    return value, max(0.0, value - z * se), min(1.0, value + z * se)


def delong_test(y: npt.ArrayLike, p1: npt.ArrayLike, p2: npt.ArrayLike) -> dict[str, float]:
    """Two-sided DeLong test of ``AUC_1 = AUC_2`` for two models scored on the same loans."""
    aucs, cov = delong_covariance(y, [p1, p2])
    variance = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = float(aucs[0] - aucs[1])
    if variance <= 0:
        z, p_value = 0.0, 1.0
    else:
        z = diff / np.sqrt(variance)
        p_value = float(2 * norm.sf(abs(z)))
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "z": float(z), "p_value": p_value}


def brier_score(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Mean squared difference between predicted PD and observed default."""
    y_arr, p_arr = _arrays(y, p)
    return float(np.mean((p_arr - y_arr) ** 2))


def performance_summary(y: npt.ArrayLike, p: npt.ArrayLike) -> dict[str, float]:
    """Sample size, default rates, AUC, Gini (95% DeLong CI), KS and Brier."""
    y_arr, p_arr = _arrays(y, p)
    auc_value, auc_low, auc_high = auc_confidence_interval(y_arr, p_arr)
    return {
        "n": float(len(y_arr)),
        "default_rate": float(y_arr.mean()),
        "mean_pd": float(p_arr.mean()),
        "auc": auc_value,
        "gini": 2.0 * auc_value - 1.0,
        "gini_ci_low": 2.0 * auc_low - 1.0,
        "gini_ci_high": 2.0 * auc_high - 1.0,
        "ks": ks_statistic(y_arr, p_arr),
        "brier": brier_score(y_arr, p_arr),
    }


def performance_table(samples: Mapping[str, Sample]) -> pd.DataFrame:
    """One row of :func:`performance_summary` per named sample."""
    rows = {name: performance_summary(y, p) for name, (y, p) in samples.items()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["n"] = table["n"].astype(int)
    return table


def calibration_table(y: npt.ArrayLike, p: npt.ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted PD against observed default rate by PD quantile bucket."""
    y_arr, p_arr = _arrays(y, p)
    buckets = pd.qcut(p_arr, q=n_bins, duplicates="drop")
    frame = pd.DataFrame({"y": y_arr, "p": p_arr, "bucket": buckets})
    table = frame.groupby("bucket", observed=True).agg(
        n=("y", "size"), mean_pd=("p", "mean"), default_rate=("y", "mean")
    )
    return table.reset_index(drop=True)


def plot_roc(samples: Mapping[str, Sample], ax: Axes | None = None) -> Axes:
    """ROC curves of several samples or models on one axis, with their Gini."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, label="Random")
    for color, (name, (y, p)) in zip(SERIES, samples.items(), strict=False):
        fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(p))
        ax.plot(fpr, tpr, color=color, label=f"{name} (Gini {gini(y, p):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate (goods)")
    ax.set_ylabel("True positive rate (bads)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return ax


def plot_calibration(
    samples: Mapping[str, Sample], n_bins: int = 10, ax: Axes | None = None
) -> Axes:
    """Observed default rate against mean predicted PD by PD decile."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.6))
    upper = 0.0
    tables = {name: calibration_table(y, p, n_bins) for name, (y, p) in samples.items()}
    for table in tables.values():
        upper = max(upper, float(table[["mean_pd", "default_rate"]].to_numpy().max()))
    upper *= 1.05
    ax.plot([0, upper], [0, upper], color=INK_MUTED, linewidth=1.0, label="Perfect calibration")
    for color, (name, table) in zip(SERIES, tables.items(), strict=False):
        ax.plot(table["mean_pd"], table["default_rate"], color=color, marker="o", label=name)
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    return ax


def ks_curve(
    y: npt.ArrayLike, score: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cumulative score distributions of bads and goods on the grid of distinct scores.

    Returns ``(grid, cum_bad, cum_good)`` with ``cum_*(s) = P(score <= s | class)``.
    A higher score means lower risk, so ``cum_bad`` lies above ``cum_good``.
    """
    y_arr, s_arr = _arrays(y, score)
    grid = np.unique(s_arr)
    bad_scores = np.sort(s_arr[y_arr == 1])
    good_scores = np.sort(s_arr[y_arr == 0])
    cum_bad = np.searchsorted(bad_scores, grid, side="right") / len(bad_scores)
    cum_good = np.searchsorted(good_scores, grid, side="right") / len(good_scores)
    return grid, cum_bad, cum_good


def plot_ks_curve(y: npt.ArrayLike, score: npt.ArrayLike, ax: Axes | None = None) -> Axes:
    """Cumulative share of bads and goods by score, with the KS gap marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))
    grid, cum_bad, cum_good = ks_curve(y, score)
    ax.step(grid, cum_bad, where="post", color=SERIES[1], label="Bads")
    ax.step(grid, cum_good, where="post", color=SERIES[0], label="Goods")
    gap = cum_bad - cum_good
    i = int(np.argmax(gap))
    ax.vlines(grid[i], cum_good[i], cum_bad[i], color=INK_MUTED, linewidth=1.5)
    ax.annotate(
        f"KS = {gap[i]:.3f}\nat score {grid[i]:.0f}",
        xy=(grid[i], (cum_bad[i] + cum_good[i]) / 2),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set(xlabel="Score", ylim=(0, 1))
    ax.set_ylabel("Cumulative share with score <= s")
    ax.legend(loc="lower right")
    return ax
