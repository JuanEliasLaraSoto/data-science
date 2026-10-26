"""Monotonic binning, Weight of Evidence (WoE) and Information Value (IV).

Conventions (Siddiqi, *Credit Risk Scorecards*, 2006):

* the target is ``y = 1`` for a *bad* (default) and ``y = 0`` for a *good*;
* ``WoE_i = ln(%good_i / %bad_i)``, so a positive WoE means lower risk than average;
* ``IV = sum_i (%good_i - %bad_i) * WoE_i``.

Binning algorithm
-----------------
1. Fine classing. Numeric variables are cut at up to ``max_initial_bins``
   quantiles. Every category of a nominal variable starts as its own bin, and the
   categories are ordered by bad rate.
2. Monotonicity (numeric only). Adjacent bins are pooled with the
   pool-adjacent-violators algorithm (the isotonic-regression algorithm) until
   the bad rate is strictly monotonic in the variable. Both directions are tried
   and the one that keeps more IV wins.
3. Stability. Bins holding fewer than ``min_bin_frac`` of the observations, or
   with no goods or no bads, are merged with the adjacent bin whose bad rate is
   closest.
4. Parsimony. While there are more than ``max_bins`` bins, the adjacent pair whose
   merge loses the least IV is merged.

Merging two adjacent bins of a monotonic sequence gives a bad rate between the
two, so steps 3 and 4 never break the monotonicity obtained in step 2.

Missing values get their own bin when they are frequent enough and contain both
goods and bads. Otherwise they are merged into the bin with the closest bad
rate. Missing values or categories never seen in training are assigned to the
riskiest bin, which is the conservative choice.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

Monotonic = Literal["auto", "increasing", "decreasing"]
Kind = Literal["numeric", "categorical"]

MISSING_LABEL = "Missing"


# ---------------------------------------------------------------------------- groups


@dataclass(frozen=True)
class _Group:
    """A run of consecutive fine bins (``atoms``) with its counts."""

    start: int
    stop: int  # exclusive
    n: int
    bads: int

    @property
    def goods(self) -> int:
        return self.n - self.bads

    @property
    def bad_rate(self) -> float:
        return self.bads / self.n


def _merge(a: _Group, b: _Group) -> _Group:
    return _Group(min(a.start, b.start), max(a.stop, b.stop), a.n + b.n, a.bads + b.bads)


def _violates(left: _Group, right: _Group, increasing: bool) -> bool:
    # Cross-multiplied comparison of bad rates (exact on integers). Ties are
    # pooled as well, so the result is strictly monotonic.
    lhs, rhs = left.bads * right.n, right.bads * left.n
    return lhs >= rhs if increasing else lhs <= rhs


def _pool_adjacent_violators(groups: Sequence[_Group], increasing: bool) -> list[_Group]:
    stack: list[_Group] = []
    for group in groups:
        stack.append(group)
        while len(stack) >= 2 and _violates(stack[-2], stack[-1], increasing):
            right, left = stack.pop(), stack.pop()
            stack.append(_merge(left, right))
    return stack


def _closest_neighbour(groups: Sequence[_Group], i: int) -> int:
    candidates = [k for k in (i - 1, i + 1) if 0 <= k < len(groups)]
    return min(candidates, key=lambda k: abs(groups[k].bad_rate - groups[i].bad_rate))


def _enforce_min_size(groups: Sequence[_Group], min_count: int) -> list[_Group]:
    merged = list(groups)
    while len(merged) > 1:
        weak = [i for i, g in enumerate(merged) if g.n < min_count or g.bads == 0 or g.goods == 0]
        if not weak:
            break
        i = min(weak, key=lambda k: merged[k].n)
        lo, hi = sorted((i, _closest_neighbour(merged, i)))
        merged[lo : hi + 1] = [_merge(merged[lo], merged[hi])]
    return merged


def _iv_term(goods: int, bads: int, total_goods: int, total_bads: int) -> float:
    dist_good, dist_bad = goods / total_goods, bads / total_bads
    if dist_good == 0 or dist_bad == 0:
        return float("inf")
    return (dist_good - dist_bad) * float(np.log(dist_good / dist_bad))


def _groups_iv(groups: Sequence[_Group], total_goods: int, total_bads: int) -> float:
    return sum(_iv_term(g.goods, g.bads, total_goods, total_bads) for g in groups)


def _reduce_to_max_bins(
    groups: Sequence[_Group], max_bins: int, total_goods: int, total_bads: int
) -> list[_Group]:
    merged = list(groups)
    while len(merged) > max_bins:
        losses = [
            _iv_term(a.goods, a.bads, total_goods, total_bads)
            + _iv_term(b.goods, b.bads, total_goods, total_bads)
            - _iv_term(a.goods + b.goods, a.bads + b.bads, total_goods, total_bads)
            for a, b in itertools.pairwise(merged)
        ]
        i = int(np.argmin(losses))
        merged[i : i + 2] = [_merge(merged[i], merged[i + 1])]
    return merged


# ---------------------------------------------------------------------------- binning


@dataclass(frozen=True, eq=False)
class Binning:
    """Fitted binning of one characteristic.

    Attributes
    ----------
    feature
        Column name.
    kind
        ``"numeric"`` (right-closed intervals defined by ``cuts``) or
        ``"categorical"`` (explicit ``categories`` mapping).
    cuts
        Interior cut points; bin ``i`` is ``(cuts[i-1], cuts[i]]``.
    categories
        Category -> bin index (categorical only).
    missing_bin
        Bin receiving missing values, or ``None`` if training had none (missing
        values are then sent to the riskiest bin).
    table
        One row per bin: counts, bad rate, WoE and IV contribution.
    """

    feature: str
    kind: Kind
    cuts: tuple[float, ...]
    categories: Mapping[str, int]
    missing_bin: int | None
    table: pd.DataFrame

    @property
    def iv(self) -> float:
        """Information value of the characteristic."""
        return float(self.table["iv"].sum())

    @property
    def n_bins(self) -> int:
        """Number of bins, including a dedicated missing bin if there is one."""
        return len(self.table)

    @property
    def riskiest_bin(self) -> int:
        """Index of the bin with the lowest WoE (highest bad rate)."""
        return int(np.argmin(self.table["woe"].to_numpy()))

    def assign(self, x: pd.Series) -> npt.NDArray[np.intp]:
        """Return the bin index of every observation in ``x``."""
        missing = x.isna().to_numpy()
        out = np.full(len(x), self.riskiest_bin, dtype=np.intp)
        present = ~missing
        if self.kind == "numeric":
            values = pd.to_numeric(x[present]).to_numpy(dtype=float)
            out[present] = np.searchsorted(np.asarray(self.cuts), values, side="left")
        else:
            mapped = x[present].astype(str).map(self.categories)
            out[present] = mapped.fillna(self.riskiest_bin).to_numpy(dtype=np.intp)
        if self.missing_bin is not None:
            out[missing] = self.missing_bin
        return out

    def transform(self, x: pd.Series) -> npt.NDArray[np.float64]:
        """Replace every observation in ``x`` by the WoE of its bin."""
        woe = self.table["woe"].to_numpy(dtype=float)
        return woe[self.assign(x)]


def _is_numeric(x: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x))


def _format_edge(value: float) -> str:
    return f"{value:.6g}"


def _numeric_labels(cuts: Sequence[float]) -> list[str]:
    edges = [-np.inf, *cuts, np.inf]
    labels = []
    for lo, hi in itertools.pairwise(edges):
        right = ")" if np.isinf(hi) else "]"
        labels.append(f"({_format_edge(lo)}, {_format_edge(hi)}{right}")
    return labels


def _woe_table(labels: Sequence[str], n: Sequence[int], bads: Sequence[int]) -> pd.DataFrame:
    table = pd.DataFrame({"bin": list(labels), "count": n, "bads": bads})
    table["goods"] = table["count"] - table["bads"]
    total_goods, total_bads = table["goods"].sum(), table["bads"].sum()
    table["count_pct"] = table["count"] / table["count"].sum()
    table["bad_rate"] = table["bads"] / table["count"]
    # Zero cells only arise in degenerate cases (e.g. a single bin); a 0.5
    # continuity correction keeps the WoE finite there.
    goods_adj = table["goods"].where(table["goods"] > 0, 0.5)
    bads_adj = table["bads"].where(table["bads"] > 0, 0.5)
    table["dist_good"] = goods_adj / total_goods
    table["dist_bad"] = bads_adj / total_bads
    table["woe"] = np.log(table["dist_good"] / table["dist_bad"])
    table["iv"] = (table["dist_good"] - table["dist_bad"]) * table["woe"]
    columns = ["bin", "count", "count_pct", "goods", "bads", "bad_rate"]
    return table[[*columns, "dist_good", "dist_bad", "woe", "iv"]]


def _validate_target(y: pd.Series | npt.ArrayLike, n: int) -> npt.NDArray[np.int64]:
    target = np.asarray(y)
    if n == 0:
        raise ValueError("Cannot fit a binning on an empty sample.")
    if target.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {target.shape}.")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("y must be binary (0 = good, 1 = bad).")
    if target.min() == target.max():
        raise ValueError("y must contain both goods and bads.")
    return target.astype(np.int64)


def fit_binning(
    x: pd.Series,
    y: pd.Series | npt.ArrayLike,
    *,
    min_bin_frac: float = 0.05,
    max_bins: int = 8,
    max_initial_bins: int = 20,
    monotonic: Monotonic = "auto",
) -> Binning:
    """Fit a monotonic (numeric) or bad-rate-ordered (categorical) binning.

    Parameters
    ----------
    x
        Characteristic to bin. Numeric dtypes are binned into intervals; anything
        else is treated as nominal.
    y
        Binary target (1 = bad).
    min_bin_frac
        Minimum share of all observations in each bin.
    max_bins
        Maximum number of non-missing bins.
    max_initial_bins
        Number of quantile intervals used for fine classing.
    monotonic
        Direction of the bad rate for numeric variables; ``"auto"`` keeps the
        direction with the larger IV.

    Returns
    -------
    Binning
        The fitted binning, including its WoE/IV table.
    """
    if not 0 < min_bin_frac < 0.5:
        raise ValueError("min_bin_frac must be in (0, 0.5).")
    if max_bins < 1 or max_initial_bins < 1:
        raise ValueError("max_bins and max_initial_bins must be positive.")
    target = _validate_target(y, len(x))
    total_bads = int(target.sum())
    total_goods = len(target) - total_bads
    min_count = max(1, int(np.ceil(min_bin_frac * len(x))))

    missing = x.isna().to_numpy()
    xs, ys = x[~missing], target[~missing]
    kind: Kind = "numeric" if _is_numeric(x) else "categorical"

    # 1. Fine classing into "atoms" ordered along the variable (or bad rate).
    atom_cuts: npt.NDArray[np.float64] = np.empty(0)
    atom_labels: list[str] = []
    atom_index: npt.NDArray[np.intp]
    if kind == "numeric":
        values = pd.to_numeric(xs).to_numpy(dtype=float)
        if len(values):
            probs = np.linspace(0, 1, max_initial_bins + 1)[1:-1]
            atom_cuts = np.unique(np.quantile(values, probs, method="inverted_cdf"))
        atom_index = np.searchsorted(atom_cuts, values, side="left")
        n_atoms = len(atom_cuts) + 1
    else:
        as_str = xs.astype(str).to_numpy()
        bad_rates = pd.Series(ys).groupby(as_str).mean().sort_values(kind="stable")
        atom_labels = [str(c) for c in bad_rates.index]
        lookup = {c: i for i, c in enumerate(atom_labels)}
        atom_index = np.array([lookup[c] for c in as_str], dtype=np.intp)
        n_atoms = len(atom_labels)

    atom_n = np.bincount(atom_index, minlength=n_atoms)
    atom_bads = np.bincount(atom_index, weights=ys, minlength=n_atoms).astype(np.int64)
    occupied = np.flatnonzero(atom_n)
    groups = [_Group(i, i + 1, int(atom_n[a]), int(atom_bads[a])) for i, a in enumerate(occupied)]

    # 2-4. Monotonicity, stability, parsimony.
    if groups:
        if kind == "numeric":
            directions = [True, False] if monotonic == "auto" else [monotonic == "increasing"]
            candidates = [
                _enforce_min_size(_pool_adjacent_violators(groups, inc), min_count)
                for inc in directions
            ]
            groups = max(candidates, key=lambda g: _groups_iv(g, total_goods, total_bads))
        else:
            groups = _enforce_min_size(groups, min_count)
        groups = _reduce_to_max_bins(groups, max_bins, total_goods, total_bads)

    # Translate groups of atoms into cut points / category mappings.
    cuts: tuple[float, ...] = ()
    categories: dict[str, int] = {}
    if kind == "numeric":
        cuts = tuple(float(atom_cuts[occupied[g.stop - 1]]) for g in groups[:-1])
        labels = _numeric_labels(cuts) if groups else []
    else:
        labels = []
        for b, g in enumerate(groups):
            members = [atom_labels[a] for a in occupied[g.start : g.stop]]
            categories.update(dict.fromkeys(members, b))
            labels.append(", ".join(members))
    n = [g.n for g in groups]
    bads = [g.bads for g in groups]

    # Missing values: own bin if material, otherwise pooled with the closest bin.
    missing_bin: int | None = None
    n_missing, bads_missing = int(missing.sum()), int(target[missing].sum())
    if n_missing:
        material = n_missing >= min_count and 0 < bads_missing < n_missing
        if material or not groups:
            missing_bin = len(groups)
            labels.append(MISSING_LABEL)
            n.append(n_missing)
            bads.append(bads_missing)
        else:
            rate = bads_missing / n_missing
            missing_bin = int(np.argmin([abs(g.bad_rate - rate) for g in groups]))
            labels[missing_bin] += f" | {MISSING_LABEL}"
            n[missing_bin] += n_missing
            bads[missing_bin] += bads_missing

    return Binning(
        feature=str(x.name),
        kind=kind,
        cuts=cuts,
        categories=categories,
        missing_bin=missing_bin,
        table=_woe_table(labels, n, bads),
    )


# ---------------------------------------------------------------------------- IV


def iv_strength(iv: float) -> str:
    """Label an IV with the usual rule of thumb (Siddiqi, 2006).

    ``< 0.02`` unpredictive, ``0.02-0.1`` weak, ``0.1-0.3`` medium,
    ``0.3-0.5`` strong, ``> 0.5`` suspicious (too good to be true: check for leakage).
    """
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv <= 0.5:
        return "strong"
    return "suspicious"


class WoEEncoder:
    """Fit one :class:`Binning` per column and encode a frame into WoE values.

    Parameters
    ----------
    min_bin_frac, max_bins, max_initial_bins, monotonic
        Passed to :func:`fit_binning`.
    """

    def __init__(
        self,
        *,
        min_bin_frac: float = 0.05,
        max_bins: int = 8,
        max_initial_bins: int = 20,
        monotonic: Monotonic = "auto",
    ) -> None:
        self.min_bin_frac = min_bin_frac
        self.max_bins = max_bins
        self.max_initial_bins = max_initial_bins
        self.monotonic: Monotonic = monotonic
        self.binnings_: dict[str, Binning] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | npt.ArrayLike) -> Self:
        """Fit a binning for every column of ``X``."""
        self.binnings_ = {
            str(column): fit_binning(
                X[column],
                y,
                min_bin_frac=self.min_bin_frac,
                max_bins=self.max_bins,
                max_initial_bins=self.max_initial_bins,
                monotonic=self.monotonic,
            )
            for column in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Encode ``X`` (restricted to ``columns`` if given) into WoE values."""
        if not self.binnings_:
            raise RuntimeError("WoEEncoder is not fitted.")
        selected = list(self.binnings_) if columns is None else list(columns)
        return pd.DataFrame({c: self.binnings_[c].transform(X[c]) for c in selected}, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        """IV, number of bins and strength label per feature, sorted by IV."""
        rows = [
            {"feature": f, "iv": b.iv, "n_bins": b.n_bins, "strength": iv_strength(b.iv)}
            for f, b in self.binnings_.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)
