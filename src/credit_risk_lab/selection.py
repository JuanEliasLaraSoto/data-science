"""Variable selection for the scorecard: IV screening, correlation and VIF.

The filters are applied in sequence on the *training* sample only:

1. IV screening. Drop characteristics with ``IV < min_iv`` (no predictive
   power). An ``IV > max_iv`` is too good to be true for an application
   variable and usually signals leakage, so such characteristics are dropped
   *unless* a documented review concluded that the information is genuinely
   available at application time. The threshold triggers a review; it is not
   proof of leakage.
2. Pairwise correlation of the WoE-encoded characteristics. Going through
   them in decreasing IV, keep one only if ``|rho| <= max_corr`` with every
   characteristic already kept. Of each correlated pair this keeps the more
   predictive one.
3. Variance inflation factor. While the largest VIF exceeds ``max_vif``, drop
   that characteristic. Multicollinearity inflates the standard errors of the
   logit and can flip coefficient signs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class SelectionResult:
    """Outcome of :func:`select_features`.

    Attributes
    ----------
    selected
        Kept characteristics, in decreasing IV.
    log
        One row per dropped or flagged characteristic: filter (``stage``),
        ``decision`` and reason.
    """

    selected: list[str]
    log: pd.DataFrame


def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """VIF of every column, from the inverse of the correlation matrix.

    For a regression with intercept, ``VIF_j = 1 / (1 - R_j^2)`` equals the
    ``j``-th diagonal element of the inverse correlation matrix.
    """
    if X.shape[1] == 1:
        return pd.Series(1.0, index=X.columns)
    corr = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(corr)), index=X.columns)


def select_features(
    woe: pd.DataFrame,
    iv: pd.Series,
    *,
    min_iv: float = 0.02,
    max_iv: float = 0.5,
    max_corr: float = 0.6,
    max_vif: float = 5.0,
    reviewed_high_iv: Mapping[str, str] | None = None,
) -> SelectionResult:
    """Apply the IV, correlation and VIF filters.

    Parameters
    ----------
    woe
        WoE-encoded training sample, one column per candidate characteristic.
    iv
        Information value indexed by characteristic.
    min_iv, max_iv
        IV screening thresholds.
    max_corr
        Maximum absolute Pearson correlation between kept characteristics.
    max_vif
        Maximum variance inflation factor among kept characteristics.
    reviewed_high_iv
        Characteristics with ``IV > max_iv`` that passed a leakage review,
        mapped to the review's conclusion. They are kept and the note is logged.

    Returns
    -------
    SelectionResult
        Selected characteristics and a log of the dropped ones.
    """
    reviewed = dict(reviewed_high_iv or {})
    log: list[dict[str, object]] = []

    def record(feature: str, stage: str, reason: str, decision: str = "dropped") -> None:
        log.append({"feature": feature, "stage": stage, "decision": decision, "reason": reason})

    candidates: list[str] = []
    for feature, value in iv.sort_values(ascending=False).items():
        name = str(feature)
        if value < min_iv:
            record(name, "iv", f"IV {value:.4f} < {min_iv}")
        elif value > max_iv and name in reviewed:
            record(name, "iv", f"IV {value:.4f} > {max_iv}; review: {reviewed[name]}", "kept")
            candidates.append(name)
        elif value > max_iv:
            record(name, "iv", f"IV {value:.4f} > {max_iv}: suspicious, review for leakage")
        else:
            candidates.append(name)

    corr = woe[candidates].corr().abs()
    kept: list[str] = []
    for feature in candidates:
        clashes = corr.loc[feature, kept] if kept else pd.Series(dtype=float)
        if (clashes > max_corr).any():
            partner = str(clashes.idxmax())
            record(
                feature, "correlation", f"|rho| = {clashes.max():.2f} with {partner} (higher IV)"
            )
        else:
            kept.append(feature)

    while len(kept) > 1:
        vif = variance_inflation_factors(woe[kept])
        worst = str(vif.idxmax())
        if vif[worst] <= max_vif:
            break
        record(worst, "vif", f"VIF {vif[worst]:.2f} > {max_vif}")
        kept.remove(worst)

    return SelectionResult(
        selected=kept,
        log=pd.DataFrame(log, columns=["feature", "stage", "decision", "reason"]),
    )
