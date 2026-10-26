"""Logistic-regression scorecard and points scaling.

Model
-----
``ln(odds_bad) = b0 + sum_j b_j * WoE_j``, fitted by maximum likelihood with
statsmodels to get standard errors and p-values. With the WoE convention
``ln(%good / %bad)``, a higher WoE means lower risk, so every coefficient must be
**negative**. A positive coefficient means the variable works against its own
univariate evidence, almost always because of multicollinearity. Such variables
are removed by backward elimination together with non-significant ones.

Scaling
-------
``score = offset + factor * ln(odds_good)`` with ``factor = PDO / ln 2`` and
``offset = base_score - factor * ln(base_odds)``. With the defaults, a score of
600 means good:bad odds of 50:1, and every 20 points double the odds.
Distributing the intercept evenly across the ``n`` characteristics gives the
points of bin ``i`` of characteristic ``j`` (Siddiqi, 2006)::

    points_ij = offset / n - factor * (b_j * WoE_ij + b0 / n)

and the points of a loan add up to its score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm

from credit_risk_lab.woe import Binning, WoEEncoder

INTERCEPT = "const"


@dataclass(frozen=True)
class PointsScaling:
    """Linear map between log-odds and score points."""

    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        """Points per unit of log-odds: ``PDO / ln 2``."""
        return self.pdo / float(np.log(2.0))

    @property
    def offset(self) -> float:
        """Score at even odds: ``base_score - factor * ln(base_odds)``."""
        return self.base_score - self.factor * float(np.log(self.base_odds))

    def score_from_pd(self, pd_: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Convert probabilities of default to scores."""
        p = np.asarray(pd_, dtype=float)
        return np.asarray(self.offset + self.factor * np.log((1.0 - p) / p), dtype=float)

    def pd_from_score(self, score: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Convert scores back to probabilities of default."""
        log_odds_good = (np.asarray(score, dtype=float) - self.offset) / self.factor
        return 1.0 / (1.0 + np.exp(log_odds_good))


def fit_logit(X: pd.DataFrame, y: pd.Series | npt.ArrayLike) -> Any:
    """Fit ``P(bad) = logistic(b0 + X b)`` by maximum likelihood (statsmodels)."""
    design = sm.add_constant(X, has_constant="add")
    return sm.Logit(np.asarray(y, dtype=float), design).fit(disp=0, maxiter=200)


def backward_elimination(
    X: pd.DataFrame, y: pd.Series | npt.ArrayLike, *, max_pvalue: float = 0.05
) -> tuple[Any, list[str], pd.DataFrame]:
    """Refit the logit, dropping one offending characteristic at a time.

    Wrong-signed coefficients (``>= 0``) are removed first, the least significant
    of them first. Then characteristics with ``p > max_pvalue`` are removed,
    largest p-value first.

    Returns
    -------
    tuple
        Final statsmodels result, kept characteristics and a log of removals.
    """
    features = list(X.columns)
    log: list[dict[str, object]] = []
    while True:
        result = fit_logit(X[features], y)
        coefs = result.params.drop(INTERCEPT)
        pvalues = result.pvalues.drop(INTERCEPT)
        wrong_sign = coefs[coefs >= 0].index
        if len(wrong_sign):
            worst = str(pvalues[wrong_sign].idxmax())
            reason = f"coefficient {coefs[worst]:+.3f} >= 0 (inconsistent with WoE)"
        elif len(pvalues) and pvalues.max() > max_pvalue:
            worst = str(pvalues.idxmax())
            reason = f"p-value {pvalues[worst]:.3g} > {max_pvalue}"
        else:
            break
        log.append({"feature": worst, "stage": "logit", "decision": "dropped", "reason": reason})
        features.remove(worst)
        if not features:
            raise ValueError("Backward elimination removed every characteristic.")
    columns = ["feature", "stage", "decision", "reason"]
    return result, features, pd.DataFrame(log, columns=columns)


@dataclass(eq=False)
class Scorecard:
    """WoE logistic scorecard.

    Attributes
    ----------
    binnings
        Binning of each characteristic in the model.
    params
        Logit coefficients, including the intercept (``"const"``).
    scaling
        Points scaling.
    summary
        Coefficient table (estimate, standard error, z, p-value, 95% CI).
    fit_stats
        Log-likelihood, pseudo R-squared, AIC, BIC and sample size.
    elimination_log
        Characteristics removed during backward elimination.
    """

    binnings: Mapping[str, Binning]
    params: pd.Series
    scaling: PointsScaling = field(default_factory=PointsScaling)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    fit_stats: dict[str, float] = field(default_factory=dict)
    elimination_log: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        y: pd.Series | npt.ArrayLike,
        encoder: WoEEncoder,
        features: Sequence[str],
        *,
        scaling: PointsScaling | None = None,
        max_pvalue: float = 0.05,
    ) -> Scorecard:
        """Fit the scorecard on raw characteristics ``X`` using a fitted ``encoder``."""
        woe = encoder.transform(X, features)
        result, kept, log = backward_elimination(woe, y, max_pvalue=max_pvalue)
        ci = result.conf_int()
        summary = pd.DataFrame(
            {
                "coef": result.params,
                "std_err": result.bse,
                "z": result.tvalues,
                "p_value": result.pvalues,
                "ci_low": ci[0],
                "ci_high": ci[1],
            }
        )
        fit_stats = {
            "n_obs": float(result.nobs),
            "log_likelihood": float(result.llf),
            "pseudo_r2": float(result.prsquared),
            "aic": float(result.aic),
            "bic": float(result.bic),
        }
        return cls(
            binnings={f: encoder.binnings_[f] for f in kept},
            params=result.params.copy(),
            scaling=scaling or PointsScaling(),
            summary=summary,
            fit_stats=fit_stats,
            elimination_log=log,
        )

    @property
    def features(self) -> list[str]:
        """Characteristics in the model."""
        return [str(f) for f in self.params.index if f != INTERCEPT]

    def woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """WoE-encode the model's characteristics."""
        return pd.DataFrame(
            {f: self.binnings[f].transform(X[f]) for f in self.features}, index=X.index
        )

    def log_odds_bad(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Linear predictor ``b0 + sum_j b_j WoE_j``."""
        coefs = self.params[self.features].to_numpy(dtype=float)
        linear = float(self.params[INTERCEPT]) + self.woe(X).to_numpy(dtype=float) @ coefs
        return np.asarray(linear, dtype=float)

    def predict_pd(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Probability of default."""
        return 1.0 / (1.0 + np.exp(-self.log_odds_bad(X)))

    def score(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Exact (unrounded) score: ``offset - factor * ln(odds_bad)``."""
        return np.asarray(
            self.scaling.offset - self.scaling.factor * self.log_odds_bad(X), dtype=float
        )

    def points_table(self) -> pd.DataFrame:
        """Points of every bin of every characteristic (rounded and exact)."""
        n = len(self.features)
        intercept = float(self.params[INTERCEPT])
        frames = []
        for feature in self.features:
            table = self.binnings[feature].table
            coef = float(self.params[feature])
            exact = self.scaling.offset / n - self.scaling.factor * (
                coef * table["woe"] + intercept / n
            )
            frames.append(
                pd.DataFrame(
                    {
                        "feature": feature,
                        "bin": table["bin"],
                        "count_pct": table["count_pct"],
                        "bad_rate": table["bad_rate"],
                        "woe": table["woe"],
                        "coef": coef,
                        "points_exact": exact,
                        "points": exact.round().astype(int),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def points(self, X: pd.DataFrame) -> npt.NDArray[np.int64]:
        """Integer score obtained by adding the rounded points of each bin."""
        table = self.points_table()
        total = np.zeros(len(X), dtype=np.int64)
        for feature in self.features:
            pts = table.loc[table["feature"] == feature, "points"].to_numpy(dtype=np.int64)
            total += pts[self.binnings[feature].assign(X[feature])]
        return total
