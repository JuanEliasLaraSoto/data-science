"""Challenger models used to benchmark the scorecard.

The challengers (decision tree, random forest, gradient boosting) get the same
characteristics as the final scorecard, but in raw form. They can choose their
own split points and model interactions, which the additive WoE logit cannot.
The gap in Gini is therefore the performance cost of the scorecard's
interpretability constraints.

Hyperparameters are tuned with a small grid and stratified 3-fold
cross-validation (AUC) on the training sample only. No class weights are used,
so the predicted probabilities remain comparable with the scorecard's PD.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# Leaf sizes are fractions of the training sample, so the same grid spans strong and
# weak regularisation on both a 700-loan and a 200,000-loan training sample.
PARAM_GRIDS: Mapping[str, Mapping[str, Sequence[Any]]] = {
    "Decision tree": {"model__max_depth": [3, 5, 7], "model__min_samples_leaf": [0.005, 0.02]},
    "Random forest": {"model__max_depth": [6, 10], "model__min_samples_leaf": [0.001, 0.01]},
    "Gradient boosting": {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 31]},
}


@dataclass(frozen=True, eq=False)
class ChallengerResult:
    """A tuned challenger model.

    Attributes
    ----------
    name
        Display name.
    model
        Fitted pipeline (encoding + estimator), refitted on the full training sample.
    best_params
        Hyperparameters chosen by cross-validation.
    cv_auc
        Mean cross-validated AUC of the chosen hyperparameters.
    """

    name: str
    model: Pipeline
    best_params: dict[str, Any]
    cv_auc: float

    def predict_pd(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Predicted probability of default."""
        return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)


def categorical_columns(X: pd.DataFrame) -> list[str]:
    """Columns that are not numeric (treated as nominal)."""
    return [
        str(c)
        for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])
    ]


def _encoder(X: pd.DataFrame) -> ColumnTransformer:
    """Ordinal-encode nominal columns (first), pass numeric columns through.

    Missing and unseen categories become ``NaN``, which all three estimators
    handle natively, so no imputation is needed.
    """
    categorical = categorical_columns(X)
    numeric = [str(c) for c in X.columns if c not in categorical]
    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
    )
    return ColumnTransformer(
        [("categorical", ordinal, categorical), ("numeric", "passthrough", numeric)],
        verbose_feature_names_out=False,
    )


def make_challengers(X: pd.DataFrame, *, random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Unfitted challenger pipelines for the columns of ``X``."""
    n_categorical = len(categorical_columns(X))
    is_categorical = [i < n_categorical for i in range(X.shape[1])]
    estimators: dict[str, Any] = {
        "Decision tree": DecisionTreeClassifier(random_state=random_state),
        "Random forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=random_state
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            categorical_features=is_categorical if any(is_categorical) else None,
            random_state=random_state,
        ),
    }
    return {
        name: Pipeline([("encode", _encoder(X)), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def fit_challengers(
    X: pd.DataFrame,
    y: pd.Series | npt.ArrayLike,
    *,
    cv: int = 3,
    random_state: int = RANDOM_STATE,
    param_grids: Mapping[str, Mapping[str, Sequence[Any]]] = PARAM_GRIDS,
) -> dict[str, ChallengerResult]:
    """Tune every challenger by grid search and refit it on the whole of ``X``."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    target = np.asarray(y)
    results = {}
    for name, pipeline in make_challengers(X, random_state=random_state).items():
        search = GridSearchCV(
            pipeline,
            {k: list(v) for k, v in param_grids[name].items()},
            scoring="roc_auc",
            cv=folds,
            refit=True,
        )
        search.fit(X, target)
        results[name] = ChallengerResult(
            name=name,
            model=search.best_estimator_,
            best_params={k.removeprefix("model__"): v for k, v in search.best_params_.items()},
            cv_auc=float(search.best_score_),
        )
    return results
