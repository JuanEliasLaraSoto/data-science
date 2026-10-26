from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk_lab.benchmark import categorical_columns, fit_challengers, make_challengers

SMALL_GRIDS = {
    "Decision tree": {"model__max_depth": [3]},
    "Random forest": {"model__max_depth": [4], "model__n_estimators": [20]},
    "Gradient boosting": {"model__learning_rate": [0.1], "model__max_iter": [30]},
}


def test_categorical_columns(credit_sample: pd.DataFrame) -> None:
    X = credit_sample.drop(columns="target").assign(flag=True)
    assert categorical_columns(X) == ["purpose", "flag"]


def test_make_challengers_has_three_models(credit_sample: pd.DataFrame) -> None:
    models = make_challengers(credit_sample.drop(columns="target"))
    assert list(models) == ["Decision tree", "Random forest", "Gradient boosting"]


def test_challengers_learn_and_handle_missing_and_unseen(credit_sample: pd.DataFrame) -> None:
    X = credit_sample.drop(columns="target")
    results = fit_challengers(X, credit_sample["target"], param_grids=SMALL_GRIDS)
    new = X.head(5).assign(purpose=["boat", None, "car", "debt", "home"], dti=np.nan)
    for name, result in results.items():
        assert result.name == name
        assert result.cv_auc > 0.6
        pds = result.predict_pd(new)
        assert pds.shape == (5,)
        assert ((pds >= 0) & (pds <= 1)).all()
    assert results["Decision tree"].best_params == {"max_depth": 3}


def test_challengers_without_categorical_columns(credit_sample: pd.DataFrame) -> None:
    X = credit_sample[["fico", "dti"]]
    results = fit_challengers(X, credit_sample["target"], param_grids=SMALL_GRIDS)
    assert results["Gradient boosting"].cv_auc > 0.6
