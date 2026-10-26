from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_lab.woe import MISSING_LABEL, WoEEncoder, fit_binning, iv_strength


def non_missing(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["bin"] != MISSING_LABEL]


def is_strictly_monotonic(values: pd.Series) -> bool:
    diffs = np.diff(values.to_numpy())
    return bool((diffs > 0).all() or (diffs < 0).all())


# ------------------------------------------------------------------ numeric


def test_bad_rate_decreases_with_fico(credit_sample: pd.DataFrame) -> None:
    binning = fit_binning(credit_sample["fico"], credit_sample["target"])
    rates = binning.table["bad_rate"]
    assert binning.kind == "numeric"
    assert binning.n_bins >= 3
    assert (np.diff(rates) < 0).all()
    assert (np.diff(binning.table["woe"]) > 0).all()  # higher WoE = lower risk


def test_bad_rate_increases_with_dti(credit_sample: pd.DataFrame) -> None:
    binning = fit_binning(credit_sample["dti"], credit_sample["target"])
    assert (np.diff(non_missing(binning.table)["bad_rate"]) > 0).all()


@pytest.mark.parametrize("seed", range(5))
def test_binning_is_monotonic_and_respects_size_limits(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(size=3000), name="x")
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * x - 1.5))))
    binning = fit_binning(x, y, min_bin_frac=0.05, max_bins=6)
    table = binning.table
    assert is_strictly_monotonic(table["bad_rate"])
    assert len(table) <= 6
    assert (table["count"] >= 0.05 * len(x)).all()
    assert (table["goods"] > 0).all()
    assert (table["bads"] > 0).all()
    assert table["count"].sum() == len(x)


def test_forced_direction_against_the_data_pools_everything() -> None:
    x = pd.Series(np.arange(1000, dtype=float), name="x")
    y = (x < 300).astype(int)  # risk decreases with x
    binning = fit_binning(x, y, monotonic="increasing")
    assert binning.n_bins == 1
    assert binning.iv == pytest.approx(0.0)


def test_auto_direction_picks_the_informative_one() -> None:
    x = pd.Series(np.arange(1000, dtype=float), name="x")
    tenth = np.arange(1000) % 10 == 0
    y = np.where(x < 300, ~tenth, tenth).astype(int)  # bad rate 90% below 300, 10% above
    binning = fit_binning(x, y)
    assert binning.table["bad_rate"].tolist() == pytest.approx([0.9, 0.1], abs=0.01)
    assert binning.iv > 0.5


def test_intervals_are_right_closed() -> None:
    x = pd.Series([1.0] * 50 + [2.0] * 50 + [3.0] * 50, name="x")
    y = np.array([0] * 45 + [1] * 5 + [0] * 35 + [1] * 15 + [0] * 20 + [1] * 30)
    binning = fit_binning(x, y, min_bin_frac=0.1)
    assert binning.cuts == (1.0, 2.0)
    assert binning.table["bin"].tolist() == ["(-inf, 1]", "(1, 2]", "(2, inf)"]
    np.testing.assert_array_equal(
        binning.assign(pd.Series([0.5, 1.0, 1.5, 2.0, 9.0])), [0, 0, 1, 1, 2]
    )


def test_transform_maps_to_table_woe(credit_sample: pd.DataFrame) -> None:
    binning = fit_binning(credit_sample["fico"], credit_sample["target"])
    woe = binning.transform(credit_sample["fico"])
    assert set(np.round(woe, 12)) == set(np.round(binning.table["woe"], 12))


def test_iv_matches_hand_computation() -> None:
    x = pd.Series(["A"] * 100 + ["B"] * 100, name="x")
    y = np.array([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
    binning = fit_binning(x, y, min_bin_frac=0.1)
    # WoE_A = ln(0.8 / 0.2), WoE_B = ln(0.2 / 0.8); IV = 2 * 0.6 * ln 4.
    assert binning.table["woe"].tolist() == pytest.approx([np.log(4), -np.log(4)])
    assert binning.iv == pytest.approx(1.2 * np.log(4))


def test_noise_has_negligible_iv(credit_sample: pd.DataFrame) -> None:
    binning = fit_binning(credit_sample["noise"], credit_sample["target"])
    assert binning.iv < 0.02


# ------------------------------------------------------------------ missing values


def test_material_missing_values_get_their_own_bin(credit_sample: pd.DataFrame) -> None:
    binning = fit_binning(credit_sample["dti"], credit_sample["target"])
    assert binning.table["bin"].iloc[-1] == MISSING_LABEL
    assert binning.missing_bin == binning.n_bins - 1
    missing_rate = binning.table["bad_rate"].iloc[-1]
    assert missing_rate > credit_sample["target"].mean()


def test_rare_missing_values_are_pooled_with_closest_bin() -> None:
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=2000), name="x")
    y = rng.binomial(1, 1 / (1 + np.exp(-(x - 1.0))))
    x[:20] = np.nan
    binning = fit_binning(x, y)
    assert MISSING_LABEL not in binning.table["bin"].tolist()
    assert binning.missing_bin is not None
    assert binning.table["bin"].iloc[binning.missing_bin].endswith(f"| {MISSING_LABEL}")
    assert binning.table["count"].sum() == len(x)


def test_unseen_missing_values_go_to_riskiest_bin(credit_sample: pd.DataFrame) -> None:
    binning = fit_binning(credit_sample["fico"], credit_sample["target"])
    assert binning.missing_bin is None
    assigned = binning.assign(pd.Series([np.nan, 700.0]))
    assert assigned[0] == binning.riskiest_bin


def test_all_missing_column_is_a_single_missing_bin() -> None:
    x = pd.Series([np.nan] * 10, name="x")
    binning = fit_binning(x, np.array([0, 1] * 5))
    assert binning.table["bin"].tolist() == [MISSING_LABEL]
    assert binning.iv == pytest.approx(0.0)


# ------------------------------------------------------------------ categorical


def test_categories_are_ordered_by_bad_rate(credit_sample: pd.DataFrame) -> None:
    binning = fit_binning(credit_sample["purpose"], credit_sample["target"], min_bin_frac=0.05)
    assert binning.kind == "categorical"
    assert (np.diff(binning.table["bad_rate"]) > 0).all()
    assert set(binning.categories) == {"debt", "card", "home", "car"}


def test_rare_categories_are_merged() -> None:
    x = pd.Series(["a"] * 500 + ["b"] * 480 + ["rare"] * 20, name="x")
    y = np.array(([0] * 9 + [1]) * 50 + ([0] * 7 + [1] * 3) * 48 + [0, 1] * 10)
    binning = fit_binning(x, y, min_bin_frac=0.05)
    assert binning.n_bins == 2
    assert binning.categories["rare"] == binning.categories["b"]


def test_unseen_category_goes_to_riskiest_bin(credit_sample: pd.DataFrame) -> None:
    binning = fit_binning(credit_sample["purpose"], credit_sample["target"])
    assigned = binning.assign(pd.Series(["boat", "car", None]))
    assert assigned[0] == binning.riskiest_bin
    assert assigned[1] == binning.categories["car"]
    assert assigned[2] == binning.riskiest_bin


def test_boolean_is_treated_as_categorical() -> None:
    x = pd.Series([True] * 50 + [False] * 50, name="flag")
    y = np.array([0] * 40 + [1] * 10 + [0] * 25 + [1] * 25)
    assert fit_binning(x, y).kind == "categorical"


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("y", "message"),
    [
        (np.array([0, 1, 2, 1]), "binary"),
        (np.array([0, 0, 0, 0]), "both goods and bads"),
        (np.array([0, 1, 0]), "shape"),
    ],
)
def test_invalid_targets_are_rejected(y: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_binning(pd.Series([1.0, 2.0, 3.0, 4.0]), y)


def test_empty_sample_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_binning(pd.Series([], dtype=float), np.array([]))


@pytest.mark.parametrize("kwargs", [{"min_bin_frac": 0.0}, {"min_bin_frac": 0.6}, {"max_bins": 0}])
def test_invalid_parameters_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        fit_binning(pd.Series([1.0, 2.0]), np.array([0, 1]), **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------ encoder & IV labels


def test_encoder_fits_transforms_and_ranks(credit_sample: pd.DataFrame) -> None:
    X = credit_sample.drop(columns="target")
    encoder = WoEEncoder().fit(X, credit_sample["target"])
    woe = encoder.transform(X)
    assert list(woe.columns) == list(X.columns)
    assert woe.index.equals(X.index)
    ranking = encoder.iv_table()
    assert ranking["feature"].iloc[0] == "fico"
    assert ranking["feature"].iloc[-1] == "noise"
    assert list(encoder.transform(X, ["dti"]).columns) == ["dti"]


def test_unfitted_encoder_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        WoEEncoder().transform(pd.DataFrame({"x": [1.0]}))


@pytest.mark.parametrize(
    ("iv", "label"),
    [
        (0.01, "unpredictive"),
        (0.05, "weak"),
        (0.2, "medium"),
        (0.4, "strong"),
        (0.5, "strong"),
        (0.8, "suspicious"),
    ],
)
def test_iv_strength(iv: float, label: str) -> None:
    assert iv_strength(iv) == label
