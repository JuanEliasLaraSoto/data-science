"""Tests for Lending Club / German Credit cleaning and the sample split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk_lab import data


def raw_lending_club() -> pd.DataFrame:
    """A tiny frame in the format of ``accepted_2007_to_2018Q4.csv.gz``."""
    rows = [
        # id, term, issue_d, loan_status, application_type
        ("1", " 36 months", "Jan-2013", "Fully Paid", "Individual"),
        ("2", " 36 months", "Feb-2013", "Charged Off", "Individual"),
        ("3", " 60 months", "Mar-2013", "Fully Paid", "Individual"),  # wrong term
        ("4", " 36 months", "Apr-2013", "Current", "Individual"),  # unresolved
        ("5", " 36 months", "May-2013", "Fully Paid", "Joint App"),  # joint
        (
            "6",
            " 36 months",
            "Jun-2010",
            "Does not meet the credit policy. Status:Charged Off",
            "Individual",
        ),
        ("7", " 36 months", "Jul-2015", "Default", "Individual"),
        ("8", " 36 months", "Aug-2015", "Late (31-120 days)", "Individual"),
        ("Total amount funded in policy code 1: 123", None, None, None, None),  # footer
    ]
    frame = pd.DataFrame(rows, columns=["id", "term", "issue_d", "loan_status", "application_type"])
    n = len(frame)
    frame["loan_amnt"] = [10000.0] * n
    frame["annual_inc"] = [50000.0, 0.0, *[40000.0] * (n - 2)]
    frame["fico_range_low"] = [700.0] * n
    frame["fico_range_high"] = [704.0] * n
    frame["earliest_cr_line"] = ["Jan-2003"] * n
    frame["emp_length"] = ["10+ years", "< 1 year", "3 years", "n/a", *["1 year"] * (n - 4)]
    frame["revol_util"] = ["45.3%"] * n
    frame["home_ownership"] = ["RENT"] * n
    return frame


def test_load_lending_club_skips_leaky_and_excluded_columns(tmp_path: Path) -> None:
    raw = raw_lending_club().assign(total_pymnt=1.0, recoveries=0.0, grade="A", zip_code="123xx")
    path = tmp_path / "accepted.csv.gz"
    raw.to_csv(path, index=False)
    loaded = data.load_lending_club(path)
    assert not {"total_pymnt", "recoveries", "grade", "zip_code"} & set(loaded.columns)
    assert {"loan_status", "issue_d", "fico_range_low"} <= set(loaded.columns)


def test_clean_lending_club_filters_and_target() -> None:
    clean, log = data.clean_lending_club(raw_lending_club())
    assert clean["id"].tolist() == ["1", "2", "7"]
    assert clean[data.TARGET].tolist() == [0, 1, 1]
    assert list(clean.columns[:4]) == list(data.META_COLUMNS)
    assert log["rows"].tolist() == sorted(log["rows"], reverse=True)
    assert log["rows"].iloc[0] == 9
    assert log["rows"].iloc[-1] == 3
    assert not set(data.SAMPLE_DEFINITION_COLUMNS) & set(clean.columns)
    assert str(clean[data.VINTAGE].iloc[0]) == "2013Q1"


def test_clean_lending_club_engineers_features() -> None:
    clean, _ = data.clean_lending_club(raw_lending_club())
    first = clean.iloc[0]
    assert first["fico_score"] == 702
    assert first["credit_history_months"] == 120  # Jan-2003 -> Jan-2013
    assert clean["emp_length"].tolist()[:2] == [10.0, 0.0]
    assert first["revol_util"] == pytest.approx(45.3)
    assert first["loan_to_income"] == pytest.approx(0.2)
    assert np.isnan(clean["loan_to_income"].iloc[1])  # zero income
    assert not {"fico_range_low", "fico_range_high", "earliest_cr_line"} & set(clean.columns)


def test_emp_length_handles_missing() -> None:
    clean, _ = data.clean_lending_club(raw_lending_club().assign(emp_length="n/a"))
    assert clean["emp_length"].isna().all()


def test_exclusion_lists_are_documented_and_disjoint() -> None:
    table = data.exclusion_table()
    assert not set(data.LEAKAGE_COLUMNS) & set(data.EXCLUDED_COLUMNS)
    assert table["reason"].str.len().gt(10).all()
    assert {"recoveries", "last_pymnt_d", "last_fico_range_high", "funded_amnt_inv"} <= set(
        data.LEAKAGE_COLUMNS
    )
    assert {"grade", "sub_grade", "int_rate"} <= set(data.EXCLUDED_COLUMNS)


def test_maturity_by_issue_year() -> None:
    table = data.maturity_by_issue_year(raw_lending_club())
    assert table.loc[2013, "loans"] == 4  # 60-month and policy loans excluded
    assert table.loc[2013, "resolved_share"] == pytest.approx(0.75)
    assert table.loc[2013, "default_rate_resolved"] == pytest.approx(1 / 3)
    assert table.loc[2015, "resolved_share"] == pytest.approx(0.5)


# ------------------------------------------------------------------ split


def dated_sample(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    issue = pd.to_datetime("2012-01-01") + pd.to_timedelta(rng.integers(0, 4 * 365, n), "D")
    return pd.DataFrame(
        {
            data.LOAN_ID: np.arange(n),
            data.ISSUE_DATE: issue,
            data.VINTAGE: issue.to_period("Q"),
            data.TARGET: rng.binomial(1, 0.15, n),
            "x": rng.normal(size=n),
        }
    )


def test_split_respects_windows_and_stratification() -> None:
    df = dated_sample()
    design = data.SampleDesign(
        dev_start="2012-01", dev_end="2014-12", oot_start="2015-01", oot_end="2015-12"
    )
    parts = data.split_sample(df, design)
    assert set(parts) == {"train", "test", "oot"}
    assert parts["oot"][data.ISSUE_DATE].dt.year.eq(2015).all()
    assert parts["train"][data.ISSUE_DATE].dt.year.le(2014).all()
    ids = [set(p[data.LOAN_ID]) for p in parts.values()]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    dev = len(parts["train"]) + len(parts["test"])
    assert len(parts["test"]) / dev == pytest.approx(0.3, abs=0.01)
    rates = [parts[k][data.TARGET].mean() for k in ("train", "test")]
    assert rates[0] == pytest.approx(rates[1], abs=0.01)


def test_split_is_reproducible() -> None:
    df = dated_sample()
    first = data.split_sample(df)["test"][data.LOAN_ID].tolist()
    assert first == data.split_sample(df)["test"][data.LOAN_ID].tolist()


def test_overlapping_windows_are_rejected() -> None:
    design = data.SampleDesign(dev_end="2015-06", oot_start="2015-01")
    with pytest.raises(ValueError, match="overlap"):
        data.split_sample(dated_sample(), design)


def test_split_without_dates_is_random() -> None:
    df = dated_sample().drop(columns=[data.ISSUE_DATE, data.VINTAGE])
    parts = data.split_sample(df)
    assert set(parts) == {"train", "test"}
    assert len(parts["train"]) + len(parts["test"]) == len(df)


def test_rare_strata_fall_back_to_target_only() -> None:
    df = dated_sample(n=200)
    df.loc[df.index[0], data.ISSUE_DATE] = pd.Timestamp("2008-03-01")  # lonely year
    design = data.SampleDesign(dev_start="2008-01")
    parts = data.split_sample(df, design)
    assert len(parts["train"]) + len(parts["test"]) == (df[data.ISSUE_DATE].dt.year <= 2014).sum()


# ------------------------------------------------------------------ German Credit


def test_clean_german_credit() -> None:
    raw = pd.DataFrame(
        {
            "duration": [6, 48, 12],
            "personal_status": ["male single"] * 3,
            "foreign_worker": ["yes"] * 3,
            "class": ["good", "bad", "good"],
        }
    )
    clean, log = data.clean_german_credit(raw)
    assert list(clean.columns) == [data.LOAN_ID, data.TARGET, "duration"]
    assert clean[data.TARGET].tolist() == [0, 1, 0]
    assert len(log) == 2
