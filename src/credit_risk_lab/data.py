"""Data acquisition, cleaning and sample splitting.

Sources
-------
* Lending Club accepted loans 2007-2018 (Kaggle ``wordsforthewise/lending-club``),
  downloaded through the public Kaggle REST API (v1) with the user's API token.
* Macro series from FRED through the public ``fredgraph.csv`` endpoint, which
  requires no API key.
* German Credit (OpenML ``credit-g``) as a fallback dataset. It has no dates, so
  it cannot support an out-of-time sample.

Lending Club sample design
--------------------------
* **Target.** ``1`` for ``Charged Off`` or ``Default``, ``0`` for ``Fully Paid``.
  It is a lifetime (36-month) default flag. Unresolved loans (``Current``,
  ``Late``, ``In Grace Period``) are dropped.
* **Segment.** Individual applications for 36-month loans, issued under the
  current credit policy. Loans flagged "Does not meet the credit policy" were
  underwritten under older rules, and joint applications are a different
  population.
* **Maturity.** The development window and the out-of-time window only contain
  vintages that have had 36 months to run off. Otherwise the "resolved" loans of
  recent vintages would over-represent early defaults and early repayments.
* **Leakage.** Every field known only after origination is dropped; see
  :data:`LEAKAGE_COLUMNS` for the list and the reason for each.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import logging
import os
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DATA_DIR: Final = PROJECT_ROOT / "data"
RAW_DIR: Final = DATA_DIR / "raw"
PROCESSED_DIR: Final = DATA_DIR / "processed"

KAGGLE_API_URL: Final = "https://www.kaggle.com/api/v1"
LENDING_CLUB_DATASET: Final = "wordsforthewise/lending-club"
LENDING_CLUB_FILE: Final = "accepted_2007_to_2018Q4.csv.gz"
# Minimal set of columns that must be present for the file to be usable.
LENDING_CLUB_REQUIRED_COLUMNS: Final = frozenset({"loan_amnt", "issue_d", "loan_status"})

FRED_CSV_URL: Final = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES: Final[Mapping[str, str]] = {
    "UNRATE": "Unemployment rate, %, monthly, seasonally adjusted",
    "A191RL1Q225SBEA": "Real GDP growth, % change from previous quarter (SAAR), quarterly",
    "FEDFUNDS": "Effective federal funds rate, %, monthly",
}

OPENML_GERMAN_CREDIT: Final = "credit-g"

USER_AGENT: Final = "credit-risk-lab/0.1 (+https://github.com/JuanEliasLaraSoto/data-science)"
_CHUNK_SIZE: Final = 1 << 20  # 1 MiB
_LOG_EVERY_BYTES: Final = 50 << 20  # 50 MiB
_TIMEOUT_SECONDS: Final = 60.0
_GZIP_MAGIC: Final = b"\x1f\x8b"
_ZIP_MAGIC: Final = b"PK\x03\x04"


class DataDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or fails validation.

    Parameters
    ----------
    message
        Human-readable description of the failure.
    status
        HTTP status code, when the failure was an HTTP error response.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class KaggleCredentialsError(DataDownloadError):
    """Raised when no usable Kaggle API credentials can be found."""


@dataclass(frozen=True)
class KaggleCredentials:
    """Kaggle API token. The key is excluded from ``repr`` to avoid leaking it in logs."""

    username: str
    key: str = field(repr=False)

    def basic_auth_header(self) -> str:
        """Return the value of the HTTP ``Authorization`` header (basic auth)."""
        token = base64.b64encode(f"{self.username}:{self.key}".encode()).decode("ascii")
        return f"Basic {token}"


def load_kaggle_credentials(
    env: Mapping[str, str] | None = None, config_dir: Path | None = None
) -> KaggleCredentials:
    """Find Kaggle API credentials.

    Resolution order mirrors the official Kaggle client: the environment
    variables ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` first, then
    ``kaggle.json`` in ``config_dir`` (default ``$KAGGLE_CONFIG_DIR`` or
    ``~/.kaggle``).

    Parameters
    ----------
    env
        Environment mapping; defaults to ``os.environ``.
    config_dir
        Directory containing ``kaggle.json``.

    Returns
    -------
    KaggleCredentials
        The username and API key.

    Raises
    ------
    KaggleCredentialsError
        If no credentials are found or ``kaggle.json`` is malformed.
    """
    env = os.environ if env is None else env
    username, key = env.get("KAGGLE_USERNAME"), env.get("KAGGLE_KEY")
    if username and key:
        return KaggleCredentials(username=username, key=key)

    if config_dir is None:
        config_dir = Path(env.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    path = config_dir / "kaggle.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return KaggleCredentials(username=str(payload["username"]), key=str(payload["key"]))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise KaggleCredentialsError(
                f"{path} is not a valid kaggle.json (expected 'username' and 'key')."
            ) from exc

    raise KaggleCredentialsError(
        "No Kaggle credentials found. Set KAGGLE_USERNAME and KAGGLE_KEY, or place "
        f"kaggle.json in {config_dir} (create a token at https://www.kaggle.com/settings)."
    )


def kaggle_file_url(dataset: str, file_name: str | None = None) -> str:
    """Build the Kaggle API v1 download URL for a whole dataset or a single file."""
    url = f"{KAGGLE_API_URL}/datasets/download/{dataset}"
    if file_name is not None:
        url += "/" + urllib.parse.quote(file_name)
    return url


def fred_csv_url(series_id: str) -> str:
    """Build the public (key-less) FRED CSV download URL for a series."""
    return f"{FRED_CSV_URL}?{urllib.parse.urlencode({'id': series_id})}"


def _request(url: str, credentials: KaggleCredentials | None = None) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if credentials is not None:
        # Unredirected: Kaggle redirects to a pre-signed storage URL, which must not
        # receive (and may reject) the Kaggle credentials.
        request.add_unredirected_header("Authorization", credentials.basic_auth_header())
    return request


def _download(request: urllib.request.Request, destination: Path) -> Path:
    """Stream ``request`` to ``destination`` atomically (via a ``.part`` file)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    url = request.full_url
    try:
        with (
            urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response,
            partial.open("wb") as fh,
        ):
            total = response.headers.get("Content-Length")
            copied, next_log = 0, _LOG_EVERY_BYTES
            while chunk := response.read(_CHUNK_SIZE):
                fh.write(chunk)
                copied += len(chunk)
                if copied >= next_log:
                    logger.info("  %s: %d MiB / %s", destination.name, copied >> 20, total or "?")
                    next_log += _LOG_EVERY_BYTES
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise DataDownloadError(
            f"HTTP {exc.code} while downloading {url}: {exc.reason}", status=exc.code
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        host = urllib.parse.urlsplit(url).hostname
        raise DataDownloadError(
            f"Could not download {url}: {exc}. Check that network access to {host} is allowed."
        ) from exc
    partial.replace(destination)
    logger.info("Downloaded %s (%d bytes)", destination, destination.stat().st_size)
    return destination


def _read_csv_header(lines: Iterable[str]) -> list[str]:
    reader = csv.reader(lines)
    return [column.strip() for column in next(reader, [])]


def _validate_lending_club(path: Path) -> None:
    try:
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as fh:
            header = _read_csv_header(fh)
    except (OSError, UnicodeDecodeError) as exc:
        raise DataDownloadError(f"{path} is not a readable gzip-compressed CSV.") from exc
    missing = LENDING_CLUB_REQUIRED_COLUMNS - set(header)
    if missing:
        raise DataDownloadError(f"{path} is missing expected columns: {sorted(missing)}.")


def _extract_from_zip(archive: Path, member_name: str, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        matches = [n for n in zf.namelist() if Path(n).name.lower() == member_name.lower()]
        if not matches:
            raise DataDownloadError(
                f"{member_name} not found in Kaggle archive; members: {zf.namelist()}."
            )
        with zf.open(matches[0]) as src, destination.open("wb") as dst:
            while chunk := src.read(_CHUNK_SIZE):
                dst.write(chunk)


def _finalise_kaggle_file(downloaded: Path, member_name: str, destination: Path) -> None:
    """Turn a Kaggle payload (raw file or zip archive) into ``destination``."""
    with downloaded.open("rb") as fh:
        magic = fh.read(4)
    try:
        if magic.startswith(_ZIP_MAGIC):
            _extract_from_zip(downloaded, member_name, destination)
        elif magic.startswith(_GZIP_MAGIC):
            downloaded.replace(destination)
        else:
            raise DataDownloadError(
                f"Unexpected payload for {member_name}: neither a gzip file nor a zip archive."
            )
    finally:
        downloaded.unlink(missing_ok=True)


def download_lending_club(
    raw_dir: Path = RAW_DIR,
    *,
    force: bool = False,
    credentials: KaggleCredentials | None = None,
) -> Path:
    """Download the Lending Club accepted-loans file (2007-2018Q4) from Kaggle.

    The single-file endpoint is tried first (about 0.4 GB). If Kaggle answers 404
    for it, the whole dataset archive is downloaded and the file is extracted.

    Parameters
    ----------
    raw_dir
        Root directory for raw data; the file goes to ``raw_dir/lending_club``.
    force
        Re-download even if the file already exists.
    credentials
        Kaggle credentials; resolved with :func:`load_kaggle_credentials` if omitted.

    Returns
    -------
    Path
        Path to ``accepted_2007_to_2018Q4.csv.gz``.
    """
    target = raw_dir / "lending_club" / LENDING_CLUB_FILE
    if target.exists() and not force:
        logger.info("Lending Club file already present at %s; skipping.", target)
        return target

    credentials = credentials or load_kaggle_credentials()
    staging = target.with_name(target.name + ".download")
    try:
        _download(
            _request(kaggle_file_url(LENDING_CLUB_DATASET, LENDING_CLUB_FILE), credentials), staging
        )
    except DataDownloadError as exc:
        if exc.status != 404:
            raise
        logger.warning("Single-file download not available; fetching the whole dataset archive.")
        _download(_request(kaggle_file_url(LENDING_CLUB_DATASET), credentials), staging)

    _finalise_kaggle_file(staging, LENDING_CLUB_FILE, target)
    try:
        _validate_lending_club(target)
    except DataDownloadError:
        target.unlink()
        raise
    return target


def _validate_fred_csv(path: Path, series_id: str) -> None:
    with path.open(encoding="utf-8", newline="") as fh:
        header = _read_csv_header(fh)
    if len(header) != 2 or header[1] != series_id:
        raise DataDownloadError(
            f"FRED response for {series_id} is not the expected CSV (header: {header[:5]})."
        )


def download_fred_series(series_id: str, raw_dir: Path = RAW_DIR, *, force: bool = False) -> Path:
    """Download one FRED series as CSV (``date, value``) to ``raw_dir/fred``.

    Parameters
    ----------
    series_id
        FRED series identifier, e.g. ``"UNRATE"``.
    raw_dir
        Root directory for raw data.
    force
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Path to ``<series_id>.csv``.
    """
    target = raw_dir / "fred" / f"{series_id}.csv"
    if target.exists() and not force:
        logger.info("FRED series %s already present at %s; skipping.", series_id, target)
        return target
    _download(_request(fred_csv_url(series_id)), target)
    try:
        _validate_fred_csv(target, series_id)
    except DataDownloadError:
        target.unlink()
        raise
    return target


def download_fred(
    series_ids: Sequence[str] = tuple(FRED_SERIES), raw_dir: Path = RAW_DIR, *, force: bool = False
) -> dict[str, Path]:
    """Download several FRED series; see :func:`download_fred_series`."""
    return {sid: download_fred_series(sid, raw_dir, force=force) for sid in series_ids}


# Fallback-of-the-fallback: the same UCI Statlog German Credit data (decoded labels)
# shipped inside a pinned PyPI release of scorecardpy. Only the CSV is read from the
# archive (nothing is executed) and its SHA-256 is checked.
GERMAN_CREDIT_MIRROR_URL: Final = (
    "https://files.pythonhosted.org/packages/56/5a/"
    "7bae50d0875192a86d2153535fa221596c9f0c83b86dd7c25b005aa37de9/scorecardpy-0.1.9.7.tar.gz"
)
GERMAN_CREDIT_MIRROR_MEMBER: Final = "scorecardpy-0.1.9.7/scorecardpy/data/germancredit.csv"
GERMAN_CREDIT_MIRROR_SHA256: Final = (
    "2c0bae00275c028fc853a1ea72cc7a68002c3f6876c41300c5c948711540c8c6"
)
#: Mirror column names -> OpenML ``credit-g`` column names.
GERMAN_CREDIT_MIRROR_COLUMNS: Final[Mapping[str, str]] = {
    "status_of_existing_checking_account": "checking_status",
    "duration_in_month": "duration",
    "credit_history": "credit_history",
    "purpose": "purpose",
    "credit_amount": "credit_amount",
    "savings_account_and_bonds": "savings_status",
    "present_employment_since": "employment",
    "installment_rate_in_percentage_of_disposable_income": "installment_commitment",
    "personal_status_and_sex": "personal_status",
    "other_debtors_or_guarantors": "other_parties",
    "present_residence_since": "residence_since",
    "property": "property_magnitude",
    "age_in_years": "age",
    "other_installment_plans": "other_payment_plans",
    "housing": "housing",
    "number_of_existing_credits_at_this_bank": "existing_credits",
    "job": "job",
    "number_of_people_being_liable_to_provide_maintenance_for": "num_dependents",
    "telephone": "own_telephone",
    "foreign_worker": "foreign_worker",
    "creditability": "class",
}


def _german_credit_from_mirror(work_dir: Path) -> pd.DataFrame:
    """Read German Credit from the pinned PyPI mirror and verify its checksum."""
    archive = _download(_request(GERMAN_CREDIT_MIRROR_URL), work_dir / "mirror.tar.gz")
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            member = tar.extractfile(GERMAN_CREDIT_MIRROR_MEMBER)
            if member is None:
                raise DataDownloadError(f"{GERMAN_CREDIT_MIRROR_MEMBER} is not a regular file.")
            payload = member.read()
    except (tarfile.TarError, KeyError) as exc:
        raise DataDownloadError(f"Could not read {GERMAN_CREDIT_MIRROR_MEMBER}: {exc}") from exc
    finally:
        archive.unlink(missing_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != GERMAN_CREDIT_MIRROR_SHA256:
        raise DataDownloadError(f"German Credit mirror checksum mismatch ({digest}).")
    return pd.read_csv(io.BytesIO(payload)).rename(columns=GERMAN_CREDIT_MIRROR_COLUMNS)


def download_german_credit(raw_dir: Path = RAW_DIR, *, force: bool = False) -> Path:
    """Download German Credit as a CSV fallback dataset, in the OpenML ``credit-g`` schema.

    OpenML (``credit-g``, version 1) is the primary source. If it is unreachable,
    the same UCI data are read from a pinned PyPI release (checksum-verified).

    Parameters
    ----------
    raw_dir
        Root directory for raw data; the file goes to ``raw_dir/german_credit``.
    force
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Path to ``credit_g.csv``.
    """
    target = raw_dir / "german_credit" / "credit_g.csv"
    if target.exists() and not force:
        logger.info("German Credit already present at %s; skipping.", target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        bunch = fetch_openml(
            name=OPENML_GERMAN_CREDIT,
            version=1,
            as_frame=True,
            data_home=str(target.parent / "openml_cache"),
        )
        frame, source = bunch.frame, "OpenML"
    except (OSError, ValueError) as openml_error:
        logger.warning("OpenML unreachable (%s); trying the PyPI mirror.", openml_error)
        try:
            frame, source = _german_credit_from_mirror(target.parent), "PyPI mirror"
        except DataDownloadError as mirror_error:
            raise DataDownloadError(
                f"Could not download German Credit from OpenML ({openml_error}) nor from the "
                f"PyPI mirror ({mirror_error}). Check network access to www.openml.org and "
                "api.openml.org."
            ) from mirror_error
    frame.to_csv(target, index=False)
    logger.info("Saved German Credit from %s (%d rows) to %s", source, len(frame), target)
    return target


# ============================================================================ cleaning

TARGET: Final = "target"
VINTAGE: Final = "vintage"
ISSUE_DATE: Final = "issue_d"
LOAN_ID: Final = "id"
META_COLUMNS: Final = (LOAN_ID, ISSUE_DATE, VINTAGE, TARGET)

GOOD_STATUSES: Final = ("Fully Paid",)
BAD_STATUSES: Final = ("Charged Off", "Default")
POLICY_STATUS_PREFIX: Final = "Does not meet the credit policy"


def _by_reason(groups: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {column: reason for reason, columns in groups.items() for column in columns}


#: Fields only known after origination: using them would leak the outcome.
LEAKAGE_COLUMNS: Final[Mapping[str, str]] = _by_reason(
    {
        "Repayment performance observed after origination": (
            "out_prncp",
            "out_prncp_inv",
            "total_pymnt",
            "total_pymnt_inv",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
            "recoveries",
            "collection_recovery_fee",
            "last_pymnt_d",
            "last_pymnt_amnt",
            "next_pymnt_d",
            "pymnt_plan",
        ),
        "Credit bureau data refreshed after origination": (
            "last_credit_pull_d",
            "last_fico_range_high",
            "last_fico_range_low",
        ),
        "Hardship programme: only exists once the borrower is in difficulty": (
            "hardship_flag",
            "hardship_type",
            "hardship_reason",
            "hardship_status",
            "deferral_term",
            "hardship_amount",
            "hardship_start_date",
            "hardship_end_date",
            "payment_plan_start_date",
            "hardship_length",
            "hardship_dpd",
            "hardship_loan_status",
            "orig_projected_additional_accrued_interest",
            "hardship_payoff_balance_amount",
            "hardship_last_payment_amount",
        ),
        "Debt settlement: only exists after charge-off or serious delinquency": (
            "debt_settlement_flag",
            "debt_settlement_flag_date",
            "settlement_status",
            "settlement_date",
            "settlement_amount",
            "settlement_percentage",
            "settlement_term",
        ),
        "Funding outcome, determined after the credit decision": (
            "funded_amnt",
            "funded_amnt_inv",
        ),
    }
)

#: Fields known at origination but excluded from the candidate characteristics.
EXCLUDED_COLUMNS: Final[Mapping[str, str]] = _by_reason(
    {
        "Lending Club's own risk grade and pricing: using it would make the scorecard "
        "a recalibration of the incumbent model": ("grade", "sub_grade", "int_rate", "installment"),
        "Identifier or free text": ("member_id", "url", "desc", "title", "emp_title"),
        "Geography: proxy for protected characteristics (fair lending)": (
            "zip_code",
            "addr_state",
        ),
        "Platform decision, not an applicant characteristic": (
            "initial_list_status",
            "disbursement_method",
            "policy_code",
        ),
        "Joint-application field (joint applications are outside the segment)": (
            "annual_inc_joint",
            "dti_joint",
            "verification_status_joint",
            "revol_bal_joint",
            "sec_app_fico_range_low",
            "sec_app_fico_range_high",
            "sec_app_earliest_cr_line",
            "sec_app_inq_last_6mths",
            "sec_app_mort_acc",
            "sec_app_open_acc",
            "sec_app_revol_util",
            "sec_app_open_act_il",
            "sec_app_num_rev_accts",
            "sec_app_chargeoff_within_12_mths",
            "sec_app_collections_12_mths_ex_med",
            "sec_app_mths_since_last_major_derog",
        ),
    }
)

#: Fields used to define the sample; constant (or the target) afterwards.
SAMPLE_DEFINITION_COLUMNS: Final = ("loan_status", "term", "application_type")


def exclusion_table() -> pd.DataFrame:
    """Every dropped Lending Club field with its category and reason."""
    rows = [
        *({"column": c, "category": "leakage", "reason": r} for c, r in LEAKAGE_COLUMNS.items()),
        *({"column": c, "category": "excluded", "reason": r} for c, r in EXCLUDED_COLUMNS.items()),
    ]
    return pd.DataFrame(rows)


def load_lending_club(path: Path) -> pd.DataFrame:
    """Read the accepted-loans file, skipping the leaky and excluded columns."""
    header = pd.read_csv(path, nrows=0).columns
    dropped = set(LEAKAGE_COLUMNS) | set(EXCLUDED_COLUMNS)
    usecols = [c for c in header if c not in dropped]
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _parse_month(values: pd.Series) -> pd.Series:
    """Parse Lending Club month strings such as ``"Dec-2015"``."""
    return pd.to_datetime(values, format="%b-%Y", errors="coerce")


def _parse_percent(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    return pd.to_numeric(values.astype(str).str.strip().str.rstrip("%"), errors="coerce")


def _emp_length_years(values: pd.Series) -> pd.Series:
    """``"< 1 year"`` -> 0, ``"5 years"`` -> 5, ``"10+ years"`` -> 10, other -> NaN."""
    text = values.astype(str)
    years = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce")
    return years.where(~text.str.startswith("<"), 0.0)


def engineer_lending_club_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive application-time characteristics and drop their raw inputs.

    * ``fico_score``: midpoint of the FICO range at application.
    * ``credit_history_months``: months between the oldest credit line and issue.
    * ``emp_length``: employment length in years (``10+`` capped at 10).
    * ``loan_to_income``: requested amount over annual income.
    """
    out = df.copy()
    if {"fico_range_low", "fico_range_high"} <= set(out.columns):
        out["fico_score"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
        out = out.drop(columns=["fico_range_low", "fico_range_high"])
    if "earliest_cr_line" in out.columns:
        opened = _parse_month(out["earliest_cr_line"])
        issued = out[ISSUE_DATE]
        out["credit_history_months"] = (issued.dt.year - opened.dt.year) * 12 + (
            issued.dt.month - opened.dt.month
        )
        out = out.drop(columns=["earliest_cr_line"])
    if "emp_length" in out.columns:
        out["emp_length"] = _emp_length_years(out["emp_length"])
    if "revol_util" in out.columns:
        out["revol_util"] = _parse_percent(out["revol_util"])
    if {"loan_amnt", "annual_inc"} <= set(out.columns):
        income = out["annual_inc"].where(out["annual_inc"] > 0)
        out["loan_to_income"] = out["loan_amnt"] / income
    return out


def _log_step(log: list[dict[str, object]], step: str, frame: pd.DataFrame) -> None:
    log.append({"step": step, "rows": len(frame)})


def clean_lending_club(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the segment filters, build the target and the characteristics.

    Parameters
    ----------
    raw
        Output of :func:`load_lending_club`.

    Returns
    -------
    tuple
        The modelling table (:data:`META_COLUMNS` plus candidate characteristics)
        and a log with the number of rows after each filter.
    """
    log: list[dict[str, object]] = []
    df = raw.copy()
    _log_step(log, "Raw file", df)

    status = df["loan_status"].astype(str)
    df = df[~status.str.startswith(POLICY_STATUS_PREFIX)]
    _log_step(log, "Exclude loans not meeting the current credit policy", df)

    term = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)", expand=False))
    df = df[term == 36]
    _log_step(log, "36-month term only", df)

    if "application_type" in df.columns:
        df = df[df["application_type"].astype(str).str.lower() == "individual"]
        _log_step(log, "Individual applications only", df)

    df = df.assign(**{ISSUE_DATE: _parse_month(df[ISSUE_DATE])})
    df = df[df[ISSUE_DATE].notna()]
    _log_step(log, "Valid issue date", df)

    df = df[df["loan_status"].isin([*GOOD_STATUSES, *BAD_STATUSES])]
    _log_step(log, "Resolved loans only (Fully Paid, Charged Off, Default)", df)

    df = df.assign(
        **{
            TARGET: df["loan_status"].isin(BAD_STATUSES).astype(np.int64),
            VINTAGE: df[ISSUE_DATE].dt.to_period("Q"),
        }
    )
    df = engineer_lending_club_features(df)
    df = df.drop(columns=[c for c in SAMPLE_DEFINITION_COLUMNS if c in df.columns])
    features = [c for c in df.columns if c not in META_COLUMNS]
    df = df[[c for c in META_COLUMNS if c in df.columns] + features].reset_index(drop=True)
    return df, pd.DataFrame(log)


def maturity_by_issue_year(raw: pd.DataFrame) -> pd.DataFrame:
    """Share of resolved loans and default rate by issue year (36-month, current policy).

    Shows which vintages have run off, and therefore which ones can be used
    without maturity bias.
    """
    status = raw["loan_status"].astype(str)
    term = pd.to_numeric(raw["term"].astype(str).str.extract(r"(\d+)", expand=False))
    keep = ~status.str.startswith(POLICY_STATUS_PREFIX) & (term == 36)
    frame = pd.DataFrame(
        {
            "year": _parse_month(raw.loc[keep, ISSUE_DATE]).dt.year,
            "resolved": status[keep].isin([*GOOD_STATUSES, *BAD_STATUSES]),
            "bad": status[keep].isin(BAD_STATUSES),
        }
    ).dropna(subset=["year"])
    table = frame.groupby("year").agg(loans=("resolved", "size"), resolved=("resolved", "sum"))
    table["resolved_share"] = table["resolved"] / table["loans"]
    resolved = frame[frame["resolved"]]
    table["default_rate_resolved"] = resolved.groupby("year")["bad"].mean()
    table.index = table.index.astype(int)
    return table


#: German Credit fields excluded on fair-lending grounds.
GERMAN_CREDIT_EXCLUDED: Final[Mapping[str, str]] = {
    "personal_status": "Sex and marital status: protected characteristic",
    "foreign_worker": "National origin: protected characteristic",
}


def clean_german_credit(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the modelling table for OpenML ``credit-g`` (target 1 = ``bad``)."""
    log: list[dict[str, object]] = []
    _log_step(log, "Raw file", raw)
    df = raw.drop(columns=[c for c in GERMAN_CREDIT_EXCLUDED if c in raw.columns])
    df[LOAN_ID] = np.arange(len(df))
    df[TARGET] = (df["class"] == "bad").astype(np.int64)
    df = df.drop(columns=["class"])
    features = [c for c in df.columns if c not in META_COLUMNS]
    _log_step(log, "Exclude protected characteristics", df)
    return df[[LOAN_ID, TARGET, *features]], pd.DataFrame(log)


# ============================================================================ split


@dataclass(frozen=True)
class SampleDesign:
    """Development and out-of-time windows (inclusive, ``YYYY-MM``).

    Development loans are split at random into train and test, stratified by
    target and issue year. Out-of-time loans are never used for fitting.
    """

    dev_start: str = "2007-01"
    dev_end: str = "2014-12"
    oot_start: str = "2015-01"
    oot_end: str = "2015-12"
    test_size: float = 0.3
    random_state: int = 42


def _stratified_split(
    df: pd.DataFrame, strata: pd.Series, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strata.value_counts().min() < 2:
        strata = df[TARGET]
    train, test = train_test_split(
        df, test_size=test_size, stratify=strata, random_state=random_state
    )
    return train, test


def split_sample(df: pd.DataFrame, design: SampleDesign | None = None) -> dict[str, pd.DataFrame]:
    """Split into ``train``/``test`` (and ``oot`` when the data have issue dates).

    Parameters
    ----------
    df
        Modelling table from :func:`clean_lending_club` or :func:`clean_german_credit`.
    design
        Sample windows. Without issue dates only a stratified random split is made.

    Returns
    -------
    dict
        ``{"train": ..., "test": ..., "oot": ...}`` (``oot`` only with dates).
    """
    design = design or SampleDesign()
    if ISSUE_DATE not in df.columns:
        train, test = _stratified_split(df, df[TARGET], design.test_size, design.random_state)
        return {"train": train, "test": test}

    month = df[ISSUE_DATE].dt.to_period("M")
    in_dev = (month >= pd.Period(design.dev_start, "M")) & (month <= pd.Period(design.dev_end, "M"))
    in_oot = (month >= pd.Period(design.oot_start, "M")) & (month <= pd.Period(design.oot_end, "M"))
    if (in_dev & in_oot).any():
        raise ValueError("Development and out-of-time windows overlap.")
    dev = df[in_dev]
    strata = dev[TARGET].astype(str) + "_" + dev[ISSUE_DATE].dt.year.astype(str)
    train, test = _stratified_split(dev, strata, design.test_size, design.random_state)
    return {"train": train, "test": test, "oot": df[in_oot]}
