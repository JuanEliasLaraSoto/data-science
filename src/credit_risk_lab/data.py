"""Data acquisition, cleaning and sample splitting.

Phase 0 implements data acquisition only:

* Lending Club accepted loans 2007-2018 (Kaggle ``wordsforthewise/lending-club``),
  downloaded through the public Kaggle REST API (v1) with the user's API token.
* Macro series from FRED through the public ``fredgraph.csv`` endpoint, which
  requires no API key.
* German Credit (OpenML ``credit-g``) as a fallback dataset. It has no dates, so
  it cannot support an out-of-time sample.

Cleaning, target definition, removal of post-origination (leaky) fields and the
train/test/out-of-time split are added in Phase 1.
"""

from __future__ import annotations

import base64
import csv
import gzip
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from sklearn.datasets import fetch_openml

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


def download_german_credit(raw_dir: Path = RAW_DIR, *, force: bool = False) -> Path:
    """Download German Credit (OpenML ``credit-g``, version 1) as a CSV fallback dataset.

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
    except (OSError, ValueError) as exc:
        raise DataDownloadError(
            f"Could not download OpenML '{OPENML_GERMAN_CREDIT}': {exc}. "
            "Check that network access to www.openml.org and api.openml.org is allowed."
        ) from exc
    bunch.frame.to_csv(target, index=False)
    logger.info("Saved German Credit (%d rows) to %s", len(bunch.frame), target)
    return target
