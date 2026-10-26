"""Tests for the download layer. All network access is replaced by fakes."""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from credit_risk_lab import data

LC_HEADER = "id,member_id,loan_amnt,issue_d,loan_status\n1,,1000,Dec-2015,Fully Paid\n"


class FakeResponse(io.BytesIO):
    """Minimal stand-in for the object returned by ``urllib.request.urlopen``."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


def install_urlopen(monkeypatch: pytest.MonkeyPatch, *responses: bytes | Exception) -> list[Any]:
    """Replace urlopen with a fake that returns ``responses`` in order; return the requests."""
    queue = list(responses)
    seen: list[Any] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    return seen


def gz(text: str) -> bytes:
    return gzip.compress(text.encode())


def zipped(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buffer.getvalue()


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.org", code, "error", {}, None)  # type: ignore[arg-type]


CREDS = data.KaggleCredentials(username="user", key="secret")


# ------------------------------------------------------------------ credentials


def test_credentials_from_environment(tmp_path: Path) -> None:
    env = {"KAGGLE_USERNAME": "alice", "KAGGLE_KEY": "k123"}
    creds = data.load_kaggle_credentials(env=env, config_dir=tmp_path)
    assert creds == data.KaggleCredentials("alice", "k123")


def test_credentials_from_kaggle_json(tmp_path: Path) -> None:
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "bob", "key": "k456"}))
    creds = data.load_kaggle_credentials(env={"KAGGLE_CONFIG_DIR": str(tmp_path)})
    assert creds == data.KaggleCredentials("bob", "k456")


def test_missing_credentials_raise(tmp_path: Path) -> None:
    with pytest.raises(data.KaggleCredentialsError, match="No Kaggle credentials"):
        data.load_kaggle_credentials(env={}, config_dir=tmp_path)


def test_malformed_kaggle_json_raises(tmp_path: Path) -> None:
    (tmp_path / "kaggle.json").write_text(json.dumps({"user": "bob"}))
    with pytest.raises(data.KaggleCredentialsError, match=r"not a valid kaggle\.json"):
        data.load_kaggle_credentials(env={}, config_dir=tmp_path)


def test_credentials_repr_hides_key() -> None:
    assert "secret" not in repr(CREDS)


def test_basic_auth_header() -> None:
    # base64("user:secret")
    assert CREDS.basic_auth_header() == "Basic dXNlcjpzZWNyZXQ="


# ------------------------------------------------------------------ URLs


def test_kaggle_urls() -> None:
    assert data.kaggle_file_url("a/b") == "https://www.kaggle.com/api/v1/datasets/download/a/b"
    assert data.kaggle_file_url("a/b", "x y.csv").endswith("/a/b/x%20y.csv")


def test_fred_url() -> None:
    assert (
        data.fred_csv_url("UNRATE") == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    )


def test_authorization_header_is_not_forwarded_on_redirect() -> None:
    request = data._request("https://www.kaggle.com/x", CREDS)
    assert request.unredirected_hdrs["Authorization"] == CREDS.basic_auth_header()
    assert "Authorization" not in request.headers


# ------------------------------------------------------------------ FRED


def test_download_fred_series(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = install_urlopen(monkeypatch, b"observation_date,UNRATE\n2020-01-01,3.6\n")
    path = data.download_fred_series("UNRATE", tmp_path)
    assert path == tmp_path / "fred" / "UNRATE.csv"
    assert pd.read_csv(path)["UNRATE"].tolist() == [3.6]
    assert seen[0].full_url.endswith("id=UNRATE")
    assert not path.with_name("UNRATE.csv.part").exists()


def test_download_fred_rejects_unexpected_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_urlopen(monkeypatch, b"<html>Error</html>\n")
    with pytest.raises(data.DataDownloadError, match="not the expected CSV"):
        data.download_fred_series("UNRATE", tmp_path)
    assert not (tmp_path / "fred" / "UNRATE.csv").exists()


def test_download_fred_skips_existing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = install_urlopen(monkeypatch)
    existing = tmp_path / "fred" / "UNRATE.csv"
    existing.parent.mkdir(parents=True)
    existing.write_text("observation_date,UNRATE\n")
    assert data.download_fred_series("UNRATE", tmp_path) == existing
    assert seen == []


def test_download_fred_many(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_urlopen(monkeypatch, b"DATE,UNRATE\n", b"DATE,FEDFUNDS\n")
    paths = data.download_fred(("UNRATE", "FEDFUNDS"), tmp_path)
    assert set(paths) == {"UNRATE", "FEDFUNDS"}


# ------------------------------------------------------------------ errors


def test_network_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_urlopen(monkeypatch, urllib.error.URLError("Tunnel connection failed: 403"))
    with pytest.raises(data.DataDownloadError, match=r"fred\.stlouisfed\.org"):
        data.download_fred_series("UNRATE", tmp_path)
    assert list((tmp_path / "fred").iterdir()) == []


def test_http_errors_keep_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_urlopen(monkeypatch, http_error(500))
    with pytest.raises(data.DataDownloadError) as info:
        data.download_fred_series("UNRATE", tmp_path)
    assert info.value.status == 500


# ------------------------------------------------------------------ Lending Club


def lc_target(tmp_path: Path) -> Path:
    return tmp_path / "lending_club" / data.LENDING_CLUB_FILE


def test_lending_club_gzip_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = install_urlopen(monkeypatch, gz(LC_HEADER))
    path = data.download_lending_club(tmp_path, credentials=CREDS)
    assert path == lc_target(tmp_path)
    assert pd.read_csv(path)["loan_status"].tolist() == ["Fully Paid"]
    assert seen[0].full_url.endswith(f"/{data.LENDING_CLUB_DATASET}/{data.LENDING_CLUB_FILE}")


def test_lending_club_zipped_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = zipped({data.LENDING_CLUB_FILE: gz(LC_HEADER)})
    install_urlopen(monkeypatch, payload)
    path = data.download_lending_club(tmp_path, credentials=CREDS)
    assert pd.read_csv(path).shape == (1, 5)
    assert sorted(p.name for p in path.parent.iterdir()) == [data.LENDING_CLUB_FILE]


def test_lending_club_falls_back_to_full_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = zipped(
        {
            "rejected_2007_to_2018Q4.csv.gz": gz("x\n"),
            "accepted_2007_to_2018Q4.csv.gz": gz(LC_HEADER),
        }
    )
    seen = install_urlopen(monkeypatch, http_error(404), archive)
    path = data.download_lending_club(tmp_path, credentials=CREDS)
    assert pd.read_csv(path)["issue_d"].tolist() == ["Dec-2015"]
    assert seen[1].full_url.endswith(f"/datasets/download/{data.LENDING_CLUB_DATASET}")


def test_lending_club_other_http_errors_propagate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_urlopen(monkeypatch, http_error(401))
    with pytest.raises(data.DataDownloadError) as info:
        data.download_lending_club(tmp_path, credentials=CREDS)
    assert info.value.status == 401


def test_lending_club_archive_without_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_urlopen(monkeypatch, zipped({"other.csv": b"a\n"}))
    with pytest.raises(data.DataDownloadError, match="not found in Kaggle archive"):
        data.download_lending_club(tmp_path, credentials=CREDS)
    assert list(lc_target(tmp_path).parent.iterdir()) == []


def test_lending_club_unexpected_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_urlopen(monkeypatch, b"<html>login</html>")
    with pytest.raises(data.DataDownloadError, match="neither a gzip file nor a zip"):
        data.download_lending_club(tmp_path, credentials=CREDS)
    assert list(lc_target(tmp_path).parent.iterdir()) == []


def test_lending_club_missing_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_urlopen(monkeypatch, gz("id,loan_amnt\n1,1000\n"))
    with pytest.raises(data.DataDownloadError, match="missing expected columns"):
        data.download_lending_club(tmp_path, credentials=CREDS)
    assert not lc_target(tmp_path).exists()


def test_lending_club_corrupt_gzip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_urlopen(monkeypatch, b"\x1f\x8b" + b"\x00" * 20)
    with pytest.raises(data.DataDownloadError, match="not a readable gzip"):
        data.download_lending_club(tmp_path, credentials=CREDS)


def test_lending_club_skips_existing_and_needs_no_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    target = lc_target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(gz(LC_HEADER))
    assert data.download_lending_club(tmp_path) == target


def test_lending_club_resolves_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    monkeypatch.setenv("KAGGLE_KEY", "env-key")
    seen = install_urlopen(monkeypatch, gz(LC_HEADER))
    data.download_lending_club(tmp_path)
    expected = data.KaggleCredentials("env-user", "env-key").basic_auth_header()
    assert seen[0].unredirected_hdrs["Authorization"] == expected


# ------------------------------------------------------------------ German Credit


def test_download_german_credit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frame = pd.DataFrame({"duration": [6, 48], "class": ["good", "bad"]})
    calls: list[dict[str, Any]] = []

    def fake_fetch_openml(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(frame=frame)

    monkeypatch.setattr(data, "fetch_openml", fake_fetch_openml)
    path = data.download_german_credit(tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(path), frame)
    assert calls[0]["name"] == "credit-g"
    assert calls[0]["version"] == 1

    # Second call is a no-op.
    assert data.download_german_credit(tmp_path) == path
    assert len(calls) == 1


def test_download_german_credit_network_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def failing_fetch_openml(**kwargs: Any) -> SimpleNamespace:
        raise urllib.error.URLError("blocked")

    monkeypatch.setattr(data, "fetch_openml", failing_fetch_openml)
    with pytest.raises(data.DataDownloadError, match=r"api\.openml\.org"):
        data.download_german_credit(tmp_path)
