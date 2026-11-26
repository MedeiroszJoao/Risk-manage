from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest
import requests

from riskmanage.eia import (
    DEFAULT_PETROLEUM_SERIES,
    EIA_PETROLEUM_SPOT_BULK_URL,
    download_eia_petroleum_spot_prices,
    save_eia_petroleum_spot_prices,
)


class FakeResponse:
    def __init__(self, content: bytes, headers: dict[str, str] | None = None, status_code: int = 200):
        self.content = content
        self.headers = headers or {"Content-Length": str(len(content)), "Content-Type": "application/zip"}
        self.status_code = status_code
        self.url = EIA_PETROLEUM_SPOT_BULK_URL
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size=8192):
        step = max(1, min(chunk_size, 37_000))
        for offset in range(0, len(self.content), step):
            yield self.content[offset : offset + step]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, content: bytes, status_code: int = 200, headers: dict[str, str] | None = None):
        self.response = FakeResponse(content, headers=headers, status_code=status_code)
        self.calls = []

    def get(self, url, headers, stream, timeout):
        self.calls.append({"url": url, "headers": headers, "stream": stream, "timeout": timeout})
        return self.response


def _series(series_id: str, rows: list[list[object]], *, units: str = "Dollars per Barrel"):
    return {
        "series_id": series_id,
        "name": f"{series_id} daily spot price",
        "units": units,
        "f": "D",
        "unitsshort": "$/bbl",
        "description": f"Description for {series_id}",
        "start": "20070101",
        "end": "20261120",
        "last_updated": "2026-11-21T17:00:00-05:00",
        "data": rows,
    }


def _pet_zip(*objects: dict) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open("PET.txt", "w") as member:
            for obj in objects:
                member.write(json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n")
    return output.getvalue()


def test_bulk_download_filters_wti_brent_and_preserves_negative_spot_price(tmp_path, monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    content = _pet_zip(
        _series("PET.RWTC.D", [
            ["20261120", "60.25"],
            ["20200421", "8.91"],
            ["20200420", "-36.98"],
            ["20061229", "61.05"],
        ]),
        _series("PET.RBRTE.D", [
            ["20261120", "64.12"],
            ["20200421", "9.12"],
            ["20200420", "25.22"],
        ]),
        {"category_id": "123", "name": "Petroleum category", "childseries": ["PET.RWTC.D"]},
    )
    session = FakeSession(content)

    frame, manifest = download_eia_petroleum_spot_prices(
        start="2007-01-01",
        end="2026-11-20",
        session=session,
        raw_dir=tmp_path / "raw",
    )

    assert len(session.calls) == 1
    assert session.calls[0]["url"] == EIA_PETROLEUM_SPOT_BULK_URL
    assert session.calls[0]["stream"] is True
    assert "api_key" not in session.calls[0]
    assert session.response.closed is True
    assert set(frame["series_id"]) == {"PET.RWTC.D", "PET.RBRTE.D"}
    assert "2006-12-29" not in frame["date"].tolist()

    wti = frame.loc[frame["series_id"] == "PET.RWTC.D"].set_index("date")
    assert wti.loc["2020-04-20", "value"] == -36.98
    assert wti.loc["2020-04-21", "value"] == 8.91
    assert wti.index.tolist() == sorted(wti.index.tolist())

    assert manifest["request"]["api_key_required"] is False
    assert manifest["row_count"] == 6
    assert manifest["parser"]["matched_series_lines"] == 2
    assert manifest["parser"]["json_lines"] == 3
    assert manifest["series"][0]["units"] == "Dollars per Barrel"
    archive_path = Path(manifest["download"]["archive_path"])
    assert archive_path.exists()
    assert sha256(archive_path.read_bytes()).hexdigest() == manifest["download"]["sha256"]
    assert "test-secret" not in json.dumps(manifest)


def test_bulk_download_counts_null_values_without_forward_fill(tmp_path):
    content = _pet_zip(
        _series("PET.RWTC.D", [
            ["20240102", "70.0"],
            ["20240103", None],
            ["20240104", "null"],
            ["20240105", "72.0"],
            ["20240106", "w"],
            ["20240107", "*"],
        ]),
        _series("PET.RBRTE.D", [["20240102", "76.0"], ["20240104", "77.0"]]),
    )
    frame, manifest = download_eia_petroleum_spot_prices(
        start="2024-01-01", end="2024-01-31", session=FakeSession(content), raw_dir=tmp_path
    )

    wti_dates = frame.loc[frame["series_id"] == "PET.RWTC.D", "date"].tolist()
    assert wti_dates == ["2024-01-02", "2024-01-05"]
    wti_metadata = next(row for row in manifest["series"] if row["series_id"] == "PET.RWTC.D")
    assert wti_metadata["null_observations_in_range"] == 2
    assert wti_metadata["withheld_observations_in_range"] == 1
    assert wti_metadata["insignificant_observations_in_range"] == 1
    assert manifest["parser"]["null_observations"] == 2
    assert manifest["parser"]["withheld_observations"] == 1
    assert manifest["parser"]["insignificant_observations"] == 1


def test_bulk_download_saves_archive_manifest_and_csv(tmp_path):
    content = _pet_zip(
        _series("PET.RWTC.D", [["20240102", "70.0"]]),
        _series("PET.RBRTE.D", [["20240102", "76.0"]]),
    )
    frame, manifest = download_eia_petroleum_spot_prices(
        session=FakeSession(content), raw_dir=tmp_path / "raw"
    )
    csv_path = tmp_path / "processed" / "prices.csv"
    manifest_path = tmp_path / "processed" / "manifest.json"
    save_eia_petroleum_spot_prices(frame, manifest, csv_path, manifest_path)

    assert pd.read_csv(csv_path).shape[0] == 2
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["row_count"] == 2
    assert Path(saved["download"]["archive_path"]).exists()


def test_save_accepts_negative_wti_spot_values(tmp_path):
    frame = pd.DataFrame([{
        "date": "2020-04-20",
        "series": "wti_cushing_spot",
        "series_id": "PET.RWTC.D",
        "value": -36.98,
        "units": "Dollars per Barrel",
        "series_description": "WTI Cushing spot",
    }])
    csv_path = tmp_path / "negative.csv"
    manifest_path = tmp_path / "negative.json"
    save_eia_petroleum_spot_prices(frame, {"row_count": 1}, csv_path, manifest_path)
    assert pd.read_csv(csv_path).iloc[0]["value"] == -36.98


def test_bulk_download_rejects_missing_series(tmp_path):
    content = _pet_zip(_series("PET.RWTC.D", [["20240102", "70.0"]]))
    with pytest.raises(ValueError, match="missing requested daily series"):
        download_eia_petroleum_spot_prices(session=FakeSession(content), raw_dir=tmp_path)


def test_bulk_download_rejects_duplicate_dates(tmp_path):
    content = _pet_zip(
        _series("PET.RWTC.D", [["20240102", "70.0"], ["20240102", "71.0"]]),
        _series("PET.RBRTE.D", [["20240102", "76.0"]]),
    )
    with pytest.raises(ValueError, match="Duplicate daily observations"):
        download_eia_petroleum_spot_prices(session=FakeSession(content), raw_dir=tmp_path)


def test_bulk_download_rejects_non_finite_values(tmp_path):
    content = _pet_zip(
        _series("PET.RWTC.D", [["20240102", "NaN"]]),
        _series("PET.RBRTE.D", [["20240102", "76.0"]]),
    )
    with pytest.raises(ValueError, match="Non-finite"):
        download_eia_petroleum_spot_prices(session=FakeSession(content), raw_dir=tmp_path)


def test_bulk_download_rejects_bad_zip_and_http_failure(tmp_path):
    with pytest.raises(ValueError, match="not a valid ZIP"):
        download_eia_petroleum_spot_prices(
            session=FakeSession(b"not a zip file"), raw_dir=tmp_path / "bad-zip"
        )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        download_eia_petroleum_spot_prices(
            session=FakeSession(b"ignored", status_code=503), raw_dir=tmp_path / "http"
        )


def test_bulk_download_rejects_incomplete_download(tmp_path):
    content = _pet_zip(
        _series("PET.RWTC.D", [["20240102", "70.0"]]),
        _series("PET.RBRTE.D", [["20240102", "76.0"]]),
    )
    headers = {"Content-Length": str(len(content) + 1)}
    with pytest.raises(ValueError, match="size mismatch"):
        download_eia_petroleum_spot_prices(
            session=FakeSession(content, headers=headers), raw_dir=tmp_path
        )


def test_default_series_are_daily_petroleum_bulk_identifiers():
    assert DEFAULT_PETROLEUM_SERIES == {
        "wti_cushing_spot": "PET.RWTC.D",
        "brent_europe_spot": "PET.RBRTE.D",
    }


def test_invalid_date_range_series_and_duplicate_ids_are_rejected(tmp_path):
    content = _pet_zip(
        _series("PET.RWTC.D", [["20240102", "70.0"]]),
        _series("PET.RBRTE.D", [["20240102", "76.0"]]),
    )
    session = FakeSession(content)
    with pytest.raises(ValueError, match="start must be"):
        download_eia_petroleum_spot_prices(
            start="2024-02-01", end="2024-01-01", session=session, raw_dir=tmp_path / "date"
        )
    with pytest.raises(ValueError, match="daily EIA Petroleum"):
        download_eia_petroleum_spot_prices(
            series={"wti": "PET.RWTC.W"}, session=session, raw_dir=tmp_path / "frequency"
        )
    with pytest.raises(ValueError, match="duplicate EIA series IDs"):
        download_eia_petroleum_spot_prices(
            series={"wti": "PET.RWTC.D", "again": "PET.RWTC.D"},
            session=session,
            raw_dir=tmp_path / "duplicate",
        )


def test_bulk_download_rejects_observations_outside_requested_start(tmp_path):
    content = _pet_zip(
        _series("PET.RWTC.D", [["20231229", "70.0"], ["20240102", "71.0"]]),
        _series("PET.RBRTE.D", [["20240102", "76.0"]]),
    )
    frame, manifest = download_eia_petroleum_spot_prices(
        start="2024-01-01", session=FakeSession(content), raw_dir=tmp_path
    )
    assert frame["date"].min() == "2024-01-02"
    assert next(x for x in manifest["series"] if x["series_id"] == "PET.RWTC.D")["filtered_out_of_range"] == 1


def test_each_bulk_download_gets_distinct_snapshot_directory(tmp_path):
    content = _pet_zip(
        _series("PET.RWTC.D", [["20240102", "70.0"]]),
        _series("PET.RBRTE.D", [["20240102", "76.0"]]),
    )
    _, first = download_eia_petroleum_spot_prices(session=FakeSession(content), raw_dir=tmp_path)
    _, second = download_eia_petroleum_spot_prices(session=FakeSession(content), raw_dir=tmp_path)
    assert first["snapshot_id"] != second["snapshot_id"]
    assert Path(first["download"]["archive_path"]).is_dir() is False
    assert Path(first["download"]["archive_path"]).exists()
    assert Path(second["download"]["archive_path"]).exists()


def test_bulk_parser_requires_pet_txt_member(tmp_path):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("OTHER.txt", "{}\n")
    with pytest.raises(ValueError, match="exactly one PET.txt"):
        download_eia_petroleum_spot_prices(session=FakeSession(output.getvalue()), raw_dir=tmp_path)
