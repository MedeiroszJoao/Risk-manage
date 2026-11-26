import json

import pandas as pd
import pytest

from riskmanage.eia import (
    DEFAULT_PETROLEUM_SERIES,
    download_eia_petroleum_spot_prices,
    save_eia_petroleum_spot_prices,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.content = json.dumps(payload, sort_keys=True).encode("utf-8")

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, params, timeout):
        params_dict = dict(params)
        self.calls.append((url, params_dict, timeout))
        series_id = params_dict["facets[series][]"]
        offset = int(params_dict["offset"])
        return FakeResponse(self.pages[(series_id, offset)])


def _row(period, value, series_id):
    return {
        "period": period,
        "value": str(value),
        "series": series_id,
        "value-units": "dollars per barrel",
        "seriesDescription": f"{series_id} daily spot price",
    }


def test_eia_pagination_provenance_and_no_forward_fill(tmp_path):
    pages = {
        ("RWTC", 0): {
            "response": {
                "total": "3",
                "data": [_row("2024-01-02", 70.0, "RWTC"), _row("2024-01-03", 71.0, "RWTC")],
            }
        },
        ("RWTC", 2): {
            "response": {"total": "3", "data": [_row("2024-01-05", 72.0, "RWTC")]}
        },
        ("RBRTE", 0): {
            "response": {
                "total": "2",
                "data": [_row("2024-01-02", 76.0, "RBRTE"), _row("2024-01-04", 77.0, "RBRTE")],
            }
        },
    }
    session = FakeSession(pages)
    frame, manifest = download_eia_petroleum_spot_prices(
        start="2024-01-01",
        end="2024-01-31",
        api_key="test-secret-not-to-persist",
        session=session,
        page_size=2,
        raw_dir=tmp_path / "raw",
    )

    assert len(frame) == 5
    assert set(frame["series_id"]) == {"RWTC", "RBRTE"}
    assert frame.loc[frame["series_id"] == "RWTC", "date"].tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-05",
    ]
    assert "2024-01-04" not in frame.loc[frame["series_id"] == "RWTC", "date"].tolist()
    assert [call[1]["offset"] for call in session.calls] == ["0", "2", "0"]
    assert all(call[1]["api_key"] == "test-secret-not-to-persist" for call in session.calls)
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 3
    assert len(manifest["series"]) == 2
    manifest_text = json.dumps(manifest)
    assert "test-secret-not-to-persist" not in manifest_text
    assert manifest["row_count"] == 5
    assert len(manifest["normalized_sha256"]) == 64


def test_eia_requires_environment_or_explicit_key(monkeypatch, tmp_path):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EIA_API_KEY"):
        download_eia_petroleum_spot_prices(session=FakeSession({}), raw_dir=tmp_path)


def test_eia_rejects_api_error_payload(tmp_path):
    session = FakeSession({
        ("RWTC", 0): {"error": "invalid api key", "code": 403}
    })
    with pytest.raises(RuntimeError, match="invalid api key"):
        download_eia_petroleum_spot_prices(
            api_key="test-key",
            session=session,
            raw_dir=tmp_path,
        )


def test_eia_rejects_inconsistent_pagination_total(tmp_path):
    session = FakeSession({
        ("RWTC", 0): {
            "response": {"total": "2", "data": [_row("2024-01-02", 70.0, "RWTC")]}
        },
        ("RWTC", 1): {
            "response": {"total": "3", "data": [_row("2024-01-03", 71.0, "RWTC")]}
        },
    })
    with pytest.raises(ValueError, match="total changed"):
        download_eia_petroleum_spot_prices(
            api_key="test-key",
            session=session,
            page_size=1,
            series={"wti": "RWTC"},
            raw_dir=tmp_path,
        )


def test_eia_rejects_duplicate_dates_and_invalid_values(tmp_path):
    duplicate_session = FakeSession({
        ("RWTC", 0): {
            "response": {
                "total": "2",
                "data": [_row("2024-01-02", 70.0, "RWTC"), _row("2024-01-02", 71.0, "RWTC")],
            }
        }
    })
    with pytest.raises(ValueError, match="Duplicate daily"):
        download_eia_petroleum_spot_prices(
            api_key="test-key",
            session=duplicate_session,
            raw_dir=tmp_path / "duplicate",
        )

    invalid_session = FakeSession({
        ("RWTC", 0): {
            "response": {"total": "1", "data": [_row("2024-01-02", 0.0, "RWTC")]}
        }
    })
    with pytest.raises(ValueError, match="Non-positive"):
        download_eia_petroleum_spot_prices(
            api_key="test-key",
            session=invalid_session,
            raw_dir=tmp_path / "invalid",
        )


def test_eia_save_writes_csv_and_manifest(tmp_path):
    session = FakeSession({
        ("RWTC", 0): {
            "response": {"total": "1", "data": [_row("2024-01-02", 70.0, "RWTC")]}
        },
        ("RBRTE", 0): {
            "response": {"total": "1", "data": [_row("2024-01-02", 76.0, "RBRTE")]}
        },
    })
    frame, manifest = download_eia_petroleum_spot_prices(
        api_key="test-key",
        session=session,
        raw_dir=tmp_path / "raw",
    )
    csv_path = tmp_path / "processed" / "prices.csv"
    manifest_path = tmp_path / "processed" / "manifest.json"
    save_eia_petroleum_spot_prices(frame, manifest, csv_path, manifest_path)

    assert pd.read_csv(csv_path).shape[0] == 2
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["row_count"] == 2
    assert "test-key" not in manifest_path.read_text(encoding="utf-8")


def test_default_series_are_the_documented_wti_and_brent_markets():
    assert DEFAULT_PETROLEUM_SERIES == {
        "wti_cushing_spot": "RWTC",
        "brent_europe_spot": "RBRTE",
    }


def test_invalid_date_range_and_page_size_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="start must be"):
        download_eia_petroleum_spot_prices(
            start="2024-02-01",
            end="2024-01-01",
            api_key="test-key",
            session=FakeSession({}),
            raw_dir=tmp_path / "date",
        )
    with pytest.raises(ValueError, match="page_size"):
        download_eia_petroleum_spot_prices(
            api_key="test-key",
            session=FakeSession({}),
            page_size=5001,
            raw_dir=tmp_path / "size",
        )
