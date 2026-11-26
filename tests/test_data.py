from hashlib import sha256
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from riskmanage.data import (
    REQUIRED_COLUMNS,
    load_market_data,
    download_fred_series,
    merge_eia_petroleum_into_market_data,
    validate_market_data,
)
from riskmanage.fixtures import build_synthetic_market_data
from riskmanage.pnl import enrich_market_data


def test_fixture_is_deterministic():
    left = build_synthetic_market_data(n=300, seed=20261124)
    right = build_synthetic_market_data(n=300, seed=20261124)
    pd.testing.assert_frame_equal(left, right)


def test_fixture_has_required_schema_and_sorted_unique_dates():
    frame = build_synthetic_market_data(n=300)
    assert set(REQUIRED_COLUMNS).issubset(frame.columns)
    assert frame["date"].is_monotonic_increasing
    assert not frame["date"].duplicated().any()


def test_invalid_nonpositive_spot_is_rejected():
    frame = build_synthetic_market_data(n=300)
    frame.loc[10, "sp500"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_market_data(frame)


def test_duplicate_dates_are_rejected():
    frame = build_synthetic_market_data(n=300)
    frame.loc[10, "date"] = frame.loc[9, "date"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_market_data(frame)


def test_load_market_data_checks_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01"], "sp500": [100.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_market_data(path)


def test_negative_wti_spot_is_valid_and_volatility_estimator_remains_finite():
    frame = build_synthetic_market_data(n=300)
    frame.loc[150, "wti"] = -36.98
    validate_market_data(frame)

    enriched = enrich_market_data(frame)
    assert enriched.loc[150, "wti"] == -36.98
    assert enriched["wti_vol"].notna().all()
    assert enriched["wti_vol"].map(lambda value: pd.notna(value) and value >= 0).all()


def test_nonpositive_fx_and_equity_levels_are_still_rejected():
    frame = build_synthetic_market_data(n=300)
    frame.loc[10, "eurusd"] = 0
    with pytest.raises(ValueError, match="FX rates must be positive"):
        validate_market_data(frame)



def test_brent_must_be_positive_while_wti_may_be_negative():
    frame = build_synthetic_market_data(n=300)
    frame.loc[150, "wti"] = -36.98
    frame.loc[150, "brent"] = 25.0
    validate_market_data(frame)

    frame.loc[150, "brent"] = 0.0
    with pytest.raises(ValueError, match="Brent spot must be positive"):
        validate_market_data(frame)


def test_eia_oil_series_merge_into_fred_market_panel():
    fred = build_synthetic_market_data(n=300).drop(columns=["wti", "brent"])
    eia = pd.DataFrame(
        [
            ("2016-01-04", "wti_cushing_spot", "PET.RWTC.D", 37.0),
            ("2016-01-04", "brent_europe_spot", "PET.RBRTE.D", 37.5),
            ("2016-01-05", "wti_cushing_spot", "PET.RWTC.D", 36.0),
            ("2016-01-05", "brent_europe_spot", "PET.RBRTE.D", 36.8),
            ("2016-01-06", "wti_cushing_spot", "PET.RWTC.D", -36.98),
            ("2016-01-06", "brent_europe_spot", "PET.RBRTE.D", 35.9),
        ],
        columns=["date", "series", "series_id", "value"],
    )

    merged = merge_eia_petroleum_into_market_data(fred, eia)

    assert {"wti", "brent"}.issubset(merged.columns)
    assert merged.loc[2, "wti"] == -36.98
    assert merged.loc[2, "brent"] == 35.9
    validate_market_data(merged)


def test_eia_oil_merge_rejects_wrong_series_id_and_duplicate_observation():
    fred = build_synthetic_market_data(n=300).drop(columns=["wti", "brent"])
    eia = pd.DataFrame(
        [
            ("2016-01-04", "wti_cushing_spot", "PET.RWTC.D", 37.0),
            ("2016-01-04", "brent_europe_spot", "PET.RBRTE.D", 37.5),
        ],
        columns=["date", "series", "series_id", "value"],
    )
    eia.loc[0, "series_id"] = "PET.WRONG.D"
    with pytest.raises(ValueError, match="mapped to PET.RWTC.D"):
        merge_eia_petroleum_into_market_data(fred, eia)



def test_adding_brent_preserves_original_seeded_market_factor_paths():
    frame = build_synthetic_market_data(n=300, seed=20261124)
    first = frame.iloc[0]
    assert first["sp500"] == pytest.approx(2101.329780593683, abs=1e-10)
    assert first["vix"] == pytest.approx(22.061546606465672, abs=1e-10)
    assert first["eurusd"] == pytest.approx(1.076672149961795, abs=1e-10)
    assert first["usdjpy"] == pytest.approx(117.54013573262118, abs=1e-10)
    assert first["wti"] == pytest.approx(48.633322510891944, abs=1e-10)
    assert first["ust2y"] == pytest.approx(2.417550522693182, abs=1e-10)
    assert first["ust10y"] == pytest.approx(3.022248441738302, abs=1e-10)



class FakeFREDResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.status_code = 200

    def raise_for_status(self):
        return None


class FakeFREDSession:
    def __init__(self):
        self.responses = {
            "SP500": "observation_date,SP500\n2024-01-01,.\n2024-01-02,4700\n2024-01-03,4705\n",
            "VIXCLS": "observation_date,VIXCLS\n2024-01-01,12\n2024-01-02,13\n2024-01-03,12.5\n",
            "DEXUSEU": "observation_date,DEXUSEU\n2024-01-01,1.10\n2024-01-02,1.11\n2024-01-03,1.12\n",
            "DEXJPUS": "observation_date,DEXJPUS\n2024-01-01,140\n2024-01-02,141\n2024-01-03,142\n",
            "DGS2": "observation_date,DGS2\n2024-01-01,4.0\n2024-01-02,4.1\n2024-01-03,4.2\n",
            "DGS10": "observation_date,DGS10\n2024-01-01,4.1\n2024-01-02,4.2\n2024-01-03,4.3\n",
        }
        self.calls = []

    def get(self, url, timeout):
        query = parse_qs(urlparse(url).query)
        series_id = query["id"][0]
        self.calls.append({"url": url, "timeout": timeout})
        return FakeFREDResponse(self.responses[series_id].encode("utf-8"))



def test_fred_ingestion_keeps_immutable_raw_snapshots_and_true_valid_coverage(tmp_path):
    session = FakeFREDSession()

    first_frame, first_manifest = download_fred_series(
        start="2007-01-01",
        end="2024-01-03",
        output_dir=tmp_path / "fred",
        session=session,
    )
    second_frame, second_manifest = download_fred_series(
        start="2007-01-01",
        end="2024-01-03",
        output_dir=tmp_path / "fred",
        session=session,
    )

    assert first_frame["date"].min() == pd.Timestamp("2024-01-02")
    assert second_frame["date"].min() == pd.Timestamp("2024-01-02")
    assert first_manifest["snapshot_id"] != second_manifest["snapshot_id"]
    assert first_manifest["raw_snapshot_directory"] != second_manifest["raw_snapshot_directory"]

    sp500 = next(item for item in first_manifest["series"] if item["series_id"] == "SP500")
    assert sp500["first_date"] == "2024-01-02"
    assert sp500["last_date"] == "2024-01-03"
    raw_path = Path(sp500["raw_path"])
    assert raw_path.exists()
    assert sha256(raw_path.read_bytes()).hexdigest() == sp500["raw_sha256"]
    assert raw_path.parent == Path(first_manifest["raw_snapshot_directory"])

    parsed_query = parse_qs(urlparse(session.calls[0]["url"]).query)
    assert set(parsed_query) == {"id", "cosd", "coed"}
    assert all("api_key" not in parse_qs(urlparse(call["url"]).query) for call in session.calls)
