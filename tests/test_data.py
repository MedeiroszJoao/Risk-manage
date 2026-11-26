import pandas as pd
import pytest

from riskmanage.data import (
    REQUIRED_COLUMNS,
    load_market_data,
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
