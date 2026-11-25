import pandas as pd
import pytest

from riskmanage.data import REQUIRED_COLUMNS, load_market_data, validate_market_data
from riskmanage.fixtures import build_synthetic_market_data


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
