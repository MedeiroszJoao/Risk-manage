from __future__ import annotations

import pandas as pd
import pytest

from riskmanage.pnl import build_desk_pnl
from riskmanage.portfolio import Position, build_default_portfolio


def test_wti_spot_proxy_pnl_handles_negative_price_crossing_exactly():
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-04-17", "2020-04-20", "2020-04-21"]),
            "sp500": [2874.56, 2823.16, 2736.56],
            "vix": [38.15, 43.83, 45.41],
            "eurusd": [1.0874, 1.0866, 1.0850],
            "usdjpy": [107.52, 107.67, 107.78],
            "wti": [18.31, -36.98, 8.91],
            "ust2y": [0.20, 0.20, 0.20],
            "ust10y": [0.65, 0.65, 0.63],
        }
    )
    position = Position(
        desk="Commodities",
        instrument="WTI_SPOT_PROXY",
        quantity=1_000.0,
        label="Test linear exposure, 1,000 barrels",
    )

    result = build_desk_pnl(market, "Commodities", positions=[position])

    assert result["hpl"].tolist() == pytest.approx([-55_290.0, 45_890.0])
    assert result["rtpl"].tolist() == pytest.approx([-55_290.0, 45_890.0])
    assert result["residual"].tolist() == pytest.approx([0.0, 0.0])


def test_default_commodity_book_is_explicitly_a_spot_proxy_not_futures():
    positions = build_default_portfolio({"sp500": 5_000.0, "eurusd": 1.10})
    commodity_positions = [p for p in positions if p.desk == "Commodities"]

    assert len(commodity_positions) == 1
    assert commodity_positions[0].instrument == "WTI_SPOT_PROXY"
    assert "not a listed futures contract" in commodity_positions[0].label
