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
            "brent": [28.50, 25.10, 27.30],
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
    assert result["residual"].tolist() == pytest.approx([0.0, 0.0], abs=1e-10)


def test_default_commodity_book_is_explicitly_a_spot_proxy_not_futures():
    positions = build_default_portfolio({"sp500": 5_000.0, "eurusd": 1.10})
    commodity_positions = [p for p in positions if p.desk == "Commodities"]

    assert [p.instrument for p in commodity_positions] == [
        "WTI_SPOT_PROXY", "BRENT_SPOT_PROXY"
    ]
    assert all("not a listed futures contract" in p.label for p in commodity_positions)



def test_wti_brent_spot_spread_pnl_explains_both_benchmarks_exactly():
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-04-17", "2020-04-20", "2020-04-21"]),
            "sp500": [2874.56, 2823.16, 2736.56],
            "vix": [38.15, 43.83, 45.41],
            "eurusd": [1.0874, 1.0866, 1.0850],
            "usdjpy": [107.52, 107.67, 107.78],
            "wti": [18.31, -36.98, 8.91],
            "brent": [28.50, 29.00, 27.50],
            "ust2y": [0.20, 0.20, 0.20],
            "ust10y": [0.65, 0.65, 0.63],
        }
    )
    positions = [
        Position("Commodities", "WTI_SPOT_PROXY", 1_000.0),
        Position("Commodities", "BRENT_SPOT_PROXY", -2_000.0),
    ]

    result = build_desk_pnl(market, "Commodities", positions=positions)

    assert result["hpl"].tolist() == pytest.approx([-56_290.0, 48_890.0])
    assert result["rtpl"].tolist() == pytest.approx([-56_290.0, 48_890.0])
    assert result["residual"].tolist() == pytest.approx([0.0, 0.0], abs=1e-10)



def test_oil_collapse_scenario_uses_additive_spot_shocks_when_wti_is_negative():
    import runpy
    from pathlib import Path

    from riskmanage.fixtures import build_synthetic_market_data

    run_case_path = Path(__file__).resolve().parents[1] / "scripts" / "run_case.py"
    build_scenario_reports = runpy.run_path(str(run_case_path))["build_scenario_reports"]
    market = build_synthetic_market_data(n=300)
    market.loc[market.index[-1], "wti"] = -36.98
    market.loc[market.index[-1], "brent"] = 10.0

    scenarios = build_scenario_reports(market).set_index("scenario")
    oil_collapse = scenarios.loc["oil_collapse"]

    # Long 150k WTI loses 25 USD/bbl; short 100k Brent gains 5 USD/bbl
    # after the Brent stress floor is applied.
    assert oil_collapse["Commodities_pnl"] == pytest.approx(-3_250_000.0)
