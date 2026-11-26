import math

import pytest

from riskmanage.calibration import (
    HestonOptionQuote,
    _quote_error,
    calibrate_heston,
)
from riskmanage.heston import HestonParameters, heston_price_fourier


def parameters(**overrides):
    values = dict(
        spot=100.0,
        variance0=0.04,
        mean_reversion=1.5,
        long_run_variance=0.04,
        vol_of_variance=0.5,
        correlation=-0.7,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    values.update(overrides)
    return HestonParameters(**values)


def synthetic_panel():
    target = parameters()
    quotes = []
    for maturity in (0.25, 0.5, 1.0):
        for strike in (85.0, 92.5, 100.0, 107.5, 115.0):
            option_type = "call" if strike >= 100.0 else "put"
            price = heston_price_fourier(
                strike, maturity, target, option_type
            )["price"]
            spread = max(0.02, 0.015 * price)
            quotes.append(
                HestonOptionQuote(
                    strike,
                    maturity,
                    option_type,
                    price,
                    bid=max(0.0, price - spread / 2.0),
                    ask=price + spread / 2.0,
                )
            )
    return quotes


def test_bid_ask_residual_is_zero_inside_and_signed_outside():
    quote = HestonOptionQuote(100.0, 1.0, "call", 10.0, bid=9.9, ask=10.1)
    assert _quote_error(10.0, quote, "bid_ask") == 0.0
    assert _quote_error(9.8, quote, "bid_ask") == pytest.approx(-0.1)
    assert _quote_error(10.2, quote, "bid_ask") == pytest.approx(0.1)


def test_heston_calibration_recovers_synthetic_surface_prices():
    market = parameters()
    initial = parameters(
        variance0=0.05,
        mean_reversion=1.2,
        long_run_variance=0.05,
        vol_of_variance=0.6,
        correlation=-0.6,
    )
    result = calibrate_heston(
        synthetic_panel(),
        market,
        initial,
        objective="mid",
        n_starts=2,
        max_nfev=45,
        seed=11,
    )
    assert result["success"]
    assert result["price_rmse"] < 0.02
    assert result["bid_ask_hit_ratio"] >= 0.8
    assert result["jacobian_rank"] == 5
    assert len(result["jacobian_singular_values"]) == 5
    assert "not a parameter" in result["uncertainty_boundary"]


def test_bid_ask_objective_and_quote_validation():
    quotes = synthetic_panel()
    result = calibrate_heston(
        quotes,
        parameters(),
        parameters(),
        objective="bid_ask",
        n_starts=1,
        max_nfev=3,
    )
    assert result["success"]
    assert result["price_rmse"] < 0.01
    with pytest.raises(ValueError, match="bid and ask"):
        HestonOptionQuote(100, 1, "call", 10, bid=9.9)
    with pytest.raises(ValueError, match="inside"):
        HestonOptionQuote(100, 1, "call", 10, bid=10.1, ask=10.2)
