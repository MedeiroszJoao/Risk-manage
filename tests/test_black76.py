import numpy as np
import pytest

from riskmanage.black76 import black76_implied_volatility, black76_price, black76_vega


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("volatility", [0.05, 0.2, 0.8, 1.5])
def test_black76_implied_volatility_recovers_volatility(option_type, volatility):
    price = black76_price(102.0, 100.0, 0.75, 0.965, volatility, option_type)
    recovered = black76_implied_volatility(price, 102.0, 100.0, 0.75, 0.965, option_type)
    assert recovered == pytest.approx(volatility, abs=1e-9)


def test_black76_put_call_parity():
    forward, strike, maturity, discount, volatility = 97.0, 100.0, 1.4, 0.93, 0.32
    call = black76_price(forward, strike, maturity, discount, volatility, "call")
    put = black76_price(forward, strike, maturity, discount, volatility, "put")
    assert call-put == pytest.approx(discount*(forward-strike), abs=1e-12)


def test_black76_zero_volatility_is_discounted_intrinsic_value():
    assert black76_price(105.0, 100.0, 1.0, 0.95, 0.0, "call") == pytest.approx(4.75)
    assert black76_price(105.0, 100.0, 1.0, 0.95, 0.0, "put") == pytest.approx(0.0)


def test_black76_vectorization_and_mixed_option_types():
    values = black76_price(
        [100.0, 100.0, 105.0], [100.0, 110.0, 100.0],
        1.0, 0.97, [0.2, 0.25, 0.3], ["call", "put", "call"]
    )
    assert values.shape == (3,)
    assert np.isfinite(values).all()
    assert np.all(values > 0.0)


def test_black76_vega_matches_finite_difference():
    args = (105.0, 100.0, 0.8, 0.97)
    volatility, bump = 0.27, 1e-5
    analytic = black76_vega(*args, volatility)
    up = black76_price(*args, volatility+bump)
    down = black76_price(*args, volatility-bump)
    numeric = (up-down)/(2.0*bump)
    assert analytic >= 0.0
    assert analytic == pytest.approx(numeric, rel=1e-7, abs=1e-9)


def test_black76_implied_volatility_at_intrinsic_is_zero():
    assert black76_implied_volatility(5.0, 105.0, 100.0, 1.0, 1.0, "call") == 0.0


@pytest.mark.parametrize(
    "args",
    [
        (-0.5, 100.0, 100.0, 1.0, 0.95, "call"),
        (100.0, 100.0, 100.0, 1.0, 0.95, "call"),
        (1.0, 100.0, 100.0, 0.0, 0.95, "put"),
    ],
)
def test_black76_implied_volatility_rejects_invalid_price_or_contract(args):
    with pytest.raises(ValueError):
        black76_implied_volatility(*args)


def test_black76_rejects_invalid_contract_inputs():
    with pytest.raises(ValueError):
        black76_price(-1.0, 100.0, 1.0, 0.95, 0.2)
    with pytest.raises(ValueError):
        black76_price(100.0, 100.0, 1.0, 0.95, -0.2)
