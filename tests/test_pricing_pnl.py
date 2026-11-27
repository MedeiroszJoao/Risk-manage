import numpy as np
import pytest
import pytest
from riskmanage.pricing import black_scholes_call, garman_kohlhagen_call
from riskmanage.fixtures import build_synthetic_market_data
from riskmanage.portfolio import Position
from riskmanage.pnl import (
    build_desk_pnl,
    _mixed_second_derivative,
    _taylor_rtpl,
    value_position,
)

def test_black_scholes_call_is_monotone_in_spot_and_nonnegative():
    low = black_scholes_call(90, 100, 1.0, .03, .20)
    high = black_scholes_call(110, 100, 1.0, .03, .20)
    assert 0 <= low < high

def test_garman_kohlhagen_call_respects_positive_bounds():
    price = garman_kohlhagen_call(1.10, 1.05, 1.0, .04, .02, .12)
    assert 0 <= price <= 1.10

def test_expired_call_is_intrinsic():
    assert black_scholes_call(110, 100, 0.0, .03, .2) == pytest.approx(10)

def test_desk_pnl_has_hpl_rtpl_and_residual():
    data = build_synthetic_market_data(500)
    result = build_desk_pnl(data, "Equity Options", model="delta_only")
    assert {"date", "hpl", "rtpl", "residual"}.issubset(result.columns)
    assert result.hpl.notna().all() and result.rtpl.notna().all()

def test_linear_commodity_desk_has_exact_delta_explain():
    data = build_synthetic_market_data(500)
    result = build_desk_pnl(data, "Commodities", model="diagonal_second_order")
    assert np.max(np.abs(result.residual.to_numpy())) < 1e-7

def test_delta_only_option_model_misses_non_linear_risk_vs_diagonal_second_order():
    data = build_synthetic_market_data(800, seed=88)
    delta = build_desk_pnl(data, "Equity Options", model="delta_only")
    full = build_desk_pnl(data, "Equity Options", model="full_greeks")
    assert delta.residual.abs().mean() > full.residual.abs().mean()

def test_equity_options_delta_only_fails_pla_and_diagonal_second_order_repairs_it():
    from riskmanage.validation import pla_test
    data=build_synthetic_market_data(1800,seed=20261124)
    delta=build_desk_pnl(data,"Equity Options",model="delta_only")
    full=build_desk_pnl(data,"Equity Options",model="full_greeks")
    assert pla_test(delta.hpl,delta.rtpl)["zone"]=="red"
    assert pla_test(full.hpl,full.rtpl)["zone"]=="green"
    assert full.residual.abs().mean() < .2*delta.residual.abs().mean()


def test_mixed_second_derivative_matches_analytic_black_scholes_vanna():
    spot = 100.0
    strike = 100.0
    maturity = 0.75
    volatility = 0.24
    position = Position(
        "Equity Options",
        "EQUITY_CALL",
        quantity=1.0,
        maturity=maturity,
        strike=strike,
        multiplier=1.0,
    )
    state = {"sp500": spot, "equity_vol": volatility}
    observed = _mixed_second_derivative(
        [position], "Equity Options", state, 0,
        "sp500", "equity_vol", 0.01, 0.00024
    )

    d1 = (
        np.log(spot / strike) + (0.035 + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    normal_density = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    analytic_vanna = -normal_density * d2 / volatility

    assert observed == pytest.approx(analytic_vanna, rel=1e-4, abs=1e-7)


def test_mixed_second_derivative_converges_toward_analytic_vanna():
    spot = 100.0
    strike = 100.0
    maturity = 0.75
    volatility = 0.24
    position = Position(
        "Equity Options",
        "EQUITY_CALL",
        quantity=1.0,
        maturity=maturity,
        strike=strike,
        multiplier=1.0,
    )
    state = {"sp500": spot, "equity_vol": volatility}
    d1 = (
        np.log(spot / strike) + (0.035 + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    analytic_vanna = (
        -np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi) * d2 / volatility
    )

    errors = []
    for scale in (4.0, 2.0, 1.0):
        observed = _mixed_second_derivative(
            [position], "Equity Options", state, 0,
            "sp500", "equity_vol", 0.01 * scale, 0.00024 * scale
        )
        errors.append(abs(observed - analytic_vanna))

    assert errors[2] < errors[1] < errors[0]


def test_full_hessian_cross_term_reduces_bivariate_taylor_error():
    position = Position(
        "Equity Options",
        "EQUITY_CALL",
        quantity=1.0,
        maturity=0.75,
        strike=100.0,
        multiplier=1.0,
    )
    positions = [position]
    previous = {"sp500": 100.0, "equity_vol": 0.24}
    current = {"sp500": 105.0, "equity_vol": 0.27}
    hpl = (
        value_position(position, current, 1)
        - value_position(position, previous, 0)
    )
    diagonal = _taylor_rtpl(
        positions, "Equity Options", previous, current, 0, "diagonal_second_order"
    )
    full = _taylor_rtpl(
        positions, "Equity Options", previous, current, 0, "full_hessian"
    )

    assert abs(hpl - full) < abs(hpl - diagonal)


def test_full_hessian_mode_reports_its_cross_term_scope():
    data = build_synthetic_market_data(500, seed=440)
    result = build_desk_pnl(data, "Equity Options", model="full_hessian")
    assert set(result["risk_model"]) == {"full_hessian"}
    assert set(result["rtpl_approximation"]) == {
        "full_market_factor_hessian_no_time_market_cross_terms"
    }
    assert result["residual"].notna().all()


def test_legacy_full_greeks_name_is_deprecated_diagonal_alias():
    data = build_synthetic_market_data(500, seed=441)
    with pytest.warns(DeprecationWarning, match="legacy name"):
        legacy = build_desk_pnl(data, "Equity Options", model="full_greeks")
    diagonal = build_desk_pnl(data, "Equity Options", model="diagonal_second_order")

    assert legacy["risk_model"].eq("diagonal_second_order").all()
    assert np.allclose(legacy["rtpl"].to_numpy(), diagonal["rtpl"].to_numpy())
