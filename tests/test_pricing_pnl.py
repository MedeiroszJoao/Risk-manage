import numpy as np
import pytest
from riskmanage.pricing import black_scholes_call, garman_kohlhagen_call
from riskmanage.fixtures import build_synthetic_market_data
from riskmanage.pnl import build_desk_pnl

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
    result = build_desk_pnl(data, "Commodities", model="full_greeks")
    assert np.max(np.abs(result.residual.to_numpy())) < 1e-7

def test_delta_only_option_model_misses_non_linear_risk():
    data = build_synthetic_market_data(800, seed=88)
    delta = build_desk_pnl(data, "Equity Options", model="delta_only")
    full = build_desk_pnl(data, "Equity Options", model="full_greeks")
    assert delta.residual.abs().mean() > full.residual.abs().mean()

def test_equity_options_delta_only_fails_pla_and_full_greeks_repairs_it():
    from riskmanage.validation import pla_test
    data=build_synthetic_market_data(1800,seed=20261124)
    delta=build_desk_pnl(data,"Equity Options",model="delta_only")
    full=build_desk_pnl(data,"Equity Options",model="full_greeks")
    assert pla_test(delta.hpl,delta.rtpl)["zone"]=="red"
    assert pla_test(full.hpl,full.rtpl)["zone"]=="green"
    assert full.residual.abs().mean() < .2*delta.residual.abs().mean()
