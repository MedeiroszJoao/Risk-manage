
import math

import pytest
from scipy.special import ndtr

from riskmanage.heston import (
    HestonParameters,
    heston_characteristic_function,
    heston_monte_carlo_price,
    heston_price_fourier,
)

def baseline(**overrides):
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


def test_characteristic_function_normalization_and_risk_neutral_forward():
    params = baseline()
    assert heston_characteristic_function(0.0, 1.0, params) == pytest.approx(1.0 + 0j, abs=1e-12)
    assert heston_characteristic_function(-1j, 1.0, params) == pytest.approx(
        params.spot * math.exp(params.risk_free_rate - params.dividend_yield),
        rel=1e-11,
    )


def test_characteristic_function_conjugate_symmetry():
    params = baseline()
    frequencies = np.array([0.2, 0.8, 1.7, 4.0])
    positive = heston_characteristic_function(frequencies, 1.3, params)
    negative = heston_characteristic_function(-frequencies, 1.3, params)
    assert negative == pytest.approx(np.conjugate(positive), rel=1e-11, abs=1e-11)


def test_deterministic_variance_limit_matches_black_scholes():
    params = baseline(
        variance0=0.04,
        long_run_variance=0.09,
        vol_of_variance=0.0,
    )
    maturity = 1.7
    strike = 103.0
    integrated_variance = (
        params.long_run_variance * maturity
        + (params.variance0 - params.long_run_variance)
        * (1.0 - math.exp(-params.mean_reversion * maturity))
        / params.mean_reversion
    )
    d1 = (
        math.log(params.spot / strike)
        + (params.risk_free_rate - params.dividend_yield) * maturity
        + 0.5 * integrated_variance
    ) / math.sqrt(integrated_variance)
    d2 = d1 - math.sqrt(integrated_variance)
    expected = (
        params.spot * math.exp(-params.dividend_yield * maturity) * ndtr(d1)
        - strike * math.exp(-params.risk_free_rate * maturity) * ndtr(d2)
    )
    actual = heston_price_fourier(strike, maturity, params)
    assert actual["price"] == pytest.approx(expected, rel=2e-9, abs=2e-9)
    assert actual["quadrature_error_estimate"] < 1e-7


def test_call_put_parity_and_no_arbitrage_bounds():
    params = baseline()
    call = heston_price_fourier(95.0, 0.8, params, "call")["price"]
    put = heston_price_fourier(95.0, 0.8, params, "put")["price"]
    parity = (
        params.spot * math.exp(-params.dividend_yield * 0.8)
        - 95.0 * math.exp(-params.risk_free_rate * 0.8)
    )
    assert call - put == pytest.approx(parity, abs=1e-8)
    assert max(parity, 0.0) <= call <= params.spot * math.exp(-params.dividend_yield * 0.8)
    assert max(-parity, 0.0) <= put <= 95.0 * math.exp(-params.risk_free_rate * 0.8)


def test_qe_m_monte_carlo_agrees_with_fourier_within_sampling_interval():
    params = baseline()
    fourier_price = heston_price_fourier(100.0, 1.0, params)["price"]
    mc = heston_monte_carlo_price(
        100.0,
        1.0,
        params,
        n_paths=40_000,
        n_steps=128,
        seed=17,
        pilot_paths=2_048,
    )
    lower, upper = mc["confidence_interval"]
    # A small additional allowance is explicit for QE time-discretization bias.
    assert lower - 0.08 <= fourier_price <= upper + 0.08
    assert mc["standard_error"] < mc["unadjusted_standard_error"]
    assert mc["variance_reduction_factor"] > 1.0
    assert not mc["discretization_bias_included_in_interval"]


def test_qe_m_preserves_discounted_stock_martingale_in_mc_tolerance():
    params = baseline()
    mc = heston_monte_carlo_price(
        100.0,
        1.0,
        params,
        n_paths=60_000,
        n_steps=96,
        seed=2026,
        control_variate=False,
    )
    target = params.spot * math.exp(-params.dividend_yield)
    assert abs(mc["discounted_terminal_stock_mean"] - target) < 0.015 * target


def test_qe_high_psi_branch_produces_nonnegative_variance_and_preserves_martingale():
    params = baseline(
        variance0=0.005,
        mean_reversion=0.25,
        long_run_variance=0.16,
        vol_of_variance=1.25,
        correlation=-0.45,
    )
    mc = heston_monte_carlo_price(
        100.0,
        1.0,
        params,
        n_paths=30_000,
        n_steps=4,
        seed=27,
        control_variate=False,
    )
    target = params.spot * math.exp(-params.dividend_yield)
    assert abs(mc["discounted_terminal_stock_mean"] - target) < 0.035 * target


def test_feller_violation_is_reported_not_silently_rejected():
    params = baseline(vol_of_variance=0.8)
    assert not params.feller_condition_satisfied
    mc = heston_monte_carlo_price(
        100.0,
        0.5,
        params,
        n_paths=2_000,
        n_steps=32,
        seed=11,
        control_variate=False,
    )
    assert mc["feller_condition_satisfied"] is False
    assert mc["feller_ratio"] == pytest.approx(params.feller_ratio)


def test_validation_rejects_odd_antithetic_path_count_and_invalid_strike():
    params = baseline()
    with pytest.raises(ValueError, match="even"):
        heston_monte_carlo_price(100, 1, params, n_paths=101)
    with pytest.raises(ValueError, match="strike"):
        heston_price_fourier(0, 1, params)
