
import math

import numpy as np
import pytest
from scipy.special import ndtr

from riskmanage.heston import (
    HestonParameters,
    heston_characteristic_function,
    heston_monte_carlo_price,
    heston_price_fourier,
    heston_prices_fourier_batch,
    heston_option_price_delta_fourier_batch,
    heston_simulate_paths,
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


def test_qe_variance_transition_matches_conditional_first_two_moments():
    from riskmanage.heston import _qe_variance_step

    rng = np.random.default_rng(91)
    cases = [
        (baseline(vol_of_variance=0.35), 0.04, 1.0 / 252.0),
        (
            baseline(
                variance0=0.005,
                mean_reversion=0.25,
                long_run_variance=0.16,
                vol_of_variance=1.25,
                correlation=-0.45,
            ),
            0.005,
            0.25,
        ),
    ]
    for params, variance_now, dt in cases:
        draws = np.full(150_000, variance_now)
        shocks = rng.standard_normal(draws.size)
        simulated, mean, psi = _qe_variance_step(draws, shocks, params, dt, 1.5)
        decay = math.exp(-params.mean_reversion * dt)
        target_mean = params.long_run_variance + (
            variance_now - params.long_run_variance
        ) * decay
        target_variance = (
            variance_now
            * params.vol_of_variance**2
            * decay
            * (1.0 - decay)
            / params.mean_reversion
            + params.long_run_variance
            * params.vol_of_variance**2
            * (1.0 - decay) ** 2
            / (2.0 * params.mean_reversion)
        )
        mean_tolerance = 0.012 if np.mean(psi) <= 1.5 else 0.035
        variance_tolerance = 0.035 if np.mean(psi) <= 1.5 else 0.07
        assert np.mean(simulated) == pytest.approx(target_mean, rel=mean_tolerance)
        assert np.var(simulated, ddof=1) == pytest.approx(
            target_variance, rel=variance_tolerance
        )
        assert np.all(simulated >= 0.0)
        assert np.allclose(mean, target_mean)
        assert np.all(psi >= 0.0)


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



def test_vector_fourier_prices_match_scalar_calls_and_puts():
    params = baseline()
    strikes = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
    option_types = ["put", "put", "call", "call", "call"]
    batch = heston_prices_fourier_batch(
        strikes, 0.75, params, option_types
    )
    scalar = np.array(
        [
            heston_price_fourier(strike, 0.75, params, option_type)["price"]
            for strike, option_type in zip(strikes, option_types, strict=True)
        ]
    )
    assert batch["prices"] == pytest.approx(scalar, rel=1e-8, abs=1e-8)
    assert batch["quadrature_error_norm_units"] == "vector_integrand_l2"


def test_vector_fourier_validation_rejects_mismatched_option_types():
    params = baseline()
    with pytest.raises(ValueError, match="one 'call' or 'put'"):
        heston_prices_fourier_batch([90.0, 100.0], 1.0, params, ["call"])



def test_state_vector_fourier_prices_and_deltas_match_scalar_reference():
    params = baseline()
    spots = np.array([90.0, 100.0, 110.0])
    variances = np.array([0.025, 0.04, 0.075])
    batch = heston_option_price_delta_fourier_batch(
        spots, variances, 100.0, 0.8, params, "call"
    )
    for index, (spot, variance) in enumerate(zip(spots, variances, strict=True)):
        state = baseline(spot=float(spot), variance0=float(variance))
        scalar = heston_price_fourier(100.0, 0.8, state, "call")["price"]
        bump = 0.02
        up = heston_price_fourier(
            100.0, 0.8, baseline(spot=float(spot + bump), variance0=float(variance))
        )["price"]
        down = heston_price_fourier(
            100.0, 0.8, baseline(spot=float(spot - bump), variance0=float(variance))
        )["price"]
        delta_reference = (up - down) / (2.0 * bump)
        assert batch["prices"][index] == pytest.approx(scalar, rel=2e-7, abs=2e-7)
        assert batch["deltas"][index] == pytest.approx(delta_reference, rel=2e-5, abs=2e-5)


def test_state_vector_fourier_put_call_parity():
    params = baseline()
    spots = np.array([90.0, 100.0, 110.0])
    variances = np.array([0.025, 0.04, 0.075])
    call = heston_option_price_delta_fourier_batch(
        spots, variances, 100.0, 0.8, params, "call"
    )
    put = heston_option_price_delta_fourier_batch(
        spots, variances, 100.0, 0.8, params, "put"
    )
    assert call["prices"] - put["prices"] == pytest.approx(
        spots * math.exp(-params.dividend_yield * 0.8)
        - 100.0 * math.exp(-params.risk_free_rate * 0.8),
        abs=2e-7,
    )
    assert call["deltas"] - put["deltas"] == pytest.approx(
        math.exp(-params.dividend_yield * 0.8), abs=2e-7
    )


def test_full_path_simulation_risk_neutral_stock_martingale():
    params = baseline()
    paths = heston_simulate_paths(
        params, 1.0, n_steps=32, n_paths=30_000, seed=717
    )
    discounted = math.exp(-params.risk_free_rate) * paths["spots"][:, -1]
    target = params.spot * math.exp(-params.dividend_yield)
    assert np.mean(discounted) == pytest.approx(target, rel=0.008)


def test_full_path_simulation_supports_deterministic_variance_limit():
    params = baseline(
        variance0=0.04,
        long_run_variance=0.09,
        vol_of_variance=0.0,
    )
    paths = heston_simulate_paths(
        params, 1.0, n_steps=24, n_paths=3_000, seed=81
    )
    expected_variance = params.long_run_variance + (
        params.variance0 - params.long_run_variance
    ) * math.exp(-params.mean_reversion)
    assert paths["variances"][:, -1] == pytest.approx(
        expected_variance, abs=1e-12
    )
