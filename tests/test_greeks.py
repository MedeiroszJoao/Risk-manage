import math

import pytest

from riskmanage.greeks import heston_monte_carlo_greeks
from riskmanage.heston import HestonParameters, heston_price_fourier


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


def test_pathwise_delta_agrees_with_fourier_finite_difference():
    params = baseline()
    result = heston_monte_carlo_greeks(
        100.0, 1.0, params, n_paths=100_000, n_steps=128, seed=123
    )
    bump = 0.05
    up = heston_price_fourier(
        100.0, 1.0, baseline(spot=100.0 + bump)
    )["price"]
    down = heston_price_fourier(
        100.0, 1.0, baseline(spot=100.0 - bump)
    )["price"]
    reference_delta = (up - down) / (2.0 * bump)
    pathwise = result["delta_pathwise"]
    crn = result["delta_common_random_numbers"]
    assert abs(pathwise["estimate"] - reference_delta) < (
        4.0 * pathwise["standard_error"] + 0.003
    )
    assert abs(crn["estimate"] - reference_delta) < (
        4.0 * crn["standard_error"] + 0.003
    )


def test_gamma_vega_and_price_are_reported_with_sampling_errors():
    result = heston_monte_carlo_greeks(
        100.0, 0.75, baseline(), n_paths=30_000, n_steps=96, seed=77
    )
    assert result["price"]["standard_error"] > 0.0
    assert result["gamma_common_random_numbers"]["standard_error"] >= 0.0
    assert (
        result["vega_initial_volatility_common_random_numbers"]["standard_error"]
        > 0.0
    )
    assert result["simulation"]["common_random_numbers"] is True
    assert "not implied-volatility vega" in result["greek_conventions"]["vega"]
    assert "truncation bias" in result["uncertainty_boundary"]


def test_put_call_pathwise_delta_parity():
    params = baseline()
    call = heston_monte_carlo_greeks(
        100.0, 0.8, params, option_type="call", n_paths=40_000,
        n_steps=96, seed=41
    )
    put = heston_monte_carlo_greeks(
        100.0, 0.8, params, option_type="put", n_paths=40_000,
        n_steps=96, seed=41
    )
    target = math.exp(-params.dividend_yield * 0.8)
    observed = (
        call["delta_pathwise"]["estimate"] - put["delta_pathwise"]["estimate"]
    )
    tolerance = (
        4.0
        * math.sqrt(
            call["delta_pathwise"]["standard_error"] ** 2
            + put["delta_pathwise"]["standard_error"] ** 2
        )
        + 0.002
    )
    assert abs(observed - target) < tolerance


def test_validation_rejects_invalid_antithetic_count_and_bump():
    params = baseline()
    with pytest.raises(ValueError, match="even"):
        heston_monte_carlo_greeks(100.0, 1.0, params, n_paths=101)
    with pytest.raises(ValueError, match="smaller"):
        heston_monte_carlo_greeks(
            100.0, 1.0, params, initial_volatility_bump=0.3
        )
