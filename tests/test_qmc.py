import numpy as np
import pytest

from riskmanage.heston import HestonParameters, heston_price_fourier
from riskmanage.qmc import _brownian_bridge_increments, heston_randomized_qmc_price


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


def test_brownian_bridge_preserves_independent_standard_normal_increments():
    rng = np.random.default_rng(71)
    increments = _brownian_bridge_increments(
        rng.standard_normal((40_000, 8)), maturity=1.0
    )
    assert np.max(np.abs(np.mean(increments, axis=0))) < 0.015
    assert np.var(increments, axis=0, ddof=1) == pytest.approx(
        np.ones(8), abs=0.025
    )
    covariance = np.cov(increments, rowvar=False)
    off_diagonal = covariance - np.diag(np.diag(covariance))
    assert np.max(np.abs(off_diagonal)) < 0.015


def test_randomized_sobol_heston_price_agrees_with_fourier_within_error_budget():
    params = baseline()
    qmc_result = heston_randomized_qmc_price(
        100.0,
        1.0,
        params,
        sobol_power=9,
        n_scrambles=8,
        n_steps=32,
        seed=17,
        control_variate=True,
        brownian_bridge=True,
    )
    fourier = heston_price_fourier(100.0, 1.0, params, "call")["price"]
    assert np.isfinite(qmc_result["price"])
    assert qmc_result["standard_error_across_scrambles"] > 0.0
    assert abs(qmc_result["price"] - fourier) < (
        5.0 * qmc_result["standard_error_across_scrambles"] + 0.15
    )
    assert qmc_result["n_scrambles"] == 8
    assert qmc_result["total_valuation_points"] == 4096
    assert qmc_result["brownian_bridge"] is True
    assert np.isfinite(qmc_result["control_beta_from_independent_pilot"])


def test_randomized_sobol_is_reproducible_for_fixed_seed():
    params = baseline()
    arguments = dict(
        strike=100.0,
        maturity=0.5,
        params=params,
        sobol_power=7,
        n_scrambles=4,
        n_steps=16,
        seed=909,
    )
    first = heston_randomized_qmc_price(**arguments)
    second = heston_randomized_qmc_price(**arguments)
    assert first["price"] == second["price"]
    assert first["confidence_interval"] == second["confidence_interval"]


def test_qmc_rejects_insufficient_scrambles():
    with pytest.raises(ValueError, match="n_scrambles"):
        heston_randomized_qmc_price(
            100.0, 1.0, baseline(), sobol_power=8, n_scrambles=2, n_steps=16
        )



@pytest.mark.parametrize("n_steps", [3, 5, 7, 8, 13])
def test_brownian_bridge_handles_non_power_of_two_time_grids(n_steps):
    rng = np.random.default_rng(300 + n_steps)
    increments = _brownian_bridge_increments(
        rng.standard_normal((25_000, n_steps)), maturity=2.0
    )
    assert increments.shape == (25_000, n_steps)
    assert np.max(np.abs(np.mean(increments, axis=0))) < 0.025
    assert np.var(increments, axis=0, ddof=1) == pytest.approx(
        np.ones(n_steps), abs=0.04
    )



def test_randomized_qmc_put_call_parity_on_common_scrambled_paths():
    params = baseline()
    common = dict(
        maturity=0.75,
        params=params,
        sobol_power=8,
        n_scrambles=6,
        n_steps=12,
        seed=712,
        control_variate=False,
        brownian_bridge=True,
    )
    call = heston_randomized_qmc_price(100.0, option_type="call", **common)
    put = heston_randomized_qmc_price(100.0, option_type="put", **common)
    parity = (
        params.spot * np.exp(-params.dividend_yield * common["maturity"])
        - 100.0 * np.exp(-params.risk_free_rate * common["maturity"])
    )
    assert call["price"] - put["price"] == pytest.approx(parity, abs=1e-10)


def test_randomized_qmc_deterministic_variance_limit_matches_fourier():
    params = baseline(
        variance0=0.04,
        long_run_variance=0.09,
        vol_of_variance=0.0,
    )
    result = heston_randomized_qmc_price(
        103.0,
        1.25,
        params,
        sobol_power=9,
        n_scrambles=8,
        n_steps=16,
        seed=188,
        control_variate=True,
        brownian_bridge=True,
    )
    reference = heston_price_fourier(103.0, 1.25, params)["price"]
    assert abs(result["price"] - reference) < (
        5.0 * result["standard_error_across_scrambles"] + 0.08
    )
