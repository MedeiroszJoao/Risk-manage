import numpy as np
import pytest

from riskmanage.volatility_surface import (
    fit_ssvi_power_law,
    ssvi_implied_volatility,
    ssvi_phi_power_law,
    ssvi_static_arbitrage_certificate,
    ssvi_total_variance,
)


def test_ssvi_atm_total_variance_equals_theta():
    theta = np.array([0.02, 0.05, 0.10, 0.25])
    result = ssvi_total_variance(0.0, theta, rho=-0.4, eta=0.8, gamma=0.4)
    assert np.allclose(result, theta, rtol=0.0, atol=1e-14)


def test_ssvi_is_symmetric_when_rho_is_zero():
    k = np.linspace(-0.8, 0.8, 31)
    left = ssvi_total_variance(k, 0.12, rho=0.0, eta=0.7, gamma=0.4)
    right = ssvi_total_variance(-k, 0.12, rho=0.0, eta=0.7, gamma=0.4)
    assert np.allclose(left, right, rtol=1e-13, atol=1e-14)


def test_ssvi_broadcasting_and_implied_volatility_conversion():
    k = np.array([-0.2, 0.0, 0.2])
    theta = np.array([[0.04], [0.09]])
    total = ssvi_total_variance(k, theta, rho=-0.3, eta=0.8, gamma=0.4)
    implied = ssvi_implied_volatility(
        k, theta, np.array([[0.5], [1.0]]), rho=-0.3, eta=0.8, gamma=0.4
    )
    assert total.shape == (2, 3)
    assert implied.shape == (2, 3)
    assert np.allclose(implied**2 * np.array([[0.5], [1.0]]), total)


def test_power_law_shape_function_is_positive_and_finite():
    theta = np.array([1e-5, 0.01, 0.1, 1.0, 10.0])
    result = ssvi_phi_power_law(theta, eta=0.9, gamma=0.45)
    assert np.isfinite(result).all()
    assert np.all(result > 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": -1.0, "eta": 0.8, "gamma": 0.4},
        {"rho": 0.2, "eta": 0.0, "gamma": 0.4},
        {"rho": 0.2, "eta": 0.8, "gamma": 1.0},
    ],
)
def test_ssvi_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ssvi_total_variance(0.1, 0.1, **kwargs)


def test_power_law_certificate_passes_global_sufficient_region():
    theta = np.array([0.02, 0.05, 0.10, 0.20, 0.35])
    result = ssvi_static_arbitrage_certificate(
        theta, rho=-0.4, eta=0.9, gamma=0.45
    )
    assert result["certified_static_arbitrage_free"]
    assert result["theta_nondecreasing"]
    assert result["calendar_shape_condition"]
    assert result["butterfly_condition_1"]
    assert result["butterfly_condition_2"]
    assert result["power_law_global_region"]


def test_certificate_does_not_call_non_sufficient_parameters_arbitrage_free():
    theta = np.array([0.02, 0.05, 0.10, 0.20])
    result = ssvi_static_arbitrage_certificate(
        theta, rho=-0.4, eta=1.8, gamma=0.45
    )
    assert not result["certified_static_arbitrage_free"]
    assert not result["power_law_global_region"]


def test_certificate_rejects_decreasing_atm_total_variance_term_structure():
    theta = np.array([0.02, 0.08, 0.07, 0.15])
    result = ssvi_static_arbitrage_certificate(
        theta, rho=-0.4, eta=0.8, gamma=0.45
    )
    assert not result["certified_static_arbitrage_free"]
    assert not result["theta_nondecreasing"]


def test_calibration_recovers_synthetic_ssvi_parameters():
    true_rho, true_eta, true_gamma = -0.42, 0.90, 0.45
    expiries_theta = np.array([0.02, 0.05, 0.10, 0.20])
    strike_grid = np.linspace(-0.6, 0.6, 9)
    log_moneyness = np.tile(strike_grid, expiries_theta.size)
    theta = np.repeat(expiries_theta, strike_grid.size)
    market = ssvi_total_variance(
        log_moneyness, theta, true_rho, true_eta, true_gamma
    )

    expiry_times = np.repeat(np.array([0.25, 0.5, 1.0, 2.0]), strike_grid.size)
    result = fit_ssvi_power_law(
        log_moneyness,
        theta,
        market,
        expiry_times=expiry_times,
        n_starts=5,
        loss="linear",
    )

    assert result["success"]
    assert result["rho"] == pytest.approx(true_rho, abs=2e-4)
    assert result["eta"] == pytest.approx(true_eta, abs=2e-4)
    assert result["gamma"] == pytest.approx(true_gamma, abs=2e-4)
    assert result["rmse_total_variance"] < 1e-8
    assert result["arbitrage_certificate"]["certified_static_arbitrage_free"]


def test_calibration_rejects_insufficient_or_misaligned_quotes():
    with pytest.raises(ValueError):
        fit_ssvi_power_law(
            np.array([-0.1, 0.1, 0.2, 0.3, 0.4]),
            np.array([0.04, 0.04, 0.08, 0.08, 0.08]),
            np.array([0.04, 0.05, 0.07, 0.08, 0.09]),
            expiry_times=np.array([0.25, 0.25, 0.25, 0.5, 0.5]),
        )
    with pytest.raises(ValueError):
        fit_ssvi_power_law(
            np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]),
            np.array([0.04, 0.04, 0.04, 0.04, 0.08, 0.08]),
            np.array([0.04, 0.04, 0.04, 0.04, 0.08, 0.08]),
            expiry_times=np.array([0.25, 0.25, 0.25, 0.25, 0.5, 0.5]),
            weights=np.array([1, 1, -1, 1, 1, 1]),
        )



def test_calibration_rejects_decreasing_atm_total_variance_by_expiry():
    log_moneyness = np.array([-0.2, 0.0, 0.2, -0.2, 0.0, 0.2])
    theta = np.array([0.08, 0.08, 0.08, 0.05, 0.05, 0.05])
    expiry_times = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.5])
    market = ssvi_total_variance(
        log_moneyness, theta, rho=-0.3, eta=0.8, gamma=0.4
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        fit_ssvi_power_law(
            log_moneyness, theta, market, expiry_times=expiry_times
        )


def test_calibration_rejects_inconsistent_atm_theta_within_expiry():
    log_moneyness = np.array([-0.2, 0.0, 0.2, -0.2, 0.0, 0.2])
    theta = np.array([0.04, 0.04, 0.05, 0.08, 0.08, 0.08])
    expiry_times = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.5])
    market = np.array([0.04, 0.05, 0.06, 0.08, 0.09, 0.10])
    with pytest.raises(ValueError, match="constant within each expiry"):
        fit_ssvi_power_law(
            log_moneyness, theta, market, expiry_times=expiry_times
        )



def test_price_calibration_recovers_synthetic_ssvi_parameters():
    from riskmanage.black76 import black76_price
    from riskmanage.volatility_surface import fit_ssvi_power_law_to_black76_prices

    rho, eta, gamma = -0.42, 0.90, 0.45
    expiry_grid = np.array([0.25, 0.5, 1.0, 2.0])
    theta_grid = np.array([0.02, 0.05, 0.10, 0.20])
    k_grid = np.linspace(-0.6, 0.6, 9)
    expiry = np.repeat(expiry_grid, 9)
    theta = np.repeat(theta_grid, 9)
    log_moneyness = np.tile(k_grid, expiry_grid.size)
    forward = np.full(log_moneyness.size, 100.0)
    strike = forward*np.exp(log_moneyness)
    discount = np.exp(-0.03*expiry)
    kinds = np.where(log_moneyness < 0.0, "put", "call")
    total = ssvi_total_variance(log_moneyness, theta, rho, eta, gamma)
    volatility = np.sqrt(total/expiry)
    prices = black76_price(forward, strike, expiry, discount, volatility, kinds)

    result = fit_ssvi_power_law_to_black76_prices(
        forward, strike, expiry, discount, theta, kinds, prices, n_starts=5
    )
    assert result["success"]
    assert result["rho"] == pytest.approx(rho, abs=3e-3)
    assert result["eta"] == pytest.approx(eta, abs=3e-3)
    assert result["gamma"] == pytest.approx(gamma, abs=3e-3)
    assert result["rmse_mid_price"] < 1e-7
    assert result["arbitrage_certificate"]["certified_static_arbitrage_free"]


def test_price_calibration_bid_ask_interval_loss_is_zero_inside_spread():
    from riskmanage.black76 import black76_price
    from riskmanage.volatility_surface import fit_ssvi_power_law_to_black76_prices

    expiry_grid = np.array([0.25, 0.5])
    theta_grid = np.array([0.02, 0.05])
    k_grid = np.linspace(-0.5, 0.5, 7)
    expiry = np.repeat(expiry_grid, k_grid.size)
    theta = np.repeat(theta_grid, k_grid.size)
    log_moneyness = np.tile(k_grid, expiry_grid.size)
    forward = np.full(log_moneyness.size, 100.0)
    strike = forward*np.exp(log_moneyness)
    discount = np.exp(-0.02*expiry)
    kinds = np.where(log_moneyness < 0.0, "put", "call")
    total = ssvi_total_variance(log_moneyness, theta, -0.3, 0.8, 0.4)
    volatility = np.sqrt(total/expiry)
    mid = black76_price(forward, strike, expiry, discount, volatility, kinds)
    intrinsic = np.where(kinds == "call", discount*np.maximum(forward-strike, 0.0),
                         discount*np.maximum(strike-forward, 0.0))
    bid = np.maximum(mid-0.02, intrinsic+1e-8)
    ask = mid+0.02

    result = fit_ssvi_power_law_to_black76_prices(
        forward, strike, expiry, discount, theta, kinds, mid,
        bid_prices=bid, ask_prices=ask, n_starts=3,
    )
    assert result["success"]
    assert result["objective"] == "bid_ask_interval_distance"
    assert result["quotes_inside_bid_ask"] == len(mid)
    assert result["cost"] == pytest.approx(0.0, abs=1e-14)


def test_price_calibration_rejects_decreasing_atm_variance_by_expiry():
    from riskmanage.volatility_surface import fit_ssvi_power_law_to_black76_prices

    with pytest.raises(ValueError, match="non-decreasing"):
        fit_ssvi_power_law_to_black76_prices(
            [100]*6, [90, 100, 110, 90, 100, 110],
            [0.25]*3+[0.5]*3, [0.99]*6,
            [0.08]*3+[0.04]*3, ["put", "call", "call"]*2,
            [10, 5, 2, 12, 6, 3],
        )


@pytest.mark.parametrize("values", [[True, 0.1, 0.2], ["0.1", "0.2", "0.3"]])
def test_ssvi_numeric_inputs_reject_boolean_and_string_coercion(values):
    with pytest.raises(ValueError, match="numeric|boolean"):
        ssvi_total_variance(values, 0.1, rho=-0.3, eta=0.8, gamma=0.4)
