import numpy as np
import pytest

from riskmanage.calibration import HestonOptionQuote
from riskmanage.heston import HestonParameters, heston_price_fourier
from riskmanage.uncertainty import (
    bootstrap_heston_calibration,
    profile_heston_parameter,
    propagate_heston_parameter_uncertainty,
)


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


def quote_panel():
    true = parameters()
    rng = np.random.default_rng(29)
    quotes = []
    for maturity in (0.25, 0.5, 1.0):
        for strike in (85.0, 92.5, 100.0, 107.5, 115.0):
            option_type = "call" if strike >= 100.0 else "put"
            price = heston_price_fourier(
                strike, maturity, true, option_type
            )["price"]
            spread = max(0.02, 0.015 * price)
            sigma = max(spread / 2.0, 0.01) / np.sqrt(3.0)
            mid = max(1e-6, price + rng.normal(0.0, sigma))
            quotes.append(
                HestonOptionQuote(
                    strike,
                    maturity,
                    option_type,
                    float(mid),
                    bid=max(0.0, float(mid) - spread / 2.0),
                    ask=float(mid) + spread / 2.0,
                )
            )
    return quotes


def test_parametric_bootstrap_returns_parameter_uncertainty_samples():
    market = parameters()
    result = bootstrap_heston_calibration(
        quote_panel(),
        market,
        market,
        n_bootstrap=4,
        seed=103,
        n_starts=1,
        max_nfev=100,
    )
    assert result["successful_replicates"] >= 3
    assert result["failed_replicates"] <= 1
    assert result["success_rate"] >= 0.75
    assert len(result["bootstrap_parameter_samples"]) >= 3
    assert result["parameter_summaries"]["correlation"]["standard_deviation"] > 0.0
    assert len(result["natural_parameter_covariance"]) == 5
    assert "Gaussian quote errors" in result["noise_model"]


def test_profile_likelihood_reoptimizes_nuisance_parameters():
    market = parameters()
    result = profile_heston_parameter(
        quote_panel(),
        market,
        market,
        "correlation",
        [-0.85, -0.70, -0.55],
        max_nfev=80,
    )
    assert result["parameter"] == "correlation"
    assert len(result["points"]) == 3
    assert all(point["optimizer_success"] for point in result["points"])
    for requested, point in zip(
        result["parameter_grid"], result["points"], strict=True
    ):
        assert point["profiled_parameters"]["correlation"] == pytest.approx(
            requested, abs=1e-12
        )
        assert point["profiled_parameters"]["mean_reversion"] > 0.0
    assert result["likelihood_ratio_cutoff_delta_chi_square"] > 0.0
    assert result["profile_interval_refined"] is not None
    assert result["profile_interval_refined"][0] < -0.70
    assert result["profile_interval_refined"][1] > -0.70
    assert result["profile_root_evaluations"] >= 2


def test_bootstrap_parameter_draws_propagate_to_price_and_delta_distributions():
    market = parameters()
    samples = [
        {
            "variance0": 0.038,
            "mean_reversion": 1.35,
            "long_run_variance": 0.038,
            "vol_of_variance": 0.44,
            "correlation": -0.76,
        },
        {
            "variance0": 0.042,
            "mean_reversion": 1.65,
            "long_run_variance": 0.042,
            "vol_of_variance": 0.56,
            "correlation": -0.64,
        },
        {
            "variance0": 0.039,
            "mean_reversion": 1.45,
            "long_run_variance": 0.041,
            "vol_of_variance": 0.48,
            "correlation": -0.72,
        },
        {
            "variance0": 0.041,
            "mean_reversion": 1.55,
            "long_run_variance": 0.039,
            "vol_of_variance": 0.52,
            "correlation": -0.68,
        },
    ]
    result = propagate_heston_parameter_uncertainty(
        samples,
        market,
        [
            {"strike": 90.0, "maturity": 0.5, "option_type": "call"},
            {"strike": 100.0, "maturity": 1.0, "option_type": "put"},
        ],
    )
    assert result["bootstrap_sample_count"] == 4
    assert len(result["contracts"]) == 2
    for contract in result["contracts"]:
        assert contract["price_distribution"]["standard_deviation"] > 0.0
        assert contract["delta_distribution"]["standard_deviation"] > 0.0
        assert (
            contract["price_distribution"]["quantiles"]["p2_5"]
            <= contract["price_distribution"]["quantiles"]["p97_5"]
        )


def test_parameter_uncertainty_requires_bid_ask_noise_information():
    market = parameters()
    quote = HestonOptionQuote(100.0, 1.0, "call", 10.0)
    with pytest.raises(ValueError, match="bid and ask"):
        bootstrap_heston_calibration(
            [quote] * 6,
            market,
            market,
            n_bootstrap=2,
        )



def test_positive_quote_simulation_rejects_nonpositive_draws_without_clipping():
    from riskmanage.uncertainty import _positive_quote_mids

    class FixedNoise:
        def normal(self, location, scale):
            return np.array([-0.5, 0.25, -2.0])

    mids, invalid = _positive_quote_mids(
        np.array([0.1, 1.0, 2.0]),
        np.array([0.2, 0.2, 0.2]),
        FixedNoise(),
    )

    np.testing.assert_allclose(mids, [-0.4, 1.25, 0.0])
    np.testing.assert_array_equal(invalid, [True, False, True])
    assert mids[0] < 0.0
    assert mids[2] == 0.0


@pytest.mark.parametrize("kwargs", [
    {"n_bootstrap": 2.5},
    {"n_bootstrap": True},
    {"seed": False},
    {"n_starts": 1.5},
    {"max_nfev": True},
])
def test_heston_bootstrap_rejects_non_integer_controls(kwargs):
    market = parameters()
    options = {"n_bootstrap": 2, "n_starts": 1, "max_nfev": 20, "seed": 9}
    options.update(kwargs)
    with pytest.raises(ValueError, match="integer"):
        bootstrap_heston_calibration(quote_panel(), market, market, **options)
