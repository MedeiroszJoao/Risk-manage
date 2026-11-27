import numpy as np
import pytest

from riskmanage.heston import HestonParameters
from riskmanage.nested_hedge_risk import (
    decompose_heston_hedge_risk_uncertainty,
    two_way_random_effects_decomposition,
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


def bootstrap_samples():
    return [
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
    ]


def test_two_way_random_effects_decomposition_recovers_balanced_components():
    parameter_effect = np.array([-1.0, 0.0, 1.0])
    scenario_effect = np.array([-2.0, 0.0, 2.0])
    matrix = 10.0 + parameter_effect[:, None] + scenario_effect[None, :]
    result = two_way_random_effects_decomposition(matrix)
    components = result["nonnegative_variance_components"]
    assert components["calibration_parameter"] == pytest.approx(1.0)
    assert components["scenario_sampling"] == pytest.approx(4.0)
    assert components["interaction_and_within_cell_mc"] == pytest.approx(0.0)
    assert result["variance_component_proportions"] == pytest.approx(
        {
            "calibration_parameter": 0.2,
            "scenario_sampling": 0.8,
            "interaction_and_within_cell_mc": 0.0,
        }
    )


def test_two_way_random_effects_decomposition_rejects_unbalanced_input_shape():
    with pytest.raises(ValueError, match="B x R"):
        two_way_random_effects_decomposition(np.ones((1, 3)))
    with pytest.raises(ValueError, match="finite"):
        two_way_random_effects_decomposition([[1.0, np.nan], [2.0, 3.0]])


def test_nested_hedge_risk_crosses_calibration_draws_and_scenario_seeds():
    pricing = parameters()
    physical = parameters(
        mean_reversion=1.25,
        long_run_variance=0.05,
        vol_of_variance=0.55,
        correlation=-0.65,
    )
    result = decompose_heston_hedge_risk_uncertainty(
        bootstrap_samples(),
        pricing,
        physical,
        strike=100.0,
        maturity=0.5,
        scenario_seeds=[701, 702],
        physical_total_return=0.04,
        n_paths=120,
        n_steps=2,
        transaction_cost_bps=2.0,
        confidence_level=0.90,
    )
    assert result["design"]["calibration_bootstrap_draws"] == 3
    assert result["design"]["independent_physical_scenario_seeds"] == 2
    assert result["design"]["same_gaussian_innovations_across_calibration_draws_within_seed"]
    for matrix in result["metric_matrices"].values():
        assert np.asarray(matrix).shape == (3, 2)
        assert np.isfinite(matrix).all()
    for metric in ("heston_net_es", "frozen_bs_net_es", "heston_minus_bs_net_es"):
        decomposition = result["variance_decompositions"][metric]
        assert decomposition["n_calibration_draws"] == 3
        assert decomposition["n_scenario_seeds"] == 2
        assert decomposition["estimated_total_variance"] >= 0.0


def test_nested_hedge_risk_requires_independent_scenario_seeds():
    with pytest.raises(ValueError, match="unique"):
        decompose_heston_hedge_risk_uncertainty(
            bootstrap_samples(),
            parameters(),
            parameters(),
            strike=100.0,
            maturity=0.5,
            scenario_seeds=[7, 7],
            n_paths=120,
            n_steps=2,
            confidence_level=0.90,
        )


def test_two_way_random_effects_retains_negative_raw_components_and_truncates():
    interaction_only = np.array([[1.0, -1.0], [-1.0, 1.0]])
    result = two_way_random_effects_decomposition(interaction_only)
    raw = result["raw_variance_component_estimates"]
    clipped = result["nonnegative_variance_components"]
    assert raw["calibration_parameter"] < 0.0
    assert raw["scenario_sampling"] < 0.0
    assert clipped["calibration_parameter"] == 0.0
    assert clipped["scenario_sampling"] == 0.0
    assert clipped["interaction_and_within_cell_mc"] > 0.0


def test_two_way_random_effects_zero_total_variance_has_no_proportions():
    result = two_way_random_effects_decomposition(np.ones((2, 2)))
    assert result["estimated_total_variance"] == 0.0
    assert result["variance_component_proportions"] is None
