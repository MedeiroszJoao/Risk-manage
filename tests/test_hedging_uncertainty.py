import numpy as np
import pytest

from riskmanage.hedging_uncertainty import (
    propagate_heston_uncertainty_to_hedging,
)
from riskmanage.heston import HestonParameters


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


def run_experiment():
    pricing = parameters()
    physical = parameters(
        mean_reversion=1.25,
        long_run_variance=0.05,
        vol_of_variance=0.55,
        correlation=-0.65,
    )
    return propagate_heston_uncertainty_to_hedging(
        bootstrap_samples(),
        pricing,
        physical,
        strike=100.0,
        maturity=0.5,
        physical_total_return=0.04,
        n_paths=180,
        n_steps=3,
        transaction_cost_bps=2.0,
        confidence_level=0.90,
        seed=404,
    )


def test_bootstrap_parameter_draws_propagate_to_hedge_var_es():
    result = run_experiment()
    assert result["classification"] == (
        "synthetic_common_scenario_parameter_uncertainty_experiment"
    )
    assert result["bootstrap_parameter_draw_count"] == 3
    assert result["scenario"][
        "same_standard_normal_innovations_for_all_parameter_draws"
    ] is True
    assert result["scenario"]["path_states_depend_on_draw_specific_initial_variance"] is True
    assert result["scenario"][
        "identical_parameter_draws_produce_identical_path_states"
    ] is True
    for draw in result["draw_level_results"]:
        assert draw["physical_scenario_initial_variance"] == pytest.approx(
            draw["risk_neutral_parameters"]["variance0"]
        )
    assert result["scenario"]["n_paths"] == 180
    for strategy in ("heston_state_delta", "frozen_bs_delta"):
        distributions = result["strategies"][strategy][
            "parameter_conditional_risk_distribution"
        ]
        assert distributions["net_es"]["standard_deviation"] > 0.0
        assert distributions["net_var"]["standard_deviation"] > 0.0
        assert distributions["average_turnover"]["standard_deviation"] >= 0.0
    comparison = result["paired_parameter_draw_comparison"]
    assert 0.0 <= comparison["probability_heston_es_below_bs_es"] <= 1.0
    assert np.isfinite(
        comparison["heston_minus_bs_es_distribution"]["mean"]
    )
    assert len(result["draw_level_results"]) == 3


def test_same_seed_reproduces_parameter_conditional_hedge_risk():
    first = run_experiment()
    second = run_experiment()
    first_es = first["strategies"]["heston_state_delta"][
        "parameter_conditional_risk_distribution"
    ]["net_es"]
    second_es = second["strategies"]["heston_state_delta"][
        "parameter_conditional_risk_distribution"
    ]["net_es"]
    assert first_es == second_es
    assert (
        first["paired_parameter_draw_comparison"]
        == second["paired_parameter_draw_comparison"]
    )


def test_hedge_uncertainty_rejects_too_few_bootstrap_parameter_draws():
    with pytest.raises(ValueError, match="at least two"):
        propagate_heston_uncertainty_to_hedging(
            bootstrap_samples()[:1],
            parameters(),
            parameters(),
            strike=100.0,
            maturity=0.5,
            n_paths=100,
            n_steps=2,
            confidence_level=0.90,
        )



def test_identical_parameter_draws_reuse_identical_gaussian_scenario_innovations():
    pricing = parameters()
    physical = parameters(
        mean_reversion=1.25,
        long_run_variance=0.05,
        vol_of_variance=0.55,
        correlation=-0.65,
    )
    one_draw = bootstrap_samples()[0]
    result = propagate_heston_uncertainty_to_hedging(
        [one_draw, one_draw],
        pricing,
        physical,
        strike=100.0,
        maturity=0.5,
        physical_total_return=0.04,
        n_paths=120,
        n_steps=2,
        transaction_cost_bps=2.0,
        confidence_level=0.90,
        seed=909,
    )
    first, second = result["draw_level_results"]
    assert first["physical_scenario_initial_variance"] == pytest.approx(
        second["physical_scenario_initial_variance"]
    )
    assert first["heston_state_delta_strategy"] == second["heston_state_delta_strategy"]
    assert first["frozen_bs_delta_strategy"] == second["frozen_bs_delta_strategy"]
    assert first["paired_comparison"] == second["paired_comparison"]
