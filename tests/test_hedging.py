import numpy as np
import pytest

from riskmanage.hedging import heston_delta_hedge_experiment
from riskmanage.heston import HestonParameters


def parameter_set(**overrides):
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


def test_heston_delta_hedge_runs_two_policies_on_same_physical_paths():
    pricing = parameter_set()
    physical = parameter_set()
    result = heston_delta_hedge_experiment(
        pricing,
        physical,
        strike=100.0,
        maturity=0.5,
        physical_total_return=0.03,
        n_paths=500,
        n_steps=6,
        transaction_cost_bps=0.0,
        confidence_level=0.95,
        seed=21,
    )
    assert result["classification"] == "synthetic_physical_scenario_not_market_backtest"
    assert result["simulation"]["n_paths"] == 500
    assert set(result["strategies"]) == {
        "heston_fourier_delta",
        "frozen_black_scholes_delta",
    }
    for strategy in result["strategies"].values():
        assert np.isfinite(strategy["net_pnl_mean"])
        assert strategy["net_loss_var_es"]["expected_shortfall"] >= (
            strategy["net_loss_var_es"]["var"]
        )
        assert strategy["average_transaction_cost"] == 0.0
    assert np.isfinite(
        result["paired_comparison"]["mean_pnl_difference_heston_minus_bs"]
    )


def test_transaction_costs_reduce_net_hedging_pnl():
    pricing = parameter_set()
    physical = parameter_set()
    arguments = dict(
        pricing_params=pricing,
        physical_params=physical,
        strike=100.0,
        maturity=0.5,
        physical_total_return=0.03,
        n_paths=500,
        n_steps=6,
        confidence_level=0.95,
        seed=22,
    )
    no_cost = heston_delta_hedge_experiment(
        **arguments, transaction_cost_bps=0.0
    )
    with_cost = heston_delta_hedge_experiment(
        **arguments, transaction_cost_bps=5.0
    )
    for strategy in no_cost["strategies"]:
        free = no_cost["strategies"][strategy]
        charged = with_cost["strategies"][strategy]
        assert charged["average_transaction_cost"] > 0.0
        assert charged["net_pnl_mean"] < free["net_pnl_mean"]
        assert charged["gross_pnl_mean"] == pytest.approx(
            free["gross_pnl_mean"], rel=1e-10, abs=1e-10
        )
        assert (
            charged["gross_pnl_mean"] - charged["net_pnl_mean"]
        ) == pytest.approx(
            charged["average_transaction_cost_future_value"], abs=1e-10
        )


def test_hedging_rejects_inconsistent_initial_state_and_insufficient_tail_sample():
    pricing = parameter_set()
    with pytest.raises(ValueError, match="initial variance"):
        heston_delta_hedge_experiment(
            pricing,
            parameter_set(variance0=0.05),
            100.0,
            0.5,
            n_paths=500,
            n_steps=4,
        )
    with pytest.raises(ValueError, match="at least 40"):
        heston_delta_hedge_experiment(
            pricing,
            pricing,
            100.0,
            0.5,
            n_paths=20,
            n_steps=4,
        )


def test_constant_volatility_heston_delta_matches_frozen_black_scholes_hedge():
    deterministic = parameter_set(
        variance0=0.04,
        long_run_variance=0.04,
        vol_of_variance=0.0,
        correlation=0.0,
    )
    result = heston_delta_hedge_experiment(
        deterministic,
        deterministic,
        strike=100.0,
        maturity=0.5,
        physical_total_return=deterministic.risk_free_rate,
        n_paths=250,
        n_steps=5,
        transaction_cost_bps=2.0,
        confidence_level=0.95,
        seed=121,
    )
    heston = result["strategies"]["heston_fourier_delta"]
    black_scholes = result["strategies"]["frozen_black_scholes_delta"]
    assert heston["net_pnl_mean"] == pytest.approx(
        black_scholes["net_pnl_mean"], abs=2e-7
    )
    assert heston["net_loss_var_es"]["expected_shortfall"] == pytest.approx(
        black_scholes["net_loss_var_es"]["expected_shortfall"], abs=2e-7
    )


def test_hedge_experiment_can_return_pathwise_losses_for_tail_models():
    result = heston_delta_hedge_experiment(
        parameter_set(),
        parameter_set(),
        strike=100.0,
        maturity=0.5,
        physical_total_return=0.03,
        n_paths=120,
        n_steps=2,
        confidence_level=0.90,
        seed=812,
        include_path_samples=True,
    )
    samples = result["path_samples"]
    for strategy in ("heston_fourier_delta", "frozen_black_scholes_delta"):
        assert len(samples[strategy]["net_losses"]) == 120
        assert len(samples[strategy]["gross_losses"]) == 120
        assert np.isfinite(samples[strategy]["net_losses"]).all()
        assert np.isfinite(samples[strategy]["gross_losses"]).all()
