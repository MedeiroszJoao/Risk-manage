import numpy as np
import pytest

from riskmanage.risk import (
    EMPIRICAL_QUANTILE_METHOD,
    historical_var_es,
    liquidity_adjusted_es,
    rolling_historical_var,
    select_stress_window,
)


def test_historical_var_es_known_vector_uses_integrated_empirical_quantile():
    result = historical_var_es(np.arange(1, 101, dtype=float), 0.95)

    assert result["var"] == pytest.approx(95.0)
    assert result["expected_shortfall"] == pytest.approx(98.0)
    assert result["tail_mass_observations"] == pytest.approx(5.0)
    assert result["fractional_var_boundary_mass"] == pytest.approx(0.0)
    assert result["quantile_method"] == EMPIRICAL_QUANTILE_METHOD


def test_historical_es_uses_fractional_mass_when_tail_cuts_between_observations():
    result = historical_var_es(np.arange(1, 11, dtype=float), 0.75)

    assert result["var"] == pytest.approx(8.0)
    assert result["strict_tail_observations"] == 2
    assert result["fractional_var_boundary_mass"] == pytest.approx(0.5)
    assert result["tail_mass_observations"] == pytest.approx(2.5)
    assert result["expected_shortfall"] == pytest.approx(9.2)


def test_historical_es_does_not_overweight_ties_at_var():
    losses = np.array([0, 1, 2, 3, 4, 4, 4, 5, 8, 10], dtype=float)
    result = historical_var_es(losses, 0.65)

    naive_inclusive_tail_mean = float(losses[losses >= result["var"]].mean())

    assert result["var"] == pytest.approx(4.0)
    assert result["var_tie_observations"] == 3
    assert result["strict_tail_observations"] == 3
    assert result["fractional_var_boundary_mass"] == pytest.approx(0.5)
    assert result["expected_shortfall"] == pytest.approx(50.0 / 7.0)
    assert result["expected_shortfall"] > naive_inclusive_tail_mean


def test_historical_es_is_well_defined_when_all_losses_are_tied():
    result = historical_var_es(np.full(10, 2.0), 0.95)

    assert result["var"] == pytest.approx(2.0)
    assert result["expected_shortfall"] == pytest.approx(2.0)
    assert result["fractional_var_boundary_mass"] == pytest.approx(0.5)
    assert result["var_tie_observations"] == 10


def test_historical_var_es_obeys_translation_and_positive_scale_invariance():
    losses = np.array([-3.0, -1.0, 0.5, 2.0, 4.0, 5.0, 8.0, 13.0])
    baseline = historical_var_es(losses, 0.75)
    shifted = historical_var_es(losses + 7.0, 0.75)
    scaled = historical_var_es(3.0 * losses, 0.75)

    assert shifted["var"] == pytest.approx(baseline["var"] + 7.0)
    assert shifted["expected_shortfall"] == pytest.approx(
        baseline["expected_shortfall"] + 7.0
    )
    assert scaled["var"] == pytest.approx(3.0 * baseline["var"])
    assert scaled["expected_shortfall"] == pytest.approx(
        3.0 * baseline["expected_shortfall"]
    )


def test_rolling_var_has_no_lookahead_and_uses_same_quantile_convention():
    losses = np.arange(1, 301, dtype=float)
    forecast = rolling_historical_var(losses, window=50, confidence=0.95)

    assert np.isnan(forecast.iloc[:50]).all()
    assert forecast.iloc[50] == pytest.approx(
        np.quantile(losses[:50], 0.95, method=EMPIRICAL_QUANTILE_METHOD)
    )

    changed = losses.copy()
    changed[50] = 1e9
    forecast_after_change = rolling_historical_var(changed, window=50, confidence=0.95)
    assert forecast_after_change.iloc[50] == forecast.iloc[50]


def test_liquidity_adjusted_es_increases_with_longer_horizon_factors():
    rng = np.random.default_rng(7)
    factors = {
        "equity": rng.normal(0, 1, 800),
        "credit_vol": rng.normal(0, 2, 800),
    }
    short = liquidity_adjusted_es(factors, {"equity": 10, "credit_vol": 10}, 0.975)
    long = liquidity_adjusted_es(factors, {"equity": 10, "credit_vol": 120}, 0.975)

    assert long["liquidity_adjusted_es"] >= short["liquidity_adjusted_es"]
    assert (
        long["expected_shortfall_convention"]
        == "integrated_empirical_quantile_fractional_var_mass"
    )


def test_historical_var_es_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        historical_var_es(np.arange(1, 11, dtype=float), 1.0)


def test_stress_window_selects_maximum_10_observation_es_and_earliest_tie():
    losses = np.array([0] * 10 + [1] * 20, dtype=float)
    result = select_stress_window({"desk": losses}, window_days=20, confidence=0.75)

    candidates = [
        historical_var_es(np.convolve(losses[i : i + 20], np.ones(10), mode="valid"), 0.75)["expected_shortfall"]
        for i in range(11)
    ]
    assert result["stress_es_10d"] == pytest.approx(max(candidates))
    assert result["start_index"] == candidates.index(max(candidates))
    assert result["selection_method"] == "maximum_historical_10_observation_expected_shortfall"

    tied = select_stress_window({"desk": np.ones(30)}, window_days=20)
    assert tied["start_index"] == 0
    assert tied["tie_break"] == "earliest_window"


def test_stress_window_rejects_window_shorter_than_es_horizon():
    with pytest.raises(ValueError, match="at least 11"):
        select_stress_window({"desk": np.arange(20, dtype=float)}, window_days=10)



def test_ten_day_loss_horizon_rejects_fractional_and_boolean_inputs():
    from riskmanage.risk import _ten_day_losses

    with pytest.raises(ValueError, match="horizon must be an integer"):
        _ten_day_losses(np.arange(1.0, 25.0), horizon=10.5)
    with pytest.raises(ValueError, match="horizon must be an integer"):
        _ten_day_losses(np.arange(1.0, 25.0), horizon=True)


def test_liquidity_adjusted_es_requires_ten_day_base_horizon():
    factors = {"equity": np.random.default_rng(91).normal(size=300)}
    with pytest.raises(ValueError, match="fixed 10-day base horizon"):
        liquidity_adjusted_es(factors, {"equity": 10}, base_horizon=5)


@pytest.mark.parametrize("values", [["1", "2", "3"], [True, False, True]])
def test_risk_rejects_string_and_boolean_losses_before_numeric_coercion(values):
    with pytest.raises(ValueError, match="1D numeric vector"):
        historical_var_es(values, 0.95)


@pytest.mark.parametrize("confidence", [True, "0.95", np.nan, np.inf])
def test_risk_rejects_non_numeric_or_non_finite_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        historical_var_es(np.arange(1, 51, dtype=float), confidence)


@pytest.mark.parametrize("window", [20.5, True, "50"])
def test_rolling_var_rejects_non_integer_window(window):
    with pytest.raises(ValueError, match="window must be an integer"):
        rolling_historical_var(np.arange(1, 101, dtype=float), window=window)


@pytest.mark.parametrize("window_days", [20.5, True])
def test_stress_window_rejects_non_integer_window(window_days):
    with pytest.raises(ValueError, match="window_days must be an integer"):
        select_stress_window(
            {"desk": np.arange(1, 101, dtype=float)}, window_days=window_days
        )


def test_liquidity_adjusted_es_rejects_fractional_liquidity_horizon():
    factors = {"equity": np.random.default_rng(93).normal(size=300)}
    with pytest.raises(ValueError, match="liquidity_horizon"):
        liquidity_adjusted_es(factors, {"equity": 10.5})


def test_empirical_es_matches_independent_fractional_tail_sum():
    losses = np.array([1, 2, 2, 3, 4, 5, 6, 7], dtype=float)
    confidence = 0.80
    ordered = np.sort(losses)[::-1]
    mass = len(losses) * (1.0 - confidence)
    whole = int(np.floor(mass))
    fraction = mass - whole
    independent_es = (ordered[:whole].sum() + fraction * ordered[whole]) / mass
    result = historical_var_es(losses, confidence)
    assert result["expected_shortfall"] == pytest.approx(independent_es)
    assert result["expected_shortfall"] >= result["var"]


def test_risk_rejects_boolean_mixed_with_float_before_numpy_promotion():
    with pytest.raises(ValueError, match="boolean"):
        historical_var_es([True, 0.5, 1.0, 2.0], 0.75)
