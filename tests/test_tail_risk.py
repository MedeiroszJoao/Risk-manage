import numpy as np
import pytest
from scipy.stats import genpareto

from riskmanage.tail_risk import fit_pot_gpd_tail_risk


def test_pot_gpd_recovers_finite_mean_tail_and_reports_bootstrap_intervals():
    rng = np.random.default_rng(321)
    losses = genpareto.rvs(
        c=0.20, loc=0.0, scale=1.0, size=30_000, random_state=rng
    )
    result = fit_pot_gpd_tail_risk(
        losses,
        confidence_levels=(0.975, 0.99),
        threshold_quantiles=(0.80, 0.85, 0.90),
        min_exceedances=100,
        n_bootstrap=30,
        seed=99,
    )
    assert len(result["threshold_fits"]) == 3
    stability = result["threshold_stability"]
    assert stability["automatic_threshold_selection"] is False
    assert stability["shape"]["successful_threshold_fits"] == 3
    assert stability["shape"]["range"] < 0.10
    assert stability["confidence_level_metrics"]["0.975"]["expected_shortfall"][
        "successful_threshold_fits"
    ] == 3
    for fit in result["threshold_fits"]:
        assert fit["status"] == "fitted"
        assert abs(fit["fit"]["shape"] - 0.20) < 0.10
        assert fit["fit"]["scale"] > 0.0
        for alpha in ("0.975", "0.99"):
            estimate = fit["confidence_levels"][alpha]
            assert estimate["status"] == "estimated"
            assert estimate["expected_shortfall"] > estimate["var"]
            interval = fit["conditional_exceedance_bootstrap"][
                "conditional_percentile_intervals"
            ][alpha]["var"]
            assert interval["percentile_95_interval"][0] <= interval["percentile_95_interval"][1]


def test_pot_gpd_flags_infinite_expected_shortfall_for_shape_above_one():
    rng = np.random.default_rng(552)
    losses = genpareto.rvs(
        c=1.15, loc=0.0, scale=1.0, size=20_000, random_state=rng
    )
    result = fit_pot_gpd_tail_risk(
        losses,
        confidence_levels=(0.975,),
        threshold_quantiles=(0.90,),
        min_exceedances=50,
        n_bootstrap=0,
        n_full_sample_bootstrap=0,
    )
    fit = result["threshold_fits"][0]
    assert fit["status"] == "fitted"
    assert fit["fit"]["shape"] >= 1.0
    assert fit["confidence_levels"]["0.975"]["expected_shortfall"] is None
    assert any("infinite" in warning for warning in fit["warnings"])


def test_pot_gpd_does_not_fit_below_minimum_exceedance_count():
    rng = np.random.default_rng(22)
    losses = rng.standard_normal(1_000)
    result = fit_pot_gpd_tail_risk(
        losses,
        confidence_levels=(0.99,),
        threshold_quantiles=(0.98,),
        min_exceedances=50,
        n_bootstrap=0,
        n_full_sample_bootstrap=0,
    )
    fit = result["threshold_fits"][0]
    assert fit["status"] == "not_fitted"
    assert fit["exceedance_count"] < 50


def test_pot_gpd_rejects_bad_thresholds_and_confidence_levels():
    losses = np.random.default_rng(2).standard_normal(1_000)
    with pytest.raises(ValueError, match="exceed"):
        fit_pot_gpd_tail_risk(losses, confidence_levels=(0.95,), threshold_quantiles=(0.95,))
    with pytest.raises(ValueError, match="unique"):
        fit_pot_gpd_tail_risk(losses, confidence_levels=(0.99,), threshold_quantiles=(0.90, 0.90))



def test_pot_gpd_bootstrap_is_reproducible_for_fixed_seed():
    rng = np.random.default_rng(932)
    losses = genpareto.rvs(
        c=0.15, loc=0.0, scale=1.2, size=8_000, random_state=rng
    )
    kwargs = dict(
        confidence_levels=(0.975,),
        threshold_quantiles=(0.90,),
        min_exceedances=50,
        n_bootstrap=20,
        seed=81,
    )
    first = fit_pot_gpd_tail_risk(losses, **kwargs)
    second = fit_pot_gpd_tail_risk(losses, **kwargs)
    first_fit = first["threshold_fits"][0]
    second_fit = second["threshold_fits"][0]
    assert first_fit["fit"] == second_fit["fit"]
    assert first_fit["conditional_exceedance_bootstrap"] == second_fit[
        "conditional_exceedance_bootstrap"
    ]
    assert first_fit["full_sample_iid_bootstrap"] == second_fit[
        "full_sample_iid_bootstrap"
    ]


def test_pot_gpd_reports_empirical_tail_and_no_extrapolation_p_value():
    losses = np.random.default_rng(71).standard_t(df=5, size=5_000)
    result = fit_pot_gpd_tail_risk(
        losses,
        confidence_levels=(0.975,),
        threshold_quantiles=(0.90, 0.95),
        min_exceedances=20,
        n_bootstrap=0,
    )
    assert result["empirical_var_es"]["0.975"]["status"] == "estimated"
    for fit in result["threshold_fits"]:
        if fit["status"] == "fitted":
            assert "ks_uniform_distance_descriptive" in fit["fit"]
            assert "p_value" not in fit["fit"]
            assert fit["exceedance_count"] >= 20
