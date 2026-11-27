import numpy as np
import pytest

from riskmanage.capital import (
    aggregate_nmrf_ses,
    build_ima_capital,
    calculate_imcc,
    select_reduced_factor_set,
    select_stress_window,
)


def test_stress_window_selects_worst_cluster():
    rng = np.random.default_rng(10)
    values = rng.normal(0, 1, 700)
    values[330:360] += 5
    result = select_stress_window({"market": values}, window_days=100, confidence=0.975)
    overlap = min(result["end_index"] + 1, 360) - max(result["start_index"], 330)
    assert overlap >= 25
    assert result["supervisory_status"] == "not_assessed"


def test_imcc_formula_exact():
    result = calculate_imcc(
        es_full_current=120,
        es_reduced_current=80,
        es_reduced_stress=100,
        class_components={
            "equity": {"full_current": 60, "reduced_current": 40, "reduced_stress": 50},
            "rates": {"full_current": 60, "reduced_current": 40, "reduced_stress": 50},
        },
        rho=0.5,
    )
    assert result["unconstrained"] == pytest.approx(150)
    assert result["constrained_sum"] == pytest.approx(150)
    assert result["imcc"] == pytest.approx(150)
    assert result["status"] == "formula_primitive_only"


def test_nmrf_aggregation_separates_special_groups():
    charges = {"credit1": 3.0, "equity1": 4.0, "other1": 5.0, "other2": 12.0}
    groups = {
        "credit1": "credit_idio",
        "equity1": "equity_idio",
        "other1": "other",
        "other2": "other",
    }
    result = aggregate_nmrf_ses(charges, groups, 0.6)
    expected = 3 + 4 + np.sqrt((0.6 * 17) ** 2 + (1 - 0.6**2) * (5**2 + 12**2))
    assert result["aggregate"] == pytest.approx(expected)
    assert result["status"] == "formula_primitive_only"


def test_reduced_factor_coverage_uses_a_forward_untouched_holdout():
    rng = np.random.default_rng(77)
    common = rng.normal(0, 1, 700)
    factors = {
        "equity": common,
        "rates": 0.7 * common + rng.normal(0, 0.4, 700),
        "fx": rng.normal(0, 0.25, 700),
    }
    result = select_reduced_factor_set(factors, 0.975, 0.75)
    coverage = result["coverage_12w"]

    assert coverage["status"] == "computed_out_of_sample"
    assert coverage["observations"] == 60
    assert coverage["out_of_sample"] is True
    assert coverage["selection_overlap"] is False
    assert result["coverage_is_out_of_sample"] is True
    assert result["factor_set_frozen_during_validation"] is True
    assert result["selection_window"]["end_index"] < result["validation_window"]["start_index"]
    assert coverage["validation_window"]["observations"] == 60
    assert result["target_met"] == (coverage["mean_ratio"] >= 0.75)
    assert result["explained_ratio"] == pytest.approx(
        result["selection_sample_explained_ratio"]
    )


def test_holdout_observations_cannot_change_factor_selection():
    rng = np.random.default_rng(20261126)
    common = rng.normal(0.0, 1.0, 500)
    factors = {
        "equity": common,
        "rates": 0.65 * common + rng.normal(0.0, 0.35, 500),
        "fx": rng.normal(0.0, 0.20, 500),
    }
    altered = {name: values.copy() for name, values in factors.items()}
    altered["fx"][-60:] *= 25.0

    baseline = select_reduced_factor_set(factors)
    changed_holdout = select_reduced_factor_set(altered)

    assert baseline["selected_factors"] == changed_holdout["selected_factors"]
    assert baseline["selection_sample_explained_ratio"] == pytest.approx(
        changed_holdout["selection_sample_explained_ratio"]
    )
    assert baseline["selection_window"] == changed_holdout["selection_window"]
    assert baseline["coverage_12w"]["mean_ratio"] != pytest.approx(
        changed_holdout["coverage_12w"]["mean_ratio"]
    )

def test_reduced_factor_selection_requires_full_estimation_window():
    with pytest.raises(ValueError, match="insufficient history"):
        select_reduced_factor_set({"equity": np.ones(249)})


def test_reduced_factor_selection_requires_full_310_observation_holdout_design():
    rng = np.random.default_rng(20)
    factors_309 = {"equity": rng.normal(size=309)}
    with pytest.raises(ValueError, match="at least 310"):
        select_reduced_factor_set(factors_309)

    factors_310 = {"equity": rng.normal(size=310)}
    result = select_reduced_factor_set(factors_310)
    assert result["coverage_12w"]["status"] == "computed_out_of_sample"
    assert result["coverage_12w"]["observations"] == 60

def test_ima_research_aggregation_fails_closed_on_short_history():
    rng = np.random.default_rng(21)
    factors = {"equity": rng.normal(0, 1, 250)}
    stress = {"start_index": 0, "end_index": 249, "window_days": 250}
    with pytest.raises(ValueError, match="at least 310"):
        build_ima_capital(factors, stress, {"equity": "equity"})


def test_ima_research_aggregation_fails_closed_on_invalid_stress_indices():
    rng = np.random.default_rng(22)
    factors = {"equity": rng.normal(0, 1, 400)}
    stress = {"start_index": 200, "end_index": 399, "window_days": 250}
    with pytest.raises(ValueError, match="outside|do not match"):
        build_ima_capital(factors, stress, {"equity": "equity"})


def test_ima_aggregation_is_never_labelled_as_regulatory_capital():
    rng = np.random.default_rng(23)
    common = rng.normal(0, 1, 500)
    factors = {
        "equity": common + rng.normal(0, 0.05, 500),
        "fx": common * 0.1 + rng.normal(0, 0.2, 500),
    }
    stress = select_stress_window(factors, window_days=250)
    result = build_ima_capital(factors, stress, {"equity": "equity", "fx": "fx"})
    assert result["capital_label"] == "research_only_non_regulatory"
    assert result["regulatory_status"] == "not_assessed"
    assert result["imcc"] is None
    assert "credit_spread" in result["missing_regulatory_risk_classes"]
    assert result["regulatory_requirements_not_evidenced"]


@pytest.mark.parametrize(
    "stress",
    [
        {"start_index": 0.9, "end_index": 249, "window_days": 250},
        {"start_index": 0, "end_index": 249.9, "window_days": 250},
        {"start_index": 0, "end_index": 249, "window_days": 250.9},
        {"start_index": True, "end_index": 249, "window_days": 250},
    ],
)
def test_ima_research_aggregation_rejects_non_integer_stress_metadata(stress):
    rng = np.random.default_rng(31)
    factors = {"equity": rng.normal(0, 1, 500)}
    with pytest.raises(ValueError, match="integer"):
        build_ima_capital(factors, stress, {"equity": "equity"})


def test_reduced_factor_selection_rejects_boolean_window_values():
    rng = np.random.default_rng(32)
    factors = {"equity": rng.normal(0, 1, 400)}
    with pytest.raises(ValueError, match="estimation_window must be an integer"):
        select_reduced_factor_set(factors, estimation_window=True)
    with pytest.raises(ValueError, match="validation_days must be an integer"):
        select_reduced_factor_set(factors, validation_days=True)


def test_imcc_uses_fixed_basel_weight():
    with pytest.raises(ValueError, match="fixed rho = 0.5"):
        calculate_imcc(
            120, 80, 100,
            {
                "equity": {"full_current": 60, "reduced_current": 40, "reduced_stress": 50},
                "rates": {"full_current": 60, "reduced_current": 40, "reduced_stress": 50},
            },
            rho=0.6,
        )


def test_nmrf_uses_fixed_basel_weight():
    with pytest.raises(ValueError, match="fixed rho = 0.6"):
        aggregate_nmrf_ses(
            {"credit": 3.0}, {"credit": "credit_idio"}, rho=0.5
        )


def test_ima_rejects_unknown_risk_class_mapping():
    rng = np.random.default_rng(33)
    factors = {"equity": rng.normal(0, 1, 500)}
    stress = select_stress_window(factors, window_days=250)
    with pytest.raises(ValueError, match="unsupported regulatory risk class"):
        build_ima_capital(factors, stress, {"equity": "misc"})


@pytest.mark.parametrize("bad_value", ["1.0", True, np.nan, np.inf])
def test_capital_rejects_non_numeric_or_non_finite_factor_losses(bad_value):
    values = np.array([1.0, 2.0, 3.0], dtype=object)
    values[1] = bad_value
    with pytest.raises(ValueError, match="numeric vector"):
        select_reduced_factor_set({"equity": values}, estimation_window=2)


@pytest.mark.parametrize("bad_confidence", ["0.975", True, np.nan])
def test_capital_rejects_non_numeric_confidence(bad_confidence):
    rng = np.random.default_rng(44)
    with pytest.raises(ValueError, match="confidence"):
        select_reduced_factor_set({"equity": rng.normal(size=300)}, confidence=bad_confidence)


def test_capital_rejects_string_es_and_rho_inputs():
    with pytest.raises(ValueError, match="es_full_current must be numeric"):
        calculate_imcc(
            "120", 80, 100,
            {"equity": {"full_current": 60, "reduced_current": 40, "reduced_stress": 50}},
        )
    with pytest.raises(ValueError, match="rho must be numeric"):
        calculate_imcc(
            120, 80, 100,
            {"equity": {"full_current": 60, "reduced_current": 40, "reduced_stress": 50}},
            rho="0.5",
        )


def test_build_ima_capital_validates_rho_even_if_required_class_is_missing():
    rng = np.random.default_rng(45)
    factors = {"equity": rng.normal(size=500)}
    stress = select_stress_window(factors, window_days=250)
    with pytest.raises(ValueError, match="rho must be numeric"):
        build_ima_capital(factors, stress, {"equity": "equity"}, rho="0.5")


def test_capital_rejects_boolean_mixed_with_float_before_numpy_promotion():
    with pytest.raises(ValueError, match="boolean"):
        select_reduced_factor_set({"equity": [True, 0.5, 1.0, 2.0]})


def test_imcc_does_not_apply_portfolio_floor_to_individual_risk_classes():
    result = calculate_imcc(
        es_full_current=100.0,
        es_reduced_current=80.0,
        es_reduced_stress=90.0,
        class_components={
            "equity": {"full_current": 40.0, "reduced_current": 80.0, "reduced_stress": 10.0},
            "rates": {"full_current": 90.0, "reduced_current": 60.0, "reduced_stress": 20.0},
        },
    )

    # Unconstrained: 90 * max(100/80, 1) = 112.5.
    # Constrained: 10 * (40/80) + 20 * (90/60) = 35.
    # The class-level ratio below one must not be floored independently.
    assert result["unconstrained"] == pytest.approx(112.5)
    assert result["class_components"]["equity"] == pytest.approx(5.0)
    assert result["class_components"]["rates"] == pytest.approx(30.0)
    assert result["constrained_sum"] == pytest.approx(35.0)
    assert result["imcc"] == pytest.approx(73.75)


def test_imcc_does_not_floor_individual_risk_class_ratios_at_one():
    result = calculate_imcc(
        es_full_current=100.0,
        es_reduced_current=80.0,
        es_reduced_stress=120.0,
        class_components={
            "rates": {
                "full_current": 30.0,
                "reduced_current": 60.0,
                "reduced_stress": 40.0,
            },
            "equity": {
                "full_current": 70.0,
                "reduced_current": 20.0,
                "reduced_stress": 30.0,
            },
        },
    )

    # MAR33.6 floors the portfolio-wide scaling ratio; MAR33.15 does not
    # independently floor each partial risk-class ratio.
    assert result["unconstrained"] == pytest.approx(150.0)
    assert result["class_components"]["rates"] == pytest.approx(20.0)
    assert result["class_components"]["equity"] == pytest.approx(105.0)
    assert result["constrained_sum"] == pytest.approx(125.0)
    assert result["imcc"] == pytest.approx(137.5)


def test_imcc_portfolio_scaling_ratio_below_one_is_floored_by_mar33_6():
    result = calculate_imcc(
        es_full_current=80.0,
        es_reduced_current=100.0,
        es_reduced_stress=120.0,
        class_components={
            "equity": {
                "full_current": 40.0,
                "reduced_current": 80.0,
                "reduced_stress": 10.0,
            },
        },
    )

    # MAR33.6: the portfolio-wide full/reduced current ES ratio is floored at 1.
    # MAR33.15: the partial class ratio is not independently floored.
    assert result["unconstrained"] == pytest.approx(120.0)
    assert result["class_components"]["equity"] == pytest.approx(5.0)
    assert result["constrained_sum"] == pytest.approx(5.0)
    assert result["imcc"] == pytest.approx(62.5)


def test_imcc_portfolio_scaling_ratio_equal_to_one_is_unchanged():
    result = calculate_imcc(
        es_full_current=100.0,
        es_reduced_current=100.0,
        es_reduced_stress=120.0,
        class_components={
            "equity": {
                "full_current": 50.0,
                "reduced_current": 50.0,
                "reduced_stress": 20.0,
            },
        },
    )

    assert result["unconstrained"] == pytest.approx(120.0)
    assert result["class_components"]["equity"] == pytest.approx(20.0)
    assert result["imcc"] == pytest.approx(70.0)


def test_imcc_applies_floor_to_portfolio_ratio_but_not_class_ratios():
    result = calculate_imcc(
        es_full_current=80.0,
        es_reduced_current=100.0,
        es_reduced_stress=120.0,
        class_components={
            "rates": {
                "full_current": 30.0,
                "reduced_current": 60.0,
                "reduced_stress": 40.0,
            },
            "equity": {
                "full_current": 50.0,
                "reduced_current": 40.0,
                "reduced_stress": 20.0,
            },
        },
    )

    # MAR33.6 floors the portfolio-wide ratio at one. MAR33.15's
    # constrained partial risk-class terms retain their own ratios.
    assert result["unconstrained"] == pytest.approx(120.0)
    assert result["class_components"]["rates"] == pytest.approx(20.0)
    assert result["class_components"]["equity"] == pytest.approx(25.0)
    assert result["constrained_sum"] == pytest.approx(45.0)
    assert result["imcc"] == pytest.approx(82.5)
