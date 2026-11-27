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


def test_reduced_factor_coverage_is_explicitly_in_sample_not_oos():
    rng = np.random.default_rng(77)
    common = rng.normal(0, 1, 700)
    factors = {
        "equity": common,
        "rates": 0.7 * common + rng.normal(0, 0.4, 700),
        "fx": rng.normal(0, 0.25, 700),
    }
    result = select_reduced_factor_set(factors, 0.975, 0.75)
    coverage = result["coverage_12w"]
    assert coverage["status"] == "descriptive_in_sample"
    assert coverage["observations"] == 60
    assert result["coverage_is_out_of_sample"] is False
    assert result["target_met_in_sample"]


def test_reduced_factor_selection_requires_full_estimation_window():
    with pytest.raises(ValueError, match="insufficient history"):
        select_reduced_factor_set({"equity": np.ones(249)})


def test_reduced_factor_coverage_does_not_claim_validation_without_309_rows():
    rng = np.random.default_rng(20)
    common = rng.normal(0, 1, 300)
    result = select_reduced_factor_set({"equity": common}, estimation_window=250, validation_days=60)
    assert result["coverage_12w"]["status"] == "insufficient_history"
    assert result["coverage_is_out_of_sample"] is False


def test_ima_research_aggregation_fails_closed_on_short_history():
    rng = np.random.default_rng(21)
    factors = {"equity": rng.normal(0, 1, 250)}
    stress = {"start_index": 0, "end_index": 249, "window_days": 250}
    with pytest.raises(ValueError, match="at least 309"):
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
