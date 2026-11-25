import numpy as np
import pytest
from riskmanage.capital import select_stress_window, calculate_imcc, aggregate_nmrf_ses

def test_stress_window_selects_worst_cluster():
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, 700); x[330:360] += 5
    result = select_stress_window({"market": x}, window_days=100, confidence=.975)
    assert result["start_index"] <= 330
    assert result["end_index"] >= 359

def test_imcc_formula_exact():
    result = calculate_imcc(
        es_full_current=120, es_reduced_current=80, es_reduced_stress=100,
        class_components={
            "equity": {"full_current": 60, "reduced_current": 40, "reduced_stress": 50},
            "rates": {"full_current": 60, "reduced_current": 40, "reduced_stress": 50},
        }, rho=.5)
    assert result["unconstrained"] == pytest.approx(150)
    assert result["constrained_sum"] == pytest.approx(150)
    assert result["imcc"] == pytest.approx(150)

def test_nmrf_aggregation_separates_special_groups():
    charges={"credit1":3.0,"equity1":4.0,"other1":5.0,"other2":12.0}
    groups={"credit1":"credit_idio","equity1":"equity_idio","other1":"other","other2":"other"}
    result=aggregate_nmrf_ses(charges,groups,.6)
    expected=3+4+np.sqrt((.6*17)**2+(1-.6**2)*(5**2+12**2))
    assert result["aggregate"] == pytest.approx(expected)
