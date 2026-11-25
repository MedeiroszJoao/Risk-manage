import numpy as np
import pytest
from riskmanage.risk import historical_var_es, rolling_historical_var, liquidity_adjusted_es

def test_historical_var_es_known_vector():
    result = historical_var_es(np.arange(1, 101, dtype=float), .95)
    assert result["var"] == pytest.approx(96.0, abs=1.0)
    assert result["expected_shortfall"] > result["var"]

def test_rolling_var_has_no_lookahead():
    losses = np.arange(1, 301, dtype=float)
    forecast = rolling_historical_var(losses, window=50, confidence=.95)
    assert np.isnan(forecast.iloc[:50]).all()
    assert forecast.iloc[50] == pytest.approx(np.quantile(losses[:50], .95, method="higher"))
    changed = losses.copy(); changed[50] = 1e9
    forecast2 = rolling_historical_var(changed, window=50, confidence=.95)
    assert forecast2.iloc[50] == forecast.iloc[50]

def test_liquidity_adjusted_es_increases_with_longer_horizon_factors():
    rng = np.random.default_rng(7)
    factors = {"equity": rng.normal(0, 1, 800), "credit_vol": rng.normal(0, 2, 800)}
    short = liquidity_adjusted_es(factors, {"equity": 10, "credit_vol": 10}, .975)
    long = liquidity_adjusted_es(factors, {"equity": 10, "credit_vol": 120}, .975)
    assert long["liquidity_adjusted_es"] >= short["liquidity_adjusted_es"]
