import numpy as np
import pandas as pd
import pytest

from riskmanage.validation import (
    backtest_var,
    christoffersen_independence,
    kupiec_pof,
    pla_test,
    rfet_diagnostic,
)


def test_kupiec_no_exceptions_is_finite():
    result = kupiec_pof(0, 250, 0.99)
    assert np.isfinite(result["lr_statistic"])
    assert 0 <= result["p_value"] <= 1


def test_kupiec_rejects_non_integer_counts():
    with pytest.raises(ValueError, match="integers"):
        kupiec_pof(1.5, 250, 0.99)


def test_christoffersen_clustered_exceptions_have_larger_independence_statistic():
    spread = np.array([1, 0, 0, 0, 1, 0, 0, 0] * 30)
    clustered = np.array([1, 1, 1, 0, 0, 0, 0, 0] * 30)
    assert (
        christoffersen_independence(clustered)["lr_statistic"]
        > christoffersen_independence(spread)["lr_statistic"]
    )


@pytest.mark.parametrize("sequence", [[0.8, 1.7, 0], [0, 1, 2], ["0", "1", "0"]])
def test_christoffersen_rejects_values_before_integer_coercion(sequence):
    with pytest.raises(ValueError, match="0/1"):
        christoffersen_independence(sequence)


@pytest.mark.parametrize("sequence", [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1]])
def test_christoffersen_returns_not_estimable_when_transition_row_is_missing(sequence):
    result = christoffersen_independence(sequence)
    assert result["status"] == "not_estimable"
    assert result["lr_statistic"] is None
    assert result["p_value"] is None


def test_pla_detects_perfect_match():
    values = np.linspace(-5, 5, 250)
    result = pla_test(values, values, 250)
    assert result["zone"] == "green"
    assert result["spearman"] > 0.999


def test_pla_requires_250_observations():
    assert pla_test(np.arange(100), np.arange(100))["zone"] == "insufficient_sample"


def test_pla_does_not_invent_correlation_for_constant_series():
    result = pla_test(np.ones(250), np.arange(250))
    assert result["zone"] == "not_classifiable"
    assert result["status"] == "undefined_spearman"
    assert result["spearman"] is None


def test_backtest_is_explicitly_hpl_only_and_preserves_original_indices():
    hpl = np.zeros(25)
    hpl[4] = -5.0
    var = np.ones(25)
    var[0] = np.nan
    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    result = backtest_var(hpl, var, 0.99, dates)
    assert result["status"] == "computed_hpl_only"
    assert result["regulatory_status"] == "not_assessed_missing_APL"
    # Missing VaR is a counted outlier, not a dropped observation.
    assert result["exceptions"] == 2
    assert result["unmatched_observations_counted_as_exceptions"] == 1
    assert result["exception_indices"] == [0, 4]
    assert result["exception_dates"] == [str(dates[0]), str(dates[4])]
    assert result["desk_threshold_status"] == "not_classified_insufficient_history"
    assert "capital_multiplier" not in result


def test_backtest_rejects_misaligned_dates():
    with pytest.raises(ValueError, match="align"):
        backtest_var(np.zeros(25), np.ones(25), observation_dates=["2026-01-01"])


def test_rfet_public_closes_do_not_count():
    ledger = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=120, freq="3D"),
            "factor_id": "SPX",
            "evidence_type": "public_close",
            "verified": True,
        }
    )
    result = rfet_diagnostic(ledger)
    assert result["factors"]["SPX"]["classification"] == "not_evidenced"
    assert result["regulatory_status"] == "not_assessed"


def test_rfet_does_not_treat_string_false_as_verified():
    ledger = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=120, freq="D"),
            "factor_id": "SPX",
            "evidence_type": "transaction",
            "verified": "False",
        }
    )
    result = rfet_diagnostic(ledger)
    assert result["factors"]["SPX"]["eligible_candidate_real_price_days"] == 0


def test_rfet_count_threshold_is_only_a_review_candidate_not_a_pass():
    ledger = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=120, freq="D"),
            "factor_id": "SPX",
            "evidence_type": "transaction",
            "verified": True,
        }
    )
    result = rfet_diagnostic(ledger)
    factor = result["factors"]["SPX"]
    assert factor["numerical_count_screen_met"]
    assert factor["classification"] == "candidate_for_independent_RFET_review"
    assert "modellable" not in factor["classification"]
    assert result["regulatory_status"] == "not_assessed"


def test_desk_backtest_uses_mar32_19_thresholds_not_bankwide_green_zone():
    hpl = np.zeros(250)
    hpl[:13] = -2.0
    var = np.ones(250)
    result_99 = backtest_var(hpl, var, 0.99)
    assert result_99["desk_threshold_exceeded_on_hpl_alone"] is True
    assert result_99["desk_threshold_limit"] == 12
    assert result_99["desk_threshold_exceptions"] == 13
    assert result_99["regulatory_status"] == "hpl_only_exceeds_numeric_desk_limit_apl_unavailable"
    assert "zone" not in result_99
    assert "capital_multiplier" not in result_99

    hpl_975 = np.zeros(250)
    hpl_975[:30] = -2.0
    result_975 = backtest_var(hpl_975, var, 0.975)
    assert result_975["desk_threshold_exceeded_on_hpl_alone"] is False
    assert result_975["desk_threshold_limit"] == 30
    assert result_975["desk_threshold_status"] == "below_hpl_limit_apl_unknown"


def test_backtest_counts_missing_hpl_as_outlier():
    hpl = np.zeros(25)
    hpl[3] = np.nan
    result = backtest_var(hpl, np.ones(25))
    assert result["missing_hpl_observations"] == 1
    assert result["exceptions"] == 1
    assert result["exception_indices"] == [3]


def test_backtest_rejects_unordered_or_duplicate_dates():
    with pytest.raises(ValueError, match="unique and increasing"):
        backtest_var(
            np.zeros(25),
            np.ones(25),
            observation_dates=pd.date_range("2026-01-01", periods=25, freq="B")[::-1],
        )
    dates = list(pd.date_range("2026-01-01", periods=25, freq="B"))
    dates[-1] = dates[-2]
    with pytest.raises(ValueError, match="unique and increasing"):
        backtest_var(np.zeros(25), np.ones(25), observation_dates=dates)


def test_pla_does_not_silently_drop_missing_rows():
    hpl = np.linspace(-5, 5, 250)
    rtpl = hpl.copy()
    hpl[7] = np.nan
    result = pla_test(hpl, rtpl)
    assert result["status"] == "incomplete_data"
    assert result["zone"] == "not_classifiable"
    assert result["missing_hpl_observations"] == 1
    assert result["spearman"] is None
