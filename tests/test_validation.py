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


def _rfet_kwargs(as_of_date):
    assessment = pd.Timestamp(as_of_date).normalize()
    calibration_end = assessment
    calibration_start = (
        calibration_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    )
    return {
        "as_of_date": assessment,
        "calibration_period_start": calibration_start,
        "calibration_period_end": calibration_end,
    }


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
    result = rfet_diagnostic(ledger, **_rfet_kwargs("2025-12-26"))
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
    result = rfet_diagnostic(ledger, **_rfet_kwargs("2026-04-30"))
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
    result = rfet_diagnostic(ledger, **_rfet_kwargs("2026-04-30"))
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


def test_pla_constant_input_is_not_classified_without_scipy_warning():
    result = pla_test(np.ones(250), np.arange(250))
    assert result["status"] == "undefined_spearman"
    assert result["zone"] == "not_classifiable"
    assert result["spearman"] is None


@pytest.mark.parametrize("confidence", [True, False, "0.99", "0.975"])
def test_validation_probability_rejects_booleans_and_strings(confidence):
    from riskmanage.validation import _probability

    with pytest.raises(ValueError, match="numeric"):
        _probability(confidence)


@pytest.mark.parametrize("values", [[True, False, True], ["1", "2", "3"]])
def test_validation_numeric_vectors_reject_boolean_and_string_coercion(values):
    from riskmanage.validation import _one_dimensional_numeric

    with pytest.raises(ValueError, match="numeric array"):
        _one_dimensional_numeric(values, "sample")


def test_backtest_missing_values_are_not_fabricated_as_statistical_exceptions():
    hpl = np.zeros(250)
    var = np.ones(250)
    hpl[20] = np.nan
    var[80] = np.nan
    hpl[120] = -2.0

    result = backtest_var(hpl, var, 0.99)

    assert result["exceptions"] == 3
    assert result["matched_observations"] == 248
    assert result["kupiec"]["observations"] == 248
    assert result["kupiec"]["exceptions"] == 1
    assert result["kupiec_inference_population"] == "finite_matched_HPL_VaR_pairs_only"
    assert result["christoffersen"]["valid_transitions"] == 245
    assert result["christoffersen_inference_population"] == (
        "adjacent_finite_matched_HPL_VaR_pairs_only"
    )


def test_backtest_does_not_bridge_missing_observation_in_christoffersen_transitions():
    hpl = np.zeros(25)
    var = np.ones(25)
    hpl[10] = np.nan

    result = backtest_var(hpl, var)

    assert result["christoffersen"]["valid_transitions"] == 22


def test_christoffersen_rejects_boolean_exception_sequence():
    with pytest.raises(ValueError, match="booleans"):
        christoffersen_independence([False, True, False])


def test_rfet_reports_invalid_rows_instead_of_silently_dropping_them():
    ledger = pd.DataFrame(
        {
            "date": ["2026-01-01", "not-a-date", "2026-01-03", "2026-01-04"],
            "factor_id": ["SPX", "SPX", None, "   "],
            "evidence_type": ["transaction"] * 4,
            "verified": [True] * 4,
        }
    )

    result = rfet_diagnostic(ledger, **_rfet_kwargs("2026-01-04"))

    assert result["input_rows"] == 4
    assert result["invalid_date_rows"] == 1
    assert result["missing_factor_id_rows"] == 2
    assert result["excluded_rows"] == 3
    assert result["factors"]["SPX"]["eligible_candidate_real_price_days"] == 1


def test_rfet_empty_valid_population_still_reports_exclusions():
    ledger = pd.DataFrame(
        {
            "date": ["bad-date"],
            "factor_id": [None],
            "evidence_type": ["transaction"],
            "verified": [True],
        }
    )

    result = rfet_diagnostic(ledger, **_rfet_kwargs("2026-01-01"))

    assert result["status"] == "not_evidenced"
    assert result["input_rows"] == 1
    assert result["excluded_rows"] == 1


def test_validation_rejects_boolean_mixed_with_numeric_before_numpy_promotion():
    from riskmanage.validation import _one_dimensional_numeric

    with pytest.raises(ValueError, match="booleans"):
        _one_dimensional_numeric([True, 0.25, 0.5], "sample")
    with pytest.raises(ValueError, match="booleans"):
        christoffersen_independence([True, 0, 1])


def test_rfet_stale_historically_dense_ledger_fails_at_current_assessment_date():
    ledger = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=120, freq="D"),
            "factor_id": "SPX",
            "evidence_type": "transaction",
            "verified": True,
        }
    )

    result = rfet_diagnostic(ledger, **_rfet_kwargs("2026-01-01"))
    factor = result["factors"]["SPX"]

    assert result["as_of_date"] == "2026-01-01"
    assert factor["eligible_candidate_real_price_days"] == 0
    assert factor["classification"] == "not_evidenced"


def test_rfet_excludes_future_evidence_relative_to_assessment_date():
    ledger = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "factor_id": ["SPX", "SPX", "SPX"],
            "evidence_type": ["transaction"] * 3,
            "verified": [True] * 3,
        }
    )

    result = rfet_diagnostic(ledger, **_rfet_kwargs("2026-01-02"))

    assert result["future_date_rows"] == 1
    assert result["excluded_rows"] == 1
    assert result["factors"]["SPX"]["eligible_candidate_real_price_days"] == 2


def test_rfet_requires_explicit_assessment_date():
    with pytest.raises(TypeError):
        rfet_diagnostic(pd.DataFrame())


def test_rfet_assessment_date_includes_all_observations_on_that_calendar_day():
    ledger = pd.DataFrame(
        {
            "date": [
                "2026-04-30T09:00:00Z",
                "2026-04-30T16:00:00Z",
                "2026-05-01T09:00:00Z",
            ],
            "factor_id": ["SPX"] * 3,
            "evidence_type": ["transaction"] * 3,
            "verified": [True] * 3,
        }
    )

    result = rfet_diagnostic(ledger, **_rfet_kwargs("2026-04-30"))

    assert result["future_date_rows"] == 1
    assert result["factors"]["SPX"]["eligible_candidate_real_price_days"] == 1
    assert result["factors"]["SPX"]["latest_eligible_observation_date"] == "2026-04-30"


def test_rfet_rejects_explicit_none_assessment_date():
    with pytest.raises(ValueError, match="explicit assessment date"):
        rfet_diagnostic(
            pd.DataFrame(),
            as_of_date=None,
            calibration_period_start="2025-01-02",
            calibration_period_end="2026-01-01",
        )


@pytest.mark.parametrize("bad_date", [None, True, 20260430, 0.5])
def test_rfet_rejects_implicit_numeric_or_boolean_assessment_dates(bad_date):
    with pytest.raises(ValueError, match="as_of_date"):
        rfet_diagnostic(
            pd.DataFrame(),
            as_of_date=bad_date,
            calibration_period_start="2025-01-02",
            calibration_period_end="2026-01-01",
        )


def test_rfet_24_observation_count_uses_explicit_calibration_period():
    ledger = pd.DataFrame(
        {
            "date": pd.date_range("2025-04-01", periods=24, freq="D"),
            "factor_id": "SPX",
            "evidence_type": "transaction",
            "verified": True,
        }
    )
    result = rfet_diagnostic(
        ledger,
        as_of_date="2026-04-30",
        calibration_period_start="2025-04-01",
        calibration_period_end="2026-03-31",
    )
    factor = result["factors"]["SPX"]

    assert result["calibration_period_lag_days"] == 30
    assert factor["calibration_period_real_price_days"] == 24
    assert factor["eligible_candidate_real_price_days"] == 0
    assert factor["minimum_candidate_days_in_any_90d_window"] == 0
    assert not factor["count_screen_24_days_and_90d_distribution_met"]


@pytest.mark.parametrize(
    "calibration_start,calibration_end",
    [
        ("2025-05-02", "2026-04-30"),  # 364 inclusive days
        ("2025-04-01", "2026-04-30"),  # 395 inclusive days
    ],
)
def test_rfet_requires_one_calendar_year_calibration_period(
    calibration_start, calibration_end
):
    with pytest.raises(ValueError, match="365 or 366 days"):
        rfet_diagnostic(
            pd.DataFrame(),
            as_of_date="2026-04-30",
            calibration_period_start=calibration_start,
            calibration_period_end=calibration_end,
        )


def test_rfet_rejects_calibration_period_ending_more_than_one_month_before_assessment():
    with pytest.raises(ValueError, match="no more than one month"):
        rfet_diagnostic(
            pd.DataFrame(),
            as_of_date="2026-04-30",
            calibration_period_start="2025-03-01",
            calibration_period_end="2026-03-29",
        )
