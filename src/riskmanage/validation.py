from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, ks_2samp, spearmanr


def _contains_boolean(values) -> bool:
    """Detect booleans before NumPy can promote them to integer/float dtype."""
    objects = np.asarray(values, dtype=object)
    return any(isinstance(value, (bool, np.bool_)) for value in objects.flat)


def _probability(confidence: float) -> float:
    """Validate a probability without accepting strings or boolean coercion."""
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(
        confidence, (int, float, np.integer, np.floating)
    ):
        raise ValueError("confidence must be a finite numeric value in (0.5, 1)")
    value = float(confidence)
    if not np.isfinite(value) or not 0.5 < value < 1.0:
        raise ValueError("confidence must be a finite numeric value in (0.5, 1)")
    return value


def _one_dimensional_numeric(values, name: str) -> np.ndarray:
    if _contains_boolean(values):
        raise ValueError(
            f"{name} must be a one-dimensional numeric array; booleans are not measurements"
        )
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} must be a one-dimensional numeric array; booleans and coercion are not allowed"
        )
    return raw.astype(float, copy=False)


def _binary_sequence(values) -> np.ndarray:
    if _contains_boolean(values):
        raise ValueError("exceptions must contain numeric 0/1 values; booleans are not allowed")
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < 3:
        raise ValueError("exceptions must be a 0/1 vector with at least 3 observations")
    if raw.dtype.kind not in "iuf":
        raise ValueError(
            "exceptions must contain numeric 0/1 values; booleans and coercion are not allowed"
        )
    numeric = raw.astype(float, copy=False)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("exceptions must contain only exact 0/1 values")
    return numeric.astype(np.int8)


def kupiec_pof(exceptions: int, observations: int, confidence: float = 0.99):
    """Kupiec unconditional-coverage likelihood-ratio test.

    This is an asymptotic diagnostic, not a forecast-quality or regulatory
    assessment by itself.
    """
    try:
        x = operator.index(exceptions)
        n = operator.index(observations)
    except TypeError as exc:
        raise ValueError("exceptions and observations must be integers") from exc
    if isinstance(exceptions, (bool, np.bool_)) or isinstance(observations, (bool, np.bool_)):
        raise ValueError("exceptions and observations must be integers, not booleans")
    c = _probability(confidence)
    if n <= 0 or x < 0 or x > n:
        raise ValueError("invalid Kupiec inputs")

    p0 = 1.0 - c
    phat = x / n
    ll_null = float(xlogy(x, p0) + xlogy(n - x, 1.0 - p0))
    ll_alt = float(xlogy(x, phat) + xlogy(n - x, 1.0 - phat))
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return {
        "lr_statistic": float(lr),
        "p_value": float(chi2.sf(lr, 1)),
        "exceptions": int(x),
        "observations": int(n),
        "expected_exceptions": float(n * p0),
        "test": "kupiec_unconditional_coverage",
        "reference_distribution": "chi_square_1_asymptotic",
    }


def _christoffersen_from_pairs(previous, following):
    """Estimate first-order independence from explicitly valid adjacent pairs."""
    previous = np.asarray(previous, dtype=np.int8)
    following = np.asarray(following, dtype=np.int8)
    n00 = int(np.sum((previous == 0) & (following == 0)))
    n01 = int(np.sum((previous == 0) & (following == 1)))
    n10 = int(np.sum((previous == 1) & (following == 0)))
    n11 = int(np.sum((previous == 1) & (following == 1)))
    den0, den1 = n00 + n01, n10 + n11
    transitions = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    if len(previous) < 2 or den0 == 0 or den1 == 0:
        return {
            "status": "not_estimable",
            "reason": "at least two valid transitions and both previous-state rows are required",
            "lr_statistic": None,
            "p_value": None,
            "transitions": transitions,
            "valid_transitions": int(len(previous)),
            "test": "christoffersen_first_order_independence",
        }

    total = den0 + den1
    p = (n01 + n11) / total
    p01, p11 = n01 / den0, n11 / den1
    ll_null = float(xlogy(n01 + n11, p) + xlogy(n00 + n10, 1.0 - p))
    ll_alt = float(
        xlogy(n01, p01)
        + xlogy(n00, 1.0 - p01)
        + xlogy(n11, p11)
        + xlogy(n10, 1.0 - p11)
    )
    lr = max(0.0, 2.0 * (ll_alt - ll_null))
    return {
        "status": "computed",
        "lr_statistic": float(lr),
        "p_value": float(chi2.sf(lr, 1)),
        "transitions": transitions,
        "valid_transitions": int(len(previous)),
        "test": "christoffersen_first_order_independence",
        "reference_distribution": "chi_square_1_asymptotic",
    }


def christoffersen_independence(exception_sequence):
    """Christoffersen first-order Markov independence diagnostic."""
    x = _binary_sequence(exception_sequence)
    return _christoffersen_from_pairs(x[:-1], x[1:])


def backtest_var(
    hpl,
    var_forecast,
    confidence: float = 0.99,
    observation_dates: Sequence | None = None,
):
    """HPL-only daily VaR backtest with desk-level MAR32.19 thresholds.

    MAR32.18 counts a missing HPL or VaR as an outlier. This function therefore
    never drops unmatched dates. It cannot establish desk eligibility because
    Actual P&L (APL) is absent; a count above the HPL limit is sufficient to
    fail the numerical desk threshold, but a count below it is not an approval.
    """
    h = _one_dimensional_numeric(hpl, "hpl")
    v = _one_dimensional_numeric(var_forecast, "var_forecast")
    c = _probability(confidence)
    if h.size != v.size:
        raise ValueError("HPL and VaR arrays must have the same length")
    n = int(h.size)
    if n < 20:
        raise ValueError("at least 20 dated HPL / VaR observations are required")

    dates = None
    if observation_dates is not None:
        if len(observation_dates) != n:
            raise ValueError("observation_dates must align with HPL and VaR arrays")
        try:
            dates = pd.DatetimeIndex(pd.to_datetime(observation_dates, errors="raise"))
        except (TypeError, ValueError) as exc:
            raise ValueError("observation_dates must contain valid dates") from exc
        if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("observation_dates must be valid, unique and increasing")

    finite_hpl = np.isfinite(h)
    finite_var = np.isfinite(v)
    matched = finite_hpl & finite_var
    missing_hpl = ~finite_hpl
    missing_var = ~finite_var
    losses = -h
    exceptions = (~matched) | (matched & (losses > v))
    exception_sequence = exceptions.astype(np.int8)
    count = int(exception_sequence.sum())
    matched_count = int(matched.sum())
    matched_exception_count = int((matched & (losses > v)).sum())

    # Inferential tests use only observed, matched HPL/VaR pairs. Missing data
    # remain exceptions for the operational desk count above, but are not
    # fabricated as realised tail events in Kupiec/Christoffersen likelihoods.
    if matched_count >= 20:
        kupiec = kupiec_pof(matched_exception_count, matched_count, c)
    else:
        kupiec = {
            "status": "not_estimable",
            "reason": "fewer than 20 finite, matched HPL/VaR observations",
            "exceptions": matched_exception_count,
            "observations": matched_count,
            "test": "kupiec_unconditional_coverage",
        }
    adjacent_matched = matched[:-1] & matched[1:]
    valid_previous = exception_sequence[:-1][adjacent_matched]
    valid_following = exception_sequence[1:][adjacent_matched]
    christoffersen = _christoffersen_from_pairs(valid_previous, valid_following)

    threshold_count = None
    threshold_limit = None
    threshold_exceeded = None
    if n < 250:
        threshold_status = "not_classified_insufficient_history"
    elif abs(c - 0.99) < 1e-10:
        threshold_count = int(exception_sequence[-250:].sum())
        threshold_limit = 12
        threshold_exceeded = threshold_count > threshold_limit
        threshold_status = (
            "hpl_only_threshold_exceeded"
            if threshold_exceeded
            else "below_hpl_limit_apl_unknown"
        )
    elif abs(c - 0.975) < 1e-10:
        threshold_count = int(exception_sequence[-250:].sum())
        threshold_limit = 30
        threshold_exceeded = threshold_count > threshold_limit
        threshold_status = (
            "hpl_only_threshold_exceeded"
            if threshold_exceeded
            else "below_hpl_limit_apl_unknown"
        )
    else:
        threshold_status = "not_classified_confidence_not_in_mar32_19"

    result = {
        "status": "computed_hpl_only",
        "scope": "HPL-only research diagnostic; Actual P&L (APL) is not available",
        "regulatory_status": (
            "hpl_only_exceeds_numeric_desk_limit_apl_unavailable"
            if threshold_exceeded
            else "not_assessed_missing_APL"
        ),
        "observations": n,
        "matched_observations": int(matched.sum()),
        "missing_hpl_observations": int(missing_hpl.sum()),
        "missing_var_observations": int(missing_var.sum()),
        "unmatched_observations_counted_as_exceptions": int((~matched).sum()),
        "exceptions": count,
        "exception_rate": float(count / n),
        "confidence": c,
        "desk_threshold_status": threshold_status,
        "desk_threshold_basis": "HPL-only numerical screen; MAR32.19; not an eligibility approval",
        "desk_threshold_observations": 250 if n >= 250 else None,
        "desk_threshold_exceptions": threshold_count,
        "desk_threshold_limit": threshold_limit,
        "desk_threshold_exceeded_on_hpl_alone": threshold_exceeded,
        "kupiec": kupiec,
        "kupiec_inference_population": "finite_matched_HPL_VaR_pairs_only",
        "kupiec_matched_exceptions": matched_exception_count,
        "christoffersen": christoffersen,
        "christoffersen_inference_population": "adjacent_finite_matched_HPL_VaR_pairs_only",
        "exception_indices": np.flatnonzero(exceptions).tolist(),
    }
    if dates is not None:
        result["exception_dates"] = [
            str(value) for value in dates[np.flatnonzero(exceptions)].tolist()
        ]
    return result

def pla_test(hpl, rtpl, window: int = 250):
    """HPL-versus-RTPL distribution diagnostic using the latest 250 pairs.

    Missing values are not silently removed: the PLA window is unclassifiable
    unless all 250 HPL/RTPL pairs are present and finite.
    """
    h = _one_dimensional_numeric(hpl, "hpl")
    r = _one_dimensional_numeric(rtpl, "rtpl")
    if h.size != r.size:
        raise ValueError("HPL and RTPL must be aligned vectors of equal length")
    if not isinstance(window, (int, np.integer)) or isinstance(window, (bool, np.bool_)):
        raise ValueError("window must be an integer")
    if window != 250:
        raise ValueError("this FRTB-inspired PLA diagnostic uses a 250-observation window")
    if h.size < window:
        return {
            "status": "insufficient_sample",
            "zone": "insufficient_sample",
            "observations": int(h.size),
            "required_observations": int(window),
            "spearman": None,
            "ks_distance": None,
        }

    h, r = h[-window:], r[-window:]
    finite_h, finite_r = np.isfinite(h), np.isfinite(r)
    if not finite_h.all() or not finite_r.all():
        return {
            "status": "incomplete_data",
            "zone": "not_classifiable",
            "observations": int(window),
            "missing_hpl_observations": int((~finite_h).sum()),
            "missing_rtpl_observations": int((~finite_r).sum()),
            "spearman": None,
            "ks_distance": None,
            "reason": "PLA requires 250 aligned finite HPL/RTPL pairs; missing rows are not dropped",
        }

    ks = float(ks_2samp(h, r, method="auto").statistic)
    if np.ptp(h) == 0.0 or np.ptp(r) == 0.0:
        return {
            "status": "undefined_spearman",
            "zone": "not_classifiable",
            "observations": int(window),
            "spearman": None,
            "ks_distance": ks,
            "reason": "Spearman correlation is undefined for a constant/degenerate series",
        }

    rho = float(spearmanr(h, r).statistic)
    if not np.isfinite(rho):
        return {
            "status": "undefined_spearman",
            "zone": "not_classifiable",
            "observations": int(window),
            "spearman": None,
            "ks_distance": ks,
            "reason": "Spearman correlation is undefined for a constant/degenerate series",
        }

    if rho > 0.80 and ks < 0.09:
        zone = "green"
    elif rho < 0.70 or ks > 0.12:
        zone = "red"
    else:
        zone = "amber"
    return {
        "status": "computed_research_diagnostic",
        "scope": "HPL-versus-RTPL research diagnostic; not a regulatory desk PLA decision",
        "zone": zone,
        "observations": int(window),
        "spearman": rho,
        "ks_distance": ks,
        "thresholds": {
            "green_spearman_gt": 0.80,
            "green_ks_lt": 0.09,
            "red_spearman_lt": 0.70,
            "red_ks_gt": 0.12,
        },
    }

def _strict_true(values: pd.Series) -> pd.Series:
    """Only actual boolean True values count as verified; strings are rejected."""
    return values.map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value))


def rfet_diagnostic(
    evidence_ledger: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
    calibration_period_start: str | pd.Timestamp,
    calibration_period_end: str | pd.Timestamp,
):
    """Compute a Basel MAR31.13-aligned RFET count screen, not a modellability pass.

    The 24-observation rule is counted over the explicit current ES calibration
    period. The 4-per-90-day and 100-observation rules use the previous
    12 calendar months ending on the explicit RFET assessment date.
    """
    required = {"date", "factor_id", "evidence_type", "verified"}
    if not required.issubset(evidence_ledger.columns):
        missing = sorted(required - set(evidence_ledger.columns))
        return {
            "status": "not_evidenced",
            "reason": f"missing fields: {missing}",
            "factors": {},
            "regulatory_status": "not_assessed",
        }

    def parse_explicit_date(value, name: str) -> pd.Timestamp:
        if (
            value is None
            or isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating))
            or not isinstance(value, (str, pd.Timestamp, np.datetime64))
            and not hasattr(value, "isoformat")
        ):
            if name == "as_of_date":
                raise ValueError("as_of_date must be an explicit assessment date or timestamp")
            raise ValueError(f"{name} must be an explicit date or timestamp")
        try:
            parsed = pd.Timestamp(pd.to_datetime(value, errors="raise", utc=True))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a valid explicit date") from exc
        if pd.isna(parsed):
            raise ValueError(f"{name} must be a valid explicit date")
        return parsed.normalize()

    assessment_date = parse_explicit_date(as_of_date, "as_of_date")
    calibration_start = parse_explicit_date(
        calibration_period_start, "calibration_period_start"
    )
    calibration_end = parse_explicit_date(
        calibration_period_end, "calibration_period_end"
    )

    if calibration_start > calibration_end:
        raise ValueError("calibration period start must not exceed its end")
    calibration_period_days = int((calibration_end - calibration_start).days + 1)
    if calibration_period_days not in (365, 366):
        raise ValueError(
            "calibration period must span one calendar year (365 or 366 days)"
        )
    earliest_allowed_end = assessment_date - pd.DateOffset(months=1)
    if calibration_end > assessment_date or calibration_end < earliest_allowed_end:
        raise ValueError(
            "calibration period end must be on or before, and no more than one month "
            "before, the RFET assessment date"
        )

    rfet_window_start = (
        assessment_date - pd.DateOffset(months=12) + pd.Timedelta(days=1)
    )

    data = evidence_ledger.copy()
    input_rows = int(len(data))
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    invalid_date = data["date"].isna()
    future_date = data["date"].dt.normalize() > assessment_date
    missing_factor = (
        data["factor_id"].isna()
        | data["factor_id"].astype("string").str.strip().eq("").fillna(True)
    )
    invalid_date_rows = int(invalid_date.sum())
    missing_factor_id_rows = int(missing_factor.sum())
    future_date_rows = int(future_date.sum())
    excluded_mask = invalid_date | missing_factor | future_date
    excluded_rows = int(excluded_mask.sum())
    data = data.loc[~excluded_mask].copy()

    common_metadata = {
        "input_rows": input_rows,
        "invalid_date_rows": invalid_date_rows,
        "missing_factor_id_rows": missing_factor_id_rows,
        "excluded_rows": excluded_rows,
        "future_date_rows": future_date_rows,
        "as_of_date": assessment_date.date().isoformat(),
        "rfet_12m_window_start": rfet_window_start.date().isoformat(),
        "calibration_period_start": calibration_start.date().isoformat(),
        "calibration_period_end": calibration_end.date().isoformat(),
        "calibration_period_days": calibration_period_days,
        "calibration_period_lag_days": int((assessment_date - calibration_end).days),
    }

    if data.empty:
        return {
            "status": "not_evidenced",
            "reason": "no rows with valid dates and non-empty factor identifiers",
            "factors": {},
            "regulatory_status": "not_assessed",
            **common_metadata,
        }

    direct_types = {"transaction", "arm_length_transaction", "committed_quote"}
    third_party_type = "qualifying_third_party_quote"
    data["_verified_bool"] = _strict_true(data["verified"])
    data["_third_party_complete"] = False
    vendor_fields = {
        "vendor_audit_verified",
        "risk_factor_mapping_verified",
        "vendor_dates_and_counts_verified",
    }
    if vendor_fields.issubset(data.columns):
        data["_third_party_complete"] = (
            _strict_true(data["vendor_audit_verified"])
            & _strict_true(data["risk_factor_mapping_verified"])
            & _strict_true(data["vendor_dates_and_counts_verified"])
        )
    data["_eligible_evidence"] = (
        data["_verified_bool"]
        & (
            data["evidence_type"].isin(direct_types)
            | (
                data["evidence_type"].eq(third_party_type)
                & data["_third_party_complete"]
            )
        )
    )

    results = {}
    for factor, group in data.groupby("factor_id"):
        observation_days = group["date"].dt.normalize()
        eligible_12m = group[
            (observation_days >= rfet_window_start)
            & (observation_days <= assessment_date)
            & group["_eligible_evidence"]
        ]
        rfet_dates = (
            eligible_12m["date"].dt.normalize().drop_duplicates().sort_values()
        )
        count_12m = int(len(rfet_dates))

        eligible_calibration = group[
            (observation_days >= calibration_start)
            & (observation_days <= calibration_end)
            & group["_eligible_evidence"]
        ]
        calibration_dates = (
            eligible_calibration["date"]
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
        )
        calibration_count = int(len(calibration_dates))

        final_window_start = assessment_date - pd.Timedelta(days=89)
        starts = pd.date_range(rfet_window_start, final_window_start, freq="D")
        min_90 = (
            min(
                int(
                    (
                        (rfet_dates >= window_start)
                        & (rfet_dates < window_start + pd.Timedelta(days=90))
                    ).sum()
                )
                for window_start in starts
            )
            if len(starts)
            else count_12m
        )

        rule_100_count = count_12m >= 100
        rule_24_and_90 = calibration_count >= 24 and min_90 >= 4
        screen_met = bool(rule_100_count or rule_24_and_90)
        latest_eligible_date = rfet_dates.max() if count_12m else None
        latest_eligible_age_days = (
            int((assessment_date - latest_eligible_date).days)
            if latest_eligible_date is not None
            else None
        )
        results[str(factor)] = {
            "eligible_candidate_real_price_days": count_12m,
            "calibration_period_real_price_days": calibration_count,
            "latest_eligible_observation_date": (
                latest_eligible_date.date().isoformat()
                if latest_eligible_date is not None
                else None
            ),
            "latest_eligible_observation_age_days": latest_eligible_age_days,
            "minimum_candidate_days_in_any_90d_window": min_90,
            "count_screen_100_days_met": bool(rule_100_count),
            "count_screen_24_days_and_90d_distribution_met": bool(rule_24_and_90),
            "numerical_count_screen_met": screen_met,
            "classification": (
                "candidate_for_independent_RFET_review"
                if screen_met
                else ("below_count_screen" if count_12m or calibration_count else "not_evidenced")
            ),
        }

    return {
        "status": "independent_review_required",
        "regulatory_status": "not_assessed",
        **common_metadata,
        "eligible_evidence_types": sorted(direct_types | {third_party_type}),
        "factors": results,
        "note": (
            "This is only a numerical evidence-count screen. It does not establish "
            "valid real-price observations, vendor audit controls, representative "
            "factor mapping, source eligibility or Basel RFET modellability. The 24-day "
            "count uses the explicit ES calibration period; the 90-day distribution "
            "and 100-day count use the previous 12 calendar months at the assessment date."
        ),
    }
