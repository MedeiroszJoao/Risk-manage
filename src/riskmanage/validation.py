from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, ks_2samp, spearmanr


def _probability(confidence: float) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a finite number in (0.5, 1)") from exc
    if not np.isfinite(value) or not 0.5 < value < 1.0:
        raise ValueError("confidence must be a finite number in (0.5, 1)")
    return value


def _one_dimensional_numeric(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    return raw.astype(float, copy=False)


def _binary_sequence(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < 3:
        raise ValueError("exceptions must be a 0/1 vector with at least 3 observations")
    if raw.dtype.kind not in "biuf":
        raise ValueError("exceptions must contain numeric 0/1 values; coercion is not allowed")
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


def christoffersen_independence(exception_sequence):
    """Christoffersen first-order Markov independence diagnostic.

    If either previous-state transition row is absent, the test is not
    estimable from the sample and no likelihood-ratio statistic is returned.
    """
    x = _binary_sequence(exception_sequence)
    previous, following = x[:-1], x[1:]
    n00 = int(np.sum((previous == 0) & (following == 0)))
    n01 = int(np.sum((previous == 0) & (following == 1)))
    n10 = int(np.sum((previous == 1) & (following == 0)))
    n11 = int(np.sum((previous == 1) & (following == 1)))
    den0, den1 = n00 + n01, n10 + n11
    transitions = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}

    if den0 == 0 or den1 == 0:
        return {
            "status": "not_estimable",
            "reason": "both previous-state rows (0 and 1) must be observed",
            "lr_statistic": None,
            "p_value": None,
            "transitions": transitions,
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
        "test": "christoffersen_first_order_independence",
        "reference_distribution": "chi_square_1_asymptotic",
    }


def _backtest_zone(exceptions: int, observations: int, confidence: float):
    if observations != 250:
        return "not_classified", None
    if abs(confidence - 0.99) < 1e-10:
        if exceptions <= 4:
            return "green", 1.50
        multipliers = {5: 1.70, 6: 1.76, 7: 1.83, 8: 1.88, 9: 1.92}
        if exceptions in multipliers:
            return "amber", multipliers[exceptions]
        return "red", 2.00
    if abs(confidence - 0.975) < 1e-10 and exceptions > 30:
        return "ineligible", None
    return "diagnostic_only", None


def backtest_var(
    hpl,
    var_forecast,
    confidence: float = 0.99,
    observation_dates: Sequence | None = None,
):
    """Backtest HPL losses against aligned one-day VaR forecasts.

    This function has no Actual P&L (APL) input. It is an HPL-only research
    diagnostic, not the Basel regulatory backtest.
    """
    h = _one_dimensional_numeric(hpl, "hpl")
    v = _one_dimensional_numeric(var_forecast, "var_forecast")
    c = _probability(confidence)
    if h.size != v.size:
        raise ValueError("HPL and VaR arrays must have the same length")
    if observation_dates is not None and len(observation_dates) != h.size:
        raise ValueError("observation_dates must align with HPL and VaR arrays")

    valid = np.isfinite(h) & np.isfinite(v)
    if int(valid.sum()) < 20:
        raise ValueError("at least 20 matched HPL / VaR observations are required")
    original_indices = np.flatnonzero(valid)
    losses = -h[valid]
    forecasts = v[valid]
    exceptions = (losses > forecasts).astype(np.int8)
    count = int(exceptions.sum())
    n = int(exceptions.size)
    zone, multiplier = _backtest_zone(count, n, c)

    result = {
        "status": "computed_hpl_only",
        "scope": "HPL-only research diagnostic; Actual P&L (APL) is not available",
        "regulatory_status": "not_assessed_missing_APL",
        "observations": n,
        "excluded_unmatched_observations": int(h.size - n),
        "exceptions": count,
        "exception_rate": float(count / n),
        "confidence": c,
        "zone": zone,
        "zone_scope": "250-observation 99% numerical diagnostic; not a regulatory multiplier",
        "capital_multiplier": multiplier,
        "kupiec": kupiec_pof(count, n, c),
        "christoffersen": christoffersen_independence(exceptions),
        "exception_indices": original_indices[exceptions.astype(bool)].tolist(),
    }
    if observation_dates is not None:
        dates = np.asarray(observation_dates, dtype=object)[original_indices[exceptions.astype(bool)]]
        result["exception_dates"] = [str(value) for value in dates.tolist()]
    return result


def pla_test(hpl, rtpl, window: int = 250):
    """HPL-versus-RTPL distribution diagnostic using a 250-pair window.

    If Spearman correlation is undefined for a constant/degenerate series,
    no green/amber/red zone is assigned and zero is not substituted.
    """
    h = _one_dimensional_numeric(hpl, "hpl")
    r = _one_dimensional_numeric(rtpl, "rtpl")
    if h.size != r.size:
        raise ValueError("HPL and RTPL must be aligned vectors of equal length")
    if not isinstance(window, (int, np.integer)) or isinstance(window, (bool, np.bool_)):
        raise ValueError("window must be an integer")
    if window != 250:
        raise ValueError("this FRTB-inspired PLA diagnostic uses a 250-observation window")

    valid = np.isfinite(h) & np.isfinite(r)
    h, r = h[valid], r[valid]
    matched = int(h.size)
    if matched < window:
        return {
            "status": "insufficient_sample",
            "zone": "insufficient_sample",
            "observations": matched,
            "required_observations": int(window),
            "spearman": None,
            "ks_distance": None,
        }

    h, r = h[-window:], r[-window:]
    rho = float(spearmanr(h, r).statistic)
    ks = float(ks_2samp(h, r, method="auto").statistic)
    if not np.isfinite(rho):
        return {
            "status": "undefined_spearman",
            "zone": "not_classifiable",
            "observations": int(window),
            "matched_observations_before_window": matched,
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
        "status": "computed",
        "scope": "HPL-versus-RTPL research diagnostic; not a bank regulatory PLA decision",
        "zone": zone,
        "observations": int(window),
        "matched_observations_before_window": matched,
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


def rfet_diagnostic(evidence_ledger: pd.DataFrame, lookback_days: int = 365):
    """Compute an RFET count screen, never a regulatory modellability pass.

    The ledger schema cannot prove valid real-price observations, vendor audit,
    identifier mapping, representativeness or complete source controls.
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
    if (
        not isinstance(lookback_days, (int, np.integer))
        or isinstance(lookback_days, (bool, np.bool_))
        or lookback_days < 1
    ):
        raise ValueError("lookback_days must be a positive integer")

    data = evidence_ledger.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    data = data.dropna(subset=["date", "factor_id"])
    if data.empty:
        return {
            "status": "not_evidenced",
            "reason": "no rows with valid dates and factor identifiers",
            "factors": {},
            "regulatory_status": "not_assessed",
        }

    latest = data["date"].max()
    cutoff = latest - pd.Timedelta(days=int(lookback_days) - 1)
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
        eligible = group[(group["date"] >= cutoff) & group["_eligible_evidence"]]
        dates = eligible["date"].dt.normalize().drop_duplicates().sort_values()
        count = int(len(dates))
        if count:
            final_window_start = latest.normalize() - pd.Timedelta(days=89)
            starts = pd.date_range(cutoff.normalize(), final_window_start, freq="D")
            min_90 = (
                min(
                    int(((dates >= start) & (dates < start + pd.Timedelta(days=90))).sum())
                    for start in starts
                )
                if len(starts)
                else count
            )
        else:
            min_90 = 0
        rule_100_count = count >= 100
        rule_24_and_90 = count >= 24 and min_90 >= 4
        screen_met = bool(rule_100_count or rule_24_and_90)
        results[str(factor)] = {
            "eligible_candidate_real_price_days": count,
            "minimum_candidate_days_in_any_90d_window": min_90,
            "count_screen_100_days_met": bool(rule_100_count),
            "count_screen_24_days_and_90d_distribution_met": bool(rule_24_and_90),
            "numerical_count_screen_met": screen_met,
            "classification": (
                "candidate_for_independent_RFET_review"
                if screen_met
                else ("below_count_screen" if count else "not_evidenced")
            ),
        }
    return {
        "status": "independent_review_required",
        "regulatory_status": "not_assessed",
        "lookback_days": int(lookback_days),
        "eligible_evidence_types": sorted(direct_types | {third_party_type}),
        "factors": results,
        "note": (
            "This is only a numerical evidence-count screen. It does not establish "
            "valid real-price observations, vendor audit controls, representative "
            "factor mapping, source eligibility or Basel RFET modellability."
        ),
    }
