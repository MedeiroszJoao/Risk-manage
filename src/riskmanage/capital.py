from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import _ten_day_losses, historical_var_es


MIN_CURRENT_OBSERVATIONS = 250
REDUCED_SET_VALIDATION_DAYS = 60
BASEL_IMCC_RHO = 0.5
BASEL_NMRF_RHO = 0.6


def _strict_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _strict_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric; strings and booleans are not accepted")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_observation_dates(observation_dates, expected_length: int) -> pd.DatetimeIndex:
    """Require a one-to-one, strictly increasing daily observation calendar."""
    if observation_dates is None:
        raise ValueError(
            "observation_dates are required for chronological out-of-sample validation"
        )
    try:
        dates = pd.DatetimeIndex(
            pd.to_datetime(observation_dates, errors="raise", utc=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_dates must contain valid timestamps") from exc
    if len(dates) != expected_length:
        raise ValueError(
            f"observation_dates length {len(dates)} does not match "
            f"factor history length {expected_length}"
        )
    if dates.hasnans:
        raise ValueError("observation_dates cannot contain missing timestamps")
    if dates.normalize().has_duplicates:
        raise ValueError("observation_dates must be unique by UTC calendar date")
    if not dates.is_monotonic_increasing:
        raise ValueError("observation_dates must be strictly increasing")
    return dates


def _normalize_risk_class(value: str) -> str:
    aliases = {
        "rates": "interest_rate",
        "interest_rate": "interest_rate",
        "interest_rates": "interest_rate",
        "equity": "equity",
        "fx": "foreign_exchange",
        "foreign_exchange": "foreign_exchange",
        "commodities": "commodity",
        "commodity": "commodity",
        "credit": "credit_spread",
        "credit_spread": "credit_spread",
    }
    if not isinstance(value, str):
        raise ValueError("risk-class mapping values must be strings")
    key = value.strip().lower()
    if key not in aliases:
        raise ValueError(f"unsupported regulatory risk class: {value!r}")
    return aliases[key]


def _align_factor_losses(factor_losses):
    if not factor_losses:
        raise ValueError("factor losses cannot be empty")
    values = {}
    for name, series in factor_losses.items():
        objects = np.asarray(series, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) for value in objects.flat):
            raise ValueError(
                f"factor losses for {name} must be a numeric vector without boolean observations"
            )
        raw = np.asarray(series)
        if raw.ndim != 1 or raw.dtype.kind not in "iuf":
            raise ValueError(
                f"factor losses for {name} must be a 1D numeric vector; "
                "boolean, string and object coercion is not allowed"
            )
        values[name] = raw.astype(float, copy=False)
    if any(series.size == 0 or not np.isfinite(series).all() for series in values.values()):
        raise ValueError("factor loss arrays must be non-empty and finite")
    if len({len(series) for series in values.values()}) != 1:
        raise ValueError("factor arrays must align")
    return values


def _confidence_level(confidence):
    value = _strict_float(confidence, "confidence")
    if not 0.5 < value < 1.0:
        raise ValueError("confidence must be finite and in (0.5, 1)")
    return value


def _portfolio_es(mapping, names, confidence):
    if not names:
        raise ValueError("at least one risk factor is required")
    missing = set(names) - set(mapping)
    if missing:
        raise ValueError(f"unknown risk factor(s): {sorted(missing)}")
    daily = np.sum(np.vstack([mapping[name] for name in names]), axis=0)
    return float(historical_var_es(_ten_day_losses(daily), confidence)["expected_shortfall"])


def _rolling_reduced_coverage(
    factors,
    selected,
    confidence=0.975,
    estimation_window=MIN_CURRENT_OBSERVATIONS,
    validation_days=REDUCED_SET_VALIDATION_DAYS,
    observation_dates=None,
):
    """Evaluate a frozen reduced set on a forward, rolling-origin holdout.

    The final validation_days observations are never used to choose selected.
    At each holdout endpoint, current full/reduced ES is recomputed on the
    trailing estimation_window observations. This is an independent research
    diagnostic, not a supervisory approval test.
    """
    n = len(next(iter(factors.values())))
    dates = pd.DatetimeIndex(observation_dates)
    required = estimation_window + validation_days
    if n < required:
        return {
            "status": "insufficient_history",
            "observations": 0,
            "required_observations": required,
            "mean_ratio": None,
            "minimum_ratio": None,
            "maximum_ratio": None,
            "ratios": [],
            "out_of_sample": False,
            "selection_overlap": None,
        }
    if not selected or not set(selected).issubset(factors):
        raise ValueError("selected must contain known risk factors")

    validation_start = n - validation_days
    selection_start = validation_start - estimation_window
    selection_end = validation_start - 1
    ratios = []
    undefined_endpoints = []
    first_window_start = None

    for end in range(validation_start, n):
        start = end - estimation_window + 1
        if first_window_start is None:
            first_window_start = start
        window = {name: series[start : end + 1] for name, series in factors.items()}
        full = _portfolio_es(window, list(window), confidence)
        reduced = _portfolio_es(window, selected, confidence)
        if not np.isfinite(full) or full <= 0.0:
            ratios.append(None)
            undefined_endpoints.append(int(end))
        else:
            ratios.append(float(reduced / full))

    if undefined_endpoints:
        return {
            "status": "unclassifiable_nonpositive_full_es",
            "observations": len(ratios),
            "expected_observations": int(validation_days),
            "undefined_full_es_endpoints": undefined_endpoints,
            "ratios": ratios,
            "mean_ratio": None,
            "minimum_ratio": None,
            "maximum_ratio": None,
            "out_of_sample": True,
            "selection_overlap": False,
            "chronology_verified": True,
            "selection_window": {
                "start_index": int(selection_start),
                "end_index": int(selection_end),
                "start_date": dates[selection_start].isoformat(),
                "end_date": dates[selection_end].isoformat(),
                "observations": int(estimation_window),
            },
            "validation_window": {
                "start_index": int(validation_start),
                "end_index": int(n - 1),
                "start_date": dates[validation_start].isoformat(),
                "end_date": dates[-1].isoformat(),
                "observations": int(validation_days),
            },
        }

    numeric_ratios = np.asarray(ratios, dtype=float)
    return {
        "status": "computed_out_of_sample",
        "observations": int(len(numeric_ratios)),
        "expected_observations": int(validation_days),
        "estimation_window": int(estimation_window),
        "validation_days": int(validation_days),
        "mean_ratio": float(np.mean(numeric_ratios)),
        "minimum_ratio": float(np.min(numeric_ratios)),
        "maximum_ratio": float(np.max(numeric_ratios)),
        "ratios": numeric_ratios.tolist(),
        "out_of_sample": True,
        "selection_overlap": False,
        "chronology_verified": True,
        "selection_window": {
            "start_index": int(selection_start),
            "end_index": int(selection_end),
            "start_date": dates[selection_start].isoformat(),
            "end_date": dates[selection_end].isoformat(),
            "observations": int(estimation_window),
        },
        "validation_window": {
            "start_index": int(validation_start),
            "end_index": int(n - 1),
            "start_date": dates[validation_start].isoformat(),
            "end_date": dates[-1].isoformat(),
            "observations": int(validation_days),
        },
        "first_validation_estimation_window": {
            "start_index": int(first_window_start),
            "end_index": int(validation_start),
            "start_date": dates[first_window_start].isoformat(),
            "end_date": dates[validation_start].isoformat(),
            "observations": int(estimation_window),
        },
        "interpretation": (
            "The factor set was frozen using the immediately preceding "
            "estimation window. The following validation_days endpoints were "
            "not used for factor selection. Each endpoint recomputes full and "
            "reduced current ES on a trailing estimation_window. This is a "
            "rolling-origin research holdout, not a supervisory IMA approval."
        ),
    }


def select_stress_window(
    factor_losses: dict[str, np.ndarray],
    window_days: int = 250,
    confidence: float = 0.975,
):
    """Select a descriptive worst historical window; not supervisory stress selection."""
    factors = _align_factor_losses(factor_losses)
    n = len(next(iter(factors.values())))
    window_days = _strict_integer(window_days, "window_days")
    if window_days < 11 or n < window_days:
        raise ValueError("insufficient observations for requested stress window")
    c = _confidence_level(confidence)

    total = np.sum(np.vstack(list(factors.values())), axis=0)
    best = None
    for start in range(0, n - window_days + 1):
        end = start + window_days
        ten_day = _ten_day_losses(total[start:end], 10)
        es = historical_var_es(ten_day, c)["expected_shortfall"]
        if best is None or es > best["stress_es"]:
            best = {"start_index": start, "end_index": end - 1, "stress_es": float(es)}
    assert best is not None
    return {
        **best,
        "window_days": int(window_days),
        "confidence": c,
        "observations": n,
        "stress_start_index": best["start_index"],
        "stress_end_index": best["end_index"],
        "selection_method": "maximum_historical_10_observation_expected_shortfall",
        "supervisory_status": "not_assessed",
    }


def select_reduced_factor_set(
    current_factor_losses: dict[str, np.ndarray],
    confidence: float = 0.975,
    target_explained: float = 0.75,
    estimation_window: int = MIN_CURRENT_OBSERVATIONS,
    validation_days: int = REDUCED_SET_VALIDATION_DAYS,
    observation_dates=None,
):
    """Select on a past window, then assess the frozen set on a forward holdout.

    Factor ranking and selection use only the estimation_window observations
    immediately preceding the final validation_days observations. The final
    holdout is not consulted until the selected set is frozen.
    """
    factors = _align_factor_losses(current_factor_losses)
    c = _confidence_level(confidence)
    target_explained = _strict_float(target_explained, "target_explained")
    if not 0.0 < target_explained <= 1.0:
        raise ValueError("target_explained must be in (0, 1]")
    estimation_window = _strict_integer(estimation_window, "estimation_window")
    validation_days = _strict_integer(validation_days, "validation_days")
    if estimation_window < 20:
        raise ValueError("estimation_window must be an integer of at least 20")
    if validation_days < 1:
        raise ValueError("validation_days must be a positive integer")

    n = len(next(iter(factors.values())))
    required = estimation_window + validation_days
    if n < required:
        raise ValueError(
            f"insufficient history: {n} observations; at least {required} "
            "are required for a non-overlapping selection/validation split"
        )

    dates = _validate_observation_dates(observation_dates, n)
    validation_start = n - validation_days
    selection_start = validation_start - estimation_window
    selection_end = validation_start - 1
    selection_sample = {
        name: series[selection_start:validation_start]
        for name, series in factors.items()
    }
    full_selection_es = _portfolio_es(selection_sample, list(selection_sample), c)
    if not np.isfinite(full_selection_es) or full_selection_es <= 0.0:
        raise ValueError("full selection-sample ES must be positive and finite")

    ranking = sorted(
        selection_sample,
        key=lambda name: historical_var_es(
            _ten_day_losses(selection_sample[name]), c
        )["expected_shortfall"],
        reverse=True,
    )
    selected = []
    training_ratio = float("nan")
    selected_es = float("nan")
    for factor in ranking:
        selected.append(factor)
        selected_es = _portfolio_es(selection_sample, selected, c)
        training_ratio = float(selected_es / full_selection_es)
        if training_ratio >= target_explained:
            break

    coverage = _rolling_reduced_coverage(
        factors, selected, c, estimation_window, validation_days, dates
    )
    oos_target_met = bool(
        coverage["status"] == "computed_out_of_sample"
        and coverage["mean_ratio"] >= target_explained
    )
    training_target_met = bool(training_ratio >= target_explained)
    return {
        "selected_factors": selected,
        "selection_sample_full_es": float(full_selection_es),
        "selection_sample_reduced_es": float(selected_es),
        "selection_sample_explained_ratio": float(training_ratio),
        # Legacy report key: this is selection-sample coverage, not OOS validation.
        "explained_ratio": float(training_ratio),
        "target_explained": float(target_explained),
        "target_met_on_selection_sample": training_target_met,
        "target_met_in_sample": training_target_met,
        "target_met": oos_target_met,
        "coverage_12w": coverage,
        "coverage_label": "rolling_origin_out_of_sample_research_diagnostic",
        "coverage_is_out_of_sample": bool(coverage.get("out_of_sample", False)),
        "factor_set_frozen_during_validation": True,
        "selection_window": {
            "start_index": int(selection_start),
            "end_index": int(selection_end),
            "start_date": dates[selection_start].isoformat(),
            "end_date": dates[selection_end].isoformat(),
            "observations": int(estimation_window),
        },
        "validation_window": {
            "start_index": int(validation_start),
            "end_index": int(n - 1),
            "start_date": dates[validation_start].isoformat(),
            "end_date": dates[-1].isoformat(),
            "observations": int(validation_days),
        },
        "coverage_method": (
            f"fixed factor set selected on {estimation_window} observations; "
            f"evaluated on the subsequent {validation_days} endpoints using "
            f"trailing {estimation_window}-observation full/reduced ES windows"
        ),
    }


def calculate_imcc(
    es_full_current: float,
    es_reduced_current: float,
    es_reduced_stress: float,
    class_components: dict[str, dict[str, float]],
    rho: float = 0.5,
):
    values = [
        _strict_float(es_full_current, "es_full_current"),
        _strict_float(es_reduced_current, "es_reduced_current"),
        _strict_float(es_reduced_stress, "es_reduced_stress"),
    ]
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("ES inputs must be positive and finite")
    rho_value = _strict_float(rho, "rho")
    if not np.isclose(rho_value, BASEL_IMCC_RHO, rtol=0.0, atol=1e-12):
        raise ValueError("Basel IMCC aggregation uses fixed rho = 0.5")
    if not class_components:
        raise ValueError("class_components cannot be empty")

    full_current, reduced_current, reduced_stress = values
    unconstrained = reduced_stress * max(full_current / reduced_current, 1.0)
    components = {}
    for name, component in class_components.items():
        try:
            class_values = [
                _strict_float(component[key], f"class_components[{name}][{key}]")
                for key in ("full_current", "reduced_current", "reduced_stress")
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid ES component for class {name}") from exc
        if any(not np.isfinite(value) or value <= 0.0 for value in class_values):
            raise ValueError(f"invalid ES component for class {name}")
        fc, rc, rs = class_values
        components[name] = float(rs * (fc / rc))

    constrained = float(sum(components.values()))
    return {
        "unconstrained": float(unconstrained),
        "constrained_sum": constrained,
        "class_components": components,
        "rho": rho_value,
        "imcc": float(rho_value * unconstrained + (1.0 - rho_value) * constrained),
        "status": "formula_primitive_only",
    }


def aggregate_nmrf_ses(
    charges_by_factor: dict[str, float],
    factor_groups: dict[str, str],
    rho: float = 0.6,
):
    if set(charges_by_factor) != set(factor_groups):
        raise ValueError("every NMRF charge needs a group")
    rho_value = _strict_float(rho, "rho")
    if not np.isclose(rho_value, BASEL_NMRF_RHO, rtol=0.0, atol=1e-12):
        raise ValueError("Basel NMRF aggregation uses fixed rho = 0.6")
    try:
        charges = {key: _strict_float(value, f"charges_by_factor[{key}]") for key, value in charges_by_factor.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("NMRF charges must be numeric") from exc
    if any(not np.isfinite(value) or value < 0.0 for value in charges.values()):
        raise ValueError("NMRF charges must be finite and nonnegative")
    credit = [charges[key] for key, group in factor_groups.items() if group == "credit_idio"]
    equity = [charges[key] for key, group in factor_groups.items() if group == "equity_idio"]
    other = [charges[key] for key, group in factor_groups.items() if group == "other"]
    unknown = set(factor_groups.values()) - {"credit_idio", "equity_idio", "other"}
    if unknown:
        raise ValueError(f"unknown NMRF group(s): {sorted(unknown)}")

    credit_agg = float(np.sqrt(np.sum(np.square(credit))))
    equity_agg = float(np.sqrt(np.sum(np.square(equity))))
    other_agg = float(
        np.sqrt((rho_value * sum(other)) ** 2 + (1.0 - rho_value**2) * sum(v * v for v in other))
    )
    return {
        "credit_idio": credit_agg,
        "equity_idio": equity_agg,
        "other": other_agg,
        "aggregate": float(credit_agg + equity_agg + other_agg),
        "rho": rho_value,
        "status": "formula_primitive_only",
    }


def build_ima_capital(
    factor_losses: dict[str, np.ndarray],
    stress_window: dict,
    class_mapping: dict[str, str],
    confidence: float = 0.975,
    rho: float = 0.5,
    observation_dates=None,
):
    """Build a non-regulatory research aggregation, failing closed on short data."""
    factors = _align_factor_losses(factor_losses)
    c = _confidence_level(confidence)
    rho = _strict_float(rho, "rho")
    if not np.isclose(rho, BASEL_IMCC_RHO, rtol=0.0, atol=1e-12):
        raise ValueError("Basel IMCC aggregation uses fixed rho = 0.5")
    if set(class_mapping) != set(factors):
        raise ValueError("each factor must map to one risk class")
    normalized_class_mapping = {
        factor: _normalize_risk_class(risk_class)
        for factor, risk_class in class_mapping.items()
    }
    n = len(next(iter(factors.values())))
    required = MIN_CURRENT_OBSERVATIONS + REDUCED_SET_VALIDATION_DAYS
    if n < required:
        raise ValueError(
            f"insufficient history for 250-day selection sample plus 60 forward validation endpoints: "
            f"{n} supplied; at least {required} required"
        )

    try:
        start = _strict_integer(stress_window["start_index"], "stress_window.start_index")
        end_inclusive = _strict_integer(stress_window["end_index"], "stress_window.end_index")
        declared_window = _strict_integer(stress_window["window_days"], "stress_window.window_days")
    except KeyError as exc:
        raise ValueError("stress_window must include start_index, end_index and window_days") from exc
    if declared_window != MIN_CURRENT_OBSERVATIONS:
        raise ValueError("research stress window must contain exactly 250 observations")
    if start < 0 or end_inclusive < start or end_inclusive >= n:
        raise ValueError("stress_window indices are outside the factor history")
    if end_inclusive - start + 1 != declared_window:
        raise ValueError("stress_window indices do not match its declared 250 observations")

    current = {name: series[-MIN_CURRENT_OBSERVATIONS:] for name, series in factors.items()}
    stressed = {name: series[start : end_inclusive + 1] for name, series in factors.items()}
    reduced = select_reduced_factor_set(
        factors,
        c,
        target_explained=0.75,
        estimation_window=MIN_CURRENT_OBSERVATIONS,
        validation_days=REDUCED_SET_VALIDATION_DAYS,
        observation_dates=observation_dates,
    )
    selected = reduced["selected_factors"]

    full_current = _portfolio_es(current, list(factors), c)
    reduced_current = _portfolio_es(current, selected, c)
    reduced_stress = _portfolio_es(stressed, selected, c)

    classes = {}
    all_classes = sorted(set(normalized_class_mapping.values()))
    missing_selected_classes = []
    for risk_class in all_classes:
        all_names = [name for name, mapped_class in normalized_class_mapping.items() if mapped_class == risk_class]
        selected_names = [name for name in all_names if name in selected]
        if not selected_names:
            missing_selected_classes.append(risk_class)
            continue
        classes[risk_class] = {
            "full_current": _portfolio_es(current, all_names, c),
            "reduced_current": _portfolio_es(current, selected_names, c),
            "reduced_stress": _portfolio_es(stressed, selected_names, c),
        }

    # MAR33.14 requires all five broad regulatory risk classes.
    mapped_regulatory_classes = set(normalized_class_mapping.values())
    required_regulatory_classes = {
        "interest_rate", "equity", "foreign_exchange", "commodity", "credit_spread"
    }
    missing_regulatory_classes = sorted(
        required_regulatory_classes - mapped_regulatory_classes
    )
    imcc_complete = not missing_selected_classes and not missing_regulatory_classes
    imcc = (
        calculate_imcc(full_current, reduced_current, reduced_stress, classes, rho)
        if imcc_complete
        else None
    )
    return {
        "reduced_set": reduced,
        "es_full_current": full_current,
        "es_reduced_current": reduced_current,
        "es_reduced_stress": reduced_stress,
        "imcc": imcc,
        "capital_label": "research_only_non_regulatory",
        "regulatory_status": "not_assessed",
        "regulatory_requirements_not_evidenced": [
            "Actual P&L and complete desk-level PLA/backtesting controls",
            "full regulatory factor mapping and eligibility",
            "MAR33 factor-specific liquidity-horizon mapping and stress calibration",
            "stress-period and broad risk-class governance",
            "modellable/non-modellable factor evidence and NMRF stress scenarios",
            "bank-specific policies, controls, independent validation and supervisory approval",
        ],
        "class_aggregation_complete_within_supplied_mapping": not missing_selected_classes,
        "missing_selected_risk_classes": missing_selected_classes,
        "required_regulatory_risk_classes": sorted(required_regulatory_classes),
        "mapped_regulatory_risk_classes": sorted(mapped_regulatory_classes),
        "missing_regulatory_risk_classes": missing_regulatory_classes,
        "imcc_status": (
            "formula_primitive_only_incomplete_regulatory_coverage"
            if imcc_complete
            else "not_computed_incomplete_risk_class_coverage"
        ),
    }
