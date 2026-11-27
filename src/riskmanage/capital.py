from __future__ import annotations

import numpy as np

from .risk import _ten_day_losses, historical_var_es


MIN_CURRENT_OBSERVATIONS = 250
REDUCED_SET_VALIDATION_DAYS = 60


def _align_factor_losses(factor_losses):
    if not factor_losses:
        raise ValueError("factor losses cannot be empty")
    values = {name: np.asarray(series, dtype=float) for name, series in factor_losses.items()}
    if any(
        series.ndim != 1 or series.size == 0 or not np.isfinite(series).all()
        for series in values.values()
    ):
        raise ValueError("factor loss arrays must be non-empty, finite 1D vectors")
    if len({len(series) for series in values.values()}) != 1:
        raise ValueError("factor arrays must align")
    return values


def _confidence_level(confidence):
    value = float(confidence)
    if not np.isfinite(value) or not 0.5 < value < 1.0:
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
):
    """Descriptive overlapping rolling coverage; deliberately not OOS validation."""
    n = len(next(iter(factors.values())))
    required = estimation_window + validation_days - 1
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
        }

    ratios = []
    for end in range(n - validation_days, n):
        start = end - estimation_window + 1
        window = {name: series[start : end + 1] for name, series in factors.items()}
        full = _portfolio_es(window, list(window), confidence)
        reduced = _portfolio_es(window, selected, confidence)
        ratios.append(float(reduced / full) if full > 0.0 else float("nan"))

    finite = [value for value in ratios if np.isfinite(value)]
    if not finite:
        return {
            "status": "undefined_full_es",
            "observations": len(ratios),
            "required_observations": required,
            "mean_ratio": None,
            "minimum_ratio": None,
            "maximum_ratio": None,
            "ratios": ratios,
            "out_of_sample": False,
        }
    return {
        "status": "descriptive_in_sample",
        "observations": len(finite),
        "estimation_window": int(estimation_window),
        "validation_days": int(validation_days),
        "mean_ratio": float(np.mean(finite)),
        "minimum_ratio": float(np.min(finite)),
        "maximum_ratio": float(np.max(finite)),
        "ratios": finite,
        "out_of_sample": False,
        "selection_overlap": True,
        "interpretation": (
            "Overlapping windows reuse observations that also participated in "
            "factor selection; this is descriptive in-sample coverage, not a "
            "holdout, rolling-origin or regulatory reduced-factor validation."
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
    if not isinstance(window_days, (int, np.integer)) or isinstance(
        window_days, (bool, np.bool_)
    ):
        raise ValueError("window_days must be an integer")
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
):
    factors = _align_factor_losses(current_factor_losses)
    c = _confidence_level(confidence)
    if not 0.0 < target_explained <= 1.0:
        raise ValueError("target_explained must be in (0, 1]")
    if not isinstance(estimation_window, (int, np.integer)) or estimation_window < 20:
        raise ValueError("estimation_window must be an integer of at least 20")
    if not isinstance(validation_days, (int, np.integer)) or validation_days < 1:
        raise ValueError("validation_days must be a positive integer")
    n = len(next(iter(factors.values())))
    if n < estimation_window:
        raise ValueError(
            f"insufficient history: {n} observations; {estimation_window} required"
        )

    selection_start = n - estimation_window
    current = {name: series[selection_start:] for name, series in factors.items()}
    full_es = _portfolio_es(current, list(current), c)
    if not np.isfinite(full_es) or full_es <= 0.0:
        raise ValueError("full portfolio ES must be positive and finite")

    ranking = sorted(
        current,
        key=lambda name: historical_var_es(
            _ten_day_losses(current[name]), c
        )["expected_shortfall"],
        reverse=True,
    )
    selected = []
    for factor in ranking:
        selected.append(factor)
        coverage = _rolling_reduced_coverage(
            factors, selected, c, estimation_window, validation_days
        )
        if coverage["status"] == "descriptive_in_sample" and (
            coverage["mean_ratio"] >= target_explained
        ):
            break

    reduced_es = _portfolio_es(current, selected, c)
    ratio = float(reduced_es / full_es)
    coverage = _rolling_reduced_coverage(
        factors, selected, c, estimation_window, validation_days
    )
    target_met = bool(
        coverage["status"] == "descriptive_in_sample"
        and coverage["mean_ratio"] >= target_explained
    )
    return {
        "selected_factors": selected,
        "full_current_es": float(full_es),
        "reduced_current_es": float(reduced_es),
        "explained_ratio": ratio,
        "target_explained": float(target_explained),
        "target_met_in_sample": target_met,
        "target_met": target_met,
        "coverage_12w": coverage,
        "coverage_label": "descriptive_overlapping_in_sample_not_OOS",
        "coverage_is_out_of_sample": False,
        "selection_window": {
            "start_index": selection_start,
            "end_index": n - 1,
            "observations": int(estimation_window),
        },
        "coverage_method": (
            f"mean reduced/full 10-day ES ratio over latest {validation_days} "
            f"overlapping endpoints, each using {estimation_window} observations; "
            "these windows overlap the factor-selection sample"
        ),
    }


def calculate_imcc(
    es_full_current: float,
    es_reduced_current: float,
    es_reduced_stress: float,
    class_components: dict[str, dict[str, float]],
    rho: float = 0.5,
):
    values = [float(es_full_current), float(es_reduced_current), float(es_reduced_stress)]
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("ES inputs must be positive and finite")
    rho_value = float(rho)
    if not np.isfinite(rho_value) or not 0.0 <= rho_value <= 1.0:
        raise ValueError("rho must be finite and in [0, 1]")
    if not class_components:
        raise ValueError("class_components cannot be empty")

    full_current, reduced_current, reduced_stress = values
    unconstrained = reduced_stress * max(full_current / reduced_current, 1.0)
    components = {}
    for name, component in class_components.items():
        try:
            class_values = [
                float(component[key])
                for key in ("full_current", "reduced_current", "reduced_stress")
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid ES component for class {name}") from exc
        if any(not np.isfinite(value) or value <= 0.0 for value in class_values):
            raise ValueError(f"invalid ES component for class {name}")
        fc, rc, rs = class_values
        components[name] = float(rs * max(fc / rc, 1.0))

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
    rho_value = float(rho)
    if not np.isfinite(rho_value) or not 0.0 <= rho_value <= 1.0:
        raise ValueError("rho must be finite and in [0, 1]")
    try:
        charges = {key: float(value) for key, value in charges_by_factor.items()}
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
):
    """Build a non-regulatory research aggregation, failing closed on short data."""
    factors = _align_factor_losses(factor_losses)
    c = _confidence_level(confidence)
    if set(class_mapping) != set(factors):
        raise ValueError("each factor must map to one risk class")
    n = len(next(iter(factors.values())))
    required = MIN_CURRENT_OBSERVATIONS + REDUCED_SET_VALIDATION_DAYS - 1
    if n < required:
        raise ValueError(
            f"insufficient history for 250-day estimation plus 60 endpoints: "
            f"{n} supplied; at least {required} required"
        )

    try:
        start = int(stress_window["start_index"])
        end_inclusive = int(stress_window["end_index"])
        declared_window = int(stress_window["window_days"])
    except (KeyError, TypeError, ValueError) as exc:
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
    )
    selected = reduced["selected_factors"]

    full_current = _portfolio_es(current, list(factors), c)
    reduced_current = _portfolio_es(current, selected, c)
    reduced_stress = _portfolio_es(stressed, selected, c)

    classes = {}
    all_classes = sorted(set(class_mapping.values()))
    missing_selected_classes = []
    for risk_class in all_classes:
        all_names = [name for name, mapped_class in class_mapping.items() if mapped_class == risk_class]
        selected_names = [name for name in all_names if name in selected]
        if not selected_names:
            missing_selected_classes.append(risk_class)
            continue
        classes[risk_class] = {
            "full_current": _portfolio_es(current, all_names, c),
            "reduced_current": _portfolio_es(current, selected_names, c),
            "reduced_stress": _portfolio_es(stressed, selected_names, c),
        }

    imcc_complete = not missing_selected_classes
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
        "class_aggregation_complete": bool(imcc_complete),
        "missing_selected_risk_classes": missing_selected_classes,
        "imcc_status": (
            "illustrative_formula_primitive_only"
            if imcc_complete
            else "not_computed_missing_selected_risk_class"
        ),
    }
