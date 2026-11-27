from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import _ten_day_losses, historical_var_es


MIN_CURRENT_OBSERVATIONS = 250
REDUCED_SET_HOLDOUT_OBSERVATIONS = MIN_CURRENT_OBSERVATIONS
DEFAULT_HOLDOUT_BOOTSTRAP_REPLICATES = 1000
DEFAULT_HOLDOUT_BOOTSTRAP_BLOCK_LENGTHS = (10, 20, 40)
DEFAULT_HOLDOUT_BOOTSTRAP_SEED = 20261127
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


def _circular_moving_block_bootstrap_es_ratio(
    daily_factor_losses,
    selected,
    confidence,
    target_explained=0.75,
    bootstrap_replicates=DEFAULT_HOLDOUT_BOOTSTRAP_REPLICATES,
    block_lengths=DEFAULT_HOLDOUT_BOOTSTRAP_BLOCK_LENGTHS,
    seed=DEFAULT_HOLDOUT_BOOTSTRAP_SEED,
):
    """Conditional block-bootstrap uncertainty for a frozen reduced-factor set.

    Each factor's overlapping ten-day losses are constructed first. The
    bootstrap resamples synchronized rows across all factors in circular
    contiguous blocks, preserving contemporaneous factor dependence and
    short-run serial dependence in the ten-day loss series.

    Intervals are conditional on the already-selected factor set and the
    observed holdout. They do not include factor-selection, model, data or
    regime uncertainty.
    """
    bootstrap_replicates = _strict_integer(
        bootstrap_replicates, "bootstrap_replicates"
    )
    seed = _strict_integer(seed, "bootstrap_seed")
    target = _strict_float(target_explained, "target_explained")
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if seed < 0:
        raise ValueError("bootstrap_seed must be nonnegative")
    if not 0.0 < target <= 1.0:
        raise ValueError("target_explained must be in (0, 1]")

    if not isinstance(block_lengths, (tuple, list)) or not block_lengths:
        raise ValueError("bootstrap_block_lengths must be a non-empty tuple or list")
    normalized_blocks = tuple(
        _strict_integer(value, "bootstrap_block_length") for value in block_lengths
    )
    if len(set(normalized_blocks)) != len(normalized_blocks):
        raise ValueError("bootstrap_block_lengths must be unique")
    if any(value < 10 for value in normalized_blocks):
        raise ValueError(
            "bootstrap block lengths must be at least 10 overlapping ten-day observations"
        )

    if not daily_factor_losses:
        raise ValueError("daily_factor_losses cannot be empty")
    if not selected or not set(selected).issubset(daily_factor_losses):
        raise ValueError("selected must contain known risk factors")

    names = list(daily_factor_losses)
    ten_day = {
        name: _ten_day_losses(np.asarray(values, dtype=float), 10)
        for name, values in daily_factor_losses.items()
    }
    lengths = {len(values) for values in ten_day.values()}
    if len(lengths) != 1:
        raise ValueError("ten-day factor loss vectors must have equal lengths")
    matrix = np.column_stack([ten_day[name] for name in names])
    observations = matrix.shape[0]
    if any(value > observations for value in normalized_blocks):
        raise ValueError(
            f"bootstrap block lengths cannot exceed {observations} ten-day observations"
        )

    selected_columns = [names.index(name) for name in selected]
    rng = np.random.default_rng(seed)
    sensitivity = {}
    required_valid = int(np.ceil(0.95 * bootstrap_replicates))

    for block_length in normalized_blocks:
        blocks_per_replication = int(np.ceil(observations / block_length))
        ratios = np.empty(bootstrap_replicates, dtype=float)
        valid = 0
        nonpositive_full_es = 0
        nonfinite_es = 0

        for _ in range(bootstrap_replicates):
            starts = rng.integers(0, observations, size=blocks_per_replication)
            indices = np.concatenate(
                [
                    (start + np.arange(block_length, dtype=int)) % observations
                    for start in starts
                ]
            )[:observations]
            sample = matrix[indices, :]
            full_losses = np.sum(sample, axis=1)
            reduced_losses = np.sum(sample[:, selected_columns], axis=1)
            full_es = historical_var_es(full_losses, confidence)["expected_shortfall"]
            reduced_es = historical_var_es(
                reduced_losses, confidence
            )["expected_shortfall"]

            if not np.isfinite(full_es) or not np.isfinite(reduced_es):
                nonfinite_es += 1
                continue
            if full_es <= 0.0:
                nonpositive_full_es += 1
                continue
            ratios[valid] = float(reduced_es / full_es)
            valid += 1

        valid_fraction = valid / bootstrap_replicates
        ci = (
            np.percentile(ratios[:valid], [2.5, 97.5]).astype(float).tolist()
            if valid >= required_valid
            else None
        )
        sensitivity[str(block_length)] = {
            "status": (
                "computed_conditional_percentile_interval"
                if ci is not None
                else "insufficient_valid_replicates"
            ),
            "block_length": int(block_length),
            "block_length_unit": (
                "consecutive overlapping ten-day loss observations; "
                "each observation starts one trading day after the prior one"
            ),
            "requested_replicates": int(bootstrap_replicates),
            "valid_replicates": int(valid),
            "failed_replicates": int(bootstrap_replicates - valid),
            "nonpositive_full_es_replicates": int(nonpositive_full_es),
            "nonfinite_es_replicates": int(nonfinite_es),
            "valid_fraction": float(valid_fraction),
            "bootstrap_mean_ratio": (
                float(np.mean(ratios[:valid])) if valid else None
            ),
            "bootstrap_median_ratio": (
                float(np.median(ratios[:valid])) if valid else None
            ),
            "confidence_interval_level": 0.95,
            "confidence_interval": ci,
            "fraction_below_target": (
                float(np.mean(ratios[:valid] < target)) if valid else None
            ),
        }

    return {
        "status": "computed_conditional_block_bootstrap",
        "method": "circular_moving_block_bootstrap_on_synchronized_ten_day_factor_losses",
        "replicates_per_block_length": int(bootstrap_replicates),
        "seed": int(seed),
        "confidence_interval_method": "two_sided_percentile_95_percent",
        "nominal_tail_observations": float(observations * (1.0 - confidence)),
        "ten_day_loss_observations": int(observations),
        "factor_columns_resampled_together": True,
        "factor_set_reselected_in_replicates": False,
        "conditional_on_frozen_factor_selection": True,
        "block_length_sensitivity": sensitivity,
        "interpretation": (
            "Intervals condition on the factor set selected before the holdout "
            "and on the observed holdout design. They quantify only sampling "
            "variation under the stated circular moving-block bootstrap; they "
            "do not include factor-selection, model, market-data or regime "
            "uncertainty. Block-length sensitivity must be read across all "
            "reported values rather than selecting the most favorable interval."
        ),
    }


def _non_overlapping_reduced_holdout(
    factors,
    selected,
    confidence=0.975,
    estimation_window=MIN_CURRENT_OBSERVATIONS,
    holdout_observations=REDUCED_SET_HOLDOUT_OBSERVATIONS,
    observation_dates=None,
    target_explained=0.75,
    bootstrap_replicates=DEFAULT_HOLDOUT_BOOTSTRAP_REPLICATES,
    bootstrap_block_lengths=DEFAULT_HOLDOUT_BOOTSTRAP_BLOCK_LENGTHS,
    bootstrap_seed=DEFAULT_HOLDOUT_BOOTSTRAP_SEED,
):
    """Evaluate a frozen factor set on a disjoint chronological holdout.

    The selection and holdout samples do not share daily observations. The
    ten-day loss aggregates overlap only within their own sample: neither the
    first nor the final holdout aggregate crosses the selection boundary.
    """
    n = len(next(iter(factors.values())))
    dates = _validate_observation_dates(observation_dates, n)
    required = estimation_window + holdout_observations
    if n < required:
        return {
            "status": "insufficient_history",
            "observations": 0,
            "required_observations": int(required),
            "ratio": None,
            "full_es": None,
            "reduced_es": None,
            "independent_holdout": False,
            "selection_overlap": None,
            "estimation_window_overlap": None,
        }
    if not selected or not set(selected).issubset(factors):
        raise ValueError("selected must contain known risk factors")

    holdout_start = n - holdout_observations
    selection_start = holdout_start - estimation_window
    selection_end = holdout_start - 1
    holdout_end = n - 1
    holdout_sample = {
        name: series[holdout_start : holdout_end + 1]
        for name, series in factors.items()
    }

    full_es = _portfolio_es(holdout_sample, list(holdout_sample), confidence)
    reduced_es = _portfolio_es(holdout_sample, selected, confidence)
    common = {
        "observations": int(holdout_observations),
        "holdout_observations": int(holdout_observations),
        "ten_day_es_observations": int(max(holdout_observations - 9, 0)),
        "selection_overlap": False,
        "estimation_window_overlap": False,
        "independent_holdout": True,
        "out_of_sample": True,
        "chronology_verified": True,
        "selection_window": {
            "start_index": int(selection_start),
            "end_index": int(selection_end),
            "start_date": dates[selection_start].isoformat(),
            "end_date": dates[selection_end].isoformat(),
            "observations": int(estimation_window),
        },
        "holdout_window": {
            "start_index": int(holdout_start),
            "end_index": int(holdout_end),
            "start_date": dates[holdout_start].isoformat(),
            "end_date": dates[holdout_end].isoformat(),
            "observations": int(holdout_observations),
        },
    }

    if not np.isfinite(full_es) or full_es <= 0.0:
        return {
            **common,
            "status": "unclassifiable_nonpositive_full_holdout_es",
            "ratio": None,
            "full_es": float(full_es) if np.isfinite(full_es) else None,
            "reduced_es": float(reduced_es) if np.isfinite(reduced_es) else None,
        }
    if not np.isfinite(reduced_es):
        return {
            **common,
            "status": "unclassifiable_nonfinite_reduced_holdout_es",
            "ratio": None,
            "full_es": float(full_es),
            "reduced_es": None,
        }

    uncertainty = _circular_moving_block_bootstrap_es_ratio(
        holdout_sample,
        selected,
        confidence,
        target_explained=target_explained,
        bootstrap_replicates=bootstrap_replicates,
        block_lengths=bootstrap_block_lengths,
        seed=bootstrap_seed,
    )
    return {
        **common,
        "status": "computed_non_overlapping_holdout",
        "ratio": float(reduced_es / full_es),
        "full_es": float(full_es),
        "reduced_es": float(reduced_es),
        "conditional_uncertainty": uncertainty,
        "interpretation": (
            "The factor set was selected using a disjoint historical sample and "
            "evaluated once on the subsequent holdout sample. Each sample's "
            "ten-day ES uses overlapping ten-day aggregates formed entirely "
            "within that sample. This is a research holdout, not a supervisory "
            "IMA approval. At 97.5%, the holdout ES tail is based on only a "
            "small number of ten-day observations; sampling uncertainty remains."
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
    holdout_observations: int = REDUCED_SET_HOLDOUT_OBSERVATIONS,
    observation_dates=None,
    bootstrap_replicates: int = DEFAULT_HOLDOUT_BOOTSTRAP_REPLICATES,
    bootstrap_block_lengths=DEFAULT_HOLDOUT_BOOTSTRAP_BLOCK_LENGTHS,
    bootstrap_seed: int = DEFAULT_HOLDOUT_BOOTSTRAP_SEED,
):
    """Select on one sample and evaluate once on a disjoint chronological holdout."""
    factors = _align_factor_losses(current_factor_losses)
    c = _confidence_level(confidence)
    target_explained = _strict_float(target_explained, "target_explained")
    if not 0.0 < target_explained <= 1.0:
        raise ValueError("target_explained must be in (0, 1]")
    estimation_window = _strict_integer(estimation_window, "estimation_window")
    holdout_observations = _strict_integer(
        holdout_observations, "holdout_observations"
    )
    if estimation_window < MIN_CURRENT_OBSERVATIONS:
        raise ValueError(
            f"estimation_window must be at least {MIN_CURRENT_OBSERVATIONS} observations"
        )
    if holdout_observations < REDUCED_SET_HOLDOUT_OBSERVATIONS:
        raise ValueError(
            "holdout_observations must be at least "
            f"{REDUCED_SET_HOLDOUT_OBSERVATIONS}"
        )

    n = len(next(iter(factors.values())))
    required = estimation_window + holdout_observations
    if n < required:
        raise ValueError(
            f"insufficient history: {n} observations; at least {required} "
            "are required for disjoint selection and holdout samples"
        )

    dates = _validate_observation_dates(observation_dates, n)
    holdout_start = n - holdout_observations
    selection_start = holdout_start - estimation_window
    selection_end = holdout_start - 1
    selection_sample = {
        name: series[selection_start:holdout_start]
        for name, series in factors.items()
    }
    full_selection_es = _portfolio_es(
        selection_sample, list(selection_sample), c
    )
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

    holdout = _non_overlapping_reduced_holdout(
        factors,
        selected,
        c,
        estimation_window,
        holdout_observations,
        dates,
        target_explained,
        bootstrap_replicates,
        bootstrap_block_lengths,
        bootstrap_seed,
    )
    holdout_target_met = bool(
        holdout["status"] == "computed_non_overlapping_holdout"
        and holdout["ratio"] >= target_explained
    )
    selection_target_met = bool(training_ratio >= target_explained)
    selection_window = {
        "start_index": int(selection_start),
        "end_index": int(selection_end),
        "start_date": dates[selection_start].isoformat(),
        "end_date": dates[selection_end].isoformat(),
        "observations": int(estimation_window),
    }
    holdout_window = {
        "start_index": int(holdout_start),
        "end_index": int(n - 1),
        "start_date": dates[holdout_start].isoformat(),
        "end_date": dates[-1].isoformat(),
        "observations": int(holdout_observations),
    }
    return {
        "selected_factors": selected,
        "selection_sample_full_es": float(full_selection_es),
        "selection_sample_reduced_es": float(selected_es),
        "selection_sample_explained_ratio": float(training_ratio),
        # Legacy report key; this is only the selection-sample statistic.
        "explained_ratio": float(training_ratio),
        "target_explained": float(target_explained),
        "target_met_on_selection_sample": selection_target_met,
        "target_met_in_sample": selection_target_met,
        "target_met": holdout_target_met,
        "holdout": holdout,
        "holdout_label": "non_overlapping_chronological_research_holdout",
        "holdout_is_independent": bool(holdout.get("independent_holdout", False)),
        "coverage_is_out_of_sample": bool(holdout.get("out_of_sample", False)),
        "factor_set_frozen_during_validation": True,
        "selection_window": selection_window,
        "holdout_window": holdout_window,
        "coverage_method": (
            f"factor set selected on {estimation_window} daily observations and "
            f"evaluated on a disjoint {holdout_observations}-observation holdout; "
            "ten-day aggregates are formed separately within each sample"
        ),
        "sampling_uncertainty_note": (
            "At 97.5% confidence, 241 ten-day holdout observations imply only "
            "about six nominal tail observations; the holdout ratio is descriptive "
            "and requires uncertainty analysis before inferential claims."
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
    bootstrap_replicates: int = DEFAULT_HOLDOUT_BOOTSTRAP_REPLICATES,
    bootstrap_block_lengths=DEFAULT_HOLDOUT_BOOTSTRAP_BLOCK_LENGTHS,
    bootstrap_seed: int = DEFAULT_HOLDOUT_BOOTSTRAP_SEED,
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
    required = MIN_CURRENT_OBSERVATIONS + REDUCED_SET_HOLDOUT_OBSERVATIONS
    if n < required:
        raise ValueError(
            f"insufficient history for 250-day selection sample plus 250-observation independent holdout: "
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
        holdout_observations=REDUCED_SET_HOLDOUT_OBSERVATIONS,
        observation_dates=observation_dates,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_block_lengths=bootstrap_block_lengths,
        bootstrap_seed=bootstrap_seed,
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
