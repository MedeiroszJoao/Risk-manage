from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import genpareto, kstest


def _empirical_var_es(losses: np.ndarray, confidence_level: float) -> dict:
    sample_size = losses.size
    tail_mass = sample_size * (1.0 - confidence_level)
    if tail_mass < 1.0:
        return {
            "var": None,
            "expected_shortfall": None,
            "tail_mass_observations": float(tail_mass),
            "status": "insufficient_sample_for_empirical_tail",
        }
    var = float(
        np.quantile(losses, confidence_level, method="inverted_cdf")
    )
    strict = losses > var
    count = int(np.sum(strict))
    es = (
        float(np.sum(losses[strict])) + (tail_mass - count) * var
    ) / tail_mass
    return {
        "var": var,
        "expected_shortfall": float(es),
        "tail_mass_observations": float(tail_mass),
        "status": "estimated",
    }


def _gpd_tail_quantiles(
    threshold: float,
    tail_probability: float,
    shape: float,
    scale: float,
    confidence_levels: Sequence[float],
) -> dict:
    output = {}
    for alpha in confidence_levels:
        target_tail_probability = 1.0 - alpha
        ratio = tail_probability / target_tail_probability
        if ratio <= 1.0:
            output[str(alpha)] = {
                "var": None,
                "expected_shortfall": None,
                "status": "confidence_level_not_beyond_threshold",
            }
            continue

        if abs(shape) < 1e-7:
            excess_quantile = scale * math.log(ratio)
        else:
            excess_quantile = scale / shape * (ratio**shape - 1.0)
        var = threshold + excess_quantile

        if shape < 1.0:
            es = var + (scale + shape * excess_quantile) / (1.0 - shape)
            es_value = float(es)
            es_status = "estimated"
        else:
            es_value = None
            es_status = "infinite_under_fitted_gpd_shape"

        output[str(alpha)] = {
            "var": float(var),
            "expected_shortfall": es_value,
            "status": "estimated",
            "expected_shortfall_status": es_status,
            "extrapolation_ratio": float(ratio),
        }
    return output


def _bootstrap_gpd_tail(
    exceedances: np.ndarray,
    threshold: float,
    tail_probability: float,
    confidence_levels: Sequence[float],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    sample_size = exceedances.size
    draws: dict[str, dict[str, list[float]]] = {
        str(alpha): {"var": [], "expected_shortfall": []}
        for alpha in confidence_levels
    }
    failures = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(exceedances, size=sample_size, replace=True)
        try:
            shape, _, scale = genpareto.fit(sample, floc=0.0)
        except (ValueError, FloatingPointError, RuntimeError):
            failures += 1
            continue
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0.0:
            failures += 1
            continue
        fitted = _gpd_tail_quantiles(
            threshold,
            tail_probability,
            float(shape),
            float(scale),
            confidence_levels,
        )
        for alpha in confidence_levels:
            result = fitted[str(alpha)]
            if result["var"] is not None and np.isfinite(result["var"]):
                draws[str(alpha)]["var"].append(float(result["var"]))
            if (
                result["expected_shortfall"] is not None
                and np.isfinite(result["expected_shortfall"])
            ):
                draws[str(alpha)]["expected_shortfall"].append(
                    float(result["expected_shortfall"])
                )

    summaries = {}
    for alpha, metrics in draws.items():
        summaries[alpha] = {}
        for metric, values in metrics.items():
            if len(values) < 2:
                summaries[alpha][metric] = None
            else:
                values_array = np.asarray(values, dtype=float)
                summaries[alpha][metric] = {
                    "bootstrap_successes": int(values_array.size),
                    "percentile_95_interval": [
                        float(np.quantile(values_array, 0.025)),
                        float(np.quantile(values_array, 0.975)),
                    ],
                    "median": float(np.median(values_array)),
                }
    return {
        "requested_replicates": int(n_bootstrap),
        "failed_replicates": int(failures),
        "successful_replicates": int(n_bootstrap - failures),
        "conditional_percentile_intervals": summaries,
        "uncertainty_boundary": (
            "Non-parametric resampling of fitted threshold exceedances, conditional on the "
            "empirical threshold, exceedance count and fitted GPD family. It does not include "
            "threshold-selection uncertainty, dependence/block uncertainty or model risk."
        ),
    }


def _stability_summary(values: Sequence[float]) -> dict:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        return {
            "successful_threshold_fits": int(sample.size),
            "minimum": float(sample[0]) if sample.size else None,
            "maximum": float(sample[0]) if sample.size else None,
            "range": 0.0 if sample.size == 1 else None,
            "mean": float(sample[0]) if sample.size else None,
            "standard_deviation": 0.0 if sample.size == 1 else None,
            "relative_range_to_absolute_mean": None,
        }
    mean = float(np.mean(sample))
    spread = float(np.max(sample) - np.min(sample))
    return {
        "successful_threshold_fits": int(sample.size),
        "minimum": float(np.min(sample)),
        "maximum": float(np.max(sample)),
        "range": spread,
        "mean": mean,
        "standard_deviation": float(np.std(sample, ddof=1)),
        "relative_range_to_absolute_mean": (
            float(spread / abs(mean)) if abs(mean) > 1e-12 else None
        ),
    }


def _threshold_stability(fits: Sequence[dict], confidence_levels: Sequence[float]) -> dict:
    successful = [row for row in fits if row["status"] == "fitted"]
    result = {
        "successful_threshold_fit_count": int(len(successful)),
        "shape": _stability_summary(
            [row["fit"]["shape"] for row in successful]
        ),
        "scale": _stability_summary(
            [row["fit"]["scale"] for row in successful]
        ),
        "confidence_level_metrics": {},
        "automatic_threshold_selection": False,
    }
    for alpha in confidence_levels:
        key = str(alpha)
        var_values = []
        es_values = []
        for row in successful:
            estimate = row["confidence_levels"][key]
            if estimate["var"] is not None and np.isfinite(estimate["var"]):
                var_values.append(estimate["var"])
            if (
                estimate["expected_shortfall"] is not None
                and np.isfinite(estimate["expected_shortfall"])
            ):
                es_values.append(estimate["expected_shortfall"])
        result["confidence_level_metrics"][key] = {
            "var": _stability_summary(var_values),
            "expected_shortfall": _stability_summary(es_values),
        }
    return result


def fit_pot_gpd_tail_risk(
    losses,
    confidence_levels: Sequence[float] = (0.975, 0.99),
    threshold_quantiles: Sequence[float] = (0.90, 0.925, 0.95),
    *,
    min_exceedances: int = 20,
    n_bootstrap: int = 100,
    seed: int = 20261126,
) -> dict:
    """Fit Peaks-over-Threshold GPD tails and report VaR/ES threshold sensitivity.

    Loss observations above each empirical threshold are fit with a zero-location
    Generalized Pareto distribution by maximum likelihood. The threshold is held
    fixed for the conditional non-parametric exceedance bootstrap. No automatic
    threshold is selected: the caller must inspect exceedance counts, parameter
    stability and goodness-of-fit diagnostics across thresholds.

    The returned KS statistic is descriptive only; its usual p-value is omitted
    because the GPD parameters were estimated from the same data. ES exists only
    when the fitted GPD shape is below one. Bootstrap intervals are conditional
    on threshold and exceedance count and do not include threshold-selection or
    time-series dependence uncertainty.
    """
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size < 50 or not np.isfinite(values).all():
        raise ValueError("losses must be a finite one-dimensional sample of at least 50")
    alphas = tuple(float(value) for value in confidence_levels)
    thresholds = tuple(float(value) for value in threshold_quantiles)
    if not alphas or any(not 0.5 < value < 1.0 for value in alphas):
        raise ValueError("confidence_levels must lie in (0.5, 1)")
    if len(set(alphas)) != len(alphas):
        raise ValueError("confidence_levels must be unique")
    if not thresholds or any(not 0.5 < value < 1.0 for value in thresholds):
        raise ValueError("threshold_quantiles must lie in (0.5, 1)")
    if any(alpha <= max(thresholds) for alpha in alphas):
        raise ValueError("each confidence level must exceed every POT threshold quantile")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold_quantiles must be unique")
    if min_exceedances < 5:
        raise ValueError("min_exceedances must be at least 5")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(seed)
    fits = []
    for quantile in sorted(thresholds):
        threshold = float(
            np.quantile(values, quantile, method="inverted_cdf")
        )
        exceedances = values[values > threshold] - threshold
        count = int(exceedances.size)
        tail_probability = float(count / values.size)
        row = {
            "threshold_quantile": quantile,
            "threshold": threshold,
            "exceedance_count": count,
            "tail_probability": tail_probability,
            "sample_size": int(values.size),
            "confidence_levels": {},
            "fit": None,
            "warnings": [],
            "status": "not_fitted",
        }
        if count < min_exceedances:
            row["warnings"].append(
                f"only {count} exceedances; minimum required is {min_exceedances}"
            )
            fits.append(row)
            continue

        try:
            shape, _, scale = genpareto.fit(exceedances, floc=0.0)
            shape = float(shape)
            scale = float(scale)
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            row["warnings"].append(
                f"GPD maximum-likelihood fit failed: {type(exc).__name__}: {exc}"
            )
            fits.append(row)
            continue
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0.0:
            row["warnings"].append("GPD fit returned non-finite shape or invalid scale")
            fits.append(row)
            continue

        fitted_cdf = genpareto.cdf(exceedances, c=shape, loc=0.0, scale=scale)
        ks_distance = float(kstest(fitted_cdf, "uniform").statistic)
        # CDF-transform KS distance is descriptive. It is not given a p-value
        # because the parameters were estimated on these same exceedances.
        row["fit"] = {
            "shape": shape,
            "scale": scale,
            "location": 0.0,
            "ks_uniform_distance_descriptive": ks_distance,
            "mean_excess_empirical": float(np.mean(exceedances)),
            "mean_excess_fitted": (
                float(scale / (1.0 - shape)) if shape < 1.0 else None
            ),
            "finite_mean_condition": bool(shape < 1.0),
            "finite_variance_condition": bool(shape < 0.5),
        }
        if shape >= 1.0:
            row["warnings"].append(
                "fitted GPD shape >= 1; Expected Shortfall is infinite under this fit"
            )
        elif shape >= 0.5:
            row["warnings"].append(
                "fitted GPD shape >= 0.5; fitted tail has infinite variance"
            )
        if count < 50:
            row["warnings"].append(
                "fewer than 50 exceedances; fitted tail parameters may be unstable"
            )
        if shape < -0.5:
            row["warnings"].append(
                "shape below -0.5; likelihood regularity and endpoint sensitivity need review"
            )

        row["confidence_levels"] = _gpd_tail_quantiles(
            threshold, tail_probability, shape, scale, alphas
        )
        if n_bootstrap:
            row["bootstrap"] = _bootstrap_gpd_tail(
                exceedances,
                threshold,
                tail_probability,
                alphas,
                n_bootstrap,
                rng,
            )
        row["status"] = "fitted"
        fits.append(row)

    empirical = {
        str(alpha): _empirical_var_es(values, alpha)
        for alpha in alphas
    }
    stability = _threshold_stability(fits, alphas)
    return {
        "method": "Peaks-over-Threshold Generalized Pareto MLE",
        "sample_size": int(values.size),
        "loss_convention": "positive values represent losses",
        "confidence_levels": list(alphas),
        "threshold_quantiles": list(sorted(thresholds)),
        "empirical_var_es": empirical,
        "threshold_fits": fits,
        "threshold_stability": stability,
        "threshold_selection": (
            "No automatic threshold is selected. Assess stability of shape, scale, "
            "VaR and ES across thresholds; inspect the mean-excess behavior and tail "
            "fit diagnostics before choosing an operational threshold."
        ),
        "uncertainty_boundary": (
            "The GPD model is an extrapolation beyond an empirical threshold. Bootstrap "
            "intervals condition on the threshold and observed exceedance count. They exclude "
            "threshold-selection uncertainty, serial dependence, structural breaks, parameter "
            "uncertainty outside the sample, and model risk."
        ),
    }
