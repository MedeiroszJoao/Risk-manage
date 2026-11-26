from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import chi2

from .calibration import (
    HestonOptionQuote,
    _decode_parameters,
    _encode_parameters,
    calibrate_heston,
)
from .heston import HestonParameters, heston_prices_fourier_batch


_PARAMETER_NAMES = (
    "variance0",
    "mean_reversion",
    "long_run_variance",
    "vol_of_variance",
    "correlation",
)
_PARAMETER_INDEX = {name: index for index, name in enumerate(_PARAMETER_NAMES)}
_LOWER = np.array(
    [
        math.log(1e-5),
        math.log(0.01),
        math.log(1e-5),
        math.log(0.01),
        np.arctanh(-0.999),
    ],
    dtype=float,
)
_UPPER = np.array(
    [
        math.log(2.0),
        math.log(20.0),
        math.log(2.0),
        math.log(5.0),
        np.arctanh(0.999),
    ],
    dtype=float,
)


def _quote_panel_prices(
    quotes: Sequence[HestonOptionQuote], params: HestonParameters
) -> np.ndarray:
    prices = np.empty(len(quotes), dtype=float)
    groups: dict[float, list[int]] = {}
    for index, quote in enumerate(quotes):
        groups.setdefault(quote.maturity, []).append(index)
    for maturity, indices in groups.items():
        result = heston_prices_fourier_batch(
            [quotes[index].strike for index in indices],
            maturity,
            params,
            [quotes[index].option_type for index in indices],
        )
        prices[indices] = result["prices"]
    return prices


def _quote_noise_scales(
    quotes: Sequence[HestonOptionQuote], min_price_scale: float
) -> np.ndarray:
    if not quotes:
        raise ValueError("at least one quote is required")
    if not np.isfinite(min_price_scale) or min_price_scale <= 0.0:
        raise ValueError("min_price_scale must be finite and positive")
    if any(quote.bid is None or quote.ask is None for quote in quotes):
        raise ValueError(
            "parameter uncertainty requires bid and ask on every quote"
        )
    half_spreads = np.array(
        [0.5 * (quote.ask - quote.bid) for quote in quotes], dtype=float
    )
    # A uniform latent price within the quoted spread has variance spread^2/12.
    # min_price_scale is the same price-scale floor used by the calibrator.
    return np.maximum(half_spreads, min_price_scale) / math.sqrt(3.0)


def _validate_market_conventions(
    market: HestonParameters, fitted: HestonParameters
) -> None:
    if not np.isclose(market.spot, fitted.spot, rtol=0.0, atol=1e-12):
        raise ValueError("market and fitted spot must agree")
    if not np.isclose(
        market.risk_free_rate, fitted.risk_free_rate, rtol=0.0, atol=1e-12
    ):
        raise ValueError("market and fitted risk-free rates must agree")
    if not np.isclose(
        market.dividend_yield, fitted.dividend_yield, rtol=0.0, atol=1e-12
    ):
        raise ValueError("market and fitted dividend yields must agree")


def _parameter_dict(params: HestonParameters) -> dict:
    return {
        "variance0": float(params.variance0),
        "mean_reversion": float(params.mean_reversion),
        "long_run_variance": float(params.long_run_variance),
        "vol_of_variance": float(params.vol_of_variance),
        "correlation": float(params.correlation),
    }


def _json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [
            float(value) if np.isfinite(value) else None
            for value in row
        ]
        for row in matrix
    ]


def _distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite observations are required")
    quantiles = np.quantile(values, [0.025, 0.05, 0.50, 0.95, 0.975])
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "quantiles": {
            "p2_5": float(quantiles[0]),
            "p5": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p95": float(quantiles[3]),
            "p97_5": float(quantiles[4]),
        },
    }


def bootstrap_heston_calibration(
    quotes: Sequence[HestonOptionQuote],
    market: HestonParameters,
    fitted_parameters: HestonParameters,
    *,
    n_bootstrap: int = 100,
    seed: int = 20261126,
    noise_scale: float = 1.0,
    min_price_scale: float = 0.01,
    n_starts: int = 1,
    max_nfev: int = 100,
) -> dict:
    """Parametric quote bootstrap conditional on a fitted Heston model.

    Each synthetic mid is generated as fitted Fourier price plus Gaussian quote
    noise. The noise standard deviation is (half-spread)/sqrt(3), equivalent to
    spread/sqrt(12), with a positive floor. Each bootstrap panel is refitted by
    weighted mid-price least squares. This quantifies parameter uncertainty
    conditional on the chosen quote-noise model and fixed quote design; it is
    not a market-data or model-risk confidence interval.
    """
    quotes = tuple(quotes)
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(noise_scale) or noise_scale <= 0.0:
        raise ValueError("noise_scale must be finite and positive")
    if not np.isfinite(min_price_scale) or min_price_scale <= 0.0:
        raise ValueError("min_price_scale must be finite and positive")
    if n_starts < 1 or max_nfev < 1:
        raise ValueError("n_starts and max_nfev must be positive")
    if fitted_parameters.vol_of_variance <= 0.0:
        raise ValueError("fitted vol_of_variance must be positive for calibration")
    _validate_market_conventions(market, fitted_parameters)

    noise_sd = _quote_noise_scales(quotes, min_price_scale) * noise_scale
    fitted_prices = _quote_panel_prices(quotes, fitted_parameters)
    rng = np.random.default_rng(seed)
    samples = []
    failures = []
    costs = []

    for replicate in range(n_bootstrap):
        simulated_mids = np.maximum(
            fitted_prices + rng.normal(0.0, noise_sd), 1e-10
        )
        bootstrap_quotes = []
        for quote, mid in zip(quotes, simulated_mids, strict=True):
            half_spread = max(0.5 * (quote.ask - quote.bid), min_price_scale)
            bootstrap_quotes.append(
                HestonOptionQuote(
                    strike=quote.strike,
                    maturity=quote.maturity,
                    option_type=quote.option_type,
                    mid_price=float(mid),
                    bid=max(0.0, float(mid) - half_spread),
                    ask=float(mid) + half_spread,
                    weight=quote.weight,
                )
            )
        try:
            fit = calibrate_heston(
                bootstrap_quotes,
                market,
                fitted_parameters,
                objective="mid",
                loss="linear",
                n_starts=n_starts,
                max_nfev=max_nfev,
                seed=seed + replicate + 1,
                min_price_scale=min_price_scale,
            )
            if not fit["success"]:
                failures.append(
                    {
                        "replicate": replicate,
                        "reason": "optimizer did not report convergence",
                        "message": fit["message"],
                    }
                )
                continue
            parameters = fit["parameters"]
            values = np.array(
                [parameters[name] for name in _PARAMETER_NAMES], dtype=float
            )
            if not np.isfinite(values).all():
                failures.append(
                    {"replicate": replicate, "reason": "non-finite parameter estimate"}
                )
                continue
            samples.append({name: float(parameters[name]) for name in _PARAMETER_NAMES})
            costs.append(float(fit["cost"]))
        except (ArithmeticError, FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(
                {
                    "replicate": replicate,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )

    if len(samples) < 2:
        raise RuntimeError(
            f"only {len(samples)} successful bootstrap fits out of {n_bootstrap}"
        )

    natural = np.array(
        [[sample[name] for name in _PARAMETER_NAMES] for sample in samples],
        dtype=float,
    )
    transformed = np.column_stack(
        (
            np.log(natural[:, 0]),
            np.log(natural[:, 1]),
            np.log(natural[:, 2]),
            np.log(natural[:, 3]),
            np.arctanh(natural[:, 4]),
        )
    )
    natural_covariance = np.cov(natural, rowvar=False, ddof=1)
    transformed_covariance = np.cov(transformed, rowvar=False, ddof=1)
    natural_correlation = np.corrcoef(natural, rowvar=False)
    transformed_correlation = np.corrcoef(transformed, rowvar=False)
    summaries = {
        name: _distribution_summary(natural[:, index])
        for index, name in enumerate(_PARAMETER_NAMES)
    }
    return {
        "method": "parametric Gaussian quote bootstrap with spread-derived noise",
        "seed": int(seed),
        "requested_replicates": int(n_bootstrap),
        "successful_replicates": int(len(samples)),
        "failed_replicates": int(len(failures)),
        "success_rate": float(len(samples) / n_bootstrap),
        "noise_model": (
            "Independent Gaussian quote errors with standard deviation "
            "max(half_bid_ask_spread, min_price_scale)/sqrt(3); this matches "
            "the variance of a uniform error over the quoted spread."
        ),
        "noise_scale_multiplier": float(noise_scale),
        "min_price_scale": float(min_price_scale),
        "quote_noise_standard_deviations": [float(value) for value in noise_sd],
        "fitted_parameters_used_as_bootstrap_generator": _parameter_dict(fitted_parameters),
        "parameter_summaries": summaries,
        "natural_parameter_covariance": _json_matrix(natural_covariance),
        "natural_parameter_correlation": _json_matrix(natural_correlation),
        "transformed_parameter_covariance": _json_matrix(transformed_covariance),
        "transformed_parameter_correlation": _json_matrix(transformed_correlation),
        "parameter_order": list(_PARAMETER_NAMES),
        "bootstrap_parameter_samples": samples,
        "optimizer_cost_summary": (
            _distribution_summary(np.asarray(costs, dtype=float))
            if len(costs) >= 2
            else None
        ),
        "failures": failures,
        "uncertainty_boundary": (
            "Intervals and covariance are conditional on the fitted Heston model, fixed quote "
            "panel, Gaussian spread-derived quote-noise model, fixed curves/carry and optimizer "
            "bounds. They exclude model misspecification, quote dependence, stale/crossed markets, "
            "curve uncertainty and physical-measure parameter uncertainty."
        ),
    }


def profile_heston_parameter(
    quotes: Sequence[HestonOptionQuote],
    market: HestonParameters,
    fitted_parameters: HestonParameters,
    parameter_name: str,
    parameter_grid: Sequence[float],
    *,
    confidence_level: float = 0.95,
    min_price_scale: float = 0.01,
    max_nfev: int = 120,
) -> dict:
    """Compute a one-parameter Gaussian profile-likelihood curve.

    One transformed parameter is fixed at each supplied physical-scale grid
    value; the remaining four parameters are re-optimized. Quote errors are
    assumed independent Gaussian with spread-derived standard deviations.
    The chi-square(1) likelihood-ratio cutoff is asymptotic and only approximate
    for a small, bounded and possibly weakly identified nonlinear model.
    """
    quotes = tuple(quotes)
    if parameter_name not in _PARAMETER_INDEX:
        raise ValueError(f"unknown Heston parameter: {parameter_name}")
    grid = np.asarray(parameter_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or not np.isfinite(grid).all():
        raise ValueError("parameter_grid must contain at least three finite values")
    if np.unique(grid).size != grid.size:
        raise ValueError("parameter_grid values must be unique")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("parameter_grid must be strictly increasing")
    _validate_market_conventions(market, fitted_parameters)
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive")

    noise_sd = _quote_noise_scales(quotes, min_price_scale)
    weights = np.array([quote.weight for quote in quotes], dtype=float)
    residual_scale = noise_sd / weights
    mids = np.array([quote.mid_price for quote in quotes], dtype=float)
    x_full = _encode_parameters(fitted_parameters)
    fixed_index = _PARAMETER_INDEX[parameter_name]
    free_indices = np.array(
        [index for index in range(5) if index != fixed_index], dtype=int
    )

    def transform_physical(value: float) -> float:
        if parameter_name == "correlation":
            if not -0.999 < value < 0.999:
                raise ValueError("correlation grid must lie inside (-0.999, 0.999)")
            return float(np.arctanh(value))
        if value <= 0.0:
            raise ValueError("positive Heston parameter grid values must be positive")
        return math.log(value)

    def residuals(free_values: np.ndarray, fixed_value: float) -> np.ndarray:
        candidate = x_full.copy()
        candidate[fixed_index] = fixed_value
        candidate[free_indices] = free_values
        params = _decode_parameters(candidate, market)
        model_prices = _quote_panel_prices(quotes, params)
        values = (model_prices - mids) / residual_scale
        if not np.isfinite(values).all():
            raise ArithmeticError("profile-likelihood residual became non-finite")
        return values

    points = []
    start_free = x_full[free_indices].copy()
    for physical_value in grid:
        fixed_value = transform_physical(float(physical_value))
        if not _LOWER[fixed_index] <= fixed_value <= _UPPER[fixed_index]:
            raise ValueError(f"{parameter_name} grid lies outside calibration bounds")
        result = least_squares(
            lambda values: residuals(values, fixed_value),
            start_free,
            bounds=(_LOWER[free_indices], _UPPER[free_indices]),
            method="trf",
            jac="3-point",
            x_scale="jac",
            loss="linear",
            max_nfev=max_nfev,
        )
        chi_square = float(np.dot(result.fun, result.fun))
        profiled_coordinates = x_full.copy()
        profiled_coordinates[fixed_index] = fixed_value
        profiled_coordinates[free_indices] = result.x
        profiled_parameters = _decode_parameters(profiled_coordinates, market)
        points.append(
            {
                "parameter_value": float(physical_value),
                "chi_square": chi_square,
                "optimizer_success": bool(result.success),
                "nfev": int(result.nfev),
                "profiled_parameters": _parameter_dict(profiled_parameters),
            }
        )
        if result.success:
            start_free = result.x

    minimum = min(point["chi_square"] for point in points)
    cutoff = float(chi2.ppf(confidence_level, df=1))
    for point in points:
        point["delta_chi_square"] = max(0.0, point["chi_square"] - minimum)
        point["inside_likelihood_ratio_set"] = (
            point["delta_chi_square"] <= cutoff
        )

    accepted = [
        point["parameter_value"]
        for point in points
        if point["inside_likelihood_ratio_set"]
    ]
    interval = (
        [float(min(accepted)), float(max(accepted))]
        if accepted
        else None
    )
    accepted_indices = [
        index for index, point in enumerate(points)
        if point["inside_likelihood_ratio_set"]
    ]
    accepted_segments = []
    if accepted_indices:
        segment_start = accepted_indices[0]
        previous = accepted_indices[0]
        for index in accepted_indices[1:]:
            if index != previous + 1:
                accepted_segments.append(
                    [float(grid[segment_start]), float(grid[previous])]
                )
                segment_start = index
            previous = index
        accepted_segments.append(
            [float(grid[segment_start]), float(grid[previous])]
        )
    grid_truncated = bool(
        accepted
        and (
            min(accepted) == float(np.min(grid))
            or max(accepted) == float(np.max(grid))
        )
    )

    # Refine each accepted profile segment by solving delta-chi-square = cutoff.
    # Nuisance parameters are re-optimized at every root evaluation.
    profile_cache: dict[float, float] = {}

    def profile_chi_square(value: float, start_free: np.ndarray) -> float:
        key = float(value)
        if key in profile_cache:
            return profile_cache[key]
        fixed_value = transform_physical(key)
        result = least_squares(
            lambda free_values: residuals(free_values, fixed_value),
            start_free.copy(),
            bounds=(_LOWER[free_indices], _UPPER[free_indices]),
            method="trf",
            jac="3-point",
            x_scale="jac",
            loss="linear",
            max_nfev=max_nfev,
        )
        if not result.success:
            raise RuntimeError(
                f"profile nuisance optimization failed at {parameter_name}={key}: "
                f"{result.message}"
            )
        statistic = float(np.dot(result.fun, result.fun))
        profile_cache[key] = statistic
        return statistic

    refined_segments = []
    for segment_start, segment_end in accepted_segments:
        start_index = int(np.flatnonzero(grid == segment_start)[0])
        end_index = int(np.flatnonzero(grid == segment_end)[0])
        start_parameters = replace(
            market, **points[start_index]["profiled_parameters"]
        )
        end_parameters = replace(
            market, **points[end_index]["profiled_parameters"]
        )
        start_free = _encode_parameters(start_parameters)[free_indices]
        end_free = _encode_parameters(end_parameters)[free_indices]

        if start_index > 0:
            lower_root = brentq(
                lambda value: profile_chi_square(value, start_free)
                - minimum
                - cutoff,
                float(grid[start_index - 1]),
                float(grid[start_index]),
                xtol=1e-7,
                rtol=1e-8,
                maxiter=60,
            )
            lower_value = float(lower_root)
            lower_is_truncated = False
        else:
            lower_value = float(grid[0])
            lower_is_truncated = True

        if end_index < grid.size - 1:
            upper_root = brentq(
                lambda value: profile_chi_square(value, end_free)
                - minimum
                - cutoff,
                float(grid[end_index]),
                float(grid[end_index + 1]),
                xtol=1e-7,
                rtol=1e-8,
                maxiter=60,
            )
            upper_value = float(upper_root)
            upper_is_truncated = False
        else:
            upper_value = float(grid[-1])
            upper_is_truncated = True

        refined_segments.append(
            {
                "lower": lower_value,
                "upper": upper_value,
                "lower_truncated": lower_is_truncated,
                "upper_truncated": upper_is_truncated,
            }
        )

    refined_interval = (
        [refined_segments[0]["lower"], refined_segments[0]["upper"]]
        if len(refined_segments) == 1
        else None
    )
    return {
        "parameter": parameter_name,
        "parameter_grid": [float(value) for value in grid],
        "confidence_level": float(confidence_level),
        "likelihood_ratio_degrees_of_freedom": 1,
        "likelihood_ratio_cutoff_delta_chi_square": cutoff,
        "minimum_profile_chi_square": float(minimum),
        "profile_interval_on_grid": interval,
        "profile_interval_refined": refined_interval,
        "refined_profile_segments": refined_segments,
        "accepted_grid_values": [float(value) for value in accepted],
        "accepted_grid_segments": accepted_segments,
        "profile_root_evaluations": int(len(profile_cache)),
        "grid_truncated_at_boundary": grid_truncated,
        "points": points,
        "noise_model": (
            "Independent Gaussian quote errors with standard deviation "
            "max(half_bid_ask_spread, min_price_scale)/sqrt(3), scaled by quote weight."
        ),
        "interpretation_boundary": (
            "The interval is a grid approximation using an asymptotic chi-square(1) cutoff. "
            "Grid truncation means an endpoint is unresolved. It is conditional on the quote-noise "
            "model, fixed curve/carry inputs and Heston model form; it is not a model-risk interval."
        ),
    }


def propagate_heston_parameter_uncertainty(
    parameter_samples: Sequence[dict],
    market: HestonParameters,
    contracts: Sequence[dict],
    *,
    spot_bump_relative: float = 0.0025,
) -> dict:
    """Propagate bootstrap Heston parameter draws into European prices and Deltas.

    Each parameter draw is repriced with Lewis vector Fourier quadrature. Delta
    is the central spot finite difference using the same parameter draw. Price
    and Delta quantiles therefore reflect fitted-parameter variation only;
    they do not include numerical integration error, bump bias or model risk.
    """
    parameter_samples = tuple(parameter_samples)
    contracts = tuple(contracts)
    if len(parameter_samples) < 2:
        raise ValueError("at least two bootstrap parameter samples are required")
    if not contracts:
        raise ValueError("at least one contract is required")
    if not np.isfinite(spot_bump_relative) or not 0.0 < spot_bump_relative < 0.05:
        raise ValueError("spot_bump_relative must lie in (0, 0.05)")

    contract_groups: dict[tuple[float, str], list[tuple[int, dict]]] = {}
    for index, contract in enumerate(contracts):
        strike = float(contract["strike"])
        maturity = float(contract["maturity"])
        option_type = contract["option_type"]
        if not np.isfinite(strike) or strike <= 0.0:
            raise ValueError("contract strikes must be finite and positive")
        if not np.isfinite(maturity) or maturity <= 0.0:
            raise ValueError("contract maturities must be finite and positive")
        if option_type not in {"call", "put"}:
            raise ValueError("contract option_type must be 'call' or 'put'")
        contract_groups.setdefault((maturity, option_type), []).append((index, contract))

    price_draws = np.empty((len(parameter_samples), len(contracts)), dtype=float)
    delta_draws = np.empty_like(price_draws)
    bump = market.spot * spot_bump_relative
    for sample_index, sample in enumerate(parameter_samples):
        try:
            params = replace(
                market,
                variance0=float(sample["variance0"]),
                mean_reversion=float(sample["mean_reversion"]),
                long_run_variance=float(sample["long_run_variance"]),
                vol_of_variance=float(sample["vol_of_variance"]),
                correlation=float(sample["correlation"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Heston parameter sample: {exc}") from exc

        for (maturity, option_type), indexed_contracts in contract_groups.items():
            strikes = [float(contract["strike"]) for _, contract in indexed_contracts]
            types = [option_type] * len(strikes)
            base = heston_prices_fourier_batch(
                strikes, maturity, params, types
            )["prices"]
            up = heston_prices_fourier_batch(
                strikes,
                maturity,
                replace(params, spot=params.spot + bump),
                types,
            )["prices"]
            down = heston_prices_fourier_batch(
                strikes,
                maturity,
                replace(params, spot=params.spot - bump),
                types,
            )["prices"]
            deltas = (up - down) / (2.0 * bump)
            for local_index, (global_index, _) in enumerate(indexed_contracts):
                price_draws[sample_index, global_index] = base[local_index]
                delta_draws[sample_index, global_index] = deltas[local_index]

    results = []
    for index, contract in enumerate(contracts):
        results.append(
            {
                "contract": {
                    "strike": float(contract["strike"]),
                    "maturity": float(contract["maturity"]),
                    "option_type": contract["option_type"],
                },
                "price_distribution": _distribution_summary(price_draws[:, index]),
                "delta_distribution": _distribution_summary(delta_draws[:, index]),
                "price_delta_correlation": float(
                    np.corrcoef(price_draws[:, index], delta_draws[:, index])[0, 1]
                )
                if np.std(price_draws[:, index]) > 0.0
                and np.std(delta_draws[:, index]) > 0.0
                else None,
            }
        )

    return {
        "bootstrap_sample_count": int(len(parameter_samples)),
        "contract_count": int(len(contracts)),
        "spot_bump_relative": float(spot_bump_relative),
        "contracts": results,
        "uncertainty_boundary": (
            "Price and Delta distributions propagate only the bootstrap parameter draws under "
            "the fixed Heston model form and fixed market conventions. They exclude quote-noise "
            "uncertainty beyond the bootstrap model, Fourier quadrature error, finite-difference "
            "bump bias, time-discretization bias, model risk and physical-measure uncertainty."
        ),
    }
