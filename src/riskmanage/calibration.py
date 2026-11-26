from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from .heston import HestonParameters, heston_prices_fourier_batch


@dataclass(frozen=True)
class HestonOptionQuote:
    """European option quote with prices expressed in the underlying's units."""

    strike: float
    maturity: float
    option_type: str
    mid_price: float
    bid: float | None = None
    ask: float | None = None
    weight: float = 1.0

    def __post_init__(self):
        values = (self.strike, self.maturity, self.mid_price, self.weight)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("quote fields must be finite")
        if self.strike <= 0.0 or self.maturity <= 0.0:
            raise ValueError("strike and maturity must be positive")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.mid_price < 0.0 or self.weight <= 0.0:
            raise ValueError("mid_price must be non-negative and weight positive")
        if (self.bid is None) != (self.ask is None):
            raise ValueError("bid and ask must be supplied together")
        if self.bid is not None:
            if not np.isfinite(self.bid) or not np.isfinite(self.ask):
                raise ValueError("bid and ask must be finite")
            if self.bid < 0.0 or self.ask < self.bid:
                raise ValueError("bid and ask must satisfy 0 <= bid <= ask")
            if not self.bid <= self.mid_price <= self.ask:
                raise ValueError("mid_price must lie inside the bid-ask interval")


def _encode_parameters(params: HestonParameters) -> np.ndarray:
    if params.vol_of_variance <= 0.0:
        raise ValueError("calibration requires positive volatility of variance")
    return np.array(
        [
            math.log(params.variance0),
            math.log(params.mean_reversion),
            math.log(params.long_run_variance),
            math.log(params.vol_of_variance),
            np.arctanh(params.correlation),
        ],
        dtype=float,
    )


def _decode_parameters(values: np.ndarray, market: HestonParameters) -> HestonParameters:
    return HestonParameters(
        spot=market.spot,
        variance0=float(np.exp(values[0])),
        mean_reversion=float(np.exp(values[1])),
        long_run_variance=float(np.exp(values[2])),
        vol_of_variance=float(np.exp(values[3])),
        correlation=float(np.tanh(values[4])),
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield,
    )


def _quote_error(model_price: float, quote: HestonOptionQuote, objective: str) -> float:
    if objective == "mid":
        return model_price - quote.mid_price
    if objective == "bid_ask":
        if quote.bid is None or quote.ask is None:
            raise ValueError("bid_ask objective requires bid and ask for every quote")
        if model_price < quote.bid:
            return model_price - quote.bid
        if model_price > quote.ask:
            return model_price - quote.ask
        return 0.0
    raise ValueError("objective must be 'mid' or 'bid_ask'")


def calibrate_heston(
    quotes: Sequence[HestonOptionQuote],
    market: HestonParameters,
    initial: HestonParameters,
    *,
    objective: str = "bid_ask",
    loss: str = "soft_l1",
    n_starts: int = 5,
    max_nfev: int = 120,
    seed: int = 20261125,
    min_price_scale: float = 0.01,
) -> dict:
    """Calibrate the five risk-neutral Heston parameters to an option panel.

    Positive parameters use logarithmic coordinates and correlation uses
    Fisher-z coordinates. A same-expiry quote panel is priced by one vector
    Lewis quadrature. Bid-ask residuals are zero inside the quoted interval and
    signed distance outside. The transformed residual Jacobian is an
    identifiability diagnostic, not a parameter covariance matrix.
    """
    quotes = tuple(quotes)
    if len(quotes) < 6:
        raise ValueError("at least six option quotes are required for five parameters")
    if len({round(q.maturity, 10) for q in quotes}) < 2:
        raise ValueError("quotes must cover at least two distinct maturities")
    if len({round(q.strike, 10) for q in quotes}) < 3:
        raise ValueError("quotes must cover at least three distinct strikes")
    if objective not in {"mid", "bid_ask"}:
        raise ValueError("objective must be 'mid' or 'bid_ask'")
    if objective == "bid_ask" and any(q.bid is None for q in quotes):
        raise ValueError("bid_ask objective requires bid and ask for every quote")
    if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise ValueError("unsupported least-squares loss")
    if n_starts < 1 or max_nfev < 1:
        raise ValueError("n_starts and max_nfev must be positive")
    if not np.isfinite(min_price_scale) or min_price_scale <= 0.0:
        raise ValueError("min_price_scale must be finite and positive")

    x0 = _encode_parameters(initial)
    lower = np.array(
        [
            math.log(1e-5),
            math.log(0.01),
            math.log(1e-5),
            math.log(0.01),
            np.arctanh(-0.999),
        ]
    )
    upper = np.array(
        [
            math.log(2.0),
            math.log(20.0),
            math.log(2.0),
            math.log(5.0),
            np.arctanh(0.999),
        ]
    )
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError("initial parameters lie outside calibration bounds")

    price_scales = np.array(
        [
            max(0.5 * (q.ask - q.bid), min_price_scale)
            if q.bid is not None
            else min_price_scale
            for q in quotes
        ],
        dtype=float,
    )
    residual_weights = np.array([q.weight for q in quotes], dtype=float) / price_scales

    maturity_groups: dict[float, list[int]] = {}
    for index, quote in enumerate(quotes):
        maturity_groups.setdefault(quote.maturity, []).append(index)

    def model_panel(params: HestonParameters) -> np.ndarray:
        prices = np.empty(len(quotes), dtype=float)
        for maturity, indices in maturity_groups.items():
            batch = heston_prices_fourier_batch(
                [quotes[index].strike for index in indices],
                maturity,
                params,
                [quotes[index].option_type for index in indices],
            )
            prices[indices] = batch["prices"]
        return prices

    evaluations = 0

    def residuals(values: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        params = _decode_parameters(values, market)
        model_prices = model_panel(params)
        output = np.array(
            [
                residual_weights[index]
                * _quote_error(model_prices[index], quote, objective)
                for index, quote in enumerate(quotes)
            ],
            dtype=float,
        )
        if not np.isfinite(output).all():
            raise ArithmeticError("Heston calibration residual became non-finite")
        return output

    rng = np.random.default_rng(seed)
    perturbation = np.array([0.45, 0.60, 0.45, 0.50, 0.55])
    starts = [x0]
    for _ in range(1, n_starts):
        candidate = x0 + rng.normal(size=5) * perturbation
        starts.append(np.clip(candidate, lower + 1e-10, upper - 1e-10))

    trials = []
    best = None
    for start in starts:
        result = least_squares(
            residuals,
            start,
            bounds=(lower, upper),
            method="trf",
            jac="3-point",
            x_scale="jac",
            loss=loss,
            f_scale=1.0,
            max_nfev=max_nfev,
        )
        trials.append(
            {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
            }
        )
        if best is None or result.cost < best.cost:
            best = result

    if best is None:
        raise ArithmeticError("Heston calibration did not produce an optimizer result")

    calibrated = _decode_parameters(best.x, market)
    model_prices = model_panel(calibrated)
    mids = np.array([q.mid_price for q in quotes], dtype=float)
    errors = model_prices - mids
    bid_ask_quotes = [q for q in quotes if q.bid is not None]
    bid_ask_hit_ratio = (
        float(
            np.mean(
                [
                    q.bid <= price <= q.ask
                    for q, price in zip(quotes, model_prices, strict=True)
                    if q.bid is not None
                ]
            )
        )
        if bid_ask_quotes
        else None
    )

    singular_values = np.linalg.svd(best.jac, compute_uv=False)
    tolerance = (
        np.finfo(float).eps * max(best.jac.shape) * singular_values[0]
        if singular_values.size
        else math.inf
    )
    rank = int(np.sum(singular_values > tolerance))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > tolerance
        else math.inf
    )
    quote_diagnostics = [
        {
            "strike": quote.strike,
            "maturity": quote.maturity,
            "option_type": quote.option_type,
            "market_mid": quote.mid_price,
            "bid": quote.bid,
            "ask": quote.ask,
            "model_price": float(price),
            "model_minus_mid": float(error),
            "inside_bid_ask": (
                bool(quote.bid <= price <= quote.ask) if quote.bid is not None else None
            ),
        }
        for quote, price, error in zip(quotes, model_prices, errors, strict=True)
    ]
    return {
        "parameters": {
            "variance0": calibrated.variance0,
            "mean_reversion": calibrated.mean_reversion,
            "long_run_variance": calibrated.long_run_variance,
            "vol_of_variance": calibrated.vol_of_variance,
            "correlation": calibrated.correlation,
            "feller_ratio": calibrated.feller_ratio,
            "feller_condition_satisfied": calibrated.feller_condition_satisfied,
        },
        "objective": objective,
        "loss": loss,
        "success": bool(best.success),
        "status": int(best.status),
        "message": str(best.message),
        "cost": float(best.cost),
        "optimality": float(best.optimality),
        "function_evaluations": int(evaluations),
        "nfev_best_start": int(best.nfev),
        "n_starts": int(n_starts),
        "seed": int(seed),
        "price_rmse": float(np.sqrt(np.mean(errors**2))),
        "price_mae": float(np.mean(np.abs(errors))),
        "maximum_absolute_price_error": float(np.max(np.abs(errors))),
        "bid_ask_hit_ratio": bid_ask_hit_ratio,
        "jacobian_singular_values": [float(value) for value in singular_values],
        "jacobian_rank": rank,
        "jacobian_condition_number_transformed_coordinates": condition_number,
        "jacobian_coordinate_system": [
            "log_variance0",
            "log_mean_reversion",
            "log_long_run_variance",
            "log_vol_of_variance",
            "atanh_correlation",
        ],
        "starts": trials,
        "quotes": quote_diagnostics,
        "uncertainty_boundary": (
            "The residual Jacobian SVD is a local identifiability diagnostic, not a parameter "
            "confidence interval. Parameter uncertainty requires profile likelihood, bootstrap "
            "or another explicit uncertainty model."
        ),
    }
