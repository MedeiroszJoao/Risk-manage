from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def _float_array(value, name):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scalar(value):
    return float(value) if value.ndim == 0 else value


def black76_price(forward, strike, maturity, discount_factor, volatility, option_type="call"):
    """European Black (1976) option price; numeric inputs support NumPy broadcasting."""
    f = _float_array(forward, "forward")
    k = _float_array(strike, "strike")
    t = _float_array(maturity, "maturity")
    df = _float_array(discount_factor, "discount_factor")
    sigma = _float_array(volatility, "volatility")
    kind = np.asarray(option_type, dtype=str)
    f, k, t, df, sigma, kind = np.broadcast_arrays(f, k, t, df, sigma, kind)

    if np.any(f <= 0.0) or np.any(k <= 0.0):
        raise ValueError("forward and strike must be strictly positive")
    if np.any(t < 0.0) or np.any(df <= 0.0) or np.any(sigma < 0.0):
        raise ValueError("maturity, discount factor or volatility is invalid")
    if not np.isin(kind, ("call", "put")).all():
        raise ValueError("option_type must contain only 'call' or 'put'")

    total_std = sigma * np.sqrt(t)
    deterministic = (t == 0.0) | (sigma == 0.0)
    safe_std = np.where(deterministic, 1.0, total_std)
    d1 = (np.log(f / k) + 0.5 * total_std**2) / safe_std
    d2 = d1 - total_std
    calls = df * (f * ndtr(d1) - k * ndtr(d2))
    puts = df * (k * ndtr(-d2) - f * ndtr(-d1))
    intrinsic = np.where(kind == "call", df * np.maximum(f-k, 0.0),
                         df * np.maximum(k-f, 0.0))
    result = np.where(deterministic, intrinsic, np.where(kind == "call", calls, puts))
    if not np.isfinite(result).all():
        raise ArithmeticError("Black-76 price is non-finite")
    return _scalar(result)


def black76_vega(forward, strike, maturity, discount_factor, volatility):
    """Analytic derivative of Black-76 price with respect to annualized volatility."""
    f = _float_array(forward, "forward")
    k = _float_array(strike, "strike")
    t = _float_array(maturity, "maturity")
    df = _float_array(discount_factor, "discount_factor")
    sigma = _float_array(volatility, "volatility")
    f, k, t, df, sigma = np.broadcast_arrays(f, k, t, df, sigma)
    if np.any(f <= 0.0) or np.any(k <= 0.0):
        raise ValueError("forward and strike must be strictly positive")
    if np.any(t < 0.0) or np.any(df <= 0.0) or np.any(sigma < 0.0):
        raise ValueError("maturity, discount factor or volatility is invalid")
    total_std = sigma * np.sqrt(t)
    safe_std = np.where(total_std > 0.0, total_std, 1.0)
    d1 = (np.log(f / k) + 0.5 * total_std**2) / safe_std
    density = np.exp(-0.5 * d1**2) / np.sqrt(2.0*np.pi)
    result = np.where(total_std > 0.0, df*f*np.sqrt(t)*density, 0.0)
    return _scalar(result)


def black76_implied_volatility(
    market_price, forward, strike, maturity, discount_factor,
    option_type="call", price_tolerance=1e-10, max_volatility=16.0,
):
    """Invert a Black-76 quote with no-arbitrage bounds and a bracketed Brent solver."""
    values = (market_price, forward, strike, maturity, discount_factor,
              price_tolerance, max_volatility)
    if not all(np.isfinite(float(x)) for x in values):
        raise ValueError("implied-volatility inputs must be finite")
    if forward <= 0.0 or strike <= 0.0 or maturity <= 0.0 or discount_factor <= 0.0:
        raise ValueError("forward, strike, maturity and discount factor must be positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if price_tolerance < 0.0 or max_volatility <= 0.0:
        raise ValueError("price_tolerance must be non-negative and max_volatility positive")

    if option_type == "call":
        lower = discount_factor * max(forward-strike, 0.0)
        upper = discount_factor * forward
    else:
        lower = discount_factor * max(strike-forward, 0.0)
        upper = discount_factor * strike
    if market_price < lower-price_tolerance:
        raise ValueError("market option price is below its no-arbitrage lower bound")
    if market_price >= upper-price_tolerance:
        raise ValueError("market option price is at or too close to its upper bound for finite IV")
    if market_price <= lower+price_tolerance:
        return 0.0

    def objective(vol):
        return black76_price(forward, strike, maturity, discount_factor, vol, option_type)-market_price

    high = min(0.5, max_volatility)
    while objective(high) < 0.0 and high < max_volatility:
        high = min(2.0*high, max_volatility)
    if objective(high) < 0.0:
        raise ValueError("could not bracket IV below max_volatility; increase the upper bound")
    return float(brentq(objective, 0.0, high, xtol=1e-12,
                        rtol=4.0*np.finfo(float).eps, maxiter=200))
