from __future__ import annotations
from math import erf, exp, log, sqrt, isfinite

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black_scholes_call(spot: float, strike: float, tau: float, rate: float,
                       volatility: float, dividend: float = 0.0) -> float:
    values=(spot,strike,tau,rate,volatility,dividend)
    if not all(isfinite(float(x)) for x in values):
        raise ValueError("Black-Scholes inputs must be finite")
    if spot<=0 or strike<=0 or tau<0 or volatility<0:
        raise ValueError("spot / strike must be positive; tau / volatility non-negative")
    if tau==0:
        return max(spot-strike,0.0)
    if volatility==0:
        return max(spot*exp(-dividend*tau)-strike*exp(-rate*tau),0.0)
    sig=volatility*sqrt(tau)
    d1=(log(spot/strike)+(rate-dividend+0.5*volatility**2)*tau)/sig
    d2=d1-sig
    return spot*exp(-dividend*tau)*_norm_cdf(d1)-strike*exp(-rate*tau)*_norm_cdf(d2)

def garman_kohlhagen_call(spot: float, strike: float, tau: float, domestic_rate: float,
                          foreign_rate: float, volatility: float) -> float:
    values=(spot,strike,tau,domestic_rate,foreign_rate,volatility)
    if not all(isfinite(float(x)) for x in values):
        raise ValueError("Garman-Kohlhagen inputs must be finite")
    if spot<=0 or strike<=0 or tau<0 or volatility<0:
        raise ValueError("spot / strike must be positive; tau / volatility non-negative")
    if tau==0:
        return max(spot-strike,0.0)
    if volatility==0:
        return max(spot*exp(-foreign_rate*tau)-strike*exp(-domestic_rate*tau),0.0)
    sig=volatility*sqrt(tau)
    d1=(log(spot/strike)+(domestic_rate-foreign_rate+0.5*volatility**2)*tau)/sig
    d2=d1-sig
    return spot*exp(-foreign_rate*tau)*_norm_cdf(d1)-strike*exp(-domestic_rate*tau)*_norm_cdf(d2)

def zero_coupon_bond(face_value: float, maturity: float, yield_percent: float) -> float:
    if not all(isfinite(float(x)) for x in (face_value,maturity,yield_percent)):
        raise ValueError("bond inputs must be finite")
    if maturity<0 or yield_percent<=-100:
        raise ValueError("invalid maturity or yield")
    return face_value*exp(-(yield_percent/100.0)*maturity)

def black_scholes_put(spot: float, strike: float, tau: float, rate: float,
                      volatility: float, dividend: float = 0.0) -> float:
    """European put from put-call parity, sharing the validated call implementation."""
    if tau < 0:
        raise ValueError("tau must be non-negative")
    call = black_scholes_call(spot, strike, tau, rate, volatility, dividend)
    return call - spot*exp(-dividend*tau) + strike*exp(-rate*tau)
