
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import t as student_t


@dataclass(frozen=True)
class HestonParameters:
    """Risk-neutral Heston parameters in annualized units.

    dS/S = (r-q)dt + sqrt(v)dW_S
    dv   = kappa(theta-v)dt + vol_of_variance*sqrt(v)dW_v
    corr(dW_S,dW_v) = rho.
    """

    spot: float
    variance0: float
    mean_reversion: float
    long_run_variance: float
    vol_of_variance: float
    correlation: float
    risk_free_rate: float
    dividend_yield: float = 0.0

    def __post_init__(self):
        values = (
            self.spot,
            self.variance0,
            self.mean_reversion,
            self.long_run_variance,
            self.vol_of_variance,
            self.correlation,
            self.risk_free_rate,
            self.dividend_yield,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("all Heston parameters must be finite")
        if self.spot <= 0.0 or self.variance0 < 0.0:
            raise ValueError("spot must be positive and initial variance non-negative")
        if self.mean_reversion <= 0.0 or self.long_run_variance <= 0.0:
            raise ValueError("mean reversion and long-run variance must be positive")
        if self.vol_of_variance < 0.0:
            raise ValueError("volatility of variance must be non-negative")
        if not -1.0 < self.correlation < 1.0:
            raise ValueError("correlation must lie strictly between -1 and 1")

    @property
    def feller_ratio(self) -> float:
        if self.vol_of_variance == 0.0:
            return math.inf
        return 2.0 * self.mean_reversion * self.long_run_variance / self.vol_of_variance**2

    @property
    def feller_condition_satisfied(self) -> bool:
        return self.feller_ratio >= 1.0


def heston_characteristic_function(frequency, maturity: float, params: HestonParameters):
    """Characteristic function E^Q[exp(i*u*log(S_T))].

    Uses the stable g representation of the affine Heston solution. The
    deterministic-variance limit (vol_of_variance=0) is handled analytically.
    """
    if not np.isfinite(maturity) or maturity < 0.0:
        raise ValueError("maturity must be finite and non-negative")
    u = np.asarray(frequency, dtype=np.complex128)
    if not np.isfinite(u.real).all() or not np.isfinite(u.imag).all():
        raise ValueError("frequency must be finite")
    if maturity == 0.0:
        result = np.exp(1j * u * np.log(params.spot))
        return complex(result) if result.ndim == 0 else result

    kappa = params.mean_reversion
    theta = params.long_run_variance
    sigma = params.vol_of_variance
    rho = params.correlation
    v0 = params.variance0
    log_forward_spot = np.log(params.spot) + (
        params.risk_free_rate - params.dividend_yield
    ) * maturity

    if sigma == 0.0:
        integrated_variance = (
            theta * maturity
            + (v0 - theta) * (-np.expm1(-kappa * maturity)) / kappa
        )
        result = np.exp(
            1j * u * log_forward_spot
            - 0.5 * (u * u + 1j * u) * integrated_variance
        )
        return complex(result) if result.ndim == 0 else result

    iu = 1j * u
    beta = kappa - rho * sigma * iu
    discriminant = np.sqrt(beta * beta + sigma * sigma * (u * u + iu))
    d = discriminant
    g = (beta - d) / (beta + d)
    exp_minus_dt = np.exp(-d * maturity)
    one_minus_g_exp = 1.0 - g * exp_minus_dt
    one_minus_g = 1.0 - g

    c = (
        iu * log_forward_spot
        + (kappa * theta / (sigma * sigma))
        * (
            (beta - d) * maturity
            - 2.0 * np.log(one_minus_g_exp / one_minus_g)
        )
    )
    d_coefficient = (
        (beta - d)
        / (sigma * sigma)
        * (-np.expm1(-d * maturity))
        / one_minus_g_exp
    )
    result = np.exp(c + d_coefficient * v0)
    if not np.isfinite(result.real).all() or not np.isfinite(result.imag).all():
        raise ArithmeticError("Heston characteristic function became non-finite")
    return complex(result) if result.ndim == 0 else result


def heston_price_fourier(
    strike: float,
    maturity: float,
    params: HestonParameters,
    option_type: str = "call",
    epsabs: float = 1e-9,
    epsrel: float = 1e-9,
    limit: int = 300,
):
    """European Heston price via Lewis' square-root Fourier inversion.

    The adaptive quadrature error estimate is returned with the price. It is a
    numerical integration diagnostic, not a model or parameter uncertainty band.
    """
    values = (strike, maturity, epsabs, epsrel)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("strike, maturity and quadrature tolerances must be finite")
    if strike <= 0.0 or maturity <= 0.0:
        raise ValueError("strike and maturity must be strictly positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if epsabs <= 0.0 or epsrel <= 0.0 or limit < 50:
        raise ValueError("quadrature tolerances must be positive and limit at least 50")

    s0 = params.spot
    r = params.risk_free_rate
    q = params.dividend_yield
    log_moneyness = math.log(s0 / strike)

    def integrand(x: float) -> float:
        shifted_cf = heston_characteristic_function(x - 0.5j, maturity, params)
        log_return_cf = shifted_cf * np.exp(-1j * (x - 0.5j) * math.log(s0))
        value = np.exp(1j * x * log_moneyness) * log_return_cf / (x * x + 0.25)
        return float(np.real(value))

    integral, error = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    call = (
        s0 * math.exp(-q * maturity)
        - math.sqrt(s0 * strike)
        * math.exp(-r * maturity)
        * integral
        / math.pi
    )
    lower = math.exp(-r * maturity) * max(
        s0 * math.exp((r - q) * maturity) - strike, 0.0
    )
    upper = s0 * math.exp(-q * maturity)
    tolerance = max(
        20.0 * error * math.sqrt(s0 * strike) * math.exp(-r * maturity) / math.pi,
        1e-8,
    )
    if not np.isfinite(call) or call < lower - tolerance or call > upper + tolerance:
        raise ArithmeticError(
            "Fourier inversion returned a price outside discounted no-arbitrage bounds; "
            "inspect quadrature convergence and characteristic-function branch stability"
        )
    call = min(max(call, lower), upper)
    if option_type == "call":
        price = call
    else:
        price = call - s0 * math.exp(-q * maturity) + strike * math.exp(-r * maturity)
    return {
        "price": float(price),
        "call_price": float(call),
        "quadrature_error_estimate": float(error),
        "method": "Lewis square-root Fourier inversion",
        "option_type": option_type,
        "maturity": float(maturity),
        "strike": float(strike),
    }


def _qe_variance_step(variance, normal, params: HestonParameters, dt: float, psi_c: float):
    kappa = params.mean_reversion
    theta = params.long_run_variance
    xi = params.vol_of_variance
    decay = math.exp(-kappa * dt)
    mean = theta + (variance - theta) * decay
    variance_of_variance = (
        variance * xi * xi * decay * (1.0 - decay) / kappa
        + theta * xi * xi * (1.0 - decay) ** 2 / (2.0 * kappa)
    )
    psi = np.maximum(variance_of_variance / (mean * mean), 0.0)
    next_variance = np.empty_like(variance)
    low = psi <= psi_c

    if np.any(low):
        psi_low = np.maximum(psi[low], 1e-14)
        two_over_psi = 2.0 / psi_low
        b_squared = two_over_psi - 1.0 + np.sqrt(two_over_psi) * np.sqrt(
            np.maximum(two_over_psi - 1.0, 0.0)
        )
        scale = mean[low] / (1.0 + b_squared)
        next_variance[low] = scale * (
            np.sqrt(b_squared) + normal[low]
        ) ** 2

    high = ~low
    if np.any(high):
        psi_high = psi[high]
        probability_zero = (psi_high - 1.0) / (psi_high + 1.0)
        rate = (1.0 - probability_zero) / mean[high]
        uniforms = np.clip(ndtr(normal[high]), 1e-15, 1.0 - 1e-15)
        positive = uniforms > probability_zero
        values = np.zeros_like(uniforms)
        values[positive] = np.log(
            (1.0 - probability_zero[positive]) / (1.0 - uniforms[positive])
        ) / rate[positive]
        next_variance[high] = values

    return np.maximum(next_variance, 0.0), mean, psi


def _qe_conditional_log_mgf(
    a: float, mean: np.ndarray, psi: np.ndarray, psi_c: float
):
    """log E[exp(a*v_next)|v] under the QE transition mixture."""
    result = np.empty_like(mean)
    low = psi <= psi_c
    if np.any(low):
        psi_low = np.maximum(psi[low], 1e-14)
        two_over_psi = 2.0 / psi_low
        b_squared = two_over_psi - 1.0 + np.sqrt(two_over_psi) * np.sqrt(
            np.maximum(two_over_psi - 1.0, 0.0)
        )
        scale = mean[low] / (1.0 + b_squared)
        denominator = 1.0 - 2.0 * a * scale
        if np.any(denominator <= 0.0):
            raise ValueError("QE-M moment does not exist for this time step; increase steps")
        result[low] = (
            -0.5 * np.log(denominator)
            + a * scale * b_squared / denominator
        )
    high = ~low
    if np.any(high):
        psi_high = psi[high]
        probability_zero = (psi_high - 1.0) / (psi_high + 1.0)
        rate = (1.0 - probability_zero) / mean[high]
        denominator = rate - a
        if np.any(denominator <= 0.0):
            raise ValueError("QE-M moment does not exist for this time step; increase steps")
        mgf = probability_zero + (1.0 - probability_zero) * rate / denominator
        result[high] = np.log(mgf)
    if not np.isfinite(result).all():
        raise ArithmeticError("QE-M conditional moment correction is non-finite")
    return result


def _heston_terminal_spots(
    params: HestonParameters,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool,
    psi_c: float,
):
    if params.vol_of_variance <= 0.0:
        raise ValueError("QE-M simulation requires positive volatility of variance")
    dt = maturity / n_steps
    independent_paths = n_paths // 2 if antithetic else n_paths
    log_spot = np.full(n_paths, math.log(params.spot), dtype=float)
    variance = np.full(n_paths, params.variance0, dtype=float)

    rho = params.correlation
    kappa = params.mean_reversion
    theta = params.long_run_variance
    xi = params.vol_of_variance
    gamma1 = 0.5
    gamma2 = 0.5
    k1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = gamma1 * dt * (1.0 - rho * rho)
    k4 = gamma2 * dt * (1.0 - rho * rho)
    mgf_exponent = k2 + 0.5 * k4

    for _ in range(n_steps):
        z_variance_base = rng.standard_normal(independent_paths)
        z_stock_base = rng.standard_normal(independent_paths)
        if antithetic:
            z_variance = np.concatenate((z_variance_base, -z_variance_base))
            z_stock = np.concatenate((z_stock_base, -z_stock_base))
        else:
            z_variance = z_variance_base
            z_stock = z_stock_base

        next_variance, conditional_mean, psi = _qe_variance_step(
            variance, z_variance, params, dt, psi_c
        )
        log_mgf = _qe_conditional_log_mgf(mgf_exponent, conditional_mean, psi, psi_c)
        martingale_correction = (
            -(k1 + 0.5 * k3) * variance - log_mgf
        )
        conditional_variance = np.maximum(k3 * variance + k4 * next_variance, 0.0)
        log_spot += (
            (params.risk_free_rate - params.dividend_yield) * dt
            + martingale_correction
            + k1 * variance
            + k2 * next_variance
            + np.sqrt(conditional_variance) * z_stock
        )
        variance = next_variance

    terminal = np.exp(log_spot)
    if not np.isfinite(terminal).all():
        raise ArithmeticError("Heston QE-M simulation produced non-finite terminal spots")
    return terminal


def heston_monte_carlo_price(
    strike: float,
    maturity: float,
    params: HestonParameters,
    n_paths: int = 100_000,
    n_steps: int = 128,
    option_type: str = "call",
    seed: int = 20261125,
    antithetic: bool = True,
    control_variate: bool = True,
    pilot_paths: int = 4_096,
    psi_c: float = 1.5,
    confidence_level: float = 0.95,
):
    """Price a European option with Andersen-style QE-M Heston simulation.

    The QE variance transition is paired with an exponential-moment correction
    to preserve the stock's one-step risk-neutral conditional expectation. If
    control variates are enabled, an independent pilot estimates beta against
    discounted terminal stock; this avoids estimating beta on the valuation
    sample and understating its standard error.

    Standard errors and t intervals quantify simulation error conditional on
    the time discretization. They do not include QE discretization bias or model
    / calibration uncertainty.
    """
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if n_paths < 4 or n_steps < 1:
        raise ValueError("n_paths must be at least 4 and n_steps at least 1")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even when antithetic variates are enabled")
    if control_variate and (pilot_paths < 4 or (antithetic and pilot_paths % 2)):
        raise ValueError("pilot_paths must be at least 4 and even for antithetic sampling")
    if not np.isfinite(psi_c) or psi_c <= 0.0:
        raise ValueError("psi_c must be finite and positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    rng = np.random.default_rng(seed)
    discount = math.exp(-params.risk_free_rate * maturity)
    control_mean = params.spot * math.exp(-params.dividend_yield * maturity)

    def independent_observations(terminal):
        if option_type == "call":
            discounted_payoff = discount * np.maximum(terminal - strike, 0.0)
        else:
            discounted_payoff = discount * np.maximum(strike - terminal, 0.0)
        discounted_stock = discount * terminal
        if antithetic:
            half = terminal.size // 2
            discounted_payoff = 0.5 * (
                discounted_payoff[:half] + discounted_payoff[half:]
            )
            discounted_stock = 0.5 * (
                discounted_stock[:half] + discounted_stock[half:]
            )
        return discounted_payoff, discounted_stock

    beta = 0.0
    if control_variate:
        pilot_terminal = _heston_terminal_spots(
            params, maturity, n_steps, pilot_paths, rng, antithetic, psi_c
        )
        pilot_payoff, pilot_stock = independent_observations(pilot_terminal)
        stock_variance = float(np.var(pilot_stock, ddof=1))
        if stock_variance <= np.finfo(float).tiny:
            raise ArithmeticError("control variate pilot has zero stock-price variance")
        beta = float(
            np.cov(pilot_payoff, pilot_stock, ddof=1)[0, 1] / stock_variance
        )

    terminal = _heston_terminal_spots(
        params, maturity, n_steps, n_paths, rng, antithetic, psi_c
    )
    payoff, discounted_stock = independent_observations(terminal)
    raw_price = float(np.mean(payoff))
    adjusted = payoff - beta * (discounted_stock - control_mean)
    price = float(np.mean(adjusted))
    raw_standard_error = float(np.std(payoff, ddof=1) / math.sqrt(payoff.size))
    standard_error = float(np.std(adjusted, ddof=1) / math.sqrt(adjusted.size))
    alpha = 1.0 - confidence_level
    critical_value = float(student_t.ppf(1.0 - alpha / 2.0, adjusted.size - 1))
    interval = (
        price - critical_value * standard_error,
        price + critical_value * standard_error,
    )
    martingale_target = control_mean
    simulated_discounted_stock = float(np.mean(discounted_stock))
    return {
        "price": price,
        "unadjusted_price": raw_price,
        "standard_error": standard_error,
        "unadjusted_standard_error": raw_standard_error,
        "confidence_interval": (float(interval[0]), float(interval[1])),
        "confidence_level": float(confidence_level),
        "n_paths": int(n_paths),
        "effective_samples": int(payoff.size),
        "n_steps": int(n_steps),
        "scheme": "Heston QE-M with conditional stock-martingale correction",
        "antithetic": bool(antithetic),
        "control_variate": bool(control_variate),
        "control_beta_from_independent_pilot": float(beta),
        "pilot_paths": int(pilot_paths if control_variate else 0),
        "variance_reduction_factor": float(
            (raw_standard_error / standard_error) ** 2
            if standard_error > 0.0 else math.inf
        ),
        "discounted_terminal_stock_mean": simulated_discounted_stock,
        "discounted_terminal_stock_target": float(martingale_target),
        "stock_martingale_relative_error": float(
            simulated_discounted_stock / martingale_target - 1.0
        ),
        "feller_ratio": float(params.feller_ratio),
        "feller_condition_satisfied": bool(params.feller_condition_satisfied),
        "discretization_bias_included_in_interval": False,
    }
