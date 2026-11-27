from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def _finite_array(values, name: str) -> np.ndarray:
    objects = np.asarray(values, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in objects.flat):
        raise ValueError(f"{name} must not contain boolean values")
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric values without implicit coercion")
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scalar_or_array(values: np.ndarray):
    return float(values) if values.ndim == 0 else values


def ssvi_phi_power_law(theta, eta: float, gamma: float):
    """Power-law SSVI shape function phi(theta).

    Uses phi(theta) = eta / [theta**gamma * (1 + theta)**(1-gamma)].
    The function accepts 0 < gamma < 1; global static-arbitrage certification
    is deliberately narrower and is handled by ssvi_static_arbitrage_certificate.
    """
    theta_array = _finite_array(theta, "theta")
    if np.any(theta_array <= 0.0):
        raise ValueError("theta must be strictly positive")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and strictly positive")
    if not np.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between 0 and 1")

    log_phi = np.log(eta) - gamma * np.log(theta_array) - (1.0 - gamma) * np.log1p(theta_array)
    return _scalar_or_array(np.exp(log_phi))


def ssvi_total_variance(
    log_moneyness,
    theta,
    rho: float,
    eta: float,
    gamma: float,
):
    """Return SSVI total implied variance w(k, theta).

    log_moneyness is k = log(K / F_T), and theta is the ATM total
    implied variance for the matching expiry. Inputs broadcast under NumPy rules.
    """
    k_array = _finite_array(log_moneyness, "log_moneyness")
    theta_array = _finite_array(theta, "theta")
    if not np.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between -1 and 1")

    k_array, theta_array = np.broadcast_arrays(k_array, theta_array)
    phi = np.asarray(ssvi_phi_power_law(theta_array, eta, gamma), dtype=float)
    root = np.hypot(phi * k_array + rho, np.sqrt(1.0 - rho * rho))
    total_variance = 0.5 * theta_array * (
        1.0 + rho * phi * k_array + root
    )
    if not np.isfinite(total_variance).all() or np.any(total_variance <= 0.0):
        raise ArithmeticError("SSVI produced non-positive or non-finite total variance")
    return _scalar_or_array(total_variance)


def ssvi_implied_volatility(
    log_moneyness,
    theta,
    maturity,
    rho: float,
    eta: float,
    gamma: float,
):
    """Convert SSVI total variance to annualized Black implied volatility."""
    total_variance = np.asarray(
        ssvi_total_variance(log_moneyness, theta, rho, eta, gamma),
        dtype=float,
    )
    maturity_array = _finite_array(maturity, "maturity")
    total_variance, maturity_array = np.broadcast_arrays(total_variance, maturity_array)
    if np.any(maturity_array <= 0.0):
        raise ValueError("maturity must be strictly positive")
    implied_volatility = np.sqrt(total_variance / maturity_array)
    return _scalar_or_array(implied_volatility)


def ssvi_static_arbitrage_certificate(
    theta_by_expiry,
    rho: float,
    eta: float,
    gamma: float,
    tolerance: float = 1e-12,
):
    """Assess the global sufficient no-static-arbitrage region for power-law SSVI.

    The certificate uses the sufficient parameter region in Gatheral & Jacquier
    (2014), Remark 4.4: 0 < gamma <= 1/2 and eta * (1 + |rho|) <= 2.
    The observed ATM total-variance term structure must also be non-decreasing.
    Failure to meet this sufficient region means 'not certified here', not
    necessarily that an arbitrage exists.
    """
    theta = _finite_array(theta_by_expiry, "theta_by_expiry")
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("theta_by_expiry must be a non-empty 1D vector")
    if np.any(theta <= 0.0):
        raise ValueError("theta_by_expiry must be strictly positive")
    if not np.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between -1 and 1")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and strictly positive")
    if not np.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between 0 and 1")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    phi = np.asarray(ssvi_phi_power_law(theta, eta, gamma), dtype=float)
    abs_rho = abs(rho)
    theta_nondecreasing = bool(np.all(np.diff(theta) >= -tolerance))

    # Theorem 4.1: calendar-spread condition in ATM-total-variance time.
    derivative_ratio = (1.0 - gamma) / (1.0 + theta)
    upper_bound = (
        np.inf
        if rho == 0.0
        else (1.0 + np.sqrt(1.0 - rho * rho)) / (rho * rho)
    )
    calendar_shape_condition = bool(
        np.all(derivative_ratio >= -tolerance)
        and np.all(derivative_ratio <= upper_bound + tolerance)
    )

    # Theorem 4.2: sufficient butterfly-arbitrage conditions, evaluated
    # at the supplied expiry grid.
    butterfly_condition_1 = bool(
        np.all(theta * phi * (1.0 + abs_rho) < 4.0 + tolerance)
    )
    butterfly_condition_2 = bool(
        np.all(theta * phi * phi * (1.0 + abs_rho) <= 4.0 + tolerance)
    )

    # Remark 4.4 gives this stronger, simple global certificate for all theta > 0.
    power_law_global_region = bool(
        0.0 < gamma <= 0.5 + tolerance
        and eta * (1.0 + abs_rho) <= 2.0 + tolerance
    )
    certified = bool(
        theta_nondecreasing
        and calendar_shape_condition
        and power_law_global_region
        and butterfly_condition_1
        and butterfly_condition_2
    )

    return {
        "certified_static_arbitrage_free": certified,
        "certificate_scope": (
            "global theta > 0 sufficient region"
            if certified
            else "not certified by the implemented sufficient region"
        ),
        "theta_nondecreasing": theta_nondecreasing,
        "calendar_shape_condition": calendar_shape_condition,
        "butterfly_condition_1": butterfly_condition_1,
        "butterfly_condition_2": butterfly_condition_2,
        "power_law_global_region": power_law_global_region,
        "max_theta_phi_1_plus_abs_rho": float(
            np.max(theta * phi * (1.0 + abs_rho))
        ),
        "max_theta_phi_squared_1_plus_abs_rho": float(
            np.max(theta * phi * phi * (1.0 + abs_rho))
        ),
        "eta_times_1_plus_abs_rho": float(eta * (1.0 + abs_rho)),
        "rho": float(rho),
        "eta": float(eta),
        "gamma": float(gamma),
        "expiries": int(theta.size),
    }


def fit_ssvi_power_law(
    log_moneyness,
    theta,
    market_total_variance,
    expiry_times,
    weights=None,
    initial_guess: tuple[float, float, float] | None = None,
    n_starts: int = 5,
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int = 10000,
):
    """Fit global (rho, eta, gamma) to observed total implied variances.

    theta and expiry_times are quote-aligned vectors. ATM total variance must
    be constant within each expiry and non-decreasing across sorted expiries;
    this makes the calendar-spread check meaningful rather than sorting theta
    values and accidentally certifying an unordered term structure.
    weights are non-negative objective weights (for example, inverse quote-error
    variance); they are normalized internally. The optimizer parameterizes
    beta = eta * (1 + abs(rho)) in (0, 2], keeping the fit inside the global
    sufficient static-arbitrage region. This fits total variance, not option prices.
    """
    k = _finite_array(log_moneyness, "log_moneyness")
    theta_array = _finite_array(theta, "theta")
    observed = _finite_array(market_total_variance, "market_total_variance")
    expiry_array = _finite_array(expiry_times, "expiry_times")
    if (
        k.ndim != 1
        or theta_array.ndim != 1
        or observed.ndim != 1
        or expiry_array.ndim != 1
    ):
        raise ValueError("calibration inputs must be 1D vectors")
    if not (
        k.size == theta_array.size == observed.size == expiry_array.size
    ):
        raise ValueError("calibration inputs must have equal lengths")
    if k.size < 6 or np.unique(expiry_array).size < 2:
        raise ValueError("calibration requires at least 6 quotes across 2 or more expiries")
    if np.any(expiry_array <= 0.0):
        raise ValueError("expiry_times must be strictly positive")

    expiry_theta = []
    for expiry in np.unique(expiry_array):
        theta_at_expiry = theta_array[expiry_array == expiry]
        if not np.allclose(theta_at_expiry, theta_at_expiry[0], rtol=1e-10, atol=1e-12):
            raise ValueError("ATM total variance must be constant within each expiry")
        expiry_theta.append(float(theta_at_expiry[0]))
    expiry_theta = np.asarray(expiry_theta, dtype=float)
    if np.any(np.diff(expiry_theta) < -1e-12):
        raise ValueError("ATM total variance must be non-decreasing with expiry")
    if np.any(theta_array <= 0.0) or np.any(observed <= 0.0):
        raise ValueError("theta and observed total variance must be strictly positive")
    if n_starts < 1 or max_nfev < 1:
        raise ValueError("n_starts and max_nfev must be positive integers")
    if not np.isfinite(f_scale) or f_scale <= 0.0:
        raise ValueError("f_scale must be finite and strictly positive")
    if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise ValueError("unsupported least-squares loss")

    if weights is None:
        root_weights = np.ones_like(observed)
    else:
        weight_array = _finite_array(weights, "weights")
        if weight_array.ndim != 1 or weight_array.size != observed.size:
            raise ValueError("weights must be a 1D vector matching the quotes")
        if np.any(weight_array < 0.0) or np.count_nonzero(weight_array) < 3:
            raise ValueError("weights must be non-negative with at least 3 positive entries")
        positive_mean = float(np.mean(weight_array[weight_array > 0.0]))
        root_weights = np.sqrt(weight_array / positive_mean)

    default_starts = [
        (-0.75, 0.75, 0.15),
        (-0.35, 1.40, 0.40),
        (0.00, 1.00, 0.25),
        (0.35, 1.40, 0.40),
        (0.75, 0.75, 0.15),
    ]
    if initial_guess is not None:
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must be (rho, eta, gamma)")
        default_starts = [tuple(float(x) for x in initial_guess)] + default_starts
    starts = default_starts[: max(n_starts, 1)]

    def decode(parameters):
        fitted_rho, beta, fitted_gamma = parameters
        fitted_eta = beta / (1.0 + abs(fitted_rho))
        return float(fitted_rho), float(fitted_eta), float(fitted_gamma)

    def residual(parameters):
        fitted_rho, fitted_eta, fitted_gamma = decode(parameters)
        model = ssvi_total_variance(
            k, theta_array, fitted_rho, fitted_eta, fitted_gamma
        )
        return (np.asarray(model) - observed) * root_weights

    fits = []
    for rho_start, eta_start, gamma_start in starts:
        if not -0.999 < rho_start < 0.999:
            raise ValueError("initial rho must lie in (-0.999, 0.999)")
        if eta_start <= 0.0 or gamma_start <= 0.0 or gamma_start > 0.5:
            raise ValueError("initial eta must be positive and initial gamma in (0, 0.5]")
        beta_start = eta_start * (1.0 + abs(rho_start))
        if beta_start > 2.0:
            beta_start = 1.95

        result = least_squares(
            residual,
            x0=np.array([rho_start, beta_start, gamma_start], dtype=float),
            bounds=(
                np.array([-0.999, 1e-8, 1e-4]),
                np.array([0.999, 2.0, 0.5]),
            ),
            method="trf",
            loss=loss,
            f_scale=f_scale,
            x_scale="jac",
            max_nfev=max_nfev,
        )
        fits.append(result)

    best = min(fits, key=lambda result: float(result.cost))
    fitted_rho, fitted_eta, fitted_gamma = decode(best.x)
    fitted = np.asarray(
        ssvi_total_variance(k, theta_array, fitted_rho, fitted_eta, fitted_gamma),
        dtype=float,
    )
    raw_residuals = fitted - observed
    singular_values = np.linalg.svd(best.jac, compute_uv=False)
    jacobian_condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    certificate = ssvi_static_arbitrage_certificate(
        expiry_theta, fitted_rho, fitted_eta, fitted_gamma
    )

    return {
        "success": bool(best.success),
        "status": int(best.status),
        "message": str(best.message),
        "rho": fitted_rho,
        "eta": fitted_eta,
        "gamma": fitted_gamma,
        "objective": "weighted least squares on total implied variance",
        "loss": loss,
        "cost": float(best.cost),
        "rmse_total_variance": float(np.sqrt(np.mean(raw_residuals**2))),
        "max_abs_residual_total_variance": float(np.max(np.abs(raw_residuals))),
        "jacobian_condition_number": jacobian_condition,
        "nfev": int(best.nfev),
        "number_of_starts": len(fits),
        "fitted_total_variance": fitted,
        "residuals_total_variance": raw_residuals,
        "arbitrage_certificate": certificate,
    }


def fit_ssvi_power_law_to_black76_prices(
    forward,
    strike,
    expiry_times,
    discount_factors,
    theta,
    option_types,
    market_prices,
    weights=None,
    bid_prices=None,
    ask_prices=None,
    initial_guess: tuple[float, float, float] | None = None,
    n_starts: int = 5,
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int = 10000,
):
    """Calibrate common SSVI parameters directly to European Black-76 option prices."""
    from .black76 import black76_price

    fwd = _finite_array(forward, "forward")
    strikes = _finite_array(strike, "strike")
    expiry = _finite_array(expiry_times, "expiry_times")
    discounts = _finite_array(discount_factors, "discount_factors")
    theta_array = _finite_array(theta, "theta")
    market = _finite_array(market_prices, "market_prices")
    kinds = np.asarray(option_types, dtype=str)
    arrays = (fwd, strikes, expiry, discounts, theta_array, market, kinds)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("price-calibration inputs must be 1D vectors")
    n = market.size
    if n < 6 or any(array.size != n for array in arrays):
        raise ValueError("price-calibration inputs must have equal lengths and at least 6 quotes")
    if np.unique(expiry).size < 2:
        raise ValueError("price calibration requires at least 2 distinct expiries")
    if (np.any(fwd <= 0.0) or np.any(strikes <= 0.0) or np.any(expiry <= 0.0)
            or np.any(discounts <= 0.0) or np.any(theta_array <= 0.0)):
        raise ValueError("forwards, strikes, expiries, discount factors and theta must be positive")
    if not np.isin(kinds, ("call", "put")).all():
        raise ValueError("option_types must contain only 'call' or 'put'")
    if np.any(market < 0.0):
        raise ValueError("market option prices must be non-negative")
    if n_starts < 1 or max_nfev < 1:
        raise ValueError("n_starts and max_nfev must be positive integers")
    if not np.isfinite(f_scale) or f_scale <= 0.0:
        raise ValueError("f_scale must be finite and strictly positive")
    if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise ValueError("unsupported least-squares loss")

    expiry_theta = []
    for tenor in np.unique(expiry):
        values = theta_array[expiry == tenor]
        if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
            raise ValueError("ATM total variance must be constant within each expiry")
        expiry_theta.append(float(values[0]))
    expiry_theta = np.asarray(expiry_theta, dtype=float)
    if np.any(np.diff(expiry_theta) < -1e-12):
        raise ValueError("ATM total variance must be non-decreasing with expiry")

    is_call = kinds == "call"
    lower = np.where(is_call, discounts*np.maximum(fwd-strikes, 0.0),
                     discounts*np.maximum(strikes-fwd, 0.0))
    upper = np.where(is_call, discounts*fwd, discounts*strikes)
    tolerance = 1e-10*np.maximum(upper, 1.0)
    if np.any(market < lower-tolerance):
        raise ValueError("market option price violates the Black-76 lower bound")
    if np.any(market >= upper-tolerance):
        raise ValueError("market option price is at or too close to the Black-76 upper bound")

    use_bid_ask = bid_prices is not None or ask_prices is not None
    if use_bid_ask and (bid_prices is None or ask_prices is None):
        raise ValueError("bid_prices and ask_prices must be supplied together")
    if use_bid_ask:
        bids = _finite_array(bid_prices, "bid_prices")
        asks = _finite_array(ask_prices, "ask_prices")
        if bids.ndim != 1 or asks.ndim != 1 or bids.size != n or asks.size != n:
            raise ValueError("bid and ask vectors must match market_prices")
        if np.any(bids < lower-tolerance) or np.any(bids < 0.0):
            raise ValueError("one or more bid prices violate the Black-76 lower bound")
        if np.any(asks >= upper-tolerance) or np.any(asks < bids):
            raise ValueError("one or more ask prices violate the Black-76 upper bound or cross the bid")
    else:
        bids = asks = None

    if weights is None:
        root_weights = np.ones(n, dtype=float)
    else:
        weight_array = _finite_array(weights, "weights")
        if weight_array.ndim != 1 or weight_array.size != n:
            raise ValueError("weights must be a 1D vector matching the quotes")
        if np.any(weight_array < 0.0) or np.count_nonzero(weight_array) < 3:
            raise ValueError("weights must be non-negative with at least 3 positive entries")
        mean_weight = float(np.mean(weight_array[weight_array > 0.0]))
        root_weights = np.sqrt(weight_array/mean_weight)

    log_moneyness = np.log(strikes/fwd)
    starts = [
        (-0.75, 0.75, 0.15), (-0.35, 1.40, 0.40), (0.00, 1.00, 0.25),
        (0.35, 1.40, 0.40), (0.75, 0.75, 0.15),
    ]
    if initial_guess is not None:
        if len(initial_guess) != 3:
            raise ValueError("initial_guess must be (rho, eta, gamma)")
        starts = [tuple(float(x) for x in initial_guess)] + starts
    starts = starts[:n_starts]

    def decode(parameters):
        rho, beta, gamma = parameters
        return float(rho), float(beta/(1.0+abs(rho))), float(gamma)

    def model_prices(parameters):
        rho, eta, gamma = decode(parameters)
        total_variance = ssvi_total_variance(log_moneyness, theta_array, rho, eta, gamma)
        volatility = np.sqrt(np.asarray(total_variance)/expiry)
        return np.asarray(black76_price(fwd, strikes, expiry, discounts, volatility, kinds))

    def residual(parameters):
        model = model_prices(parameters)
        if use_bid_ask:
            distance = np.maximum(np.maximum(bids-model, model-asks), 0.0)
        else:
            distance = model-market
        return distance*root_weights

    fits = []
    for rho0, eta0, gamma0 in starts:
        if not -0.999 < rho0 < 0.999:
            raise ValueError("initial rho must lie in (-0.999, 0.999)")
        if eta0 <= 0.0 or gamma0 <= 0.0 or gamma0 > 0.5:
            raise ValueError("initial eta must be positive and initial gamma in (0, 0.5]")
        beta0 = eta0*(1.0+abs(rho0))
        if beta0 > 2.0:
            beta0 = 1.95
        result = least_squares(
            residual,
            x0=np.array([rho0, beta0, gamma0], dtype=float),
            bounds=(np.array([-0.999, 1e-8, 1e-4]), np.array([0.999, 2.0, 0.5])),
            method="trf", loss=loss, f_scale=f_scale, x_scale="jac",
            max_nfev=max_nfev,
        )
        fits.append(result)

    best = min(fits, key=lambda result: float(result.cost))
    rho, eta, gamma = decode(best.x)
    model = model_prices(best.x)
    mid_residual = model-market
    singular = np.linalg.svd(best.jac, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular.size and singular[-1] > 0.0 else float("inf")
    certificate = ssvi_static_arbitrage_certificate(expiry_theta, rho, eta, gamma)
    inside = int(np.count_nonzero((model >= bids) & (model <= asks))) if use_bid_ask else None
    return {
        "success": bool(best.success), "status": int(best.status), "message": str(best.message),
        "rho": rho, "eta": eta, "gamma": gamma,
        "objective": "bid_ask_interval_distance" if use_bid_ask else "weighted_mid_price_least_squares",
        "loss": loss, "cost": float(best.cost),
        "rmse_mid_price": float(np.sqrt(np.mean(mid_residual**2))),
        "max_abs_mid_price_residual": float(np.max(np.abs(mid_residual))),
        "quotes_inside_bid_ask": inside,
        "jacobian_condition_number": condition,
        "nfev": int(best.nfev), "number_of_starts": len(fits),
        "fitted_prices": model, "mid_price_residuals": mid_residual,
        "arbitrage_certificate": certificate,
    }
