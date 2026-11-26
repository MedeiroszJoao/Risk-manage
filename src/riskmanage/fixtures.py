from __future__ import annotations
import numpy as np
import pandas as pd
from .data import validate_market_data


def build_synthetic_market_data(n: int = 1800, seed: int = 20261124) -> pd.DataFrame:
    """Seeded correlated daily factor history with clustered stress regimes.

    The fixture is purely synthetic and must never be described as observed market data.
    """
    if n < 300:
        raise ValueError("at least 300 business-day observations are required")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    corr = np.array([
        [1.00, -0.55, 0.18, -0.10, 0.32, -0.06, -0.08],
        [-0.55, 1.00, -0.08, 0.02, -0.22, 0.16, 0.18],
        [0.18, -0.08, 1.00, -0.25, 0.12, 0.02, 0.01],
        [-0.10, 0.02, -0.25, 1.00, -0.04, 0.02, 0.03],
        [0.32, -0.22, 0.12, -0.04, 1.00, -0.05, -0.06],
        [-0.06, 0.16, 0.02, 0.02, -0.05, 1.00, 0.72],
        [-0.08, 0.18, 0.01, 0.03, -0.06, 0.72, 1.00],
    ])
    # Preserve the original seven-factor random draw and stress-window RNG
    # sequence exactly so the existing reference paths remain unchanged.
    z = rng.multivariate_normal(np.zeros(7), corr, size=n)
    stress = np.zeros(n, dtype=bool)
    for start in (int(n * .22), int(n * .58), int(n * .82)):
        end = min(n, start + int(rng.integers(12, 28)))
        stress[start:end] = True
    brent_specific = rng.standard_normal(n)
    scale = np.where(stress, 2.8, 1.0)
    z *= scale[:, None]
    brent_specific *= scale
    equity_ret = .00028 + .0105 * z[:, 0]
    fx_ret = .00002 + .0048 * z[:, 2]
    jpy_ret = -.00001 + .0052 * z[:, 3]
    wti_ret = .00008 + .018 * z[:, 4]
    brent_ret = .00008 + .015 * (0.90 * z[:, 4] + np.sqrt(1.0 - 0.90**2) * brent_specific)
    vix = np.clip(17.5 - 6.0 * z[:, 0] + 3.5 * np.abs(z[:, 1]), 9, 75)
    y2 = np.clip(2.4 + np.cumsum(.012 * z[:, 5]), .1, 7.5)
    y10 = np.clip(3.0 + np.cumsum(.010 * z[:, 6]), .2, 8.0)
    frame = pd.DataFrame({
        "date": dates,
        "sp500": 2100 * np.exp(np.cumsum(equity_ret)),
        "vix": vix,
        "eurusd": 1.08 * np.exp(np.cumsum(fx_ret)),
        "usdjpy": 118 * np.exp(np.cumsum(jpy_ret)),
        "wti": 48 * np.exp(np.cumsum(wti_ret)),
        "brent": 52 * np.exp(np.cumsum(brent_ret)),
        "ust2y": y2,
        "ust10y": y10,
        "is_synthetic": True,
    })
    validate_market_data(frame)
    return frame
