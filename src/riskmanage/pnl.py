from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .portfolio import DESKS, Position, build_default_portfolio
from .pricing import black_scholes_call, black_scholes_put, garman_kohlhagen_call, zero_coupon_bond
from .risk import LIQUIDITY_HORIZONS

RATE=0.035
FOREIGN_RATE=0.020
TRADING_DAYS=252

def enrich_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    data=frame.copy().sort_values("date").reset_index(drop=True)
    for source,target,fallback in [
        ("eurusd","eurusd_vol",.12),("usdjpy","usdjpy_vol",.10),
        ("wti","wti_vol",.30)
    ]:
        if source == "wti":
            # WTI Cushing spot can be zero or negative. Use additive price changes
            # scaled by a trailing median absolute price, not logarithmic returns.
            scale = data[source].abs().rolling(window=60, min_periods=15).median().shift(1)
            returns = data[source].diff().div(scale.where(scale > 1e-12))
        else:
            returns = np.log(data[source]).diff()
        data[target]=returns.ewm(span=40,min_periods=15).std().shift(1)*np.sqrt(TRADING_DAYS)
        data[target]=data[target].fillna(fallback).clip(.02,2.0)
    data["equity_vol"]=(data["vix"]/100.0).clip(.05,1.5)
    return data

def market_state(row: pd.Series) -> dict[str,float]:
    return {k:float(row[k]) for k in (
        "sp500","vix","eurusd","usdjpy","wti","brent","ust2y","ust10y",
        "equity_vol","eurusd_vol","usdjpy_vol","wti_vol"
    )}

def _bond_yield(state: dict[str,float], maturity: float) -> float:
    if maturity<=2: return state["ust2y"]
    weight=min((maturity-2.0)/8.0,1.0)
    return state["ust2y"]*(1-weight)+state["ust10y"]*weight

def value_position(position: Position, state: dict[str,float], day_offset: int = 0) -> float:
    tau=max(position.maturity-day_offset/TRADING_DAYS,0.0)
    q=position.quantity
    if position.instrument=="EQUITY_CALL":
        return q*position.multiplier*black_scholes_call(
            state["sp500"],position.strike,tau,RATE,state["equity_vol"])
    if position.instrument=="EQUITY_PUT":
        return q*position.multiplier*black_scholes_put(state["sp500"],position.strike,tau,RATE,state["equity_vol"])
    if position.instrument=="VARIANCE_SWAP":
        return q*(state["equity_vol"]**2-position.strike**2)*tau
    if position.instrument=="FX_CALL":
        return q*position.multiplier*garman_kohlhagen_call(
            state["eurusd"],position.strike,tau,RATE,FOREIGN_RATE,state["eurusd_vol"])
    if position.instrument=="FX_SPOT": return q*state["eurusd"]
    if position.instrument=="ZERO_BOND_2Y":
        return zero_coupon_bond(q,2.0,state["ust2y"])
    if position.instrument=="ZERO_BOND_10Y":
        return zero_coupon_bond(q,10.0,state["ust10y"])
    if position.instrument=="WTI_SPOT_PROXY": return q*state["wti"]
    if position.instrument=="BRENT_SPOT_PROXY": return q*state["brent"]
    raise ValueError(f"unsupported instrument: {position.instrument}")

def value_desk(positions: list[Position], desk: str, state: dict[str,float],
               day_offset: int = 0) -> float:
    return float(sum(value_position(p,state,day_offset) for p in positions if p.desk==desk))

DESK_FACTORS={
    "Equity Options":("sp500","equity_vol"),
    "FX Options":("eurusd","eurusd_vol"),
    "Rates":("ust2y","ust10y"),
    "Commodities":("wti","brent"),
}
MODEL_FACTORS={
    "delta_only":{
        "Equity Options":("sp500",),
        "FX Options":("eurusd",),
        "Rates":("ust2y","ust10y"),
        "Commodities":("wti","brent"),
    },
    "diagonal_second_order":DESK_FACTORS,
    "full_hessian":DESK_FACTORS,
}
LEGACY_MODEL_ALIASES={"full_greeks":"diagonal_second_order"}

INSTRUMENT_RISK_FACTORS = {
    "EQUITY_CALL": ("sp500", "equity_vol"),
    "EQUITY_PUT": ("sp500", "equity_vol"),
    "VARIANCE_SWAP": ("equity_vol",),
    "FX_CALL": ("eurusd", "eurusd_vol"),
    "FX_SPOT": ("eurusd",),
    "ZERO_BOND_2Y": ("ust2y",),
    "ZERO_BOND_10Y": ("ust10y",),
    "WTI_SPOT_PROXY": ("wti",),
    "BRENT_SPOT_PROXY": ("brent",),
}


def _effective_position_liquidity_horizon(base_horizon: int, maturity_years: float) -> int:
    """Apply a next-longer permitted liquidity horizon when maturity is shorter."""
    if isinstance(maturity_years, (bool, np.bool_)) or not isinstance(
        maturity_years, (int, float, np.integer, np.floating)
    ):
        raise ValueError("position maturity must be a finite nonnegative number")
    maturity = float(maturity_years)
    if not np.isfinite(maturity) or maturity < 0.0:
        raise ValueError("position maturity must be a finite nonnegative number")
    if isinstance(base_horizon, (bool, np.bool_)) or not isinstance(
        base_horizon, (int, np.integer)
    ):
        raise ValueError("base liquidity horizon must be an integer")
    if int(base_horizon) not in LIQUIDITY_HORIZONS:
        raise ValueError("base liquidity horizon is not a permitted Basel horizon")
    if maturity == 0.0:
        return int(base_horizon)

    maturity_days = maturity * TRADING_DAYS
    if maturity_days >= int(base_horizon):
        return int(base_horizon)
    eligible = [horizon for horizon in LIQUIDITY_HORIZONS if horizon > maturity_days]
    if not eligible:
        raise ValueError("no permitted liquidity horizon exceeds position maturity")
    return int(min(eligible))


def _prepare_joint_liquidity_horizon_positions(
    market_data: pd.DataFrame,
    liquidity_horizons: dict[str, int],
    positions: list[Position] | None,
):
    data = enrich_market_data(market_data)
    if len(data) < 2:
        raise ValueError("at least two market observations are required")
    expected_factors = {
        factor for factor_tuple in DESK_FACTORS.values() for factor in factor_tuple
    }
    if (
        not isinstance(liquidity_horizons, dict)
        or set(liquidity_horizons) != expected_factors
    ):
        raise ValueError(
            "liquidity_horizons must map every declared desk risk factor exactly once"
        )

    normalized_horizons = {}
    for factor, value in liquidity_horizons.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"liquidity_horizon[{factor}] must be an integer")
        horizon = int(value)
        if horizon not in LIQUIDITY_HORIZONS:
            raise ValueError(
                f"liquidity_horizon[{factor}] is not a permitted horizon"
            )
        normalized_horizons[factor] = horizon

    if positions is None:
        positions = build_default_portfolio(market_state(data.iloc[0]))
    if not positions:
        raise ValueError("at least one position is required")

    position_maps = []
    for index, position in enumerate(positions):
        if position.desk not in DESKS:
            raise ValueError(f"unknown position desk: {position.desk!r}")
        if position.instrument not in INSTRUMENT_RISK_FACTORS:
            raise ValueError(
                f"no risk-factor mapping for instrument {position.instrument!r}"
            )
        base_map = {
            factor: normalized_horizons[factor]
            for factor in INSTRUMENT_RISK_FACTORS[position.instrument]
        }
        effective_map = {
            factor: _effective_position_liquidity_horizon(base, position.maturity)
            for factor, base in base_map.items()
        }
        position_maps.append({
            "position_index": int(index),
            "desk": position.desk,
            "instrument": position.instrument,
            "label": position.label or f"{position.desk}:{position.instrument}",
            "maturity_years": float(position.maturity),
            "base_factor_horizons": base_map,
            "effective_factor_horizons": effective_map,
            "position": position,
        })

    factor_subset_by_horizon = {}
    positions_by_horizon = {}
    for horizon in LIQUIDITY_HORIZONS:
        factor_subset, included_positions = set(), []
        for item in position_maps:
            selected = [
                factor for factor, eff in item["effective_factor_horizons"].items()
                if eff >= horizon
            ]
            factor_subset.update(selected)
            if selected:
                included_positions.append({
                    "position_index": item["position_index"],
                    "desk": item["desk"],
                    "instrument": item["instrument"],
                    "label": item["label"],
                    "selected_factors": selected,
                })
        factor_subset_by_horizon[str(horizon)] = sorted(factor_subset)
        positions_by_horizon[str(horizon)] = included_positions

    return data, position_maps, factor_subset_by_horizon, positions_by_horizon


def build_joint_liquidity_horizon_pnl(
    market_data: pd.DataFrame,
    liquidity_horizons: dict[str, int],
    positions: list[Position] | None = None,
) -> pd.DataFrame:
    """Daily position-level full revaluation for a constant-maturity rolling book.

    For each daily close-to-close interval and liquidity-horizon threshold, only
    mapped factors whose effective horizon is at least the threshold are shocked.
    Both marks use the same one-day aged position, so theta is excluded. These
    daily P&Ls are later summed into overlapping 10-observation totals. This is
    a daily-roll approximation, not a direct static-position 10-day revaluation.
    """
    data, position_maps, factor_subset_by_horizon, positions_by_horizon = (
        _prepare_joint_liquidity_horizon_positions(
            market_data, liquidity_horizons, positions
        )
    )

    rows = []
    for row_index in range(1, len(data)):
        previous_state = market_state(data.iloc[row_index - 1])
        current_state = market_state(data.iloc[row_index])
        row = {"date": pd.Timestamp(data.iloc[row_index]["date"])}
        for horizon in LIQUIDITY_HORIZONS:
            joint_pnl = 0.0
            for item in position_maps:
                position = item["position"]
                selected = [
                    factor for factor, eff in item["effective_factor_horizons"].items()
                    if eff >= horizon
                ]
                if not selected:
                    continue
                shocked_state = dict(previous_state)
                for factor in selected:
                    shocked_state[factor] = current_state[factor]
                joint_pnl += (
                    value_position(position, shocked_state, day_offset=1)
                    - value_position(position, previous_state, day_offset=1)
                )
            row[f"pnl_lh_{horizon}"] = float(joint_pnl)
        rows.append(row)

    result = pd.DataFrame(rows)
    result.attrs["liquidity_horizon_metadata"] = {
        "method": "position_level_joint_full_revaluation",
        "observation_horizon_days": 1,
        "time_series_aggregation": (
            "rolling sum of ten daily full-revaluation market-factor P&Ls"
        ),
        "position_policy": (
            "constant-maturity daily roll; positions are re-established at declared "
            "maturity after each daily interval"
        ),
        "risk_factor_mapping": (
            "INSTRUMENT_RISK_FACTORS plus supplied factor liquidity horizons"
        ),
        "factor_subset_by_horizon": factor_subset_by_horizon,
        "positions_by_horizon": positions_by_horizon,
        "position_factor_horizons": [
            {key: value for key, value in item.items() if key != "position"}
            for item in position_maps
        ],
        "maturity_convention": (
            "maturity_years multiplied by 252 trading days; if maturity is shorter "
            "than the mapped base liquidity horizon, use the smallest permitted "
            "horizon strictly longer than maturity; maturity zero means a non-expiring "
            "spot proxy and no maturity cap"
        ),
        "time_decay_treatment": (
            "theta is excluded by valuing both prior-state and shocked-state positions "
            "at day_offset=1; ten-day risk totals are sums of daily rolling-book P&Ls"
        ),
        "theta_included": False,
        "scope": (
            "research daily-roll alternative; instrument coverage, factor map, desk "
            "mapping and maturity conventions are simplified and not regulatory validated"
        ),
    }
    return result


def build_joint_liquidity_horizon_10d_pnl(
    market_data: pd.DataFrame,
    liquidity_horizons: dict[str, int],
    positions: list[Position] | None = None,
) -> pd.DataFrame:
    """Directly revalue overlapping ten-observation shocks for a static window book.

    Each output row is one overlapping 10-observation interval from t-10 to t.
    The supplied positions are held fixed during that interval. For each
    liquidity-horizon threshold, only the risk factors with effective horizons
    at least as long as the threshold are shocked; all other factors stay at
    their values at t-10. Both marks use the same ten-day aged maturity, which
    excludes theta while retaining full nonlinear repricing under the direct
    ten-day risk-factor move. No shorter-horizon P&L is scaled or summed.
    """
    base_horizon_days = 10
    data, position_maps, factor_subset_by_horizon, positions_by_horizon = (
        _prepare_joint_liquidity_horizon_positions(
            market_data, liquidity_horizons, positions
        )
    )
    if len(data) <= base_horizon_days:
        raise ValueError(
            "at least eleven market observations are required for a 10-day interval"
        )

    rows = []
    for end_index in range(base_horizon_days, len(data)):
        start_index = end_index - base_horizon_days
        previous_state = market_state(data.iloc[start_index])
        current_state = market_state(data.iloc[end_index])
        row = {"date": pd.Timestamp(data.iloc[end_index]["date"])}
        for horizon in LIQUIDITY_HORIZONS:
            joint_pnl = 0.0
            for item in position_maps:
                position = item["position"]
                selected = [
                    factor for factor, eff in item["effective_factor_horizons"].items()
                    if eff >= horizon
                ]
                if not selected:
                    continue
                shocked_state = dict(previous_state)
                for factor in selected:
                    shocked_state[factor] = current_state[factor]
                joint_pnl += (
                    value_position(
                        position, shocked_state, day_offset=base_horizon_days
                    )
                    - value_position(
                        position, previous_state, day_offset=base_horizon_days
                    )
                )
            row[f"pnl_lh_{horizon}"] = float(joint_pnl)
        rows.append(row)

    result = pd.DataFrame(rows)
    result.attrs["liquidity_horizon_metadata"] = {
        "method": "position_level_joint_10d_overlapping_full_revaluation",
        "observation_horizon_days": base_horizon_days,
        "base_horizon_days": base_horizon_days,
        "overlapping_windows": True,
        "window_construction": (
            "ten consecutive close-to-close observations; each row revalues the "
            "portfolio from data index t-10 to t"
        ),
        "position_policy": (
            "same supplied positions are held fixed within each ten-observation "
            "window; positions are reinitialized at the declared book state for "
            "each overlapping historical window"
        ),
        "risk_factor_mapping": (
            "INSTRUMENT_RISK_FACTORS plus supplied factor liquidity horizons"
        ),
        "factor_subset_by_horizon": factor_subset_by_horizon,
        "positions_by_horizon": positions_by_horizon,
        "position_factor_horizons": [
            {key: value for key, value in item.items() if key != "position"}
            for item in position_maps
        ],
        "maturity_convention": (
            "maturity_years multiplied by 252 trading days; if maturity is shorter "
            "than the mapped base liquidity horizon, use the smallest permitted "
            "horizon strictly longer than maturity; maturity zero means a non-expiring "
            "spot proxy and no maturity cap"
        ),
        "time_decay_treatment": (
            "theta excluded; both start and end marks use the same ten-day aged "
            "position, so only the full ten-day risk-factor shock changes value"
        ),
        "theta_included": False,
        "scope": (
            "research implementation of direct 10-day position-level joint revaluation; "
            "instrument coverage, factor map, desk mapping, maturity assumptions, stress "
            "calibration and regulatory controls remain incomplete"
        ),
    }
    return result


def _bump_size(name: str, value: float) -> float:
    if name in ("ust2y", "ust10y"):
        return 1e-3
    if name.endswith("_vol"):
        return max(abs(value) * 1e-3, 1e-4)
    return max(abs(value) * 1e-4, 1e-5)


def _mixed_second_derivative(
    positions,
    desk,
    state,
    day_offset,
    factor_i,
    factor_j,
    bump_i,
    bump_j,
):
    """Central finite-difference mixed second derivative of desk value."""
    if factor_i == factor_j or bump_i <= 0.0 or bump_j <= 0.0:
        raise ValueError("mixed derivative requires distinct factors and positive bumps")

    def value_at(sign_i, sign_j):
        scenario = dict(state)
        scenario[factor_i] = float(state[factor_i]) + sign_i * bump_i
        scenario[factor_j] = float(state[factor_j]) + sign_j * bump_j
        return value_desk(positions, desk, scenario, day_offset)

    return float(
        (
            value_at(1.0, 1.0)
            - value_at(1.0, -1.0)
            - value_at(-1.0, 1.0)
            + value_at(-1.0, -1.0)
        )
        / (4.0 * bump_i * bump_j)
    )


def _taylor_rtpl(positions, desk, previous_state, current_state, day_offset, model):
    # One-day aging is revalued exactly at unchanged market factors. No
    # market-time cross derivatives are included in this approximation.
    if desk == "Commodities":
        # Exact analytical deltas for both linear spot proxies; no finite-difference noise.
        instruments = {"wti": "WTI_SPOT_PROXY", "brent": "BRENT_SPOT_PROXY"}
        return float(sum(
            sum(p.quantity for p in positions if p.desk == desk and p.instrument == instrument)
            * (current_state[factor] - previous_state[factor])
            for factor, instrument in instruments.items()
        ))

    old_value = value_desk(positions, desk, previous_state, day_offset)
    theta_value = value_desk(positions, desk, previous_state, day_offset + 1)
    result = theta_value - old_value
    selected = MODEL_FACTORS[model][desk]
    changes = {}

    for factor in selected:
        x = float(previous_state[factor])
        dx = float(current_state[factor]) - x
        h = _bump_size(factor, x)
        up = dict(previous_state)
        down = dict(previous_state)
        up[factor] = x + h
        down[factor] = x - h
        v_up = value_desk(positions, desk, up, day_offset)
        v_down = value_desk(positions, desk, down, day_offset)
        delta = (v_up - v_down) / (2.0 * h)
        result += delta * dx
        changes[factor] = (dx, h)

        if model in ("diagonal_second_order", "full_hessian"):
            gamma = (v_up - 2.0 * old_value + v_down) / (h * h)
            result += 0.5 * gamma * dx * dx

    if model == "full_hessian":
        for i, factor_i in enumerate(selected):
            dx_i, bump_i = changes[factor_i]
            if dx_i == 0.0:
                continue
            for factor_j in selected[i + 1:]:
                dx_j, bump_j = changes[factor_j]
                if dx_j == 0.0:
                    continue
                cross_gamma = _mixed_second_derivative(
                    positions,
                    desk,
                    previous_state,
                    day_offset,
                    factor_i,
                    factor_j,
                    bump_i,
                    bump_j,
                )
                result += cross_gamma * dx_i * dx_j

    return float(result)


def build_desk_pnl(market_data: pd.DataFrame, desk: str, model: str = "delta_only",
                   positions: list[Position] | None = None) -> pd.DataFrame:
    """Mark a constant-maturity hypothetical book and construct a sensitivity RTPL.

    Models:
      - delta_only: first-order market-factor delta with separately aged theta;
      - diagonal_second_order: adds only diagonal gamma/volga terms;
      - full_hessian: adds mixed market-factor second derivatives.

    The full_hessian mode includes cross terms between declared desk market
    factors (for example spot/volatility vanna). One-day time decay is revalued
    separately; time-market mixed derivatives are not included.
    Positions stay static during each daily HPL interval and are re-established
    at declared constant maturity after the close-to-close interval.
    """
    if desk not in DESKS:
        raise ValueError(f"unknown desk: {desk}")
    if model in LEGACY_MODEL_ALIASES:
        warnings.warn(
            "model='full_greeks' is a legacy name for diagonal_second_order; "
            "use diagonal_second_order or full_hessian explicitly",
            DeprecationWarning,
            stacklevel=2,
        )
        model = LEGACY_MODEL_ALIASES[model]
    if model not in MODEL_FACTORS:
        raise ValueError(f"unknown risk model: {model}")
    data = enrich_market_data(market_data)
    if positions is None:
        positions = build_default_portfolio(market_state(data.iloc[0]))
    rows = []
    if desk == "Commodities":
        approximation = "exact_linear_delta_for_spot_proxies"
    elif model == "delta_only":
        approximation = "first_order_market_factor_delta"
    elif model == "diagonal_second_order":
        approximation = "diagonal_second_order_market_factor_expansion_no_cross_terms"
    else:
        approximation = "full_market_factor_hessian_no_time_market_cross_terms"

    for i in range(1, len(data)):
        previous = data.iloc[i - 1]
        current = data.iloc[i]
        state0 = market_state(previous)
        state1 = market_state(current)
        hpl = value_desk(positions, desk, state1, 1) - value_desk(positions, desk, state0, 0)
        rtpl = _taylor_rtpl(positions, desk, state0, state1, 0, model)
        rows.append({
            "date": pd.Timestamp(current["date"]),
            "hpl": float(hpl),
            "rtpl": float(rtpl),
            "residual": float(hpl - rtpl),
            "risk_model": model,
            "rtpl_approximation": approximation,
        })
    result = pd.DataFrame(rows)
    result.attrs["market_factor_approximation"] = approximation
    result.attrs["time_decay_treatment"] = (
        "exact one-day aging at unchanged market factors; time-market cross terms omitted"
    )
    return result


def build_all_desk_pnl(market_data: pd.DataFrame, model: str="delta_only"):
    data=enrich_market_data(market_data)
    positions=build_default_portfolio(market_state(data.iloc[0]))
    result={}
    for desk in DESKS:
        result[desk]=build_desk_pnl(data,desk,model,positions)
    return result


def build_factor_pnl(market_data: pd.DataFrame,
                     positions: list[Position] | None = None) -> pd.DataFrame:
    """Full-revaluation HPL attribution by one-factor-at-a-time market shocks.

    The remainder explicitly contains theta and cross-factor interactions; it is
    not disguised as a directly observed market risk factor.
    """
    data=enrich_market_data(market_data)
    if positions is None:
        positions=build_default_portfolio(market_state(data.iloc[0]))
    rows=[]
    desk_hpl={desk:[] for desk in DESKS}
    for i in range(1,len(data)):
        previous=data.iloc[i-1]
        current=data.iloc[i]
        s0=market_state(previous)
        s1=market_state(current)
        contribution={}
        remainder=0.0
        for desk in DESKS:
            old=value_desk(positions,desk,s0,0)
            time_only=value_desk(positions,desk,s0,1)
            full_new=value_desk(positions,desk,s1,1)
            hpl=full_new-old
            desk_hpl[desk].append(float(hpl))
            remainder += time_only-old
            partial_total=0.0
            for factor in DESK_FACTORS[desk]:
                shocked=dict(s0)
                shocked[factor]=s1[factor]
                partial=value_desk(positions,desk,shocked,1)-time_only
                contribution[factor]=contribution.get(factor,0.0)+float(partial)
                partial_total += partial
            remainder += hpl-(time_only-old)-partial_total
        rows.append({"date":pd.Timestamp(current["date"]),**contribution,
                     "unexplained_pnl":float(remainder)})
    result=pd.DataFrame(rows)
    result.attrs["desk_hpl"]={k:np.asarray(v,dtype=float) for k,v in desk_hpl.items()}
    return result
