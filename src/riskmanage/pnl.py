from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .portfolio import DESKS, Position, build_default_portfolio
from .pricing import black_scholes_call, black_scholes_put, garman_kohlhagen_call, zero_coupon_bond

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
