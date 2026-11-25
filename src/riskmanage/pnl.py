from __future__ import annotations
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
        ("eurusd","eurusd_vol",.12),("usdjpy","usdjpy_vol",.10),("wti","wti_vol",.30)
    ]:
        returns=np.log(data[source]).diff()
        data[target]=returns.ewm(span=40,min_periods=15).std().shift(1)*np.sqrt(TRADING_DAYS)
        data[target]=data[target].fillna(fallback).clip(.02,2.0)
    data["equity_vol"]=(data["vix"]/100.0).clip(.05,1.5)
    return data

def market_state(row: pd.Series) -> dict[str,float]:
    return {k:float(row[k]) for k in (
        "sp500","vix","eurusd","usdjpy","wti","ust2y","ust10y",
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
    if position.instrument=="WTI_FUTURE": return q*state["wti"]
    raise ValueError(f"unsupported instrument: {position.instrument}")

def value_desk(positions: list[Position], desk: str, state: dict[str,float],
               day_offset: int = 0) -> float:
    return float(sum(value_position(p,state,day_offset) for p in positions if p.desk==desk))

DESK_FACTORS={
    "Equity Options":("sp500","equity_vol"),
    "FX Options":("eurusd","eurusd_vol"),
    "Rates":("ust2y","ust10y"),
    "Commodities":("wti",),
}
MODEL_FACTORS={
    "delta_only":{
        "Equity Options":("sp500",),
        "FX Options":("eurusd",),
        "Rates":("ust2y","ust10y"),
        "Commodities":("wti",),
    },
    "full_greeks":DESK_FACTORS,
}

def _bump_size(name: str, value: float) -> float:
    if name in ("ust2y","ust10y"): return 1e-3
    return max(abs(value)*1e-4,1e-5)

def _taylor_rtpl(positions,desk,previous_state,current_state,day_offset,model):
    # Theta is isolated as time passage at the old market state.
    if desk=="Commodities":
        # Exact analytical delta for a linear WTI future; avoids finite-difference cancellation.
        quantity=sum(p.quantity for p in positions if p.desk==desk and p.instrument=="WTI_FUTURE")
        return float(quantity*(current_state["wti"]-previous_state["wti"]))
    old_value=value_desk(positions,desk,previous_state,day_offset)
    theta_value=value_desk(positions,desk,previous_state,day_offset+1)
    result=theta_value-old_value
    selected=MODEL_FACTORS[model][desk]
    for factor in selected:
        x=float(previous_state[factor]); dx=float(current_state[factor])-x
        h=_bump_size(factor,x)
        up=dict(previous_state); down=dict(previous_state)
        up[factor]=x+h; down[factor]=x-h
        v_up=value_desk(positions,desk,up,day_offset)
        v_down=value_desk(positions,desk,down,day_offset)
        delta=(v_up-v_down)/(2*h)
        result+=delta*dx
        if model=="full_greeks":
            gamma=(v_up-2*old_value+v_down)/(h*h)
            result+=0.5*gamma*dx*dx
    return float(result)

def build_desk_pnl(market_data: pd.DataFrame, desk: str, model: str="delta_only",
                   positions: list[Position] | None=None) -> pd.DataFrame:
    """Mark a constant-maturity hypothetical book; positions stay static during each daily HPL interval.

    After the close-to-close HPL is measured, the test book is re-established at its
    declared constant maturity for the next day. This is a research test portfolio,
    not a buy-and-hold record of one trade over the whole history.
    """
    if desk not in DESKS: raise ValueError(f"unknown desk: {desk}")
    if model not in MODEL_FACTORS: raise ValueError(f"unknown risk model: {model}")
    data=enrich_market_data(market_data)
    if positions is None: positions=build_default_portfolio(market_state(data.iloc[0]))
    rows=[]
    for i in range(1,len(data)):
        previous=data.iloc[i-1]; current=data.iloc[i]
        state0=market_state(previous); state1=market_state(current)
        hpl=value_desk(positions,desk,state1,1)-value_desk(positions,desk,state0,0)
        rtpl=_taylor_rtpl(positions,desk,state0,state1,0,model)
        rows.append({"date":pd.Timestamp(current["date"]),"hpl":float(hpl),
                     "rtpl":float(rtpl),"residual":float(hpl-rtpl),
                     "risk_model":model})
    return pd.DataFrame(rows)

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
