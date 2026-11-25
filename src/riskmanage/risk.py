from __future__ import annotations
import numpy as np
import pandas as pd

LIQUIDITY_HORIZONS=(10,20,40,60,120)

def _finite_vector(values, name="values"):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite 1D vector with at least two observations")
    return x

def historical_var_es(losses, confidence=.975):
    x=_finite_vector(losses,"losses")
    if not .5<confidence<1: raise ValueError("confidence must be between 0.5 and 1")
    var=float(np.quantile(x,confidence,method="higher"))
    tail=x[x>=var]
    return {"var":var,"expected_shortfall":float(tail.mean()),
            "tail_observations":int(len(tail)),"observations":int(len(x)),
            "confidence":float(confidence),"expected_tail_observations":float(len(x)*(1-confidence))}

def rolling_historical_var(losses, window=250, confidence=.99):
    x=np.asarray(losses,dtype=float)
    if x.ndim!=1 or window<20 or len(x)<=window: raise ValueError("need a 1D series longer than the forecast window")
    if not np.isfinite(x).all(): raise ValueError("loss series contains non-finite values")
    out=np.full(len(x),np.nan)
    for i in range(window,len(x)):
        out[i]=np.quantile(x[i-window:i],confidence,method="higher")
    return pd.Series(out)

def _ten_day_losses(daily_losses, horizon=10):
    x=_finite_vector(daily_losses,"daily losses")
    if horizon<1 or len(x)<horizon: raise ValueError("insufficient observations for requested horizon")
    return np.convolve(x,np.ones(horizon),mode="valid")

def liquidity_adjusted_es(factor_losses: dict[str,np.ndarray],
                          liquidity_horizons: dict[str,int], confidence=.975,
                          base_horizon=10):
    if not factor_losses: raise ValueError("factor loss mapping cannot be empty")
    if set(factor_losses)!=set(liquidity_horizons):
        raise ValueError("each factor must have one liquidity horizon")
    invalid={k:v for k,v in liquidity_horizons.items() if v not in LIQUIDITY_HORIZONS}
    if invalid: raise ValueError(f"invalid liquidity horizon mapping: {invalid}")
    arrays={k:_finite_vector(v,k) for k,v in factor_losses.items()}
    lengths={len(v) for v in arrays.values()}
    if len(lengths)!=1: raise ValueError("factor loss vectors must be aligned")
    all_daily=np.sum(np.vstack(list(arrays.values())),axis=0)
    base_es=historical_var_es(_ten_day_losses(all_daily,base_horizon),confidence)["expected_shortfall"]
    components=[]
    for previous,lh in zip(LIQUIDITY_HORIZONS[:-1],LIQUIDITY_HORIZONS[1:]):
        names=[name for name,h in liquidity_horizons.items() if h>=lh]
        if not names: component_es=0.0
        else:
            subtotal=np.sum(np.vstack([arrays[n] for n in names]),axis=0)
            component_es=historical_var_es(_ten_day_losses(subtotal,base_horizon),confidence)["expected_shortfall"]
        weight=np.sqrt((lh-previous)/base_horizon)
        components.append({"liquidity_horizon":lh,"factors":names,
                           "partial_es":float(component_es),"scaling":float(weight),
                           "scaled_component":float(component_es*weight)})
    total=float(np.sqrt(base_es**2+sum(x["scaled_component"]**2 for x in components)))
    return {"base_es_10d":float(base_es),"liquidity_adjusted_es":total,
            "confidence":confidence,"base_horizon":base_horizon,"components":components}
