from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2, ks_2samp, spearmanr
from scipy.special import xlogy

def kupiec_pof(exceptions: int, observations: int, confidence=.99):
    if observations<=0 or exceptions<0 or exceptions>observations or not .5<confidence<1:
        raise ValueError("invalid Kupiec inputs")
    p0=1-confidence; phat=exceptions/observations
    ll_null=float(xlogy(exceptions,p0)+xlogy(observations-exceptions,1-p0))
    ll_alt=float(xlogy(exceptions,phat)+xlogy(observations-exceptions,1-phat))
    lr=max(0.0,-2*(ll_null-ll_alt))
    return {"lr_statistic":float(lr),"p_value":float(chi2.sf(lr,1)),
            "exceptions":int(exceptions),"observations":int(observations),
            "expected_exceptions":float(observations*p0)}

def christoffersen_independence(exception_sequence):
    x=np.asarray(exception_sequence,dtype=int)
    if x.ndim!=1 or len(x)<3 or not np.isin(x,[0,1]).all():
        raise ValueError("exceptions must be a 0/1 vector with at least 3 observations")
    prev=x[:-1]; nxt=x[1:]
    n00=int(np.sum((prev==0)&(nxt==0))); n01=int(np.sum((prev==0)&(nxt==1)))
    n10=int(np.sum((prev==1)&(nxt==0))); n11=int(np.sum((prev==1)&(nxt==1)))
    den0=n00+n01; den1=n10+n11; total=den0+den1
    p=(n01+n11)/total; p01=n01/den0 if den0 else 0.0; p11=n11/den1 if den1 else 0.0
    ll_null=float(xlogy(n01+n11,p)+xlogy(n00+n10,1-p))
    ll_alt=float(xlogy(n01,p01)+xlogy(n00,1-p01)+xlogy(n11,p11)+xlogy(n10,1-p11))
    lr=max(0.0,2*(ll_alt-ll_null))
    return {"lr_statistic":float(lr),"p_value":float(chi2.sf(lr,1)),
            "transitions":{"n00":n00,"n01":n01,"n10":n10,"n11":n11}}

def _backtest_zone(exceptions,observations,confidence):
    if observations!=250: return "not_classified",None
    if abs(confidence-.99)<1e-10:
        if exceptions<=4: return "green",1.50
        multipliers={5:1.70,6:1.76,7:1.83,8:1.88,9:1.92}
        if exceptions in multipliers: return "amber",multipliers[exceptions]
        return "red",2.00
    if abs(confidence-.975)<1e-10 and exceptions>30: return "ineligible",None
    return "diagnostic_only",None

def backtest_var(hpl, var_forecast, confidence=.99):
    h=np.asarray(hpl,dtype=float); v=np.asarray(var_forecast,dtype=float)
    if h.ndim!=1 or v.ndim!=1 or len(h)!=len(v): raise ValueError("HPL and VaR arrays must align")
    valid=np.isfinite(h)&np.isfinite(v)
    if valid.sum()<20: raise ValueError("at least 20 matched HPL / VaR observations are required")
    losses=-h[valid]; forecasts=v[valid]; exc=(losses>forecasts).astype(int)
    count=int(exc.sum()); zone,multiplier=_backtest_zone(count,len(exc),confidence)
    kupiec=kupiec_pof(count,len(exc),confidence)
    christ=christoffersen_independence(exc)
    return {"observations":int(len(exc)),"exceptions":count,
            "exception_rate":float(count/len(exc)),"confidence":confidence,
            "zone":zone,"capital_multiplier":multiplier,"kupiec":kupiec,
            "christoffersen":christ,"exception_dates":np.flatnonzero(valid)[exc.astype(bool)].tolist()}

def pla_test(hpl,rtpl,window=250):
    h=np.asarray(hpl,dtype=float); r=np.asarray(rtpl,dtype=float)
    if h.ndim!=1 or r.ndim!=1 or len(h)!=len(r): raise ValueError("HPL and RTPL must be aligned vectors")
    if window!=250: raise ValueError("this FRTB-inspired PLA test uses a 250-observation window")
    valid=np.isfinite(h)&np.isfinite(r); h=h[valid];r=r[valid]
    if len(h)<window:
        return {"zone":"insufficient_sample","observations":int(len(h)),
                "required_observations":window,"spearman":None,"ks_distance":None}
    h=h[-window:];r=r[-window:]
    rho=float(spearmanr(h,r).statistic)
    if not np.isfinite(rho): rho=0.0
    ks=float(ks_2samp(h,r,method="auto").statistic)
    if rho>.80 and ks<.09: zone="green"
    elif rho<.70 or ks>.12: zone="red"
    else: zone="amber"
    return {"zone":zone,"observations":window,"spearman":rho,"ks_distance":ks,
            "thresholds":{"green_spearman_gt":.80,"green_ks_lt":.09,
                          "red_spearman_lt":.70,"red_ks_gt":.12}}

def rfet_diagnostic(evidence_ledger: pd.DataFrame, lookback_days=365):
    required={"date","factor_id","evidence_type","verified"}
    if not required.issubset(evidence_ledger.columns):
        return {"status":"not_evidenced","reason":f"missing fields: {sorted(required-set(evidence_ledger.columns))}","factors":{}}
    data=evidence_ledger.copy()
    data["date"]=pd.to_datetime(data["date"],errors="coerce")
    data=data.dropna(subset=["date","factor_id"])
    allowed={"transaction","committed_quote","qualifying_third_party_quote"}
    latest=data.date.max()
    cutoff=latest-pd.Timedelta(days=lookback_days-1)
    results={}
    for factor,group in data.groupby("factor_id"):
        eligible=group[(group.date>=cutoff)&group.verified.astype(bool)&group.evidence_type.isin(allowed)]
        dates=eligible.date.dt.normalize().drop_duplicates().sort_values()
        count=int(len(dates))
        if count:
            starts=pd.date_range(cutoff.normalize(),latest.normalize()-pd.Timedelta(days=89),freq="D")
            min_90=min(int(((dates>=start)&(dates<start+pd.Timedelta(days=90))).sum()) for start in starts) if len(starts) else count
        else: min_90=0
        rule_100=count>=100
        rule_24=count>=24 and min_90>=4
        classification="modellable_evidence_pass" if (rule_100 or rule_24) else ("non_modellable" if count>0 else "not_evidenced")
        results[str(factor)]={"eligible_real_price_observations":count,
                              "min_90d_observations":min_90,
                              "rule_100_observations":rule_100,"rule_24_plus_90d":rule_24,
                              "classification":classification}
    return {"status":"evidence_review_required","lookback_days":lookback_days,
            "eligible_evidence_types":sorted(allowed),"factors":results,
            "note":"Passing the programmed evidence count is not regulatory certification; source eligibility and evidence validity require independent review."}
