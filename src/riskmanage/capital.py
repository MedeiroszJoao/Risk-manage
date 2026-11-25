from __future__ import annotations
import numpy as np
from .risk import historical_var_es, _ten_day_losses

def _align_factor_losses(factor_losses):
    if not factor_losses: raise ValueError("factor losses cannot be empty")
    values={k:np.asarray(v,dtype=float) for k,v in factor_losses.items()}
    if any(v.ndim!=1 or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("factor loss arrays must be finite 1D vectors")
    if len({len(v) for v in values.values()})!=1:
        raise ValueError("factor arrays must align")
    return values

def _portfolio_es(mapping,names,confidence):
    daily=np.sum(np.vstack([mapping[n] for n in names]),axis=0)
    return float(historical_var_es(_ten_day_losses(daily),confidence)["expected_shortfall"])

def _rolling_reduced_coverage(factors,selected,confidence=.975,
                              estimation_window=250,validation_days=60):
    n=len(next(iter(factors.values())))
    if n<estimation_window+validation_days-1:
        return {"status":"insufficient_history","observations":0,
                "required_observations":estimation_window+validation_days-1,
                "mean_ratio":None,"minimum_ratio":None,"ratios":[]}
    ratios=[]
    for end in range(n-validation_days,n):
        start=end-estimation_window+1
        window={k:v[start:end+1] for k,v in factors.items()}
        full=_portfolio_es(window,list(window),confidence)
        reduced=_portfolio_es(window,selected,confidence)
        ratios.append(float(reduced/full) if full>0 else float("nan"))
    finite=[x for x in ratios if np.isfinite(x)]
    if not finite:
        return {"status":"undefined_full_es","observations":len(ratios),
                "required_observations":estimation_window+validation_days-1,
                "mean_ratio":None,"minimum_ratio":None,"ratios":ratios}
    return {"status":"computed","observations":len(finite),
            "estimation_window":estimation_window,"validation_days":validation_days,
            "mean_ratio":float(np.mean(finite)),"minimum_ratio":float(np.min(finite)),
            "maximum_ratio":float(np.max(finite)),"ratios":finite}

def select_stress_window(factor_losses: dict[str,np.ndarray], window_days=250, confidence=.975):
    factors=_align_factor_losses(factor_losses)
    n=len(next(iter(factors.values())))
    if window_days<60 or n<window_days:
        raise ValueError("insufficient observations for stress window")
    total=np.sum(np.vstack(list(factors.values())),axis=0)
    best=None
    for start in range(0,n-window_days+1):
        end=start+window_days
        ten_day=_ten_day_losses(total[start:end],10)
        es=historical_var_es(ten_day,confidence)["expected_shortfall"]
        if best is None or es>best["stress_es"]:
            best={"start_index":start,"end_index":end-1,"stress_es":float(es)}
    best.update({"window_days":window_days,"confidence":confidence,"observations":n,
                 "stress_start_index":best["start_index"],"stress_end_index":best["end_index"]})
    return best

def select_reduced_factor_set(current_factor_losses: dict[str,np.ndarray],
                              confidence=.975,target_explained=.75,
                              estimation_window=250,validation_days=60):
    factors=_align_factor_losses(current_factor_losses)
    if not 0<target_explained<=1:
        raise ValueError("target explained must be in (0,1]")
    current={k:v[-estimation_window:] for k,v in factors.items()}
    full_es=_portfolio_es(current,list(current),confidence)
    if full_es<=0:
        raise ValueError("full portfolio ES must be positive to define reduced-set coverage")
    ranking=sorted(current,key=lambda k:historical_var_es(
        _ten_day_losses(current[k]),confidence)["expected_shortfall"],reverse=True)
    selected=[]
    for factor in ranking:
        selected.append(factor)
        coverage=_rolling_reduced_coverage(factors,selected,confidence,
                                           estimation_window,validation_days)
        if coverage["status"]=="computed" and coverage["mean_ratio"]>=target_explained:
            break
    reduced_es=_portfolio_es(current,selected,confidence)
    ratio=float(reduced_es/full_es)
    coverage=_rolling_reduced_coverage(factors,selected,confidence,
                                       estimation_window,validation_days)
    target_met=coverage["status"]=="computed" and coverage["mean_ratio"]>=target_explained
    return {"selected_factors":selected,"full_current_es":float(full_es),
            "reduced_current_es":float(reduced_es),"explained_ratio":ratio,
            "target_explained":float(target_explained),"target_met":bool(target_met),
            "coverage_12w":coverage,
            "coverage_method":"mean of daily reduced/full 10-day ES ratios over the latest 60 business-day endpoints"}

def calculate_imcc(es_full_current: float, es_reduced_current: float,
                   es_reduced_stress: float, class_components: dict[str,dict[str,float]],
                   rho=.5):
    values=[es_full_current,es_reduced_current,es_reduced_stress]
    if any(not np.isfinite(v) or v<=0 for v in values):
        raise ValueError("ES inputs must be positive and finite")
    if not 0<=rho<=1:
        raise ValueError("rho must be in [0,1]")
    unconstrained=float(es_reduced_stress*max(es_full_current/es_reduced_current,1.0))
    components={}
    for name,c in class_components.items():
        fc,rc,rs=(float(c[k]) for k in ("full_current","reduced_current","reduced_stress"))
        if any(not np.isfinite(v) or v<=0 for v in (fc,rc,rs)):
            raise ValueError(f"invalid ES component for class {name}")
        components[name]=float(rs*max(fc/rc,1.0))
    constrained=float(sum(components.values()))
    return {"unconstrained":unconstrained,"constrained_sum":constrained,
            "class_components":components,"rho":rho,
            "imcc":float(rho*unconstrained+(1-rho)*constrained)}

def aggregate_nmrf_ses(charges_by_factor: dict[str,float],
                       factor_groups: dict[str,str], rho=.6):
    if set(charges_by_factor)!=set(factor_groups):
        raise ValueError("every NMRF charge needs a group")
    if not 0<=rho<=1:
        raise ValueError("rho must be in [0,1]")
    if any((not np.isfinite(v) or v<0) for v in charges_by_factor.values()):
        raise ValueError("NMRF charges must be finite and nonnegative")
    credit=[charges_by_factor[k] for k,g in factor_groups.items() if g=="credit_idio"]
    equity=[charges_by_factor[k] for k,g in factor_groups.items() if g=="equity_idio"]
    other=[charges_by_factor[k] for k,g in factor_groups.items() if g=="other"]
    unknown=set(factor_groups.values())-{"credit_idio","equity_idio","other"}
    if unknown:
        raise ValueError(f"unknown NMRF group(s): {sorted(unknown)}")
    credit_agg=float(np.sqrt(np.sum(np.square(credit))))
    equity_agg=float(np.sqrt(np.sum(np.square(equity))))
    other_agg=float(np.sqrt((rho*sum(other))**2+(1-rho**2)*sum(v*v for v in other)))
    return {"credit_idio":credit_agg,"equity_idio":equity_agg,"other":other_agg,
            "aggregate":float(credit_agg+equity_agg+other_agg),"rho":rho}

def build_ima_capital(factor_losses: dict[str,np.ndarray], stress_window: dict,
                      class_mapping: dict[str,str], confidence=.975, rho=.5):
    factors=_align_factor_losses(factor_losses)
    if set(class_mapping)!=set(factors):
        raise ValueError("each factor must map to one risk class")
    start,end=stress_window["start_index"],stress_window["end_index"]+1
    current={k:v[-250:] for k,v in factors.items()}
    stressed={k:v[start:end] for k,v in factors.items()}
    if min(len(v) for v in current.values())<20 or min(len(v) for v in stressed.values())<20:
        raise ValueError("insufficient current or stress observations")
    reduced=select_reduced_factor_set(factors,confidence,.75)
    selected=reduced["selected_factors"]
    full_current=_portfolio_es(current,list(factors),confidence)
    reduced_current=_portfolio_es(current,selected,confidence)
    reduced_stress=_portfolio_es(stressed,selected,confidence)
    classes={}
    for cls in sorted(set(class_mapping.values())):
        all_names=[k for k,v in class_mapping.items() if v==cls]
        names=[k for k in all_names if k in selected]
        if not names:
            continue
        classes[cls]={"full_current":_portfolio_es(current,all_names,confidence),
                      "reduced_current":_portfolio_es(current,names,confidence),
                      "reduced_stress":_portfolio_es(stressed,names,confidence)}
    imcc=calculate_imcc(full_current,reduced_current,reduced_stress,classes,rho)
    return {"reduced_set":reduced,"es_full_current":full_current,
            "es_reduced_current":reduced_current,"es_reduced_stress":reduced_stress,
            "imcc":imcc,"capital_label":"illustrative_research_output"}
