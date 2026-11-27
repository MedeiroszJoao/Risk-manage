from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from riskmanage.data import load_market_data, validate_market_data
from riskmanage.fixtures import build_synthetic_market_data
from riskmanage.pnl import (
    DESK_FACTORS, build_desk_pnl, build_factor_pnl,
    build_joint_liquidity_horizon_pnl, enrich_market_data, market_state, value_desk,
)
from riskmanage.portfolio import DESKS, build_default_portfolio
from riskmanage.risk import (
    LIQUIDITY_HORIZONS, historical_var_es, rolling_historical_var, liquidity_adjusted_es
)
from riskmanage.capital import select_stress_window, build_ima_capital
from riskmanage.validation import backtest_var, pla_test, rfet_diagnostic

LIQ_HORIZONS={"sp500":10,"equity_vol":20,"eurusd":10,"eurusd_vol":40,"ust2y":10,"ust10y":10,"wti":20,"brent":20}
RISK_CLASSES={"sp500":"equity","equity_vol":"equity","eurusd":"fx","eurusd_vol":"fx","ust2y":"rates","ust10y":"rates","wti":"commodities","brent":"commodities"}

def json_safe(obj):
    if isinstance(obj,dict): return {str(k):json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if isinstance(obj,pd.Timestamp): return obj.isoformat()
    if isinstance(obj,Path): return str(obj)
    if isinstance(obj,float) and not np.isfinite(obj): return None
    return obj

def build_scenario_reports(market):
    enriched = enrich_market_data(market)
    current = market_state(enriched.iloc[-1])
    positions=build_default_portfolio(current)
    base={d:value_desk(positions,d,current,0) for d in DESKS}
    scenarios={
      "equity_crash":{"sp500":.75,"vix_add":25},
      "fx_dislocation":{"eurusd":.90,"eurusd_vol":1.75,"usdjpy":1.08,"usdjpy_vol":1.50},
      "rates_bear_steepener":{"ust2y_add":2.0,"ust10y_add":1.5},
      "oil_collapse":{"wti_add":-25.0,"brent":.65},
      "combined_tail":{"sp500":.75,"vix_add":25,"eurusd":.90,"eurusd_vol":1.75,
        "usdjpy":1.08,"usdjpy_vol":1.50,"ust2y_add":2.0,"ust10y_add":1.5,
        "wti_add":-25.0,"brent":.65}
    }
    rows=[]
    for name,shock in scenarios.items():
        state=dict(current)
        for factor in ("sp500","eurusd","usdjpy","brent"):
            if factor in shock: state[factor]*=shock[factor]
        # The WTI shock is additive so it remains meaningful at zero or negative spot.
        state["wti"] += shock.get("wti_add", 0.0)
        for factor in ("eurusd_vol","usdjpy_vol","wti_vol"):
            if factor in shock: state[factor]=min(state[factor]*shock[factor],2.0)
        if "vix_add" in shock:
            state["vix"]=min(state["vix"]+shock["vix_add"],90.0)
            state["equity_vol"]=state["vix"]/100.0
        state["ust2y"]+=shock.get("ust2y_add",0.0)
        state["ust10y"]+=shock.get("ust10y_add",0.0)
        row={"scenario":name}; total=0.0
        for desk in DESKS:
            pnl=value_desk(positions,desk,state,0)-base[desk]
            row[f"{desk}_pnl"]=float(pnl); total-=pnl
        row["portfolio_loss"]=float(total); rows.append(row)
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser(description="Run reproducible market-risk desk validation case.")
    parser.add_argument("--synthetic",action="store_true",help="Use deterministic synthetic market history.")
    parser.add_argument("--data",default="data/market_data.csv")
    parser.add_argument("--seed",type=int,default=20261124)
    parser.add_argument("--observations",type=int,default=1800)
    parser.add_argument("--output",default="reports/generated")
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if args.synthetic:
        market=build_synthetic_market_data(args.observations,args.seed); data_kind="synthetic"
        provenance={"data_classification":"synthetic_not_observed","seed":args.seed,"rows":len(market),
          "first_date":str(market.date.min().date()),"last_date":str(market.date.max().date()),
          "generator":"riskmanage.fixtures.build_synthetic_market_data"}
    else:
        if not Path(args.data).exists():
            raise SystemExit(f"Market data not found at {args.data}. Run scripts/download_market_data.py or use --synthetic.")
        market=load_market_data(args.data); data_kind="public_fred_eia_spot_panel"
        manifest_path=Path("data/source_manifest.json")
        provenance=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
          "data_classification":"public_data_manifest_missing","data_path":args.data}
    validate_market_data(market)
    market.to_csv(out/"market_data_input.csv",index=False,date_format="%Y-%m-%d")
    quality={"rows":int(len(market)),"columns":list(market.columns),
      "duplicate_dates":int(market.date.duplicated().sum()),"missing_by_column":market.isna().sum().to_dict(),
      "first_date":str(market.date.min().date()),"last_date":str(market.date.max().date()),
      "classification":data_kind}
    (out/"market_data_quality.json").write_text(json.dumps(json_safe(quality),indent=2))

    desk_outputs={}; validation={}; combined=[]
    for desk in DESK_FACTORS:
        simple=build_desk_pnl(market,desk,"delta_only")
        diagonal=build_desk_pnl(market,desk,"diagonal_second_order")
        full=build_desk_pnl(market,desk,"full_hessian")
        for frame,model in (
            (simple,"delta_only"),
            (diagonal,"diagonal_second_order"),
            (full,"full_hessian"),
        ):
            frame["desk"]=desk
            frame["risk_model"]=model
        combined.extend([simple,diagonal,full])
        losses=-simple.hpl.to_numpy()
        var97=rolling_historical_var(losses,250,.975)
        var99=rolling_historical_var(losses,250,.99)
        hpl_tail=simple.hpl.to_numpy()[-250:]
        v97_tail=var97.to_numpy()[-250:]; v99_tail=var99.to_numpy()[-250:]
        risk97=historical_var_es(losses,.975); risk99=historical_var_es(losses,.99)
        pla_delta=pla_test(simple.hpl,simple.rtpl)
        pla_diagonal=pla_test(diagonal.hpl,diagonal.rtpl)
        pla_full=pla_test(full.hpl,full.rtpl)
        residual_delta=float(simple.residual.abs().mean())
        residual_diagonal=float(diagonal.residual.abs().mean())
        residual_full=float(full.residual.abs().mean())
        validation[desk]={
          "observations":len(simple),"hpl_mean":float(simple.hpl.mean()),
          "hpl_volatility":float(simple.hpl.std(ddof=1)),
          "absolute_residual_mean_delta_only":residual_delta,
          "absolute_residual_mean_diagonal_second_order":residual_diagonal,
          "absolute_residual_mean_full_hessian":residual_full,
          "residual_reduction_diagonal_second_order":float(1-residual_diagonal/max(residual_delta,1e-12)),
          "residual_reduction_full_hessian":float(1-residual_full/max(residual_delta,1e-12)),
          "full_hessian_reduction_vs_diagonal":float(1-residual_full/max(residual_diagonal,1e-12)),
          "historical_risk_97_5":risk97,"historical_risk_99":risk99,
          "backtest_97_5_last_250":backtest_var(hpl_tail,v97_tail,.975),
          "backtest_99_last_250":backtest_var(hpl_tail,v99_tail,.99),
          "pla_delta_only":pla_delta,
          "pla_diagonal_second_order":pla_diagonal,
          "pla_full_hessian":pla_full,
          "largest_abs_residual_dates":full.assign(abs_residual=full.residual.abs()).nlargest(5,"abs_residual")[["date","hpl","rtpl","residual"]].to_dict("records")
        }
        desk_outputs[desk]=full
    pd.concat(combined,ignore_index=True).to_csv(out/"desk_pnl.csv",index=False,date_format="%Y-%m-%d")

    factor_pnl=build_factor_pnl(market)
    factor_pnl.to_csv(out/"factor_pnl_attribution.csv",index=False,date_format="%Y-%m-%d")
    factor_names=list(RISK_CLASSES)
    factor_losses={name:-factor_pnl[name].to_numpy() for name in factor_names}
    joint_liquidity_pnl = build_joint_liquidity_horizon_pnl(market, LIQ_HORIZONS)
    joint_liquidity_pnl.to_csv(
        out/"liquidity_horizon_pnl.csv", index=False, date_format="%Y-%m-%d"
    )
    joint_horizon_losses = {
        horizon: -joint_liquidity_pnl[f"pnl_lh_{horizon}"].to_numpy()
        for horizon in LIQUIDITY_HORIZONS
    }
    liquidity=liquidity_adjusted_es(
        factor_losses, LIQ_HORIZONS, .975, 10,
        joint_horizon_losses=joint_horizon_losses,
        joint_horizon_metadata=joint_liquidity_pnl.attrs["liquidity_horizon_metadata"],
    )
    stress=select_stress_window(factor_losses,250,.975)
    stress["start_date"]=str(market.date.iloc[stress["start_index"]].date())
    stress["end_date"]=str(market.date.iloc[stress["end_index"]].date())
    capital=build_ima_capital(
      factor_losses, stress, RISK_CLASSES, .975, .5,
      observation_dates=factor_pnl["date"],
    )

    evidence=[{"date":market.date.max(),"factor_id":factor,"evidence_type":"public_close","verified":True}
              for factor in factor_names]
    rfet_as_of = pd.Timestamp(market.date.max()).normalize()
    rfet_calibration_start = (
        rfet_as_of - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    )
    rfet=rfet_diagnostic(
        pd.DataFrame(evidence),
        as_of_date=rfet_as_of,
        calibration_period_start=rfet_calibration_start,
        calibration_period_end=rfet_as_of,
    )
    desk_hpl_sum=sum(factor_pnl.attrs["desk_hpl"].values())
    factor_reconciliation={
      "max_abs_reconciliation_error":float(np.max(np.abs(factor_pnl.drop(columns=["date"]).sum(axis=1).to_numpy()-desk_hpl_sum))),
      "unexplained_pnl_mean":float(factor_pnl.unexplained_pnl.mean()),
      "unexplained_pnl_std":float(factor_pnl.unexplained_pnl.std(ddof=1)),
      "unexplained_pnl_is_theta_plus_interactions":True
    }
    desk_status={}
    for desk,metrics in validation.items():
        zone=metrics["pla_delta_only"]["zone"]
        desk_status[desk]={
          "delta_only_pla_zone":zone,
          "diagonal_second_order_pla_zone":metrics["pla_diagonal_second_order"]["zone"],
          "full_hessian_pla_zone":metrics["pla_full_hessian"]["zone"],
          "delta_only_research_pla_green":zone=="green",
          "full_hessian_reduces_residual_vs_delta_only":metrics["residual_reduction_full_hessian"]>0,
          "full_hessian_reduces_residual_vs_diagonal":metrics["full_hessian_reduction_vs_diagonal"]>0
        }
    scenario_stress=build_scenario_reports(market)
    scenario_stress.to_csv(out/"scenario_stress.csv",index=False)
    stress_history_assessment={
      "minimum_required_start_date":"2007-01-01",
      "observed_start_date":str(market.date.min().date()),
      "starts_on_or_before_2007":bool(market.date.min()<=pd.Timestamp("2007-01-01")),
      "regulatory_stress_history_requirement_met":bool(market.date.min()<=pd.Timestamp("2007-01-01")),
      "note":"If false, the dataset does not provide the full historical stress span used by the Basel IMA framework."
    }
    summary={
      "project":"RISK-MANAGE — FRTB IMA Desk Validation & Capital Lab",
      "run_utc":datetime.now(timezone.utc).isoformat(),"data_classification":data_kind,
      "market_data_provenance":provenance,"observations":len(market),
      "date_range":{"start":str(market.date.min().date()),"end":str(market.date.max().date())},
      "desks":validation,"desk_status":desk_status,"factor_pnl_reconciliation":factor_reconciliation,
      "liquidity_adjusted_es":liquidity,"stress_window":stress,"ima_capital":capital,
      "stress_history_assessment":stress_history_assessment,
      "rfet":rfet,"scenario_stress":scenario_stress.to_dict("records"),
      "scope_limitations":[
        "Synthetic runs contain no observed market data.",
        "Positions are constant-maturity hypothetical portfolios, static during each one-day HPL interval and rolled after the daily valuation.",
        "HPL only is available; Actual P&L (APL) is not available, so this is not a complete regulatory backtest.",
        "Public closing prices are not treated as RFET-qualifying transactions or committed quotes.",
        "All positions and notionals are hypothetical; capital and stress figures are illustrative.",
        "The variance-swap mark is a simplified implied-variance proxy, not a full variance-swap valuation.",
        "Liquidity-horizon ES uses position-level joint full revaluation under a simplified instrument/factor map; it is not a complete MAR33 instrument-level mapping or regulatory implementation.",
        "WTI and Brent are spot benchmarks; the commodity book uses spot-linked proxies, not futures settlements, and has no futures curve, basis, expiry or roll model.",
        "This is independent research software, not a supervisory-approved IMA implementation."
      ]
    }
    (out/"desk_validation.json").write_text(json.dumps(json_safe(validation),indent=2))
    (out/"capital_summary.json").write_text(json.dumps(json_safe({
      "liquidity_adjusted_es":liquidity,"stress_window":stress,"ima_capital":capital,
      "rfet":rfet,"factor_pnl_reconciliation":factor_reconciliation
    }),indent=2))
    (out/"executive_summary.json").write_text(json.dumps(json_safe(summary),indent=2))
    (out/"market_data_provenance.json").write_text(json.dumps(json_safe(provenance),indent=2))

    plt.figure(figsize=(11,6))
    for desk,frame in desk_outputs.items():
        plt.plot(frame.date,frame.residual.cumsum()/1e6,label=desk)
    plt.axhline(0,color="black",linewidth=.7)
    plt.title("Cumulative HPL − RTPL residual by desk")
    plt.ylabel("USD millions (illustrative positions)")
    plt.legend(); plt.tight_layout(); plt.savefig(out/"cumulative_pnl_residual.png",dpi=170); plt.close()
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,(desk,frame) in zip(axes.flat,desk_outputs.items()):
        ax.plot(frame.date,frame.hpl,label="HPL",alpha=.8)
        ax.plot(frame.date,frame.rtpl,label="RTPL",alpha=.7)
        ax.set_title(desk); ax.legend()
    fig.suptitle("Daily hypothetical P&L vs risk-theoretical P&L")
    fig.tight_layout(); fig.savefig(out/"desk_hpl_rtpl.png",dpi=170); plt.close(fig)

    md=["# RISK-MANAGE — Executive Risk Validation Report","",
      f"**Data classification:** {data_kind}  ",
      f"**Observations:** {len(market):,}  ",
      f"**Period:** {market.date.min().date()} to {market.date.max().date()}","",
      "## Desk outcomes","","| Desk | Delta-only PLA diagnostic | Diagonal second-order PLA diagnostic | Full-Hessian PLA diagnostic | Full-Hessian residual reduction vs delta-only |",
      "|---|---:|---:|---:|---:|"]
    for desk,m in validation.items():
        md.append(f"| {desk} | {m['pla_delta_only']['zone']} | {m['pla_diagonal_second_order']['zone']} | {m['pla_full_hessian']['zone']} | {m['residual_reduction_full_hessian']:.1%} |")
    imcc_report_line = (
      f"- Research IMCC formula primitive (non-regulatory): USD {capital['imcc']['imcc']:,.2f}."
      if capital["imcc"] is not None
      else f"- Research IMCC formula primitive: not computed; the supplied factor map omits regulatory classes {capital['missing_regulatory_risk_classes']} and/or selected factors omit mapped classes {capital['missing_selected_risk_classes']}."
    )
    holdout_uncertainty = capital["reduced_set"]["holdout"].get("conditional_uncertainty", {})
    block_sensitivity = holdout_uncertainty.get("block_length_sensitivity", {})
    bootstrap_ci_line = "; ".join(
        f"block {block}: {result['confidence_interval'][0]:.3f}–{result['confidence_interval'][1]:.3f}"
        for block, result in block_sensitivity.items()
        if result.get("confidence_interval") is not None
    ) or "not computed"
    md += ["","## Capital and stress outputs","",
      f"- Position-level joint-revaluation liquidity-adjusted ES research diagnostic: USD {liquidity['liquidity_adjusted_es']:,.2f} (10-day base, 97.5%).",
      f"- Liquidity ES calculation method: {liquidity['calculation_method']}; theta excluded from risk-factor shocks.",
      f"- Descriptive 250-observation high-tail window: {stress['start_date']} to {stress['end_date']}.",
      f"- Historical span starts on/before 2007: {stress_history_assessment['starts_on_or_before_2007']} (not a regulatory back-history claim).",
      imcc_report_line,
      f"- Selection-sample reduced/full ES ratio (factor-selection diagnostic only): {capital['reduced_set']['explained_ratio']:.1%}.",
      f"- Reduced/full ES ratio on the non-overlapping 250-observation holdout (research diagnostic): {capital['reduced_set']['holdout'].get('ratio')}.",
      f"- Holdout chronology verified from explicit dates: {capital['reduced_set']['holdout'].get('chronology_verified', False)}.",
      f"- Conditional 95% circular moving-block bootstrap intervals for reduced/full ES ratio (block lengths in overlapping 10-day observations): {bootstrap_ci_line}.",
      "- Bootstrap intervals condition on the frozen selected-factor set and one observed holdout; they do not include selection, model, data or regime uncertainty.",
      "- RFET: public-close-only evidence is not used to certify modellability.",
      "","## Interpretation guardrails","",
      "- HPL backtesting is not APL backtesting.",
      "- All outputs depend on hypothetical positions and declared scenario assumptions.",
      "- Capital numbers are illustrative, not bank capital requirements.",
      "- Passing a test is not proof of economic validity or regulatory approval.","",
      "See the JSON / CSV outputs for the detailed audit trail."]
    (out/"EXECUTIVE_REPORT.md").write_text("\n".join(md),encoding="utf-8")
    print(json.dumps(json_safe({
      "summary_path":str(out/"executive_summary.json"),"date_range":summary["date_range"],
      "desk_status":desk_status,"liquidity_adjusted_es":liquidity["liquidity_adjusted_es"],
      "stress_window":stress,
      "research_imcc_primitive":capital["imcc"]["imcc"] if capital["imcc"] is not None else None,
      "imcc_status":capital["imcc_status"],
      "capital_label":capital["capital_label"],
      "regulatory_status":capital["regulatory_status"],
      "reduced_factor_holdout_is_independent":capital["reduced_set"]["holdout_is_independent"],
      "rfet_status":rfet["status"],
      "max_factor_reconciliation_error":factor_reconciliation["max_abs_reconciliation_error"]
    }),indent=2))

if __name__=="__main__": main()
