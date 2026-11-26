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
from riskmanage.pnl import DESK_FACTORS, build_desk_pnl, build_factor_pnl, enrich_market_data, market_state, value_desk
from riskmanage.portfolio import DESKS, build_default_portfolio
from riskmanage.risk import historical_var_es, rolling_historical_var, liquidity_adjusted_es
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
      "oil_collapse":{"wti":.60,"brent":.65,"wti_vol":1.50},
      "combined_tail":{"sp500":.75,"vix_add":25,"eurusd":.90,"eurusd_vol":1.75,
        "usdjpy":1.08,"usdjpy_vol":1.50,"ust2y_add":2.0,"ust10y_add":1.5,
        "wti":.60,"brent":.65,"wti_vol":1.50}
    }
    rows=[]
    for name,shock in scenarios.items():
        state=dict(current)
        for factor in ("sp500","eurusd","usdjpy","wti","brent"):
            if factor in shock: state[factor]*=shock[factor]
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
        market=load_market_data(args.data); data_kind="public_fred_series"
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
        full=build_desk_pnl(market,desk,"full_greeks")
        simple["desk"]=desk; full["desk"]=desk
        simple["risk_model"]="delta_only"; full["risk_model"]="full_greeks"
        combined.extend([simple,full])
        losses=-simple.hpl.to_numpy()
        var97=rolling_historical_var(losses,250,.975)
        var99=rolling_historical_var(losses,250,.99)
        hpl_tail=simple.hpl.to_numpy()[-250:]
        v97_tail=var97.to_numpy()[-250:]; v99_tail=var99.to_numpy()[-250:]
        risk97=historical_var_es(losses,.975); risk99=historical_var_es(losses,.99)
        pla_delta=pla_test(simple.hpl,simple.rtpl); pla_full=pla_test(full.hpl,full.rtpl)
        validation[desk]={
          "observations":len(simple),"hpl_mean":float(simple.hpl.mean()),
          "hpl_volatility":float(simple.hpl.std(ddof=1)),
          "absolute_residual_mean_delta_only":float(simple.residual.abs().mean()),
          "absolute_residual_mean_full_greeks":float(full.residual.abs().mean()),
          "residual_reduction_full_greeks":float(1-full.residual.abs().mean()/max(simple.residual.abs().mean(),1e-12)),
          "historical_risk_97_5":risk97,"historical_risk_99":risk99,
          "backtest_97_5_last_250":backtest_var(hpl_tail,v97_tail,.975),
          "backtest_99_last_250":backtest_var(hpl_tail,v99_tail,.99),
          "pla_delta_only":pla_delta,"pla_full_greeks":pla_full,
          "largest_abs_residual_dates":simple.assign(abs_residual=simple.residual.abs()).nlargest(5,"abs_residual")[["date","hpl","rtpl","residual"]].to_dict("records")
        }
        desk_outputs[desk]=simple
    pd.concat(combined,ignore_index=True).to_csv(out/"desk_pnl.csv",index=False,date_format="%Y-%m-%d")

    factor_pnl=build_factor_pnl(market)
    factor_pnl.to_csv(out/"factor_pnl_attribution.csv",index=False,date_format="%Y-%m-%d")
    factor_names=list(RISK_CLASSES)
    factor_losses={name:-factor_pnl[name].to_numpy() for name in factor_names}
    liquidity=liquidity_adjusted_es(factor_losses,LIQ_HORIZONS,.975,10)
    stress=select_stress_window(factor_losses,250,.975)
    stress["start_date"]=str(market.date.iloc[stress["start_index"]].date())
    stress["end_date"]=str(market.date.iloc[stress["end_index"]].date())
    capital=build_ima_capital(factor_losses,stress,RISK_CLASSES,.975,.5)

    evidence=[{"date":market.date.max(),"factor_id":factor,"evidence_type":"public_close","verified":True}
              for factor in factor_names]
    rfet=rfet_diagnostic(pd.DataFrame(evidence))
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
        desk_status[desk]={"delta_only_pla_zone":zone,
          "full_greeks_pla_zone":metrics["pla_full_greeks"]["zone"],
          "delta_only_passes_pla":zone=="green",
          "remediation_improves_residual":metrics["residual_reduction_full_greeks"]>0}
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
      "## Desk outcomes","","| Desk | PLA delta-only | PLA with gamma / vega | Mean abs residual reduction |",
      "|---|---:|---:|---:|"]
    for desk,m in validation.items():
        md.append(f"| {desk} | {m['pla_delta_only']['zone']} | {m['pla_full_greeks']['zone']} | {m['residual_reduction_full_greeks']:.1%} |")
    md += ["","## Capital and stress outputs","",
      f"- Liquidity-adjusted ES: USD {liquidity['liquidity_adjusted_es']:,.2f} (10-day base, 97.5%).",
      f"- Selected 250-day stress window: {stress['start_date']} to {stress['end_date']}.",
      f"- Historical span starts on/before 2007: {stress_history_assessment['starts_on_or_before_2007']} (not a regulatory back-history claim).",
      f"- Illustrative IMCC: USD {capital['imcc']['imcc']:,.2f}.",
      f"- Reduced-factor current ES coverage: {capital['reduced_set']['explained_ratio']:.1%}.",
      f"- Reduced-factor 12-week mean coverage: {capital['reduced_set']['coverage_12w'].get('mean_ratio')}.",
      "- RFET: public-close-only evidence is not used to certify modellability.",
      "","## Interpretation guardrails","",
      "- HPL backtesting is not APL backtesting.",
      "- All outputs depend on hypothetical positions and declared scenario assumptions.",
      "- Capital numbers are illustrative, not bank capital requirements.",
      "- Passing a test is not proof of economic validity or regulatory approval.","",
      "See the JSON / CSV outputs for the detailed audit trail."]
    (out/"EXECUTIVE_REPORT.md").write_text("\\n".join(md),encoding="utf-8")
    print(json.dumps(json_safe({
      "summary_path":str(out/"executive_summary.json"),"date_range":summary["date_range"],
      "desk_status":desk_status,"liquidity_adjusted_es":liquidity["liquidity_adjusted_es"],
      "stress_window":stress,"illustrative_imcc":capital["imcc"]["imcc"],
      "rfet_status":rfet["status"],
      "max_factor_reconciliation_error":factor_reconciliation["max_abs_reconciliation_error"]
    }),indent=2))

if __name__=="__main__": main()
