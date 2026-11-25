import numpy as np
import pandas as pd
from riskmanage.validation import kupiec_pof, christoffersen_independence, pla_test, backtest_var, rfet_diagnostic

def test_kupiec_no_exceptions_is_finite():
    result=kupiec_pof(0,250,.99)
    assert np.isfinite(result["lr_statistic"])
    assert 0<=result["p_value"]<=1

def test_christoffersen_clustered_exceptions_are_not_more_independent():
    spread=np.array([1,0,0,0,1,0,0,0]*30,dtype=int)
    clustered=np.array([1,1,1,0,0,0,0,0]*30,dtype=int)
    assert christoffersen_independence(clustered)["lr_statistic"] > christoffersen_independence(spread)["lr_statistic"]

def test_pla_detects_perfect_match():
    x=np.linspace(-5,5,250)
    result=pla_test(x,x,250)
    assert result["zone"]=="green"
    assert result["spearman"]>0.999

def test_pla_requires_250_observations():
    assert pla_test(np.arange(100),np.arange(100))["zone"]=="insufficient_sample"

def test_price_only_rfet_evidence_does_not_pass():
    ledger=pd.DataFrame({"date":pd.date_range("2025-01-01",periods=120,freq="3D"),
                         "factor_id":"SPX","evidence_type":"public_close","verified":True})
    result=rfet_diagnostic(ledger)
    assert result["factors"]["SPX"]["classification"]=="not_evidenced"
