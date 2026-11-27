import numpy as np
from riskmanage.fixtures import build_synthetic_market_data
from riskmanage.pnl import build_factor_pnl

def test_factor_pnl_reconciles_to_aggregate_hpl():
    data=build_synthetic_market_data(450,seed=17)
    factors=build_factor_pnl(data)
    desk_total=sum(factors.attrs["desk_hpl"].values())
    reconstructed=factors.drop(columns=["date"]).sum(axis=1).to_numpy()
    assert np.allclose(reconstructed,desk_total,rtol=1e-10,atol=1e-4)
