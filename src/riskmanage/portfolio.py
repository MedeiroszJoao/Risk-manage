from __future__ import annotations
from dataclasses import dataclass

DESKS=("Equity Options","FX Options","Rates","Commodities")

@dataclass(frozen=True)
class Position:
    desk: str
    instrument: str
    quantity: float
    maturity: float = 0.0
    strike: float = 0.0
    multiplier: float = 1.0
    label: str = ""

def build_default_portfolio(initial_market_state: dict[str,float]) -> list[Position]:
    """Hypothetical constant-maturity desk books; no real institution positions."""
    s=float(initial_market_state["sp500"])
    fx=float(initial_market_state["eurusd"])
    return [
        Position("Equity Options","EQUITY_CALL",-1200,maturity=.40,strike=s,
                 multiplier=100,label="Short ATM SPX-style call"),
        Position("Equity Options","EQUITY_PUT",-1200,maturity=.40,strike=s,
                 multiplier=100,label="Short ATM SPX-style put"),
        Position("Equity Options","VARIANCE_SWAP",-35_000_000,maturity=.70,strike=.20,
                 label="Short implied variance — simplified mark-to-market proxy"),
        Position("FX Options","FX_CALL",quantity=45,maturity=.55,
                 strike=fx*1.01,multiplier=100_000,label="Long EURUSD call"),
        Position("FX Options","FX_SPOT",quantity=8_000_000,label="EURUSD spot control"),
        Position("Rates","ZERO_BOND_2Y",quantity=35_000_000,maturity=2.0,label="2Y zero coupon"),
        Position("Rates","ZERO_BOND_10Y",quantity=-27_000_000,maturity=10.0,label="10Y zero coupon short"),
        Position("Commodities","WTI_SPOT_PROXY",quantity=150_000,label="WTI Cushing spot-linked linear proxy; not a listed futures contract"),
    ]
