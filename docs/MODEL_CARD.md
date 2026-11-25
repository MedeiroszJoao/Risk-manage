# Model card — RISK-MANAGE

## Intended use
Educational quantitative market-risk modeling and validation research for portfolio demonstration.

## Not intended for
Production trading, valuation, hedging, risk limits, capital reporting, regulatory submission or lending decisions.

## Inputs
Public FRED market series for S&P 500, VIX, major FX, WTI and US Treasury yields. Hypothetical desk positions. Deterministic synthetic data for offline tests.

## Outputs
HPL / RTPL, VaR, ES, backtesting diagnostics, PLA zones, liquidity-adjusted ES, illustrative capital components and RFET evidence warnings.

## Method
Full revaluation of static positions for HPL; sensitivity-based RTPL; historical risk statistics; desk-level backtest and PLA; stress window and reduced-factor analysis; liquidity-adjusted ES / illustrative IMCC.

## Known limitations
- Public daily close history is not automatically qualifying RFET evidence.
- No Actual P&L / intraday P&L or official desk valuation data.
- No full Basel regulatory implementation, DRC, SA or production multi-curve framework.
- Historical series may not cover the required stress period back to 2007.
- Simplified instrument conventions and hypothetical positions.
- Synthetic CI outcomes do not validate predictive performance.
