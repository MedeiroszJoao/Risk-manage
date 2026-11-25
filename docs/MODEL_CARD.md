# Model card — RISK-MANAGE

## Intended use
Educational quantitative market-risk modeling and validation research for a quantitative-risk portfolio. The software demonstrates how valuation, P&L explain, backtesting, PLA, liquidity horizons and capital aggregation fit together.

## Not intended for
Production trading, valuation, hedging, risk limits, regulatory capital reporting, supervisory submission, investment advice or lending decisions.

## Inputs
- Public FRED daily series: SP500, VIXCLS, DEXUSEU, DEXJPUS, DCOILWTICO, DGS2 and DGS10.
- Seeded synthetic factor history for offline tests.
- Hypothetical constant-maturity positions across Equity Options, FX Options, Rates and Commodities.

## Outputs
HPL / RTPL, VaR, ES, backtesting diagnostics, PLA zones, liquidity-adjusted ES, historical stress window, reduced-factor coverage, illustrative IMCC, RFET evidence status and deterministic figures.

## Model choices
- Static positions during each daily HPL interval, followed by a constant-maturity roll at the next valuation point.
- Full HPL repricing.
- Delta-only RTPL versus a more complete per-factor delta / gamma / vega approximation.
- Black–Scholes, Garman–Kohlhagen, zero-coupon Treasury proxies and WTI linear exposure.
- 250-day rolling historical VaR and 250-observation PLA.

## Known limitations
- Synthetic reference data is not empirical evidence.
- HPL-only backtesting; no Actual P&L.
- Public close history is not qualifying RFET transaction / committed-quote evidence.
- The synthetic history begins in 2016 and does not satisfy the full stress history back to 2007.
- Variance-swap price is a simplified implied-variance proxy; options are European vanilla; no volatility surface / smile.
- Rates valuation is a simple Treasury-yield proxy, not production multi-curve swap pricing.
- No DRC, SA, CVA, XVA, intraday risk, valuation adjustments, bank-specific mapping or supervisor approval.
- Passing automated tests does not establish predictive performance or regulatory compliance.
