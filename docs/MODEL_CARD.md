# Model card — RISK-MANAGE

## Intended use
Educational quantitative market-risk modeling and validation research for a quantitative-risk portfolio. The software demonstrates how valuation, P&L explain, backtesting, PLA, liquidity horizons and capital aggregation fit together.

## Not intended for
Production trading, valuation, hedging, risk limits, regulatory capital reporting, supervisory submission, investment advice or lending decisions.

## Inputs
- Public FRED daily series: SP500, VIXCLS, DEXUSEU, DEXJPUS, DGS2 and DGS10.
- Public EIA daily crude-oil spot benchmarks: WTI Cushing and Europe Brent (via PET.zip bulk archive).
- Seeded synthetic factor history for offline tests.
- Hypothetical constant-maturity positions across Equity Options, FX Options, Rates and Commodities.

## Outputs
HPL / RTPL, VaR, ES, backtesting diagnostics, PLA zones, liquidity-adjusted ES, historical stress window, reduced-factor coverage, illustrative IMCC, RFET evidence status and deterministic figures.

## Model choices
- Static positions during each daily HPL interval, followed by a constant-maturity roll at the next valuation point.
- Full HPL repricing.
- Delta-only RTPL versus a more complete per-factor delta / gamma / vega approximation.
- Black–Scholes, Garman–Kohlhagen, zero-coupon Treasury proxies and long WTI / short Europe Brent linear spot-proxy exposures (not listed futures).
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


## Evidence state

This is implemented research software with unit and numerical tests. Selected numerical components have analytic-limit or controlled-simulation evidence. Empirical market validation on independent option quote panels or real desk portfolios has not been completed. The HPL / RTPL books are hypothetical; there is no APL.

The reduced-factor coverage currently reported is overlapping in-sample description, not holdout validation. RFET returns only an evidence-count screen for independent review, never a modellability pass. IMCC / NMRF outputs are research formula primitives, not regulatory capital.

See [MATURITY_REGISTER.md](MATURITY_REGISTER.md) for the component-by-component state and acceptance gaps.
