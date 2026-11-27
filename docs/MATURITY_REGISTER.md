# Maturity, scope and evidence register

**Last reviewed:** 2026-11-26  
**Purpose:** prevent implemented code, synthetic reproduction, empirical evidence and regulatory validation from being conflated.

## Project identity

RISK-MANAGE is a quantitative-finance research prototype for market-risk measurement, P&L explain and model-risk diagnostics. It is not a bank risk platform, an approved internal model, a production valuation library or a regulatory capital calculator.

## Evidence states

| State | Meaning in this repository |
|---|---|
| Implemented | Code exists and is reachable from a documented entry point. |
| Unit-tested | Relevant automated tests pass; this does not establish market validity. |
| Analytically benchmarked | Compared with an analytic identity, special case or separately derived reference. |
| Synthetic reproduction | Reproduced using generated data with known parameters. |
| Public-data ingestion verified | Download/parse/provenance pipeline executed against the named public source. |
| Empirically validated | Tested on independent observed instrument/portfolio data with point-in-time controls and an untouched holdout. |
| Regulatory assessment | Checked against the current applicable standard, complete data/evidence and independent model governance. |

These states are not interchangeable. A test of a synthetic model is not an empirical backtest. A public closing price is not an actual trading book or an RFET-qualified observation.

## Component maturity

| Component | Implemented | Independent / analytical evidence | Empirical status | Regulatory status |
|---|---|---|---|---|
| Historical VaR / ES | Yes | Quantile convention and boundary-mass/property tests | No independent trading portfolio | Not assessed |
| Rolling historical VaR | Yes | No-lookahead and quantile-convention tests | No held-out desk forecasts | Not assessed |
| Kupiec / Christoffersen | Yes | Likelihood-ratio tests and edge cases; degenerate samples return not-estimable | HPL-only diagnostic | Incomplete without APL |
| PLA | Yes | Spearman / empirical CDF checks and undefined-correlation handling | Hypothetical HPL / RTPL only | Not a regulatory desk PLA |
| Black–Scholes / GK / Black-76 | Yes | Analytic properties and selected cross-checks | No timestamped real option chain | Not production |
| Heston Fourier | Yes | Characteristic-function normalization, martingale forward, deterministic-variance Black–Scholes limit, parity and bounds | No live quotes | Not calibrated to market |
| Heston QE-M | Yes | Conditional moment / martingale diagnostics and Monte Carlo comparison against Fourier | Synthetic parameters only; CI runs 16/32/64/128/256-step sensitivity, but no accepted bias bound or market-calibrated time-grid criterion is established | Not assessed |
| SSVI | Yes | Static-arbitrage sufficient-region checks and synthetic parameter/price recovery | No real option chain or stale/bid-ask governance | Not assessed |
| POT-GPD | Yes | Synthetic known-GPD recovery, threshold grid and bootstrap diagnostics | No independent chronological financial loss series | Not assessed |
| Liquidity-adjusted ES | Research implementation | Formula-shape and unit tests | Factor horizon mapping remains illustrative | MAR33 mapping not implemented |
| Reduced-factor selection | Research implementation | In-sample overlapping descriptive coverage only | No out-of-sample factor-set validation | IMA requirements not established |
| IMCC / NMRF primitives | Formula primitives | Unit examples only | The current book lacks credit-spread factors, and the stress-period / group-level reduced-factor procedures are not implemented | Not a regulatory calculation |
| Data ingestion | FRED/EIA pipelines and SEC N-PORT accession parser | Live FRED/EIA checks; N-PORT parser unit-tested against three position records transcribed from one official filing | Full quarterly N-PORT archive and complete filing not yet reconciled; no instrument-level return reconstruction | N-PORT holdings are not RFET evidence or bank desk positions |
| Portfolio / HPL / RTPL | Yes, hypothetical constant-maturity test books | P&L reconciliation and formula/property tests | No real bank desk positions or APL | Regulatory backtest/PLA incomplete |

## Publicly disclosed fund-holdings track\n\nA first SEC N-PORT parsing path is implemented. Its test data contains three selected positions from one iShares Core S&P 500 ETF filing, not the complete holding set. The parser is not yet validated against an entire quarterly bulk archive or reconciled to every position in the original filing. The output is a point-in-time fund disclosure, not bank desk history, transactions or APL.\n\n## Critical omissions for a real desk-level validation

1. Actual P&L (APL), including the bank's treatment of fees, reserves, valuation adjustments, intraday P&L and other supervisory exclusions.
2. Actual historical positions and frozen forecasts / model versions with point-in-time market data and timestamps.
3. Full instrument identifiers, contract conventions, curves, volatility surfaces, calendars, corporate actions, futures rolls and basis.
4. Independent risk-factor mapping and source evidence for RFET, including third-party vendor audit and representativeness.
5. Complete MAR33 risk-factor liquidity-horizon mapping, IMCC-based stress-period selection, group-level reduced-factor assessment, all five broad risk classes and NMRF stress scenarios.
6. One-year APL/HPL desk-level backtesting and PLA, independent validation, governance, exception escalation, change control and supervisory requirements.

## Acceptance criteria before any stronger claim

- A clean environment can install a fully specified dependency set and reproduce all published outputs.
- Independent references or worked examples validate every pricing/risk formula claimed as verified.
- Forecasts use only information available at their decision timestamps; holdout periods are never used for model choice.
- Empirical evaluation uses observed, appropriately licensed data, with stable identifiers, point-in-time provenance and an untouched test period.
- HPL and APL are both available and reconciled for any claim of regulatory VaR backtesting.
- Desk-level HPL/RTPL, RFET, stress period, factor classes, liquidity horizons and NMRF evidence are checked against the applicable Basel standard.
- Every report carries a scope label and machine-readable provenance.

## Explicit conclusion

The codebase is a useful research implementation with tested numerical components and public-data ingestion. It is **not** yet an empirically validated trading-risk engine and does **not** implement a complete FRTB IMA capital process. No current result supports a claim of bank-level performance, model approval or regulatory capital readiness.
