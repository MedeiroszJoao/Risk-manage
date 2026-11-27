# RISK-MANAGE — FRTB IMA Desk Validation & Capital Lab

## Intent
Build a high-signal portfolio case resembling real quantitative market-risk work: transform market data into instrument-level hypothetical P&L, compare a risk model against full valuation, reject model deficiencies using desk-level tests, and calculate an explicitly illustrative FRTB-inspired capital stack.

## Goals
1. Source and validate daily public market histories for equity, implied volatility, FX, commodity and interest-rate risk factors.
2. Price a varied set of instruments across four trading desks.
3. Generate desk HPL by full revaluation and RTPL via a separately specified risk model.
4. Run 97.5% / 99% rolling one-day VaR backtests, coverage / independence diagnostics, and 250-day PLA.
5. Implement Basel-inspired 97.5% ES, 10/20/40/60/120-day liquidity adjustment, stress calibration, IMA capital aggregation and NMRF-style scenario add-ons.
6. Make deliberately incomplete risk models fail validation for explainable reasons while retaining a linear desk as a control.
7. Ship deterministic network-free tests, a source manifest, reproducible runner, machine-readable results, CI and reviewer-facing documentation.

## Non-goals
- A bank's regulatory FRTB implementation or supervisory-approved IMA model.
- Treating public market data as automatically valid RFET evidence.
- Generating Actual P&L, front-office P&L, proprietary quote evidence or model approval.
- DRC, SA, CVA, XVA, full SABR cubes, intraday risk or production curve conventions.
- Presenting synthetic fixtures as observed financial outcomes.

## Approaches considered
1. Generic VaR notebook: low effort, but lacks model governance, valuation and a demonstrable rejection test. Rejected.
2. Credit PD/LGD simulation: useful, but overlaps an earlier project and does not exercise trading-desk market-risk validation. Rejected.
3. FRTB-inspired multi-asset desk validation and capital lab: joins pricing, observed market history, HPL/RTPL, PLA/backtesting, liquidity horizons, stressed ES and NMRF governance. Selected.

## Architecture and decisions
- Python 3.11+, NumPy, pandas, SciPy, requests, matplotlib; pytest / Ruff.
- Public FRED CSV series supply equity, FX and rates; the official EIA Petroleum bulk archive supplies WTI Cushing and Europe Brent spot benchmarks. A reproducible script joins them into one market panel and retains source provenance; deterministic synthetic fixtures support offline tests.
- Risk factors: S&P 500, VIX, EUR/USD, USD/JPY, WTI Cushing spot, Europe Brent spot, US 2Y and 10Y yields.
- Desks: Equity Options, FX Options, Rates, Commodities.
- HPL: static positions from prior close, full revaluation on current-close market data. RTPL: independent risk-theoretical approximation from declared factor subset and sensitivities.
- VaR backtests use HPL only; lack of APL is disclosed.
- PLA uses 250 observations, Spearman correlation and empirical CDF distance with Basel-style zones.
- RFET consumes an evidence ledger; public closing prices alone cannot certify real-price modellability.
- Capital outputs are illustrative because a real IMA requires verified RFET evidence, full regulatory risk-factor mapping, daily controls and supervisory approval.

## Anti-goals
- No arbitrary constants silently labelled regulatory parameters.
- No look-ahead in rolling risk estimates or volatility proxies.
- No fabricated real-price observations or APL.
- No green-zone claim unless implemented metric inputs meet stated sample and threshold logic.
- No equation of software tests with predictive / regulatory validation.

## Acceptance criteria
- Fresh clone runs offline tests with pinned dependencies.
- Real-data ingestion records series IDs, source URLs, retrieval time, missingness and SHA-256.
- HPL, RTPL, VaR, ES, liquidity horizons, stress period, IMCC, PLA, backtesting and RFET diagnostics have direct tests.
- Same seed and inputs reproduce numeric outputs within documented tolerances.
- Intentional model misspecification produces different HPL and RTPL and a validation explanation identifying omitted drivers.
- Reports distinguish observed public market data, proxies, synthetic fixtures, empirical estimates and regulatory assumptions.


## Data-source and commodity-book addendum — 2026-11-26

- The EIA PET bulk download is public and requires no API key. The collector retains and hashes the exact ZIP snapshot and parses only PET.RWTC.D and PET.RBRTE.D.
- WTI Cushing spot may be zero or negative; the data contract preserves finite negative observations rather than imposing an invalid positivity constraint. The WTI volatility proxy uses scaled additive price changes, not logarithmic returns.
- The hypothetical commodities desk is long WTI Cushing and short Europe Brent using linear spot-linked proxies. These are not NYMEX/ICE futures positions, and the system does not claim to model futures basis, contract month, expiry, roll or forward curves.
- The integrated market panel uses EIA as the canonical source for both crude-oil benchmarks, avoiding mismatched WTI and Brent feed definitions.
