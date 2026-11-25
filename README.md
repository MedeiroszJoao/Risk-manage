# RISK-MANAGE — FRTB IMA Desk Validation & Capital Lab

**Independent quantitative market-risk engineering case study.** The central question:

> Can a trading desk's risk model reproduce the risks that actually drive its hypothetical P&L — and what should happen when it cannot?

This repository implements an auditable, multi-asset research laboratory inspired by the Basel Fundamental Review of the Trading Book (FRTB) Internal Models Approach (IMA). It connects market data, nonlinear instrument repricing, hypothetical P&L (HPL), risk-theoretical P&L (RTPL), historical VaR / Expected Shortfall, desk-level backtesting, P&L attribution (PLA), liquidity horizons, stress calibration and an illustrative IMA capital aggregation.

## Why this is difficult

A risk number on its own is not sufficient. A risk function must connect positions to daily market observations; the valuation layer must explain P&L; independent tests must be able to reject a deficient model; and capital calculations must preserve differences between risk classes, liquidity horizons, stressed observations and non-modellable risk factors.

`market data → curves / surfaces → full revaluation → HPL`
`market data → risk-factor model → RTPL / VaR / ES → validation`
`validation + stress calibration + liquidity horizons → illustrative IMA capital`

## Core capabilities

- **Multi-asset market data:** S&P 500, VIX, EUR/USD, USD/JPY, WTI crude and US Treasury yields from public FRED series.
- **Instrument pricing:** Black–Scholes equity options, Garman–Kohlhagen FX options, zero-coupon / duration fixed-income exposures, equity / FX / commodity delta-one positions.
- **P&L explain:** static-position HPL from full revaluation versus separately generated RTPL from a declared risk-factor subset and sensitivities.
- **Desk validation:** 97.5% / 99% rolling one-day VaR backtesting, Kupiec coverage and Christoffersen independence diagnostics, and the 250-observation PLA test.
- **Tail risk:** 97.5% ES, stressed-period selection, liquidity-adjusted ES across 10/20/40/60/120-day horizons and constrained / unconstrained IMCC.
- **Model-risk demonstrations:** deliberately incomplete option risk models, omitted gamma / vega, missing factor history, stale observations and proxy-only price histories.
- **Auditability:** data-source manifest, pinned run configuration, deterministic fixtures, formula tests, machine-readable reports and CI.

## Quick start

Python 3.11+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/download_market_data.py
python scripts/run_case.py
pytest -q
```

Offline deterministic run:

```bash
python scripts/run_case.py --synthetic
pytest -q
```

Outputs are written under `reports/generated/`: desk P&L, VaR backtests, PLA, ES / IMCC components, stress scenarios, data diagnostics, figures and an executive JSON summary.

## Integrity boundaries

This is independent research software, not a bank model, regulatory submission, capital calculator, investment recommendation or supervisory approval package. Public daily closing values are useful for experimentation, but **do not automatically satisfy Basel's definition of eligible real-price observations** for the Risk Factor Eligibility Test (RFET). The RFET component distinguishes observed price history from verified transaction / committed-quote evidence and will not certify modellability from public closes alone.

Likewise, public close-to-close hypothetical P&L is not Actual P&L (APL). This project has no bank's official desk valuation systems, intraday trades, valuation adjustments, vendor evidence or supervisory approvals. Results are labelled as an independent FRTB-inspired research implementation.

## Read next

- [Case study](docs/CASE_STUDY.md)
- [Methodology and formula map](docs/METHODOLOGY.md)
- [Model card and limitations](docs/MODEL_CARD.md)
- [Data provenance](docs/DATA_PROVENANCE.md)
- [Validation protocol](docs/VALIDATION_PROTOCOL.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Design specification](docs/superpowers/specs/2026-11-24-frtb-ima-risk-engine-design.md)
- [Implementation plan](docs/superpowers/plans/2026-11-24-frtb-ima-risk-engine.md)

## Primary standards and data

- Basel Committee, [MAR31 — model requirements and RFET](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/31/inforce/2023-01-01/published/2020-03-27)
- Basel Committee, [MAR32 — backtesting and P&L attribution](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/32/inforce/2023-01-01/published/2020-03-27)
- Basel Committee, [MAR33 — IMA capital requirements](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/33/inforce/2023-01-01/published/2020-06-05)
- ECB, [Guide to internal models, July 2025](https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.supervisory_guide202507.en.pdf)
- Federal Reserve Bank of St. Louis, [FRED API documentation](https://fred.stlouisfed.org/docs/api/fred/)

---
Independent case study. No affiliation with the BIS, ECB, Federal Reserve, FRED or any financial institution.
