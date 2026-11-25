# RISK-MANAGE — FRTB IMA Desk Validation & Capital Lab

**Independent quantitative market-risk engineering case study.** The central question:

> Can a trading desk's risk model reproduce the risks that actually drive its hypothetical P&L — and what should happen when it cannot?

This repository implements an auditable, multi-asset research lab inspired by the Basel Fundamental Review of the Trading Book (FRTB) Internal Models Approach. It connects market data, nonlinear instrument repricing, hypothetical P&L (HPL), risk-theoretical P&L (RTPL), VaR backtesting, P&L attribution (PLA), Expected Shortfall, liquidity horizons, stress calibration, reduced-factor coverage and illustrative IMA capital aggregation.

## The result that makes this case interesting

The deterministic reference run uses 1,800 **synthetic** business-day observations (2016-01-04 to 2022-11-25), not empirical bank data.

| Equity-options desk diagnostic | Delta-only model | Model with gamma + vega |
|---|---:|---:|
| PLA Spearman correlation | 0.375 | 0.999 |
| PLA KS distance | 0.396 | 0.024 |
| PLA zone | **Red** | **Green** |
| Mean absolute HPL − RTPL residual | USD 5.63m | USD 0.58m |

The full risk-factor model reduces the mean absolute P&L residual by **89.7%**. Yet the delta-only model records just **3 VaR exceptions in 250 observations at 99%**, which falls in the green backtesting zone. That contrast is deliberate: a quiet VaR exception count does not prove that the model captures the desk's economic P&L drivers.

Run it yourself with the deterministic seed to reproduce the figures.

## What is implemented

- **Market data:** reproducible public FRED CSV downloader for S&P 500, VIX, EUR/USD, USD/JPY, WTI and US Treasury yields; source URLs, retrieval timestamps, missingness and SHA-256 hashes are recorded.
- **Four hypothetical desks:** Equity Options, FX Options, Rates and Commodities.
- **Pricing:** Black–Scholes calls and puts, Garman–Kohlhagen FX calls, zero-coupon bond proxies, a clearly labelled simplified variance-swap mark, and a calibrated power-law SSVI implied-volatility surface with a sufficient static-arbitrage certificate.
- **HPL / RTPL:** close-to-close full revaluation of positions held static during each daily interval; constant-maturity test books are rolled only after daily valuation. RTPL is independently built from sensitivities and explicitly selected risk factors.
- **VaR backtesting:** rolling 97.5% / 99% historical VaR, exceptions, Kupiec unconditional-coverage and Christoffersen independence diagnostics.
- **PLA:** the latest 250 matched observations, Spearman rank correlation, empirical CDF / KS distance and Basel-style green / amber / red zones.
- **Tail risk and capital:** 97.5% ES, 10/20/40/60/120-day liquidity-adjusted ES, historical 250-day stress window, reduced-factor selection with rolling 12-week 75% coverage, constrained / unconstrained IMCC and NMRF aggregation primitives.
- **RFET evidence control:** a public close series is not automatically treated as a qualifying transaction or committed quote.
- **Reproducibility:** deterministic offline fixtures, automated formula and invariant tests, machine-readable outputs, charts and GitHub Actions.

## Run it

Python 3.11+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"

# Deterministic offline reference case
python scripts/run_case.py --synthetic

# Or retrieve public FRED series, then run the public-data case
python scripts/download_market_data.py --start 2007-01-01
python scripts/run_case.py
pytest -q
ruff check src tests scripts
```

Generated files are written to `reports/generated/`: daily HPL/RTPL, factor attribution, backtesting / PLA, capital diagnostics, RFET evidence status, stress scenarios, figures and an executive report.

## Financial engineering expansion — SSVI volatility surface

The branch now includes a power-law SSVI surface, total-variance calibration, Black-76 pricing and implied-volatility inversion, plus direct SSVI calibration to Black-76 prices with optional bid-ask interval loss and a conservative global sufficient-region certificate for static arbitrage. See [SSVI derivation, calibration and validation](docs/VOLATILITY_SURFACE.md). The total-variance and price-space calibrators are deterministic research routines; no live option chain has yet been ingested and dynamic smile evolution is not modeled.

## Important limits — read before interpreting results

This is not a bank's regulatory FRTB implementation, supervisory-approved IMA model, production pricing library or capital calculator.

- The public-data pipeline is implemented; the checked-in reference results are synthetic. Do not describe them as market observations or empirical backtesting.
- HPL is available; bank Actual P&L (APL) is not. Backtesting is therefore HPL-only and not a complete regulatory backtest.
- The synthetic history begins in 2016, and the public daily S&P 500 FRED series does not provide the full stress history back to 2007. The stress-capital outputs cannot satisfy that historical-data requirement.
- Public closes are not RFET-qualifying evidence by themselves. No factor is certified modellable without independently reviewed transaction / committed-quote evidence.
- The equity variance-swap mark is a simplified implied-variance proxy. Rates valuation is a simplified Treasury-yield proxy, not production multi-curve swap pricing.
- Position quantities are hypothetical; there is no real bank portfolio or supervisory approval.

## Technical documentation

- [Case study](docs/CASE_STUDY.md)
- [Methodology and formula map](docs/METHODOLOGY.md)
- [Model card and limitations](docs/MODEL_CARD.md)
- [Data provenance](docs/DATA_PROVENANCE.md)
- [Validation protocol](docs/VALIDATION_PROTOCOL.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Design specification](docs/superpowers/specs/2026-11-24-frtb-ima-risk-engine-design.md)
- [Implementation plan](docs/superpowers/plans/2026-11-24-frtb-ima-risk-engine.md)

## Primary standards and data

- Basel Committee, [MAR31 — risk-factor modellability / RFET](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/31/inforce/2023-01-01/published/2020-03-27)
- Basel Committee, [MAR32 — backtesting and P&L attribution](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/32/inforce/2023-01-01/published/2020-03-27)
- Basel Committee, [MAR33 — IMA capital requirements](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/33/inforce/2023-01-01/published/2020-06-05)
- ECB, [Guide to internal models, July 2025](https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.supervisory_guide202507.en.pdf)
- Federal Reserve Bank of St. Louis, [FRED API documentation](https://fred.stlouisfed.org/docs/api/fred/)

---
Independent research case study. No affiliation with the BIS, ECB, Federal Reserve, FRED or any financial institution.
