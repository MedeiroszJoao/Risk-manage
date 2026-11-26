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

- **Market data:** one integrated public-data pipeline combines FRED daily S&P 500, VIX, FX and Treasury series with EIA Petroleum bulk WTI Cushing (PET.RWTC.D) and Europe Brent (PET.RBRTE.D) spot prices. No API key is required. The exact EIA ZIP snapshot, retrieval metadata, series coverage, null counts and SHA-256 hashes are retained; negative WTI spot prices are preserved.
- **Four hypothetical desks:** Equity Options, FX Options, Rates and Commodities.
- **Pricing:** Black–Scholes calls and puts, Garman–Kohlhagen FX calls, Black-76 forward options, a calibrated power-law SSVI implied-volatility surface, and a Heston stochastic-volatility engine with Lewis Fourier inversion, QE-M Monte Carlo bid–ask-aware cross-sectional calibration, and Heston Monte Carlo Greeks (pathwise Delta plus CRN Delta/Gamma/Vega).
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

# Retrieve FRED equity/FX/rates plus EIA WTI/Brent spot, then run the public-data case
python scripts/download_market_data.py --start 2007-01-01
python scripts/run_case.py

# Optional: download the EIA petroleum archive separately (no API key required)
python scripts/download_eia_petroleum_spot.py --start 2007-01-01

pytest -q
ruff check src tests scripts
```

Generated files are written to `reports/generated/`: daily HPL/RTPL, factor attribution, backtesting / PLA, capital diagnostics, RFET evidence status, stress scenarios, figures and an executive report.

## Financial engineering expansion — Heston stochastic volatility and advanced Monte Carlo

The branch now includes Heston risk-neutral dynamics, an affine characteristic function, Lewis square-root Fourier inversion, and an Andersen-style QE-M Monte Carlo engine. Simulation includes antithetic path pairing, an independently estimated discounted-stock control variate, standard errors, confidence intervals, a stock-martingale diagnostic and explicit separation of sampling error from time-discretization bias. The reproducible synthetic-parameter benchmark is run with `python scripts/run_heston_benchmark.py`; derivation and numerical conventions are in [Heston Fourier and QE-M documentation](docs/HESTON_MONTE_CARLO.md).



## Heston calibration and identifiability

The five risk-neutral Heston parameters can now be calibrated to a cross-sectional European option panel. The objective supports weighted mid-price least squares or distance outside bid–ask intervals, deterministic multi-start, robust losses, transformed parameter constraints, and local Jacobian SVD diagnostics for rank and conditioning. Same-expiry quotes use vector-valued Lewis quadrature. The deterministic synthetic quote panel is reproduced with `python scripts/run_heston_calibration.py`; see [Heston Fourier, QE-M and calibration](docs/HESTON_MONTE_CARLO.md).

The calibration is explicitly marked synthetic. No live option quotes, timestamped quote governance, bootstrapped curves, profile likelihood or parameter confidence intervals are claimed. A good price fit is not proof that all five parameters are identifiable.

## Financial engineering expansion — SSVI volatility surface

The branch now includes a power-law SSVI surface, total-variance calibration, Black-76 pricing and implied-volatility inversion, plus direct SSVI calibration to Black-76 prices with optional bid-ask interval loss and a conservative global sufficient-region certificate for static arbitrage. See [SSVI derivation, calibration and validation](docs/VOLATILITY_SURFACE.md). The total-variance and price-space calibrators are deterministic research routines; no live option chain has yet been ingested and dynamic smile evolution is not modeled.

## Important limits — read before interpreting results

This is not a bank's regulatory FRTB implementation, supervisory-approved IMA model, production pricing library or capital calculator.

- The checked-in reference results are synthetic. The integrated public-data pipeline must complete a live FRED/EIA download and its source manifest must be reviewed before describing a risk run as empirical market evidence. FRED's SP500 series currently offers only 10 years of daily history and is copyrighted by S&P Dow Jones Indices; requesting 2007 does not yield a 2007 SP500 history. The integrated panel's actual start date and coverage warnings are written to the manifest.
- The Commodities desk now holds hypothetical linear spot proxies in both WTI Cushing and Europe Brent. These are daily spot benchmarks, not listed futures settlements, executable quotes or forward curves. They cannot directly mark futures positions without an explicit basis/curve and contract-roll model.
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
- [EIA petroleum spot ingestion](docs/EIA_DATA_INGESTION.md)
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
- U.S. Energy Information Administration, [Bulk download facility and PET archive](https://www.eia.gov/opendata/v1/bulkfiles.php)

---
Independent research case study. No affiliation with the BIS, ECB, Federal Reserve, FRED or any financial institution.


For a standalone Heston benchmark, run `python scripts/run_heston_benchmark.py --paths 20000 --steps 96`. Results are written to `reports/generated/heston_benchmark.json`. The parameter set is hypothetical; it is a numerical cross-check, not a market calibration.


## Monte Carlo Greeks

The stochastic-volatility engine estimates Delta by both the pathwise method and common-random-number bump-and-revalue; Gamma and initial-volatility Vega use CRN finite differences. Every estimate includes a sampling standard error and confidence interval, with bump and time-grid bias explicitly excluded. The reproducible comparison against Fourier finite-difference references is generated by `python scripts/run_heston_greeks.py`.


## Dynamic Heston delta hedging and tail risk

The derivatives stack now supports a discrete self-financing hedge experiment. It compares a state-dependent Heston Delta (vector Lewis Fourier derivative across simulated spot/variance states) with a frozen-volatility Black–Scholes Delta, on identical QE-M scenario paths. It separates risk-neutral pricing from physical/scenario path dynamics, includes proportional rehedging costs and reports gross/net hedge P&L, turnover, empirical VaR and fractional-tail ES. Run `python scripts/run_heston_delta_hedging.py --paths 2000 --steps 12 --cost-bps 2`. The physical parameters are hypothetical scenarios, not forecasts; the output is not a market backtest or regulatory capital result.


## Randomized quasi-Monte Carlo

Heston pricing also has an RQMC route using scrambled Sobol nets, inverse-normal mapping, Brownian bridge on both Gaussian drivers, QE-M transitions, and an independent-pilot discounted-stock control variate. The benchmark compares Brownian-bridge RQMC, standard-order RQMC and antithetic pseudo-MC at equal valuation and pilot path budgets, with independent scramble-based standard errors and Fourier reference. Run python scripts/run_heston_qmc_benchmark.py --sobol-power 10 --scrambles 8 --steps 64. RQMC is not assumed to win; results are conditional on the QE-M time grid and exclude time-discretization and model uncertainty.


## Heston calibration uncertainty and risk propagation

A calibrated point estimate is not a known truth. The uncertainty module adds a parametric quote bootstrap, one-dimensional profile likelihood with four nuisance parameters re-optimized at each grid point, and propagation of parameter draws into option prices and Deltas. It reports parameter covariance/correlation, percentile distributions, profile-likelihood curves, and price/Greek uncertainty bands. The quote-noise model is explicit: independent Gaussian quote errors with variance derived from bid–ask width. Run `python scripts/run_heston_uncertainty.py --replicates 40 --profile-points 7`. The synthetic quote panel is not live-market evidence, and these intervals do not measure model risk.


## Calibration uncertainty propagated into hedging tail risk

The uncertainty engine now carries each risk-neutral Heston bootstrap draw into a discrete hedge experiment. It reuses the same Gaussian shocks across Q parameter draws and aligns the initial variance state v0 under Q/P for each draw, while fixing other P dynamics. It recalculates option premium, state-dependent Heston Delta and a frozen-volatility Black–Scholes hedge. The output is a distribution of gross/net hedge VaR and ES, turnover and costs across calibration samples. Run `python scripts/run_heston_hedge_uncertainty.py --bootstrap-replicates 30 --paths 1000 --steps 8 --confidence 0.95`. This is conditional on the one physical scenario sample, the synthetic quote-noise model and the Heston model; it is not regulatory capital or model-risk coverage.


## Crossed calibration/scenario uncertainty decomposition

FE-010 evaluates every Heston calibration bootstrap draw over multiple independent physical/scenario seeds. Within each seed, the same Gaussian innovations are reused across Q draws; between seeds, independent scenarios are generated. A balanced two-way random-effects method-of-moments decomposition estimates calibration-parameter, scenario-sampling and interaction/within-cell Monte Carlo variance components for Heston ES, frozen-BS ES and paired ES/VaR differences. Run `python scripts/run_heston_nested_hedge_risk.py --bootstrap-replicates 20 --scenario-replicates 6 --paths 1000 --steps 8`. Small crossed designs are diagnostics; variance components are noisy and are not confidence intervals.


## Extreme-value theory for hedge-loss tails

The project now compares empirical VaR/ES with Peaks-over-Threshold Generalized Pareto (POT-GPD) extrapolation for Heston and frozen-Black–Scholes hedge losses, gross and net. It reports multiple thresholds, exceedance counts, MLE shape/scale, mean-excess diagnostics, descriptive KS distance, and conditional exceedance-bootstrap intervals. No threshold is auto-selected. Run \`python scripts/run_heston_hedge_evt.py --paths 10000 --steps 12 --cost-bps 2 --confidence-levels 0.975 0.99 --thresholds 0.90 0.925 0.95 --min-exceedances 50 --bootstrap 200\`. ES is undefined under the fitted GPD when shape \(\xi\geq1\); threshold and dependence uncertainty remain outside its bootstrap interval.
