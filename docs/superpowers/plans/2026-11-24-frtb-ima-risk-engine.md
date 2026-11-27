# FRTB IMA Desk Validation & Capital Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build an auditable FRTB-inspired multi-asset market-risk validation and capital research engine in Risk-manage.

**Architecture:** Separate data ingestion, instrument valuation, P&L production, risk metrics, validation and capital aggregation. A deterministic fixture enables offline tests; public FRED equity/FX/rates and EIA PET bulk WTI/Brent spot series are retrieved separately and combined through a provenance-preserving panel builder.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, requests, matplotlib, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-24-frtb-ima-risk-engine-design.md`

## Global Constraints
- Public historical closes are not automatically RFET-valid transaction / committed-quote observations.
- HPL is static-position full-revaluation; APL is not available.
- No claim of supervisory approval, regulatory compliance or production readiness.
- FRED source IDs, request URLs, retrieval timestamp, missingness and SHA-256 must be recorded.
- Historical data transforms must avoid look-ahead.
- Synthetic tests never access network.

## Review Focus
- Missing / stale market data: fail explicitly or flag; never hide material gaps.
- Volatility look-ahead: only information available at valuation close can price that close.
- Non-positive / implausible prices and yields: reject or report, don't silently coerce. Exception: finite zero/negative WTI Cushing spot prices are valid historical observations and must be preserved; Brent, equity indices and FX levels remain positive.
- Nonlinear options: HPL full-revalues; delta-only RTPL can fail PLA.
- Sparse tail sample: report tail observation count; don't overstate precision.

---

### Task 1: Data contracts, FRED ingestion and fixture
**Files:** `src/riskmanage/data.py`, `src/riskmanage/fixtures.py`, `scripts/download_market_data.py`, `tests/test_data.py`.
**Interfaces:** `load_market_data(path)`, `build_synthetic_market_data(n=1800, seed=20261124)`, `download_fred_series(start, end, series_ids)`. Required columns: date, sp500, vix, eurusd, usdjpy, wti, brent, ust2y, ust10y.
- [ ] Write tests for schema, reproducibility, duplicate dates, missing columns, non-positive prices and gaps.
- [ ] Run tests and verify failure.
- [ ] Implement strict data contract, seeded fixture, FRED CSV downloader, source manifest and hashes.
- [ ] Run focused and full test suite.

### Task 2: Pricing, portfolio and P&L
**Files:** `src/riskmanage/pricing.py`, `portfolio.py`, `pnl.py`, `tests/test_pricing.py`, `tests/test_pnl.py`.
**Interfaces:** Black–Scholes call; Garman–Kohlhagen call; market-state full valuation; static HPL; independent RTPL with declared risk factor subset.
- [ ] Write tests for option monotonicity / bounds / limiting cases and finite-difference Greeks.
- [ ] Test HPL static-position full-revaluation semantics.
- [ ] Test delta-only RTPL omission of nonlinear and volatility terms.
- [ ] Implement four desks: Equity Options, FX Options, Rates, Commodities.
- [ ] Run tests and verify.

### Task 3: Tail risk and capital
**Files:** `src/riskmanage/risk.py`, `capital.py`, `tests/test_risk.py`, `tests/test_capital.py`.
**Interfaces:** historical VaR/ES, liquidity-adjusted ES, stress-window selection, IMCC.
- [ ] Test finite known loss vectors, non-finite rejection, liquidity horizon set isolation, reduced-set coverage and IMCC weighting.
- [ ] Implement 97.5% ES, liquidity scaling, stress selection, constrained / unconstrained IMCC and NMRF stress aggregation.
- [ ] Run full risk and capital tests.

### Task 4: Desk validation and RFET evidence control
**Files:** `src/riskmanage/validation.py`, `tests/test_validation.py`.
**Interfaces:** VaR backtest, Kupiec POF, Christoffersen independence, PLA, RFET evidence diagnostic.
- [ ] Test 0/1/multiple exceptions and missing forecasts.
- [ ] Test Kupiec / Christoffersen on known sequences.
- [ ] Test PLA green / amber / red thresholds and sample-size rules.
- [ ] Test public price-only evidence cannot certify modellability.
- [ ] Implement desk-level validation and model-risk explanations.

### Task 5: Reproducible case, reports, docs and CI
**Files:** `scripts/run_case.py`, docs, reports, `.github/workflows/ci.yml`, `pyproject.toml`.
- [ ] Run a deterministic synthetic case and generate machine-readable outputs and figures.
- [ ] Attempt real data ingestion only when network is available; never label fixture output empirical.
- [ ] Write formula-based reviewer report, including desk failure diagnosis and limitations.
- [ ] Run `pytest -q`, `ruff check .`, `python scripts/run_case.py --synthetic`.
- [ ] Inspect final tree and open draft PR; do not merge.


## Implementation follow-up — public oil-data integration (2026-11-26)

The market-data contract was extended after the initial implementation:

- FRED supplies S&P 500, VIX, EUR/USD, USD/JPY and US 2Y/10Y Treasury yields.
- EIA PET.zip supplies both WTI Cushing and Europe Brent daily spot benchmarks from the same official bulk snapshot; no API key is required.
- The combined market panel aligns source observations on a business-day calendar and forward-fills at most four business days; raw EIA observations are preserved separately without imputation.
- WTI spot may be finite and non-positive; its volatility proxy uses additive price changes normalized by a lagged positive scale. Brent, FX and equity index levels remain positive.
- The commodities test book now has long WTI and short Brent linear spot proxies, not listed futures. Futures basis, contract expiry and roll modeling remain out of scope.
- The synthetic Brent factor is correlated with WTI using an additional innovation while preserving the original random draw for all previously existing factors, so the original deterministic equity/FX/rates/WTI reference paths remain unchanged.

The acceptance boundary remains unchanged: mocked tests and passing CI do not substitute for the first live PET.zip retrieval and review of the final EIA/FRED market panel.
