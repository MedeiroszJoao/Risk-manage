# RISK-MANAGE — Quantitative Market-Risk Research Prototype

**Scope:** a research codebase for testing how derivative valuation, hypothetical P&L (HPL), risk-theoretical P&L (RTPL), historical VaR / Expected Shortfall and model-risk diagnostics fit together.

> This is not a bank trading-risk system, production pricing library, approved internal model or FRTB IMA capital calculator. The reference portfolios are hypothetical; no bank Actual P&L (APL) is available.

## What the current case demonstrates

The deterministic reference run uses 1,800 **synthetic** business-day observations (2016-01-04 to 2022-11-25), not market observations or a real trading book.

| Equity-options desk diagnostic | Delta-only RTPL | RTPL with gamma + vega |
|---|---:|---:|
| Spearman rank correlation | 0.375 | 0.999 |
| Empirical CDF / KS distance | 0.396 | 0.024 |
| Research PLA diagnostic | Red | Green |
| Mean absolute HPL − RTPL residual | USD 5.63m | USD 0.58m |

The expanded sensitivity model reduces mean absolute residual by 89.7% in this deliberately constructed synthetic case. The delta-only model also has 3 HPL exceptions in 250 observations at 99%, below the HPL-only numeric threshold of 12; this is not desk approval because APL is unavailable. These numbers illustrate that exception counts and P&L explain answer different questions; they do not estimate real trading performance.

## Implemented scope

- **Data:** public FRED equity/FX/rates and EIA WTI/Brent spot ingestion with source metadata and hashes; deterministic synthetic fixtures for offline testing.
- **Hypothetical test books:** Equity Options, FX Options, Rates and Commodities. The commodity exposures are linear spot proxies, not listed futures.
- **Pricing:** Black–Scholes, Garman–Kohlhagen, Black-76, power-law SSVI and Heston Fourier / QE-M research implementations.
- **P&L:** close-to-close full-revaluation HPL and separately constructed sensitivity RTPL for positions static during each daily interval; constant-maturity positions are reset after each interval.
- **Risk diagnostics:** historical VaR / ES, rolling historical VaR, Kupiec and Christoffersen tests, HPL-versus-RTPL PLA diagnostics, liquidity-adjusted ES structure, descriptive stress-window selection, reduced-factor selection and formula primitives for IMCC / NMRF.
- **Tail and uncertainty research:** POT-GPD threshold diagnostics, Heston quote-noise bootstrap, profile likelihood and simulation-based hedge experiments.

See [Maturity, scope and evidence register](docs/MATURITY_REGISTER.md) for the evidence level and limitations of each component. It separates code that exists, tests that pass, synthetic reproductions, public-source ingestion checks, empirical validation and regulatory assessment.

## Run it

Python 3.11+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"

# Deterministic offline reference case
python scripts/run_case.py --synthetic

# Public FRED/EIA pipeline (market-source coverage varies by series)
python scripts/download_market_data.py --start 2007-01-01
python scripts/run_case.py

# Standalone EIA WTI/Brent spot history
python scripts/download_eia_petroleum_spot.py --start 2007-01-01
python scripts/run_commodity_spot_case.py

pytest -q
ruff check src tests scripts
```

The dependency ranges in pyproject.toml are minimum-version constraints, **not a complete lockfile**. For a forensic rerun, capture the exact Python version and resolved package versions (for example, python -m pip freeze) alongside the generated report. The project must not be described as bit-for-bit environment locked until a lockfile is committed and tested.

## What has been checked

- CI runs the test suite and Ruff on Python 3.11 and 3.12.
- Historical VaR / ES uses NumPy's empirical inverted-CDF quantile and integrates only the required fractional probability mass at the VaR boundary.
- Heston tests cover characteristic-function normalization, the risk-neutral forward condition, a deterministic-variance Black–Scholes limit, put-call parity, bounds, QE variance moments and Monte Carlo diagnostics.
- SSVI tests include static-arbitrage sufficient-region checks and synthetic calibration recovery.
- POT-GPD reports a threshold grid and bootstrap diagnostics; the bootstrap does not account for serial dependence or structural breaks.
- FRED/EIA public-data ingestion has been run in GitHub Actions. That does not create a real portfolio or qualifying transaction history.

## Non-negotiable limitations

1. All four desk portfolios are hypothetical. Public closes for SPX, FX, rates or crude oil are not actual desk positions and are not a bank transaction ledger.
2. Backtesting here is HPL-only. The Basel desk framework requires both Actual P&L and Hypothetical P&L; missing HPL/VaR rows are counted as exceptions. The code applies the desk-level 12-exception (99%) / 30-exception (97.5%) numeric limits without bank-wide multipliers, but cannot establish eligibility without APL.
3. Public daily closes do not establish RFET. Qualifying real-price evidence, vendor audit controls and defensible factor mapping need independent review.
4. The integrated history begins in 2016, not 2007. It does not establish the full historical stress span required by the IMA framework.
5. The current liquidity-horizon mapping is an illustrative research input, not the MAR33 instrument/risk-factor mapping. Stress-period selection currently maximizes portfolio ES rather than aggregate IMCC.
6. Reduced-factor coverage currently uses overlapping windows that reuse the factor-selection sample. It is descriptive in-sample coverage, not out-of-sample validation.
7. IMCC and NMRF functions are formula primitives. Their output is never a regulatory capital number; complete MAR33 mapping, IMCC-based stress-period selection, group-level 75% test, all five broad classes (including credit spread), stress scenarios and governance are absent.
8. Heston / SSVI calibration tests use synthetic quotes or analytic limits; there is no live timestamped option chain, curve bootstrap or market-quality filter.
9. No DRC, Standardised Approach, CVA/XVA, bank valuation adjustments, intraday risk, desk approval process or supervisory sign-off is implemented.

## Primary technical references

- Basel Framework MAR32 — backtesting and PLA.
- Basel Framework MAR31 — risk-factor modellability and RFET.
- Basel Framework MAR33 — liquidity horizons, IMA capital and NMRF.
- The applicable Basel text controls where an educational approximation differs from regulation.
