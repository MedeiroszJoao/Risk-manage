# Maturity, scope and evidence register

**Last reviewed:** 2026-11-27  
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
| Reduced-factor selection | Research implementation | Selection-window ES ratio and rolling-origin holdout checks | 60 forward endpoints, with the factor set frozen before the holdout; still no desk-level supervisory validation | IMA requirements not established |
| IMCC / NMRF primitives | Formula primitives | Unit examples only | The current book lacks credit-spread factors, and the stress-period / group-level reduced-factor procedures are not implemented | Not a regulatory calculation |
| Data ingestion | FRED/EIA pipelines and SEC N-PORT accession parser | Live FRED/EIA checks; N-PORT parser unit-tested against three position records transcribed from one official filing | Full quarterly N-PORT archive and complete filing not yet reconciled; no instrument-level return reconstruction | N-PORT holdings are not RFET evidence or bank desk positions |
| Portfolio / HPL / RTPL | Yes, hypothetical constant-maturity test books | P&L reconciliation and formula/property tests | No real bank desk positions or APL | Regulatory backtest/PLA incomplete; the `full_greeks` RTPL mode is currently a diagonal second-order approximation and omits mixed cross-Greeks |

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

The codebase is a useful research implementation with tested numerical components and public-data ingestion. The latest validation hardening also rejects mismatched pandas Series indexes in HPL/VaR and HPL/RTPL comparisons; its follow-up CI run is pending. It is **not** yet an empirically validated trading-risk engine and does **not** implement a complete FRTB IMA capital process. No current result supports a claim of bank-level performance, model approval or regulatory capital readiness.


## Independent maturity vector — 2026-11-27

The following is a conservative assessment after the first technical-integrity remediation pass. `CI reproducible` means the configured automated suite and experiments ran successfully; it is not independent scientific validation.

| Component | Implementation | Mathematical | Numerical | Statistical | Empirical | Independent evidence | Reproducibility | Research |
|---|---|---|---|---|---|---|---|---|
| Validation / backtesting | Implemented | Partially verified; sample semantics corrected | Tested | Tested with asymptotic/low-power limitations | HPL-only hypothetical diagnostics | No independent implementation benchmark yet | CI reproducible | Exploratory |
| Historical VaR / ES | Implemented | Convention and fractional boundary mass specified | Property/edge tests | No OOS forecast validation | No real desk book | Independent fractional-tail calculation in tests; external benchmark pending | CI reproducible | Comparative method work pending |
| Capital / IMCC | Implemented research primitives | Formula primitives tested; full process unverified | Unit-tested | Factor coverage remains in-sample | Synthetic books | Basel MAR33 source traceability; worked-example comparison pending | CI reproducible | Exploratory |
| Heston Fourier / QE-M | Implemented | Analytic identities and limiting cases tested | Benchmarked across configured methods; full error budget pending | Synthetic recovery / MC uncertainty only | No market quote validation | Analytic special cases and Fourier-vs-QE-M cross-check; independent external pricer pending | CI reproducible | Comparative |
| SSVI surface | Implemented | Sufficient static-arbitrage conditions implemented; broader proof scope limited | Synthetic recovery and dense checks | Calibration uncertainty not fully established | No real option chain | Published SSVI conditions documented; independent market benchmark pending | CI reproducible | Exploratory |
| Parameter / hedge uncertainty | Implemented | Conditional on selected model/noise assumptions | Simulation and bootstrap experiments | Conditional bootstrap diagnostics | Synthetic quotes and hypothetical hedge | Independent method comparison pending | CI reproducible | Comparative |
| POT-GPD tail risk | Implemented | GPD tail formulas implemented; dependence assumptions remain limiting | Synthetic recovery and threshold diagnostics | IID bootstrap only; temporal inference incomplete | No chronological OOS financial loss validation | Known-GPD synthetic benchmark; external empirical benchmark pending | CI reproducible | Exploratory |
| Data ingestion | Implemented | Not a model dimension | Schema/hash and ingestion checks | Not applicable | Public FRED/EIA ingestion verified; N-PORT full archive incomplete | Official public-source provenance | CI reproducible | Research infrastructure |

### Current automated verification

- [Risk-engine CI](https://github.com/MedeiroszJoao/Risk-manage/actions/runs/45924282157): successful on Python 3.11 and 3.12 including the pandas-index alignment change.
- Python 3.11: 291 tests passed; Ruff and all configured Heston experiments passed.
- Python 3.12: 291 tests passed; Ruff and all configured Heston experiments passed.
- The four new index-alignment tests passed on both Python versions in run 45924282157.
- [Live EIA ingestion workflow](https://github.com/MedeiroszJoao/Risk-manage/actions/runs/45924282153): successful.

This evidence does not change the empirical or regulatory limitations listed above.


## Audit baseline update — 2026-11-27

Use [TECHNICAL_AUDIT_REFRESH_2026-11-27.md](TECHNICAL_AUDIT_REFRESH_2026-11-27.md) as the current source-level audit baseline. The older [TECHNICAL_AUDIT_2026-11-27.md](TECHNICAL_AUDIT_2026-11-27.md) is historical and must not be read as the current code state. Its two P0 findings—class-level IMCC scaling and RFET assessment-date anchoring—are resolved in the current branch: class-level ratios are no longer floored, and RFET requires an explicit assessment date. The portfolio-wide IMCC floor remains because MAR33.6 requires it. The latest completed CI has 291 passing tests on Python 3.11 and 291 on Python 3.12; the Series index-alignment guard passed in that run. No maturity state is advanced by test count alone.


## Reduced-factor holdout remediation — 2026-11-26

The previous coverage routine evaluated the factor set on rolling windows that overlapped the factor-selection sample. That was in-sample and could not support an independent performance claim.

The current implementation now:
1. Reserves the final 60 observations as a forward validation interval.
2. Selects and freezes the reduced factor set using only the immediately preceding 250 observations.
3. Evaluates the frozen set at the 60 holdout endpoints, recomputing full and reduced ES on trailing 250-observation windows.
4. Reports an unclassifiable result when any full-set ES denominator is nonpositive; it does not discard those endpoints.
5. Requires 310 observations for the 250 + 60 design.

This is a rolling-origin research diagnostic for one frozen selection. It is not a supervisory approval, does not implement group-level factor selection across approved desks, and does not by itself establish the Basel MAR33 75% requirement.


## Chronology control for reduced-factor holdout — 2026-11-26

Out-of-sample evaluation requires an explicit observation-date vector aligned one-to-one with the factor-loss arrays. Duplicate UTC calendar dates, missing timestamps, non-monotone dates and length mismatches are rejected. Selection and validation windows expose their actual first/last dates as well as positional indices. The integrated case pipeline supplies dates from the factor P&L table.


## RTPL cross-Greek remediation — 2026-11-26

The P&L layer now exposes three distinct model levels: first-order delta, diagonal second-order, and full market-factor Hessian. The legacy full_greeks label is a deprecated alias for diagonal_second_order and no longer misstates its mathematical scope.

The full_hessian mode adds mixed second derivatives by a central four-point stencil. A separate numerical benchmark compares the mixed spot/volatility derivative with the analytic Black-Scholes call vanna and verifies finite-difference convergence as bumps are reduced. The method revalues one-day aging separately and does not claim to include market-time mixed derivatives.
