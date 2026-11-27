> **Historical audit — superseded.** This document records an earlier code state and is retained for audit history. The current source-level baseline is [TECHNICAL_AUDIT_REFRESH_2026-11-27.md](TECHNICAL_AUDIT_REFRESH_2026-11-27.md). In the current branch, the class-level IMCC ratio is no longer floored (the portfolio-wide floor remains required by Basel MAR33.6), and RFET requires an explicit assessment date. Do not treat the P0 findings recorded below as open defects in the current branch.

# RISK-MANAGE — Technical Audit Baseline

**Audit date:** 2026-11-27  
**Repository:** `MedeiroszJoao/Risk-manage`  
**Branch:** `portfolio/frtb-ima-risk-engine`  
**Audited HEAD:** `a2f534325d8ec79a3a26921a8d0d8e04e51b079e`  
**Scope:** repository inventory, static review of all source/test/script/documentation paths, deep review of the quantitative core, comparison with the Materials knowledge base and official Basel MAR31–MAR33 text.  
**Status:** baseline audit completed; independent numerical experiments and remediation are still open. No source-code change is included in this audit commit.

## 1. Executive result

The Risk-manage should be preserved. It already contains a connected research laboratory spanning multi-asset P&L, option pricing, Heston simulation, SSVI, tail risk, uncertainty, data ingestion and FRTB-inspired diagnostics. The audit does **not** recommend a rewrite or a reduction of its research identity.

The key issue is that several pieces are research primitives or synthetic demonstrations, while the surrounding interfaces can still be mistaken for complete quantitative methods. There is also at least one confirmed formula discrepancy in the IMCC class aggregation and one confirmed temporal-design defect in the RFET diagnostic.

The immediate order is:

1. Correct the confirmed IMCC formula discrepancy.
2. Anchor RFET assessment to an explicit assessment date, not the latest row found in the evidence ledger.
3. Add independent source-to-formula tests for the Basel primitives.
4. Strengthen temporal alignment, market conventions and numerical error budgets.
5. Only then advance the research experiments.

## 2. Repository inventory

The audited tree contains 83 tracked files:

| Area | Files | Scope |
|---|---:|---|
| Python source | 23 | 22 research/implementation modules plus package initializer |
| Tests | 21 | 20 test modules plus SEC fixture documentation |
| Scripts | 14 | ingestion, deterministic case, commodity case, Heston experiments |
| Documentation | 17 | methodology, model cards, maturity, audit, data and experiment notes |
| Workflows | 2 | CI and live EIA ingestion |
| Root configuration | 4 | README, pyproject, development requirements, gitignore |

### Source-module responsibility map

| Module | Current responsibility | Audit focus / evidence gap |
|---|---|---|
| `pricing.py` | Black–Scholes, Garman–Kohlhagen, zero-coupon proxy | Flat rates, dividend/forward conventions, numerical tails and independent benchmark grid |
| `black76.py` | Black-76 pricing | Boundary and independent reference checks; contract/discount conventions |
| `calibration.py` | Heston calibration from option quotes | Parameter identifiability, quote noise, residual structure, multiple minima and failed-fit reporting |
| `pnl.py` | Hypothetical book valuation, HPL/RTPL, factor attribution | VIX proxy, fixed curves, diagonal-only second-order expansion and model naming |
| `portfolio.py` | Four hypothetical desk books | Position semantics, units, maturity reset, instrument realism and reproducible book definition |
| `risk.py` | Historical VaR/ES, rolling VaR, 10-observation aggregation, liquidity ES, stress selector | Calendar/index semantics, overlap, independent ES references and exact MAR33 scope |
| `validation.py` | Kupiec, Christoffersen, PLA and RFET diagnostics | RFET as-of date; PLA date alignment; inferential sample and missing-data semantics |
| `capital.py` | Reduced-factor selection, stress window, IMCC and NMRF primitives | Confirmed class-ratio floor discrepancy; group-level 75% requirement; IMCC stress selection |
| `heston.py` | Characteristic function, Fourier, QE-M, paths, Greeks | Complex branch stability, Fourier error budget, time-step bias, strict scalar contracts |
| `volatility_surface.py` | Power-law SSVI, certificate and calibration | Identifiability, optimizer stability, quote quality and bid/ask objective diagnostics |
| `uncertainty.py` | Quote-noise bootstrap, profile likelihood, price/Greek propagation | Noise-model sensitivity, finite-sample coverage and failed-replication analysis |
| `tail_risk.py` | POT-GPD, threshold grid, conditional and full-sample bootstrap | Dependence, clustering, regime shifts, threshold uncertainty and chronological OOS |
| `greeks.py` | Heston Monte Carlo price and Greeks | Bump bias, time-discretization bias and independent Greeks references |
| `hedging.py` | Heston-delta vs frozen-volatility hedge under physical scenarios | Scenario calibration, hedge-grid convergence, costs, physical parameter uncertainty |
| `hedging_uncertainty.py` | Propagates Q-parameter bootstrap into hedge outcomes | Conditional scope and separation of parameter vs scenario uncertainty |
| `nested_hedge_risk.py` | Crossed random-effects decomposition | Small-sample stability, component estimator assumptions and within-cell MC confounding |
| `qmc.py` | Randomized Sobol Heston pricing | Replicate count, randomized-QMC error estimation and cost/accuracy comparison |
| `commodity.py` | WTI/Brent spot-linked P&L and risk | Spot/futures basis, contract rolls, units and instrument-specific curves |
| `eia.py` | EIA bulk archive ingestion and parsing | Vintage history, source revision, HTTP retry semantics and data release timing |
| `data.py` | FRED ingestion, alignment, validation and manifests | Forward-fill sensitivity, point-in-time availability and source-specific calendars |
| `sec_nport.py` | Exact-accession SEC N-PORT holdings parser | Full-quarter reconciliation, amendment handling, identifier completeness and valuation reconciliation |
| `fixtures.py` | Deterministic synthetic market data | Generator assumptions, seed stability and separation from observed data |

## 3. Confirmed findings

### P0 — IMCC class components use an incorrect floor

**Location:** `src/riskmanage/capital.py:269–303`.

The code computes each class component as:

```python
components[name] = float(rs * max(fc / rc, 1.0))
```

Basel MAR33.6 applies the floor at 1 to the *portfolio-wide unconstrained* ratio (ES_{F,C}/ES_{R,C}). MAR33.15 then aggregates the unconstrained component with the sum of constrained risk-class components. The risk-class components are the partial ES terms (ES_{R,S,i}\,ES_{F,C,i}/ES_{R,C,i}); the class-level ratio is not independently floored at 1.

**Consequence:** whenever (ES_{F,C,i}/ES_{R,C,i}<1), the current code replaces that ratio with 1 and overstates that constrained class component.

**Required correction:** remove the class-level floor; preserve the aggregate floor in the unconstrained component. Add a published-formula reference test in which at least one class ratio is below 1, another is above 1, and the final weighted aggregate is checked independently. Existing tests only use class ratios above 1 and therefore cannot expose this defect.

Official reference: [Basel Framework MAR33, paragraphs 33.6 and 33.14–33.15](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/33/inforce/2022-01-01/published/2019-12-15).

### P0 — RFET uses the latest evidence date as its assessment date

**Location:** `src/riskmanage/validation.py:351–398`.

The diagnostic defines:

```python
latest = data["date"].max()
cutoff = latest - pd.Timedelta(days=int(lookback_days) - 1)
```

This makes the end of the 12-month window depend on the newest observation present in the ledger. It is not necessarily the date on which modellability is being assessed.

**Consequence:** a stale ledger containing a dense set of old observations can be assessed against its own old endpoint and appear to satisfy the numerical count screen even when those observations are stale relative to the actual quarterly assessment date.

**Required correction:** require an explicit `as_of_date` / assessment timestamp. Define the 12-month window relative to that date; report the age of the latest eligible observation; reject or flag future observations; preserve the distinction between RFET assessment date and the evidence observation dates. Add a regression test where a historically dense but stale ledger cannot pass at a later assessment date.

The existing implementation correctly limits each date to one count per day, checks both numerical routes (24 observations plus four in every 90-day period, or 100 observations in 12 months), and labels its output as a count screen rather than modellability approval. Those protections must remain.

Official reference: [Basel Framework MAR31.12–31.15](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/31/inforce/2023-01-01/published/2020-03-27).

## 4. High-priority methodological findings

### 4.1 Backtesting and PLA

- `backtest_var` now separates operational missing-row exception counts from the matched finite population used by Kupiec and adjacent matched pairs used by Christoffersen. This is a sound correction.
- Observation dates are checked for uniqueness and ordering, but not against an explicit trading calendar. Missing business sessions can still be treated as adjacent array rows.
- `pla_test` accepts two arrays and checks equal length, but cannot independently verify that HPL and RTPL refer to the same dates. It should accept/validate aligned dates or a common dated frame.
- The code's PLA thresholds and statistics resemble MAR32.34–32.42, but this remains a research diagnostic: no desk APL, production HPL/RTPL construction, bank controls or supervisory assessment exists.
- Kupiec and Christoffersen are asymptotic likelihood-ratio tests. Their finite-sample size and power under low exception counts, dependence and overlapping forecasts are not established by the current unit tests.

Official references: [Basel MAR32, 32.19 and 32.34–32.42](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/32/inforce/2023-01-01/published/2020-03-27).

### 4.2 VaR, ES and calendar semantics

- Historical VaR now explicitly uses the empirical inverse-CDF convention; ES integrates fractional probability mass at the VaR boundary. This is mathematically coherent and addresses ties.
- The current tests cover several discrete/tied samples and metamorphic translation/scale properties.
- Rolling VaR and overlapping 10-observation losses operate on array order, not dated observations. They cannot detect missing trading sessions, stale marks, irregular calendars or a forecast/value timestamp mismatch.
- Liquidity-adjusted ES has the shape of the MAR33.4 nested-liquidity-horizon formula, but the input mapping is supplied directly by the caller. It does not derive instrument/risk-factor liquidity horizons, cap by instrument maturity, or validate the underlying factor mapping. It must remain a research implementation.
- Stress selection in `risk.py` and `capital.py` maximizes a historical 10-observation ES statistic. It is not the MAR33 stress-period process, whose stress period must be selected using the aggregate IMCC objective and at least the required historical observation span.

Official reference: [Basel MAR33.4–33.7 and 33.12](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/33/inforce/2022-01-01/published/2019-12-15).

### 4.3 Capital and reduced-factor selection

- The 75% coverage statistic is explicitly labelled overlapping and in-sample. It reuses most of the same 250-observation windows that participated in factor selection.
- The code's greedy ranking by individual factor ES is an exploratory heuristic, not a demonstrated Basel reduced-factor selection method.
- Basel's 75% requirement is evaluated at the group level across approved IMA desks, over the preceding 12 weeks. The current research book does not model that complete multi-desk group process.
- The stress selector maximizes portfolio ES, whereas the Basel FAQ says the stress period is identified by maximizing aggregate IMCC.
- The current capital builder deliberately blocks incomplete broad risk-class coverage and labels outputs non-regulatory. Keep this fail-closed behavior.

### 4.4 P&L, valuation and factor attribution

- The desk book is explicitly hypothetical and constant-maturity: it is reset after each daily interval, not a record of a held trade.
- Equity option valuation uses a flat 3.5% rate, no dividend yield, and VIX/100 as a common volatility proxy for all equity option strikes and maturities. This is not a market volatility surface.
- FX option valuation uses flat domestic/foreign rates (3.5%/2.0%). Rates books use simplified zero-coupon proxies rather than bootstrapped curves.
- The variance-swap mark is a simplified proxy, not a complete variance-swap replication/mark model.
- The `full_greeks` RTPL includes diagonal second-order terms for each selected factor but omits mixed cross-factor Hessian terms. The name overstates completeness; the implementation/documentation should describe it as diagonal second-order RTPL until cross terms are included.
- WTI and Brent exposures are spot-linked linear proxies, not exchange-traded futures. This is correctly disclosed and must remain explicit.
- The factor P&L remainder contains theta and cross-factor interaction. It is an accounting remainder, not an independently observed market factor.

### 4.5 Heston, Monte Carlo and QMC

- Heston tests cover characteristic-function normalization, conjugate symmetry, risk-neutral forward, deterministic-variance Black–Scholes limit, put-call parity, bounds, QE conditional moments, and Fourier/MC agreement.
- Fourier vs QE-M agreement is not a fully independent model validation because both implementations share the same Heston model assumptions and related discretization framework.
- The characteristic function uses a principal complex square root and logarithm in the standard affine representation. Branch continuity and moment-explosion regimes have not been systematically mapped.
- The returned Fourier quadrature estimate is not a total pricing error bound; truncation, branch, model and input errors are separate.
- QE-M Monte Carlo confidence intervals quantify sampling error conditional on the time grid; the code explicitly excludes discretization bias. A systematic time-step convergence/error budget is still required.
- Monte Carlo and Greeks public scalar inputs do not all use the strict scalar type contract used by risk/capital; booleans and non-integral controls need consistent rejection.
- Randomized Sobol has reproducibility and parity tests; the number of independent scrambles and cost/accuracy trade-off still need a statistically defensible benchmark.

### 4.6 SSVI and calibration

- The implemented SSVI sufficient-region certificate is tied to Gatheral–Jacquier power-law conditions and is appropriately described as a sufficient certificate, not a universal proof for arbitrary surfaces.
- Synthetic recovery tests do not establish market fit or parameter identifiability.
- Calibration reports a Jacobian condition number, but does not yet map parameter uncertainty, multiple local minima, profile likelihood, or stability under quote removal and bid/ask perturbation.
- The bid/ask interval objective gives zero residual to any model price inside the spread. This can be useful, but it creates flat regions and weak identification; the report must not treat zero interval loss as evidence of a uniquely identified parameter set.
- `n_starts` and `max_nfev` should be validated as strict positive integers; current checks test only numeric comparisons before later slicing/optimizer calls.

### 4.7 Uncertainty and EVT

- Heston quote bootstrap is a parametric Gaussian quote-noise procedure whose scale is derived from bid/ask spreads. It reports failed replicates, which is good; coverage is conditional on this assumed noise model.
- It does not yet compare Gaussian, uniform, interval, heteroskedastic, correlated or empirical quote-noise models.
- POT-GPD includes threshold grids, minimum exceedance requirements and both conditional exceedance and full-sample IID bootstrap diagnostics.
- IID resampling does not preserve serial dependence or clusters of exceedances. The current documentation correctly states this limitation.
- The KS/PIT diagnostic is descriptive after parameter fitting; it is not a calibrated goodness-of-fit p-value.
- Financial tail conclusions remain synthetic scenario results until tested on independent chronological loss histories with dependence-aware inference.

### 4.8 Data and empirical evidence

- FRED raw snapshots and hashes are preserved; the aligned panel forward-fills up to four business days and drops remaining incomplete rows. This choice needs sensitivity tests and explicit per-series stale-age diagnostics.
- Retrieval timestamps are recorded, but historical first-availability/publication timestamps and full point-in-time revisions are not reconstructed.
- The EIA bulk archive is a current snapshot, not a historical vintage database. Its SHA-256 proves which archive was downloaded, not what EIA had published at every historical decision date.
- FRED SP500 daily history/licensing restricts the integrated common panel; the existing documentation discloses the limitation.
- SEC N-PORT parser is careful about exact accession, series selection, identifiers and amendments. The current fixture/test covers only three selected positions from one filing, not a full quarterly archive. No total NAV/holding reconciliation or instrument-level return reconstruction has been demonstrated.
- Public fund holdings are not bank desk positions, trades, APL or RFET evidence.

## 5. Software engineering findings

| Area | Current state | Required improvement |
|---|---|---|
| Dependency control | Version ranges in `pyproject.toml`; no lockfile | Commit a tested lockfile and capture environment metadata in experiment reports |
| Lint | Ruff E4/E7/E9/F subset; several style rules ignored | Expand lint gradually without mass-formatting or identity-changing refactor |
| Types | Type hints are inconsistent; public interfaces often accept untyped scalar values | Add explicit public API contracts and static type checking |
| Test categories | Tests are grouped by module, not by evidence type | Add markers/labels for unit, property, numerical, statistical, benchmark, integration and data tests |
| Coverage | No coverage report or meaningful branch coverage gate | Measure coverage and identify untested mathematical branches; do not optimize a vanity percentage |
| Independent references | Several analytic identities; few external implementation comparisons | Add a source-to-formula benchmark suite |
| Experiment artifacts | CI uploads configured Heston JSON outputs | Include package versions, code SHA, data hashes, config, seed, run timestamp and warnings in each artifact |
| Performance | No comprehensive memory/runtime benchmark matrix | Measure before optimizing |

## 6. Existing test and CI evidence

The latest audited commit `a2f534325d8ec79a3a26921a8d0d8e04e51b079e` has a successful CI run:

- Python 3.11: **255 tests passed**, Ruff passed.
- Python 3.12: **255 tests passed**, Ruff passed.
- Deterministic synthetic case passed.
- Configured Heston Fourier/QE-M, calibration, Greeks, delta-hedging, randomized-QMC, uncertainty, nested hedge-risk and EVT experiments completed.

Run: [GitHub Actions 45921183208](https://github.com/MedeiroszJoao/Risk-manage/actions/runs/45921183208).

This verifies the configured code paths on the tested runners. It does not independently validate the mathematical formulas, market calibration, empirical performance or regulatory status.

## 7. Materials knowledge-base crosswalk

The Materials repository provides useful independent material, but it must be applied at the correct evidence level:

| Materials source | Risk-manage application |
|---|---|
| `KNOWLEDGE/FORMULA_REGISTER.md` | Formula definitions, measure, units, signs, assumptions and edge conditions |
| `KNOWLEDGE/IMPLEMENTATION_PATTERNS/fourier_pricing.py` and Carr–Madan reproduction note | Independent Fourier inversion benchmark against the Lewis Heston pricer |
| GARCH QMLE / recursive scores / robust inference note and implementation | Future volatility forecasting benchmark; sandwich/HAC inference and residual diagnostics |
| MIT 18.650 goodness-of-fit and residual notes | Calibrate KS/GOF diagnostics under estimated parameters; avoid naive reference p-values |
| MIT 18.650 MLE/Fisher/misspecification note | Heston calibration conditioning, likelihood curvature and misspecification |
| PCA eigengap/subspace stability note and reproduction | Factor-selection stability and correlation/concentration studies |
| Robust portfolio optimization and estimation-risk note | Future portfolio risk/optimization research, with chronological decision-focused evaluation |
| SDE strong-convergence reproduction | QE-M / stochastic simulation convergence study design |
| Statistical integrity / PBO / DSR note | Preserve experiment search history and correct for selection/multiple testing where applicable |
| Oxford multidimensional Itô and martingale notes | Cross-Greeks, covariance terms, pricing measure and martingale diagnostics |

Materials governance requires distinguishing `FIRST_PASS`, `DERIVED`, `REPRODUCED`, `VERIFIED` and `APPLIED`. A note or course completion is not an independent validation result.

## 8. Prioritized remediation backlog

### P0 — mathematical/data correctness

1. **CAP-04:** correct the class-level IMCC ratio floor; add a below-one independent formula test.
2. **RFET-02:** add explicit assessment date and stale-evidence controls; add stale-ledger regression test.
3. **VAL-05:** add dated HPL/RTPL alignment to PLA and explicit trading-calendar diagnostics to backtesting.
4. **PNL-01:** rename/document diagonal second-order RTPL and keep cross-Greeks visibly absent.
5. **RISK-01:** add dated observation/calendar contracts to rolling VaR and overlapping horizon aggregation.

### P1 — independent numerical/statistical evidence

6. **HES-02:** independent Lewis vs Carr–Madan benchmark, published reference cases and branch-stability maps.
7. **HES-03:** separate Fourier truncation/quadrature and QE time-step/sampling errors; establish an error budget.
8. **VOL-02:** identifiability, profile likelihood, quote perturbation and multi-start stability study.
9. **EVT-03:** block/stationary bootstrap or declustering study and chronological holdout.
10. **UNC-02:** compare quote-noise models and assess bootstrap interval coverage under controlled misspecification.
11. **CAP-05:** implement a research-only chronological group-level 75% coverage experiment and compare factor-selection methods; do not call it regulatory approval.

### P2 — empirical depth and research expansion

12. **DATA-02:** point-in-time publication/availability and revision model.
13. **NPORT-02:** complete archive parse, position-count and NAV reconciliation, amendment and identifier audit.
14. **OPT-01:** source timestamped/licensed option chains and build quote-quality filters before market calibration.
15. **FUT-01:** obtain contract-level futures settlements and build curves/roll/basis methods before claiming commodity futures risk.
16. **PERF-01:** benchmark runtime/memory and accuracy-cost trade-offs.
17. **RESEARCH-01:** preregister falsifiable comparative hypotheses, preserve all variants and publish negative results.

## 9. Acceptance rules

A backlog item is complete only when all applicable conditions are met:

1. The defect or question is reproduced with a minimal failing case.
2. The mathematical or regulatory source is recorded with exact paragraph/equation.
3. The implementation is changed without unrelated architectural rewrite.
4. A regression test fails before the fix and passes after it.
5. An independent benchmark or derivation supports the result.
6. Numerical/statistical limitations and tolerances are justified.
7. Documentation and maturity register are updated.
8. CI passes on Python 3.11 and 3.12.
9. Generated evidence includes code SHA, configuration, seed, source hashes and environment.
10. The maturity state advances only for the dimension actually demonstrated.

**No item is declared complete on the strength of a green test suite alone.**

## 10. Audit conclusion

Risk-manage is worth deepening, not rebuilding. The highest-value work is now concentrated in formula fidelity, date-aware data contracts, independent numerical benchmarks, and evidence separation.

The two confirmed P0 findings must be fixed before any capital or RFET result is used as a substantive demonstration. The remaining modules should be advanced through targeted studies, not replaced. The existing synthetic cases remain useful as controlled experiments, but they must not be presented as market validation.
