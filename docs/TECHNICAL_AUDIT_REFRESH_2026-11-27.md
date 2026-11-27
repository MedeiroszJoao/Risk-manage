# Risk-manage — Technical Audit Refresh

**Audit date:** 2026-11-27  
**Repository:** `MedeiroszJoao/Risk-manage`  
**Branch audited:** `portfolio/frtb-ima-risk-engine`  
**Audited code revision:** `876662f71195f8d047635ff8619eb8c03647fe43`  
**Pull request:** [#1](https://github.com/MedeiroszJoao/Risk-manage/pull/1)  
**Purpose:** establish a current, evidence-based remediation baseline under the Project Excellence Directive.  
**Change policy:** this audit is documentation only. No quantitative implementation was changed in this audit pass.

---

## 1. Executive verdict

The repository has a broad and technically nontrivial research implementation. Its current strongest evidence is software execution and controlled numerical experiments, not empirical market validation or regulatory validation.

The latest inspected CI run completed successfully on Python 3.11 and 3.12:

- 283 pytest tests passed on each interpreter.
- Ruff passed on both.
- The deterministic desk case and the Heston Fourier/QE-M, calibration, Monte Carlo Greeks, delta-hedging, randomized-QMC, calibration-uncertainty, hedge-uncertainty, crossed risk-decomposition and EVT experiment steps completed successfully.

CI proves that the committed code executes under those environments and passes its current test suite. It does **not** establish mathematical correctness, independent validation, market calibration, OOS performance, or regulatory compliance.

**The IMCC unconstrained floor in `capital.py` is not a defect.** I initially compared the implementation only with the compact MAR33.15 equation and missed the preceding MAR33.6 rule, which explicitly floors the current full/reduced ES ratio at 1. The implementation's `max(full_current / reduced_current, 1.0)` is consistent with MAR33.6. The current test uses a ratio above 1, however, so a regression test below 1 is still needed to prove that the floor branch is intentional and correct.

The project should retain its identity as a quantitative market-risk research laboratory. The remedy is not a rewrite. It is to close specific mathematical, statistical, numerical and empirical evidence gaps in the existing modules.

---

## 2. Audit scope and method

The PR currently contains 85 changed files:

| Area | Files inventoried |
|---|---:|
| `src/riskmanage/` modules | 23 |
| `tests/` modules | 21 |
| `scripts/` entry points | 14 |
| Documentation, workflows, configuration and fixtures | 27 |
| **Total** | **85** |

The 23 source modules were inventoried by file, public function/class signatures, size and suspicious error-handling/data-cleaning patterns. The 21 test modules were inventoried by test cases. All 14 script entry points and their principal command-line defaults were inspected. The main quantitative modules, key tests, project directives, model cards, methodology, data provenance and CI workflow were inspected in greater depth.

This is a structured technical audit pass, not a claim that every line of every documentation and configuration file has already received independent peer review. Items below are explicitly classified as confirmed defects, methodological limitations, or work still requiring verification.

### 2.1 Evidence from CI

Latest inspected run: [GitHub Actions run 45922623187](https://github.com/MedeiroszJoao/Risk-manage/actions/runs/45922623187).

| Check | Python 3.11 | Python 3.12 |
|---|---:|---:|
| pytest | 283 passed | 283 passed |
| Ruff | passed | passed |
| Deterministic case | passed | passed |
| Heston Fourier / QE-M benchmark | passed | passed |
| Heston calibration recovery | passed | passed |
| Monte Carlo Greeks | passed | passed |
| Delta hedging and tail-risk experiment | passed | passed |
| Randomized QMC benchmark | passed | passed |
| Calibration uncertainty | passed | passed |
| Hedge parameter uncertainty | passed | passed |
| Crossed hedge-risk decomposition | passed | passed |
| Hedge POT-GPD experiment | passed | passed |

The CI installs the latest versions satisfying lower-bound dependency constraints. It does not use a committed lockfile. The two Python versions are therefore useful compatibility checks, but not a frozen-environment reproduction.

---

## 3. IMCC formula review — no defect confirmed

### `src/riskmanage/capital.py::calculate_imcc`

Current implementation:

```python
unconstrained = reduced_stress * max(full_current / reduced_current, 1.0)
```

The compact MAR33.15 equation is:

[
IMCC(C)=ES_{R,S}\frac{ES_{F,C}}{ES_{R,C}}
]

However, MAR33.6 explicitly states that the ratio (ES_{F,C}/ES_{R,C}) used to scale (ES_{R,S}) is floored at 1. Therefore the implementation's `max(..., 1.0)` is consistent with the Basel rule for the unconstrained component. The constrained class components are calculated using the separate MAR33.15 class-level ratios.

**Correction to the audit process:** the earlier suspected defect was a false positive caused by reading MAR33.15 without carrying forward the MAR33.6 scaling rule. No code change is authorized for this item.

**Remaining test gap:** `test_imcc_formula_exact` exercises a ratio above 1, so it does not test the floor branch. Add a case with (ES_{F,C}<ES_{R,C}), a case at equality, and retain the above-one case. Expected unconstrained results should be calculated independently from MAR33.6. This is a test-coverage improvement, not a formula correction.

**Primary references:**
- Basel MAR33.6 — [ES stress scaling and ratio floor](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/33/inforce/2022-01-01/published/2019-12-15).
- Basel MAR33.15 — [Constrained/unconstrained IMCC aggregation](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/33/inforce/2022-01-01/published/2019-12-15).

## 4. File-by-file module audit

Status labels:
- **Confirmed defect:** current code behavior is inconsistent with its stated mathematical or data contract.
- **Methodological limitation:** implementation may be valid for its declared research purpose, but does not support a broader claim.
- **Open verification:** a dedicated independent benchmark, convergence study or empirical test is still required.

| File | Current implementation | Audit finding / next action |
|---|---|---|
| `__init__.py` | Minimal package initializer | No material issue identified. Keep the public API intentional; avoid turning the package root into an uncontrolled export surface. |
| `black76.py` | Black-76 price, vega and implied volatility | Analytic identities and finite-difference vega tests exist. Add independent library/reference-price comparisons, near-intrinsic and short-expiry conditioning tests, solver residual and convergence reporting. |
| `calibration.py` | Heston quote representation and nonlinear calibration | Current recovery evidence is synthetic. Add identifiability/profile studies, multi-start stability, optimizer-failure accounting and a real dated option-chain experiment before calling it market-calibrated. |
| `capital.py` | Reduced-factor selection, stress-window selection, IMCC and NMRF primitives | **The unconstrained IMCC floor is consistent with MAR33.6; add a regression case below 1 to cover the floor branch.** Factor ranking uses standalone ES rather than marginal contribution or dependence-aware selection; this is a research baseline, not an optimal selection method. The rolling reduced-set coverage explicitly overlaps the selection sample and is in-sample descriptive only. Stress-window selection is exploratory, not supervisory. |
| `commodity.py` | WTI/Brent spot-proxy P&L and risk case | Good handling of negative WTI values and exact-date inner joins. Portfolio quantities remain hypothetical; spot proxies are not futures settlements. No curve, roll, basis, storage or convenience-yield risk. |
| `data.py` | FRED/EIA panel validation, ingestion and merging | The panel uses business-day reindexing and forward-fill up to four rows. This requires per-series staleness and publication-calendar diagnostics; a common date index does not prove simultaneous information availability. No complete point-in-time/vintage reconstruction. |
| `eia.py` | Official EIA petroleum bulk ZIP ingestion, parser and manifest | Strong raw snapshot/hash/provenance foundation. Archive hash identifies the downloaded snapshot, not historical vintages available at each past date. Spot observations must not be represented as futures settlement history. |
| `fixtures.py` | Deterministic synthetic market-data generator | Useful for controlled experiments. All outputs must remain labelled synthetic; fixture-generated recovery is not market validation. |
| `greeks.py` | Heston Monte Carlo price/Greeks with sampling summaries | Pathwise and finite-difference comparisons exist. Quantified sampling error does not include time-discretization bias, Greek estimator bias or model error. Add bump-size and path/time-step convergence maps and independent benchmarks. |
| `hedging.py` | Heston delta-hedging experiments and transaction-cost model | A useful controlled simulation. It is not an empirical hedge backtest: paths, model parameters and hedge environment are simulated. Add replication of limiting cases, hedging-frequency convergence and independent cost-model sensitivity. |
| `hedging_uncertainty.py` | Propagates Heston calibration draws into hedge-risk distributions | Conditional on the bootstrap parameter generator, model form and scenario generator. Current defaults are exploratory; report MC error and failed fit counts, and compare alternative parameter/noise specifications. |
| `heston.py` | Affine characteristic function, Lewis Fourier inversion, QE-M, full paths | Strong internal identities: normalization, conjugate symmetry, deterministic-variance limit, parity, bounds, martingale checks and Fourier/MC comparison. Those are valuable but mostly internal consistency checks. No independent Heston implementation is currently used as a formal oracle. Build branch-continuity diagnostics, quadrature truncation/tolerance convergence, time-step bias maps and independent reference comparisons. |
| `nested_hedge_risk.py` | Two-way random-effects decomposition of calibration/scenario risk | Balanced-design recovery tests exist. Negative raw variance components are retained and truncation is reported, which is preferable to hiding them. Add unbalanced-design alternatives, finite-sample bias studies and sensitivity to dependence between calibration and scenario draws. |
| `pnl.py` | Factor-based HPL/RTPL and desk P&L | These are modelled research P&Ls, not a bank's reported HPL/APL. The factor proxy and valuation approximations need independent repricing and reconciliation evidence. Do not present the outputs as actual desk P&L. |
| `portfolio.py` | Position data class and default illustrative book | Default positions are hypothetical. The SEC N-PORT parser is not yet an end-to-end holdings-to-risk pipeline. Keep disclosed fund holdings, synthetic positions and hypothetical desk books explicitly distinct. |
| `pricing.py` | Black–Scholes, Garman–Kohlhagen and zero-coupon bond helpers | Basic property tests exist. Document rate compounding, day-count, carry and dividend conventions; compare to an independent pricing implementation and add extreme moneyness/maturity cases. |
| `qmc.py` | Randomized Sobol Heston simulation with Brownian bridge | Reproducibility and Fourier comparison exist. Eight scrambles and default (2^{10}) Sobol power are exploratory. Add RMSE-versus-cost curves over independent scramble counts and dimensions; report uncertainty across scrambles. |
| `risk.py` | Historical VaR/ES, rolling VaR, 10-observation losses, liquidity adjustment and stress window | Fractional boundary-mass ES and tie handling are explicit and tested. The module operates on arrays, not dated observations: it cannot itself enforce calendars, holidays, stale data or overlapping-return policy. Liquidity adjustment and stress-window selection require exact formula/source reconciliation and sensitivity tests. |
| `sec_nport.py` | Parser for selected SEC N-PORT dataset filings and hash-linked snapshots | Useful provenance work. Parser output is a disclosed fund-holdings snapshot, not transaction history or a bank trading book. No end-to-end instrument valuation, point-in-time market-data join or reconstructed daily portfolio P&L yet. |
| `tail_risk.py` | POT-GPD MLE, threshold sensitivity, conditional and full-sample IID bootstrap | Threshold grid and IID bootstrap are useful. Both bootstrap routes assume independent observations; neither handles volatility clustering or block dependence. The full-sample bootstrap reselects the numerical threshold quantile, but does not resolve the choice of threshold quantile itself. Add block/stationary bootstrap, regime-conditioned diagnostics and chronological OOS tail scoring. |
| `uncertainty.py` | Gaussian spread-derived quote bootstrap, profile likelihood and propagation | The model clearly records important conditionality and failed replicates. Main noise model assumes independent Gaussian quote errors; positive-price failures are rejected and reported. Profile likelihood uses an asymptotic (chi^2_1) cutoff. Compare uniform, Gaussian, dependent and heteroskedastic quote errors; study coverage and boundary effects. |
| `validation.py` | Kupiec, Christoffersen, HPL backtest, PLA diagnostics and RFET ledger diagnostic | Input coercion, missing-row handling, date alignment and RFET assessment-date issues have been materially strengthened. Backtesting is explicitly HPL-only; APL remains absent. RFET counts supplied evidence metadata and cannot authenticate a real transaction/committed quote or grant eligibility. PLA diagnostics are not supervisory acceptance. |
| `volatility_surface.py` | SSVI power-law surface, sufficient-condition certificate and synthetic calibration | Tests correctly distinguish the sufficient parameter region from a universal arbitrage-free guarantee. Calibration evidence is synthetic. Need parameter identifiability, objective/weight sensitivity, contaminated/missing quotes, calendar/butterfly stress tests and real dated bid/ask chains. |

---

## 5. Test-suite audit

All 21 test modules were inventoried. The current suite has 283 passing tests on each supported Python version.

The suite has useful deterministic, edge-case, property-like and cross-method checks. It is not yet an independent validation suite.

### 5.1 Stronger existing areas

- **Heston:** characteristic-function normalization, conjugate symmetry, deterministic-variance limit, parity, bounds, QE conditional moments, martingale checks and Fourier/MC comparison.
- **Risk:** fractional empirical ES mass, ties, translation/scale invariance, rolling no-lookahead and strict input validation.
- **Capital:** short-history fail-closed behavior, risk-class mapping, explicit in-sample coverage label and formula aggregation checks.
- **Validation:** missing-data treatment, no bridging across missing transitions, explicit assessment date and RFET evidence-window checks.
- **EIA/data:** malformed archive, missing series, duplicate dates, negative WTI, null observations and snapshot manifests.
- **Tail risk:** threshold sensitivity, infinite-ES shape condition, insufficient exceedance handling and reproducible bootstrap.

### 5.2 Test gaps that materially affect claims

1. **IMCC ratio below one:** absent; it misses the confirmed formula defect.
2. **Independent Heston oracle:** Fourier and QE-M are different methods, but both are implemented in this repository. This is cross-method evidence, not an external reference.
3. **Calibration identifiability:** synthetic parameter recovery is not proof that parameters are uniquely identified.
4. **Empirical option quotes:** no dated real bid/ask option chain is present.
5. **Temporal validation:** no comprehensive chronological OOS design for tail risk, calibration or portfolio risk.
6. **Dependence-aware uncertainty:** no block bootstrap or explicit conditional heteroskedastic error design in tail-risk bootstrap.
7. **Data point-in-time:** no complete publication-time/vintage tests for the multi-source panel.
8. **Performance:** no benchmark suite measuring time, memory and accuracy together.
9. **Property-based testing:** no Hypothesis-based randomized property suite is configured.
10. **Coverage and typing:** CI does not enforce coverage thresholds or a static type checker.
11. **Independent benchmark suite:** no dedicated reference-result registry with source, convention, precision and tolerance metadata.
12. **Reproducibility:** no dependency lockfile; CI resolves latest compatible versions.

### 5.3 Notable test concentration

- `tests/test_factor_pnl.py` has one reconciliation test. Reconciliation is necessary but does not validate economic factor attribution.
- `tests/test_calibration.py` concentrates on bid/ask residual conventions and synthetic recovery.
- `tests/test_volatility_surface.py` validates SSVI identities, sufficient-condition certificates and synthetic calibration, not market fit.
- `tests/test_sec_nport.py` tests a small official filing fixture and parser behavior, not a complete holdings-to-risk pipeline.
- `tests/test_tail_risk.py` covers core POT behavior but not serial dependence or regime shifts.

---

## 6. Data and empirical evidence audit

| Evidence stream | What it currently supports | What it does not support |
|---|---|---|
| FRED daily series | Observed public market/economic proxy histories, subject to source calendars and revisions | Point-in-time vintage reconstruction, transaction-level prices, option surfaces or full desk risk factors |
| EIA WTI/Brent spot | Historical spot-proxy diagnostics, including the negative WTI observation | Futures settlement curves, contract rolls, basis, storage or convenience-yield model validation |
| SEC N-PORT | Filed fund-holdings snapshots with report/filing metadata | Daily trading positions, intraday trades, full derivatives exposure in all cases, actual P&L or bank desk holdings |
| Synthetic market fixtures | Deterministic controlled experiments and known-truth recovery | Market calibration or empirical performance |
| Hypothetical desk positions | Demonstrations of valuation, P&L explain and risk calculations under stated assumptions | Actual HPL/APL or institutional desk validation |

### 6.1 Point-in-time requirement

A point-in-time record must distinguish at least:

- economic observation date;
- publication/filing date;
- timestamp at which the data became available;
- retrieval timestamp;
- revision/vintage identifier where applicable;
- raw source hash;
- transformation/version;
- market calendar and timezone.

A filing's report date is not its public availability date. A revised macroeconomic observation must not be backfilled into a historical simulation before the revision became available.

### 6.2 Portfolio reconstruction boundary

The N-PORT parser should be evolved into a separate reproducible holdings-to-risk experiment, not silently substituted for a trading desk book. Its first valid claim should be: “risk diagnostics on disclosed fund holdings as of a specified filing snapshot,” with all valuation and market-data limitations stated.

---

## 7. Numerical and statistical research gaps

### Heston and numerical methods

Required next experiments:

1. Characteristic-function branch/continuity maps over frequency, maturity, correlation and vol-of-variance.
2. Fourier truncation and quadrature tolerance convergence, with per-contract error budgets.
3. QE-M weak price convergence versus time steps, across Feller-satisfied and Feller-violating regimes.
4. Separate Monte Carlo sampling error from time-discretization bias using coupled paths.
5. Delta/Gamma/Vega bump sensitivity and finite-difference convergence.
6. Independent Heston reference prices (e.g. a separately maintained library or published numerical benchmark), with conventions reconciled before comparison.

### SSVI and calibration

1. Synthetic parameter-identification maps, Jacobian singular values and profile likelihoods.
2. Multiple starting points and parameter-bound sensitivity.
3. Price-space versus implied-volatility-space objectives.
4. Bid/ask-weighted and robust loss comparisons.
5. Real dated option-chain experiment, including stale/crossed/zero-bid filtering.
6. Calendar and butterfly arbitrage diagnostics with explicit sufficient versus necessary conditions.

### EVT and statistical inference

1. Block/stationary bootstrap and comparison against IID bootstrap.
2. Threshold selection uncertainty distinct from threshold-quantile uncertainty.
3. Extremal-index / declustering sensitivity.
4. Rolling-origin tail forecast scoring and exception diagnostics.
5. Confidence interval coverage experiments under known DGPs, including GARCH and regime-switching designs.
6. Power analysis for Kupiec/Christoffersen and multiple-testing treatment when many models/thresholds are compared.

### Portfolio and factor risk

1. Dependence-aware reduced-factor selection against standalone-ES ranking.
2. Marginal/component ES contribution and finite-difference consistency.
3. Correlation-breakdown and stress-dependence experiments.
4. Robust covariance and allocation methods from Materials as controlled alternative estimators, not presumed upgrades.
5. Chronological portfolio experiment with turnover, costs and survivorship/look-ahead controls.

---

## 8. Materials repository: usable independent knowledge

The Materials repository's README and GOVERNANCE were inspected. Its evidence states are explicit: `FIRST_PASS`, `DERIVED`, `REPRODUCED`, `VERIFIED` and `APPLIED`. This distinction should be mirrored in Risk-manage's component maturity register.

Relevant work already recorded in Materials:

| Materials research | Evidence recorded there | Appropriate Risk-manage use |
|---|---|---|
| GARCH(1,1) conditional QMLE | Analytic recursive score checked against finite differences; simulation recovery under Gaussian and Student-t(8) | Independent volatility forecast baseline and uncertainty comparison; not yet empirical validation |
| Carr–Madan Fourier pricing and Merton jump model | Fourier versus Poisson-mixture price agreement around (10^{-13}) in a controlled case | Independent Fourier implementation pattern and benchmark design; not a Heston oracle |
| Euler–Maruyama versus Milstein | Coupled GBM experiment with empirical strong orders near 0.5 and 1.0 | Numerical-convergence methodology for Heston QE-M; scalar GBM results do not validate Heston |
| Composite-null KS bootstrap | Refits nuisance parameters in every replicate; one controlled size experiment | Better statistical calibration patterns for model diagnostics |
| PCA/eigengap and robust scatter | Sign/subspace-invariant checks and controlled p>n portfolio experiment | Factor-selection stability and covariance-estimation alternatives |
| Almgren–Chriss execution | Continuous model with independent Simpson and discrete convex cross-checks | A future execution-cost research reference, not current hedge transaction-cost validation |
| Kalman filtering/RTS smoothing | Linear-Gaussian reference implementation and tests | Future latent-factor/regime-filtering research |

Materials itself records limitations and open work. These items must not be promoted to `VERIFIED` or `APPLIED` in Risk-manage until the corresponding code is actually integrated and independently tested.

---

## 9. Engineering and reproducibility audit

### Confirmed gaps

- No committed `uv.lock` or equivalent full dependency lock.
- CI installs latest compatible dependencies from lower bounds.
- CI runs pytest and Ruff but does not enforce coverage, typing or performance baselines.
- The test suite is deterministic in many places, but there is no single experiment manifest schema enforcing code revision, dependency versions, configuration, seed, input hashes and output hashes for every experiment.
- Reports are generated in CI and uploaded as artifacts; not every benchmark result is preserved as a versioned, reviewed reference result in the repository.

### Required engineering additions

- Lock runtime and development dependencies; define an update procedure.
- Add coverage measurement first, then set a justified threshold rather than guessing one.
- Add static typing checks for public APIs and numerical boundary functions.
- Introduce experiment manifests with immutable inputs, seed, revision, dependency versions, conventions and output hashes.
- Store a small number of canonical benchmark fixtures and expected outputs, with tolerances and source references.
- Add performance benchmarks only after correctness tests exist; measure runtime and memory together with error.

---

## 10. Maturity assessment

This is not a single overall grade. Maturity is dimension-specific.

| Component | Implementation | Mathematical | Numerical | Statistical | Empirical | Independent benchmark | Reproducibility |
|---|---|---|---|---|---|---|---|
| Black–Scholes / Garman–Kohlhagen / Black-76 | Implemented | Basic identities tested | Partial | N/A for pricing primitive | No market validation | Partial analytic identities | Partial |
| Heston Fourier | Implemented | Internal identities tested | Partial; convergence map missing | N/A for deterministic pricer | Synthetic only | No external Heston oracle | Experiment scripts/seeds |
| Heston QE-M | Implemented | Conditional moments/martingale tested | Partial; discretization bias map missing | MC sampling intervals | Synthetic only | Cross-method Fourier comparison, not independent implementation | Experiment scripts/seeds |
| SSVI | Implemented | Sufficient certificate tested | Partial | Calibration uncertainty incomplete | Synthetic only | No external surface benchmark | Partial |
| VaR / ES | Implemented | Empirical convention explicit | Basic edge cases | Basic tests | Public proxies only | Manual independent tail-sum test | Partial |
| Backtesting / PLA | Implemented | Test statistics present | N/A | HPL diagnostics; power not studied | HPL proxy only | Regulatory formula/source checks partial | Partial |
| RFET | Diagnostic implemented | Criteria encoded in part | N/A | Evidence-window counts | No authenticated transaction evidence | Basel criteria mapped; evidence authenticity absent | Ledger metadata |
| IMCC / NMRF | Formula primitives implemented | **IMCC unconstrained floor defect open** | Basic tests | N/A | No bank portfolio | Basel formula comparison found defect | Partial |
| EVT | Implemented | GPD formula conditions present | MLE/bootstrap diagnostics | IID uncertainty only | Synthetic/hedge-simulated | No independent EVT package comparison | Seeded scripts |
| Data ingestion | Implemented | Schema and provenance checks | Data-quality checks partial | N/A | FRED/EIA observed data | Official source ingestion | Hash manifests, no full PIT |
| SEC N-PORT | Parser implemented | Schema/units partially checked | Parsing tests | N/A | Public filed snapshots | SEC source fixture | Hash-linked snapshot |
| Portfolio P&L | Implemented | Approximate factor mapping | Basic reconciliation | No statistical validation | Hypothetical positions | No independent repricing | Partial |

The above table intentionally avoids a 9/10 claim. A component may be mathematically useful without empirical evidence; empirical data presence does not by itself establish a valid model.

---

## 11. Prioritized remediation backlog

### P0 — Correctness blockers

1. Add below/equal/above-one regression cases for the IMCC floor and verify the class-level ratios against MAR33.15.
2. Reconcile every remaining formula in `capital.py` against the exact applicable Basel MAR33 paragraphs, including numerator/denominator definitions and stress/current period conventions.
3. Reconcile `liquidity_adjusted_es` to the exact intended liquidity-horizon formula, including assumptions about independent/non-overlapping risk-factor shocks and scaling.
4. Audit all price/return/loss sign conventions and units end-to-end from source data through P&L and VaR/ES.
5. Confirm that no data transformation creates an implicit future-information leak.

### P1 — Independent numerical validation

6. Heston characteristic-function branch/continuity and Fourier convergence studies.
7. QE-M time-step convergence and separation of sampling error from discretization bias.
8. Independent benchmark for Heston and Black-family prices/Greeks.
9. SSVI identifiability, calibration stability and static-arbitrage stress tests.
10. Numerical error budgets for all important solver/integration outputs.

### P2 — Statistical validity

11. Dependence-aware EVT uncertainty and chronological OOS evaluation.
12. Backtest power/size experiments and missing-observation policy reconciliation with the relevant regulatory standard.
13. PLA metric and zone calculation reconciliation against MAR32.
14. Calibration uncertainty coverage experiments under controlled known-truth DGPs.
15. Reduced-factor selection comparison against dependence-aware alternatives.

### P3 — Empirical evidence and data engineering

16. Point-in-time data model with observation, publication, retrieval and vintage timestamps.
17. End-to-end SEC N-PORT holdings-to-risk experiment with explicit disclosed-fund-holdings scope.
18. Real dated options data with bid/ask quality controls and license review.
19. Futures settlement and curve data for commodity risk; retain spot proxy as a separate benchmark.
20. Chronological OOS risk experiment with a reproducible public portfolio and explicit limitations.

### P4 — Reproducibility and communication

21. Dependency lock and environment manifest.
22. Experiment manifest and canonical benchmark registry.
23. Coverage, typing and performance measurement.
24. Update model cards and maturity register from evidence, not aspirations.

---

## 12. Acceptance rules for the first remediation cycle

A correction is not complete until all applicable items below are satisfied:

- Formula and convention cited to a primary source.
- Assumptions, domain, units, sign and calendar documented.
- Counterexample or analytic benchmark added.
- Independent implementation or derivation used where feasible.
- Boundary and degenerate cases tested.
- Existing regression suite passes.
- New experiment runs and its configuration/seed/output are preserved.
- Result is reviewed for numerical stability and economic interpretation.
- Maturity register updated without overstating validation.

### First authorized code task after this audit

**Fix the IMCC unconstrained ratio floor and add the missing regression cases.** This is a small, localized correction with a clear primary-source formula and a falsifiable acceptance test. It does not require restructuring the module or changing the project's architecture.

---

## 13. Primary references

- Basel Committee, [MAR33 — Internal models approach: capital requirements calculation](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/33/inforce/2022-01-01/published/2019-12-15).
- Basel Committee, [MAR32 — Backtesting and P&L attribution test requirements](https://www.bis.org/basel_framework/chapter/MAR/32.htm?inforce=20230101&published=20200327).
- Basel Committee, [MAR31 — Internal models approach: model requirements and RFET](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/31/inforce/2023-01-01/published/2020-03-27).
- Risk-manage CI: [run 45922623187](https://github.com/MedeiroszJoao/Risk-manage/actions/runs/45922623187).
- Materials repository: [README](https://github.com/MedeiroszJoao/materials/blob/main/README.md) and [GOVERNANCE](https://github.com/MedeiroszJoao/materials/blob/main/GOVERNANCE.md).

---

**Audit conclusion:** Risk-manage should be preserved and deepened. The current test suite is a real software-quality asset, but not a substitute for independent quantitative validation. The IMCC floor is consistent with MAR33.6; the next task is to test that branch explicitly, not to remove it. Beyond that, the largest gaps are external numerical benchmarks, dependence-aware statistical validation, point-in-time empirical data and a transparent distinction between simulated books, disclosed fund holdings and actual trading-desk evidence.
