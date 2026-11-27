# Risk-manage — Technical Audit & Remediation Register

**Reviewed:** 2026-11-27  
**Branch:** `portfolio/frtb-ima-risk-engine`  
**Audit baseline:** current branch HEAD before this remediation pass: `fcdb03b48ba23d1dde374703521ba29fbdc0f91f`  
**Purpose:** record reproducible findings and acceptance criteria. This document is not a regulatory validation or a declaration of production readiness.

## Audit principles

- Preserve the existing Risk-manage identity, modules and research direction.
- Distinguish confirmed defects from methodological risks and missing evidence.
- A passing unit test is necessary but not sufficient evidence of quantitative correctness.
- Synthetic recovery, analytic identity checks, independent numerical benchmarks and empirical out-of-sample validation are separate evidence classes.
- No regulatory or production claim is permitted without the required data, controls, methodology and independent review.

## Confirmed findings from the current code

| ID | Area | Confirmed observation | Required treatment |
|---|---|---|---|
| VAL-01 | `validation._probability` | Converts strings and booleans with `float()`; e.g. `"0.99"` and `True` are accepted as confidence levels. | Reject booleans and non-numeric types before conversion. |
| VAL-02 | Numeric input boundaries | NumPy can promote mixed `[True, 0.5]` lists to floating dtype before dtype validation, silently turning a boolean into a measurement. Similar coercion existed in risk, capital, EVT and SSVI boundaries. | Detect booleans before numeric promotion; reject strings and object coercion. Preserve NaN/inf for explicit caller-specific handling. |
| VAL-03 | `validation.backtest_var` | Missing HPL/VaR observations are correctly counted as operational outliers, but the same fabricated hit sequence is passed to Kupiec and Christoffersen. | Separate operational exception accounting from statistical inference. Inferential populations must contain observed, matched HPL/VaR pairs only; Christoffersen transitions must not bridge missing observations. |
| VAL-04 | `validation.rfet_diagnostic` | Invalid dates and missing factor identifiers are dropped without reporting the number or proportion excluded. | Retain exclusion counts and data-quality diagnostics; never let invalid rows disappear silently. |
| CAP-01 | `capital.select_reduced_factor_set` | Factor selection and the reported 60-day coverage window overlap the same input sample. | Keep the current statistic explicitly descriptive/in-sample; build a chronologically separated selection/evaluation design before calling it OOS. |
| CAP-02 | `capital.select_stress_window` | Selects the worst window by a historical ES objective. | Keep it labelled exploratory; do not equate it to the FRTB MAR33 aggregate-IMCC stress-period procedure. |
| CAP-03 | `capital.build_ima_capital` | Research aggregation is gated against short histories and incomplete risk classes, but does not implement all supervisory requirements. | Maintain fail-closed research labels; independently implement and verify each required methodology before stronger claims. |
| HES-01 | `heston.py` | Existing tests cover characteristic-function identities, special limits, QE moments and Fourier/MC agreement. They do not yet establish a quantified error budget across quadrature, truncation, time discretisation and sampling. | Add independent reference values, convergence surfaces and separate error components. |
| VOL-01 | `volatility_surface.py` | Synthetic SSVI parameter/price recovery and sufficient-region checks exist. | Add identifiability, noisy/bid-ask, stale/incomplete quote and temporal OOS studies using independently sourced option data. |
| UNC-01 | `uncertainty.py` | Current bootstrap uncertainty is conditional on its chosen noise and resampling assumptions. | Quantify failed fits, compare noise models and separate parameter, model and numerical uncertainty. |
| EVT-01 | `tail_risk.py` | Threshold grids and bootstrap diagnostics exist, but the financial-loss validation is not a chronological independent test with dependence-aware inference. | Add block/stationary bootstrap or declustering where justified, regime sensitivity and untouched OOS evaluation. |
| EVT-02 | `tail_risk.fit_pot_gpd_tail_risk` | Coerced losses, confidence levels, thresholds and control parameters implicitly; booleans, strings and fractional replicate counts could enter the API or fail later inconsistently. | Reject invalid types at the boundary and validate integer controls explicitly. |
| DATA-01 | `sec_nport.py` | Parser tests use a small selected subset from one filing. | Run against a complete official quarterly archive and reconcile all positions, identifiers, totals, amendments and source hashes. |
| ENG-01 | `pyproject.toml` | Runtime and development dependencies are ranged; no lockfile fixes the full tested environment. | Establish a reproducible dependency lock and verify clean installs on supported Python versions. |

## Remediation started in this pass

### VAL-01 / VAL-02 — strict input contracts

- `_probability` now accepts numeric scalar types only and explicitly rejects booleans and strings.
- `_one_dimensional_numeric` now rejects boolean arrays and mixed boolean/numeric lists before NumPy can promote booleans to floats.
- `_binary_sequence` now rejects boolean sequences; exception indicators must be numeric exact 0/1 values.

### VAL-03 — separate operational counts from inferential samples

- Operational exception counts continue to include missing HPL/VaR observations, preserving the conservative desk-screen behavior documented in the project.
- Kupiec now uses only finite, matched HPL/VaR pairs.
- Christoffersen now uses only adjacent finite, matched pairs. A missing observation breaks the transition chain; the implementation does not bridge across it.
- Outputs identify the population used by each statistical test.
- New regression tests cover missing-data accounting, transition breaks, boolean/string inputs and confidence validation.
- The same pre-promotion boolean guard was added to risk, capital, EVT and SSVI numeric-array boundaries.

### EVT-02 — strict POT-GPD contracts

- Loss samples must be one-dimensional finite numeric arrays; booleans, strings and object coercion are rejected.
- Confidence levels and threshold quantiles must be numeric vectors with values strictly inside (0.5, 1).
- Minimum exceedance count, bootstrap counts and seed must be strict integers; booleans and fractional values are rejected.
- Added regression tests for invalid data and scalar types, including mixed boolean/numeric inputs.

**Important limitation:** the asymptotic chi-square reference distributions still require a power/sample-size study and careful interpretation for rare exceptions. These changes correct population construction; they do not establish forecast adequacy or regulatory backtesting.

## Remaining validation questions

### Validation and backtesting
- What is the exact treatment of non-trading days, market holidays and irregular observation calendars?
- How sensitive are Kupiec and Christoffersen conclusions to finite-sample effects and low expected exception counts?
- Are unconditional and conditional coverage tests adequately powered for the sample size and confidence level?
- How should missing forecasts, missing P&L and stale marks be distinguished in operational and inferential reports?
- Which PLA statistics are diagnostic only, and which regulatory definitions/thresholds would be needed for any formal assessment?

### Risk and capital
- Compare empirical quantile conventions and ES definitions against independently derived reference calculations, including ties and discrete distributions.
- Verify overlapping versus non-overlapping horizon aggregation and the calendar assumptions behind 10-day losses.
- Establish the exact scope of liquidity-adjusted ES and the risk-factor liquidity-horizon aggregation formula before interpreting it as FRTB.
- Separate factor selection from holdout evaluation and investigate correlation, concentration and marginal contributions.
- Implement stress-period selection based on the applicable aggregate IMCC methodology only after all inputs and prerequisites are available.
- NMRF calculations require a defensible inventory, stress scenarios, risk-factor mapping and documented assumptions; formula primitives alone are not a complete methodology.

### Pricing and volatility
- Heston: compare against independent published/reference prices; map errors over spot/forward moneyness, maturity and parameter regimes; separate truncation, quadrature, time-step and Monte Carlo sampling errors.
- QE-M: test branch-boundary behavior and weak/strong convergence as applicable; do not use agreement with the same model's Fourier implementation as the only reference.
- SSVI: investigate parameter identifiability and conditioning; compare price-space and total-variance-space objectives; incorporate bid/ask uncertainty and timestamp/quote-quality controls.
- No market-calibration claim until a timestamped, licensed, point-in-time option dataset is available.

### Data and empirical design
- Distinguish observation date, publication date, retrieval timestamp and first-available timestamp.
- Preserve raw source payloads, hashes, parser version, schema version and transformations.
- Build point-in-time snapshots; do not backdate revised or amended filings.
- N-PORT holdings are public fund disclosures, not bank desk positions, transactions, APL or RFET evidence.
- Public spot series are not interchangeable with listed futures settlements or instrument-level P&L.

## Acceptance gates

1. Every confirmed defect has a regression test that fails before the fix and passes after it.
2. Every quantitative formula has documented notation, assumptions, units, sign conventions and limits.
3. Every important numerical method has at least one independent reference and a justified error tolerance.
4. Every statistical experiment documents its population, null hypothesis, sample construction, missing-data policy, power limitations and multiple-testing treatment where relevant.
5. Every empirical study is chronological, point-in-time and includes an untouched holdout.
6. Every published output carries configuration, seed, source hashes, package versions and a machine-readable provenance manifest.
7. Negative results and failed replications remain reportable and are not silently discarded.
8. CI passes on supported Python versions, but CI success is not described as scientific or regulatory validation.

## Current non-claims

Risk-manage is a quantitative market-risk research laboratory. It is not a bank trading-risk system, production pricing library, approved internal model or complete FRTB IMA capital implementation. There is no bank APL, actual bank desk history, full regulatory RFET evidence or complete MAR33 validation in this repository.


## Verification after first remediation pass

**Commit tested:** `64d7dd2c7058487a8f94bd5b837282ab4f0d08a3`  
**GitHub Actions run:** [risk-engine-ci #477](https://github.com/MedeiroszJoao/Risk-manage/actions/runs/45920629074)  
**Result:** successful on Python 3.11 and 3.12.

- Python 3.11: 245 tests passed in 35.67 seconds; Ruff passed; deterministic case passed; all configured Heston benchmark, calibration, Greeks, delta-hedging, randomized-QMC, uncertainty and EVT experiment steps completed successfully.
- Python 3.12: test suite and Ruff passed; deterministic case and configured Heston experiment steps completed successfully.
- Live EIA ingestion and integrated FRED/EIA four-desk workflow also completed successfully on the same source revision: [EIA run](https://github.com/MedeiroszJoao/Risk-manage/actions/runs/45920629066).

These results verify the tested software paths and reproducibility of the configured runs. They do not establish market calibration, empirical validation of real trading positions, APL/PLA eligibility or regulatory approval.
