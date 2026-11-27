# RISK-MANAGE — Project Excellence Directive

**Effective date:** 2026-11-27  
**Applies to:** every module, experiment, correction, dataset, benchmark and report in this repository.

## 1. Project identity and non-negotiable constraint

RISK-MANAGE remains a **Quantitative Market Risk Research Laboratory**. Its existing research direction, modules, experiments and architecture are to be preserved and deepened.

This directive does not authorize a ground-up rewrite, simplification for presentation, wholesale architecture replacement, or removal of existing quantitative work. Refactoring is allowed only when a demonstrated correctness, numerical, research or maintainability problem justifies it.

The objective is to make the existing laboratory defensible for work relevant to Quantitative Research, Quantitative Development, Market Risk, Model Validation and Financial Engineering.

## 2. Quality dimensions

Every important component must be assessed independently on the dimensions that apply to it.

### 2.1 Mathematical integrity
- Document the mathematical definition, notation, assumptions, domain and conditions of validity.
- State sign conventions, units, calendars, compounding, day-count and annualization conventions.
- Verify analytical limits, identities, invariances, boundary cases and degenerate cases.
- Distinguish exact results, approximations, discretizations and heuristics.
- Quantify material approximation error.
- Do not treat plausible output or self-authored unit tests as proof of mathematical correctness.

### 2.2 Independent validation
- Use at least one genuinely independent reference for each important method where a suitable reference exists.
- Prefer analytic benchmarks and published worked examples; otherwise use independent algorithms or libraries.
- Separate calibration, validation and final test data.
- Investigate disagreements; never widen tolerances merely to obtain a pass.
- Record benchmark provenance, conventions, versions and limitations.

### 2.3 Statistical integrity
- Preserve chronological ordering and point-in-time information sets.
- Prevent look-ahead bias and leakage in all joins, transformations, selection and model fitting.
- Account for serial dependence, overlapping observations, heteroskedasticity and regime changes where relevant.
- Report uncertainty and confidence intervals where appropriate.
- Assess statistical power and finite-sample behavior.
- Distinguish statistical significance from economic importance.
- Record all attempted variants and failed experiments; account for multiple testing where relevant.

### 2.4 Data integrity and point-in-time provenance
- Record source, retrieval timestamp, observation date, publication/availability date, period, schema, license and transformations.
- Preserve immutable raw snapshots and cryptographic hashes.
- Distinguish observed, synthetic, estimated, imputed and revised data.
- Test missing values, duplicates, stale observations, invalid identifiers, units and calendar alignment.
- Preserve amendments and revisions without backdating them.
- Build point-in-time reconstruction so that a historical experiment uses only information available at its decision time.

### 2.5 Scientific honesty
No maturity claim may be inferred from code presence, visual polish or a passing CI run.

The following terms require explicit evidence and scope: `validated`, `production-ready`, `market-calibrated`, `regulatory-compliant`, `arbitrage-free`, `statistically significant`, `robust` and `industry-grade`.

Negative results, failed replications, excluded observations and uncertainty must remain visible. A component may be implemented but mathematically unverified; numerically verified but statistically untested; statistically tested but not empirically validated.

### 2.6 Software and performance engineering
- Use strict input contracts, explicit errors, consistent interfaces and clear units.
- Separate calculation, data acquisition, configuration and reporting without unnecessary architectural proliferation.
- Maintain deterministic tests, property tests, numerical tests, integration tests, regression tests and end-to-end tests.
- Record seeds, configurations, package versions, data hashes and output metadata.
- Measure runtime and memory before optimizing. Adopt vectorization, parallelism, compiled code or accelerators only when measured evidence justifies them.

## 3. Component maturity model

Maturity is a vector, not one global status.

| Dimension | Allowed states |
|---|---|
| Implementation | Not started / Implemented |
| Mathematical | Unverified / Verified |
| Numerical | Untested / Benchmarked / Validated within stated domain |
| Statistical | Untested / Tested with limitations / Validated within stated design |
| Empirical | Synthetic / Partially empirical / Chronological OOS validated |
| Independent evidence | None / Analytic reference / Independent implementation / Published benchmark |
| Reproducibility | Partial / Reproducible |
| Research | Exploratory / Comparative / Candidate contribution |

A dimension can be not applicable only with an explicit reason. CI success establishes automated execution for the tested environment; it does not establish mathematical, empirical or regulatory validity.

## 4. Required scientific record for important modules

Each important quantitative module must have:
1. Problem definition and intended use.
2. Mathematical formulation and source references.
3. Assumptions, units, sign conventions and input contracts.
4. Implementation and numerical method.
5. Independent verification and benchmark provenance.
6. Edge cases, convergence, stability and error budget as applicable.
7. Statistical design, sample construction and uncertainty as applicable.
8. Sensitivity and robustness analysis.
9. Limitations and non-claims.
10. Reproducible experiment configuration and results.

Model cards and experiment reports must follow the same evidence distinctions.

## 5. Research laboratories

### Numerical Validation Laboratory
Maintain analytic references, independent algorithms, convergence studies, stability maps, error budgets, tolerance sensitivity and precision-versus-cost benchmarks.

### Statistical Validation Laboratory
Maintain coverage and independence diagnostics, scoring rules, finite-sample analysis, power studies, temporal stability, forecast comparisons and appropriate multiple-testing corrections.

### Independent Benchmark Suite
Store external reference cases, exact conventions, expected results, tolerance rationale, source/version provenance and automated divergence reports.

### Reproducible Research
Every experiment must be numbered/versioned and retain configuration, seed, package environment, source hashes, output metadata and failed runs.

### Performance Engineering
Benchmark runtime and memory, identify actual bottlenecks and quantify any accuracy/performance trade-off.

## 6. Priority order

1. **Technical integrity:** complete audit; correct input contracts, formulas, edge cases, units, calendars and missing-data semantics.
2. **Independent validation:** external references, analytic limits, convergence, numerical stability and error budgets.
3. **Empirical research:** suitable real data, point-in-time provenance, chronological holdouts and independent out-of-sample tests.
4. **Quantitative depth:** deepen Heston, SSVI, parameter/model uncertainty, EVT and capital methodology.
5. **Research expansion:** advanced volatility, multivariate tail risk, portfolio risk/optimization, commodity/energy risk, liquidity and market microstructure.
6. **Research publication:** reproducible technical reports, benchmark datasets and transparent positive/negative findings.

Expansions are gated by evidence. Do not add a model merely because it is sophisticated or fashionable.

## 7. Existing modules: mandatory focus

- `validation.py`: distinguish operational exceptions from inferential populations; handle missing observations and irregular sequences correctly; strengthen PLA and RFET diagnostics without implying approval.
- `risk.py`: define empirical VaR/ES conventions; test ties and discrete samples; validate horizons, calendars, overlapping losses, liquidity aggregation and stress selection.
- `capital.py`: separate factor selection from evaluation; study correlation and marginal/component risk; derive IMCC, reduced-factor and NMRF procedures from authoritative sources; keep exploratory calculations explicitly non-regulatory.
- `heston.py`: investigate complex branch stability, Fourier truncation/quadrature, QE transition boundaries, martingale behavior, discretization bias, sampling error and independent reference agreement.
- `volatility_surface.py`: assess identifiability, conditioning, multiple minima, quote noise, bid/ask, stale data, static-arbitrage conditions and temporal OOS stability.
- `uncertainty.py` and `hedging_uncertainty.py`: separate parameter, model, numerical and sampling uncertainty; report failed fits and the effect of noise assumptions.
- `tail_risk.py`: test threshold stability, temporal dependence, clustering, regimes, alternative tails and OOS uncertainty.
- All other modules remain in scope for the same mathematical, numerical, statistical, data and software standards.

## 8. Empirical evidence boundaries

Public fund holdings are disclosed positions, not bank trading-desk history. Public prices do not provide actual P&L, transactions, intraday marks or RFET evidence. HPL is not APL. A public-data proxy cannot establish regulatory IMA approval.

Real option calibration requires timestamped and appropriately licensed option quotes, forwards/discount curves, quote quality controls and point-in-time availability. Spot commodity prices are not futures settlements or complete futures curves.

## 9. Prohibited shortcuts

- Do not reconstruct or replace the project architecture without demonstrated necessity.
- Do not delete or hide existing work to improve presentation.
- Do not treat synthetic data as market evidence.
- Do not label unit tests as scientific validation.
- Do not increase tolerances to conceal divergence.
- Do not silently drop observations or failed replications.
- Do not report only successful experiments.
- Do not claim regulatory compliance from formula primitives.
- Do not optimize prematurely or add technology for prestige.
- Do not advance maturity states without the required evidence.

## 10. Approval rule

A result is approved only when its evidence supports the exact claim being made. The highest maturity score is reserved for components with complete, internally consistent, independently supported, reproducible and falsifiable evidence. A component with an inapplicable validation dimension must explain why; it does not receive an automatic pass.

The audit register and maturity register are the authoritative records of defects, open questions, evidence and acceptance status. No backlog item is complete until its code, tests, independent evidence, documentation and reproducible execution have all been checked.
