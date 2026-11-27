# Audit remediation register

**Reviewed:** 2026-11-26  
**Branch:** `portfolio/frtb-ima-risk-engine`  
**Status:** corrective pass; this register does not certify regulatory compliance.

## Confirmed defects corrected

| Finding | Code / documentation action | What remains unproven |
|---|---|---|
| Fractional exception sequences silently truncated | Exact numeric 0/1 validation before casting; fractions and strings rejected | Forecast provenance and economic validity |
| Undefined PLA correlation replaced by zero | Return unclassifiable state; constant series handled without false numeric result | HPL/RTPL are synthetic; no regulatory PLA |
| Missing HPL/VaR observations silently dropped | Count missing observations as outliers | No APL; overall desk exception count unknown |
| Bank-wide zones/multipliers applied to desks | Removed bank-wide multipliers; use HPL-only desk limits (12 at 99%, 30 at 97.5%) | A below-limit HPL result is not desk approval without APL |
| Short sample could yield capital-shaped output | Require at least 309 observations and validate exact 250-row stress-window metadata | Software gate is not the full regulatory historical-stress requirement |
| In-sample reduced-factor statistic presented as validation | Label overlapping ratio as descriptive in-sample | No holdout or group-level 75% criterion |
| Incomplete capital-class mapping could create IMCC output | Suppress IMCC when any of the five broad classes or a selected class component is missing | Stress period still not selected by aggregate IMCC; complete MAR33 not implemented |
| Correlation weights could be tuned as though regulatory | Enforce fixed IMCC and NMRF weights | Formula primitives only |
| RFET screen could look like approval | Only candidate-for-review state; public closes not treated as trade/quote evidence | No complete real-price validation, vendor audit or representative mapping |

## Open blockers — no professional or regulatory claim

1. No APL or actual bank positions; backtesting and PLA cannot be completed at desk level.
2. No independent timestamped options panel; Heston/SSVI tests are analytic or synthetic, not empirical market validation.
3. Stress-period selection currently maximizes portfolio ES over a 250-observation window, not aggregate IMCC over a compliant 12-month stress period.
4. The 75% reduced/full ES statistic is single-book and in-sample, not the group-level IMA-desk test.
5. RFET is only an evidence-count screen; no regulatory modellability decision.
6. No independent NMRF inventory or stress-scenario dataset.
7. Dependencies remain ranged rather than locked; no lockfile.
8. QE-M time-grid sensitivity exists, but no accepted numerical-bias tolerance is defined. EVT has no independent chronological portfolio validation.

## Acceptance gate for the next release

- CI passes on Python 3.11 and 3.12; tests, lint and deterministic reports complete.
- Report includes configuration, source hashes, package versions and artifact provenance.
- Independent analytic references and declared error tolerances for numerical methods.
- Publicly disclosed real fund holdings snapshots, with dated filings and instrument identifiers, explicitly labelled as fund holdings rather than bank trading-book positions.
- Untouched out-of-sample evaluation period.
- No FRTB compliance claim before APL, actual desk positions, complete RFET evidence, MAR33 stress-period selection, liquidity mapping, NMRF scenarios and independent governance.
