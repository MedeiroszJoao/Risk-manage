# Validation protocol

1. Verify every market-series identifier and source URL.
2. Record dates, duplicate rows, missing values, stale gaps and source hashes.
3. Freeze instrument definitions and hypothetical position quantities before generating reports.
4. Calculate HPL by static-position full revaluation.
5. Calculate RTPL separately from its declared risk-factor subset and sensitivities.
6. Produce rolling one-day VaR using only information available at the forecast date.
7. Compare HPL to the historical VaR; label the result HPL backtesting diagnostic, not APL backtesting.
8. Calculate PLA on the most recent 250 matched observations.
9. Report tail sample counts and any insufficient-history conditions.
10. Compute liquidity-adjusted and stressed ES only where the source history supports the stated window.
11. Never certify RFET modellability from public daily closes alone.
12. Preserve config, source manifest, environment and generated report hashes for reproducibility.

## Acceptance
A reproducible run must include the config, market data provenance, desk P&L, validation diagnostics, stress and capital outputs, limitations, and tests. A passing software test suite is not evidence of predictive accuracy or regulatory acceptance.
