# Validation protocol

## Data and provenance
1. Download FRED equity / FX / rates series and the EIA Petroleum bulk archive for WTI Cushing and Europe Brent.
2. Preserve exact raw FRED CSV responses and the EIA PET.zip snapshot; record series IDs, URLs, UTC retrieval time, row counts, missingness, archive metadata and SHA-256 digests.
3. Retain finite zero / negative WTI spot observations; require positive equity / FX levels and Brent spot, finite values, unique dates and no material unresolved gaps.
4. Align the integrated panel on business dates with a maximum four-business-day forward fill, explicitly recorded in provenance. The standalone EIA commodity history uses actual common WTI / Brent observation dates without forward fill.
5. Label every run as synthetic or public-source observed data. Do not blend synthetic and observed input without an explicit rule.

## Portfolio and HPL
1. Freeze hypothetical positions before evaluating the run.
2. Hold positions static for each daily close-to-close interval.
3. Fully revalue the prior-close book using current-close market data.
4. After the HPL mark, re-establish the constant-maturity test portfolio for the following one-day interval. This is a test-book convention, not a buy-and-hold position history.
5. Preserve all instrument definitions, strikes, maturities and quantities.

## RTPL and model diagnostics
1. Calculate RTPL independently from HPL, using the declared sensitivity set.
2. Run both delta-only and expanded gamma / vega approximations.
3. Run 97.5% and 99% HPL-only diagnostics with rolling forecasts that use only prior observations. Record explicitly that this is not a regulatory backtest without APL.
4. Calculate Kupiec and Christoffersen diagnostics.
5. Run PLA on the most recent 250 matched HPL / RTPL observations.
6. Record the worst residual dates and connect them to omitted risk factors.

## Stress and capital
1. Select the 250-day window that maximizes historical 97.5% ES over available inputs.
2. Select a reduced factor set and report the latest 60 rolling 250-observation ratios only as a descriptive, overlapping in-sample diagnostic. These observations are reused in factor selection.
3. Record whether the research 75% in-sample heuristic is met. Do not describe it as out-of-sample validation or regulatory reduced-factor approval.
4. Apply 10/20/40/60/120-day scaling only as a research illustration. Regulatory liquidity horizons are mapped to instruments/risk factors under MAR33; this generic mapping is not the regulatory mapping. Compute IMCC formula primitives only when all selected risk classes are represented.
5. Report whether the available history begins on or before 2007. If not, mark the full stress-history condition as not met.

## RFET
1. A public daily close is a historical price observation, not automatically a transaction or committed quote.
2. Require a verified evidence ledger with qualifying evidence types. For third-party data, require vendor audit controls, observation counts/dates and defensible risk-factor mapping.
3. When eligible transaction / quote evidence is absent, report “not evidenced”; never claim modelled risk-factor status.

## Acceptance
A reproducible run must include the input provenance, frozen configuration, desk P&L, validation outputs, liquidity / stress / capital diagnostics, RFET evidence status, machine-readable reports and test logs. No output may claim actual bank performance or regulatory acceptance.


## Regulatory status gate

No run is eligible to be called a regulatory desk backtest or IMA capital calculation. MAR32 requires both actual and hypothetical P&L for backtesting. MAR31 RFET requires valid real-price evidence plus appropriate controls and representative factor mapping, not a database count alone. MAR33 requires factor-specific liquidity horizons, broad risk-class and stress-period governance, and the complete applicable capital process. Missing elements must stop a regulatory claim, not merely create a warning.
