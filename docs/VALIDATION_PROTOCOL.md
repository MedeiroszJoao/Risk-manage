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
3. Run 97.5% and 99% VaR backtests with rolling forecasts that use only prior observations.
4. Calculate Kupiec and Christoffersen diagnostics.
5. Run PLA on the most recent 250 matched HPL / RTPL observations.
6. Record the worst residual dates and connect them to omitted risk factors.

## Stress and capital
1. Select the 250-day window that maximizes historical 97.5% ES over available inputs.
2. Select a reduced factor set and evaluate ES coverage over the latest 60 business-day endpoints using a rolling 250-day estimation window.
3. Require average reduced/full ES coverage of at least 75%.
4. Apply 10/20/40/60/120-day liquidity horizons and compute the illustrative IMCC.
5. Report whether the available history begins on or before 2007. If not, mark the full stress-history condition as not met.

## RFET
1. A public daily close is a historical price observation, not automatically a transaction or committed quote.
2. Require a verified evidence ledger with qualifying evidence types.
3. When eligible transaction / quote evidence is absent, report “not evidenced”; never claim modelled risk-factor status.

## Acceptance
A reproducible run must include the input provenance, frozen configuration, desk P&L, validation outputs, liquidity / stress / capital diagnostics, RFET evidence status, machine-readable reports and test logs. No output may claim actual bank performance or regulatory acceptance.
