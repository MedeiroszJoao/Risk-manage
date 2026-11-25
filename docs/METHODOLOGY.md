# Methodology and formula map

## 1. Historical VaR and Expected Shortfall

For daily portfolio losses L, the historical VaR at confidence c is the empirical c-quantile. Expected Shortfall is the arithmetic mean of losses at or beyond the VaR threshold. The report preserves the number of tail observations because a 99.9% tail estimate from a small sample is not precise.

Rolling one-day VaR at date t uses only the previous 250 daily observations. The forecast at t is not allowed to use HPL observed on day t.

## 2. Backtesting

An exception occurs when one-day HPL loss exceeds the forecast VaR. The report includes:
- Exception count and frequency.
- Kupiec unconditional-coverage likelihood-ratio test.
- Christoffersen first-order independence likelihood-ratio test.
- 99% exception zones and multipliers for a 250-observation window: Green 0–4; Amber 5–9; Red 10 or more. Multipliers: 1.50 for 0–4, then 1.70 / 1.76 / 1.83 / 1.88 / 1.92 for 5–9, and 2.00 for 10 or more.

The desk report runs both 97.5% and 99%. It is explicitly **HPL-only** because the project has no Actual P&L or bank valuation adjustments. It is not a complete regulatory backtest.

## 3. P&L attribution

For the latest 250 matched observations:
- Spearman rank correlation measures whether RTPL ranks days like HPL.
- KS distance is D = sup_x |F_HPL(x) − F_RTPL(x)|, the maximum absolute difference between empirical CDFs.

The implemented Basel-style PLA zones are:
- Green: Spearman > 0.80 and KS < 0.09.
- Red: Spearman < 0.70 or KS > 0.12.
- Amber: all intermediate cases.

The code reports an insufficient-sample state unless 250 paired observations exist.

## 4. Instrument valuation and RTPL

- Equity European options: Black–Scholes calls and puts.
- FX European calls: Garman–Kohlhagen.
- Rates: zero-coupon Treasury-yield proxies, with yield inputs expressed in percent.
- Commodities: WTI delta-one future.
- Volatility exposure: a deliberately simplified variance-swap mark q × (σ² − K²) × τ. This proxy is used to test whether a model sees volatility risk; it is not a complete variance-swap fair-value model.

HPL fully revalues the constant-maturity hypothetical positions. The RTPL approximation includes first-order sensitivities; the expanded model additionally includes per-factor second-order curvature. Cross-gamma / vanna, volatility-surface dynamics and production curve conventions remain outside scope.

## 5. Liquidity-adjusted ES

Using a 10-day base horizon T and liquidity horizons 10, 20, 40, 60 and 120 days, the implemented structure is:

ES_LH = sqrt( ES_10(P)^2 + sum_{j>=2} [ ES_10(P_j) × sqrt((LH_j − LH_{j−1}) / 10) ]^2 ).

P_j contains positions whose mapped liquidity horizon is at least LH_j, while other factors are held constant. The current implementation uses historical 10-day overlapping factor P&L vectors and declared horizon mappings; it is an educational implementation of the scaling structure, not a production risk-factor mapping.

## 6. Stress and reduced-factor coverage

The engine selects the 250-day window with the highest 97.5% ES from available historical / synthetic factor P&L. A reduced factor set is selected in descending individual factor ES contribution. Its coverage is then checked at the latest 60 business-day endpoints, each using a 250-day estimation window; the average ratio ES_reduced / ES_full must be at least 75%.

The Basel stress calibration framework requires a sufficiently long historical stress data span (including the 2007 period). The synthetic reference sample begins in 2016; the public FRED SP500 daily series used here also does not supply a complete daily record back to 2007. Therefore the report explicitly marks stress-history sufficiency as **not met**.

## 7. Illustrative IMCC

The implementation follows the IMA aggregation shape with rho = 0.5:

IMCC = rho × IMCC(C) + (1 − rho) × sum_i IMCC(C_i)

IMCC(C) = ES_R,S × max(ES_F,C / ES_R,C, 1)

Each partial risk class is calculated separately using its full-current, reduced-current and reduced-stress ES. All figures are research outputs with synthetic / public proxy inputs, not regulatory capital.

## 8. RFET and NMRF

The RFET diagnostic recognizes only explicitly flagged transaction, committed-quote or qualifying third-party quote records. Public closing-price rows are not treated as proof of those records. No modellerability pass is issued when the evidence ledger is missing or contains only public closes.

The NMRF aggregation primitive is unit-tested separately. No NMRF charge is claimed from the reference case because no independently verified non-modellable risk-factor evidence / stress scenario file is available.
