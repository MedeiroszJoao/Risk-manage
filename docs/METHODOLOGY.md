# Methodology and formula map

## Backtesting
For one-day loss observations L_t and forecast VaR_t at confidence c, exception_t = 1[L_t > VaR_t]. Report exception count, empirical frequency, Kupiec unconditional-coverage likelihood ratio and Christoffersen independence diagnostic. Basel-style zones are based on 250 daily observations: green 0–4 exceptions, amber 5–9, red 10 or more for the bank-wide 99% framework.

## P&L attribution
For the most recent 250 matched HPL and RTPL observations, calculate Spearman rank correlation and the two-sample empirical CDF distance D_KS = sup_x |F_HPL(x) - F_RTPL(x)|. The Basel MAR32 PLA thresholds implemented in this project are:
- Green: Spearman > 0.80 and D_KS < 0.09.
- Red: Spearman < 0.70 or D_KS > 0.12.
- Amber: intermediate values.
A 250-observation sample is required; otherwise report insufficient data.

## Historical VaR / ES
For losses L, empirical VaR_c is the c-quantile; ES_c is the average loss at or above VaR_c. Report finite-sample tail count and do not imply precision beyond the actual number of tail observations.

## Liquidity-adjusted ES
The engine applies the Basel 10-day base-horizon formula over nested risk-factor subsets at liquidity horizons 10, 20, 40, 60 and 120 days:
ES_LH = sqrt( ES_10(P)^2 + sum_{j>=2} [ ES_10(P,j) * sqrt((LH_j-LH_{j-1})/10) ]^2 ).
P,j denotes positions exposed only to risk factors with liquidity horizon at least LH_j, holding other factors constant.

## Stressed ES and IMCC
Use an explicit stress interval and a reduced risk-factor set. The reduced set must be assessed against the full current ES; the Basel standard requires the reduced set to explain at least 75% of the full model on average over the preceding 12-week period. The illustrative IMCC function uses rho=0.5:
IMCC = rho * IMCC(C) + (1-rho) * sum_i IMCC(C_i)
where IMCC(C) = ES_R,S * max(ES_F,C / ES_R,C, 1), with a separately calculated partial risk-class expression for C_i. This implementation must clearly mark any unavailable empirical or supervisory inputs.

## RFET
A public close series provides historical price observations, but does not itself prove a transaction or committed quote. Without an evidence ledger containing qualifying real-price evidence, modellability is “not evidenced”, not “pass”.
