# Case study — A desk can pass VaR backtesting and still fail P&L attribution

## Research question
A risk model can have few daily VaR exceptions yet fail to reproduce what drives desk P&L. Nonlinear option curvature, implied-volatility moves, yield-curve twists and omitted factors can invalidate a simplified risk-theoretical P&L even when a short backtest looks quiet.

This case compares full-revaluation HPL with independent RTPL for four hypothetical desks. It is intentionally designed to expose model failure rather than manufacture universal pass results.

## Desks
| Desk | Positions / factors | Model-risk challenge |
|---|---|---|
| Equity Options | S&P 500 spot, European calls, VIX | Delta-only model omits gamma / vega |
| FX Options | EUR/USD, USD/JPY, vanilla FX calls | FX spot / implied-volatility risk |
| Rates | 2Y / 10Y Treasury exposures and zero-coupon bonds | Duration-only approximation misses convexity / curve shape |
| Commodities | WTI delta-one future and optional | Linear exposure as a control |

Position quantities and strikes are hypothetical. Real-data runs use public FRED series; CI is driven by a labelled synthetic fixture.

## P&L semantics
- **HPL:** static positions from prior close, fully revalued with current-close market observations.
- **RTPL:** a separately declared risk approximation, computed from previous-close sensitivities and a selected subset of risk-factor changes.
- **PLA:** compares the 250-day RTPL and HPL distributions using Spearman rank correlation and empirical-CDF distance.
- **Backtesting:** compares rolling one-day VaR with HPL. Actual P&L is not available; this is not a complete regulatory backtest.

## Expected evidence
The report should identify which desks fail the implemented diagnostics, list largest HPL-vs-RTPL residual days, and connect those residuals to market factors. It should propose remediation such as adding vega / gamma, curve-basis factors or nonlinear revaluation and then repeat the same validation.
