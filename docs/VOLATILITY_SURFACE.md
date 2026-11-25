# SSVI Volatility Surface — Derivation, Calibration and Validation

## Research role

This module adds a mathematically constrained implied-volatility surface to RISK-MANAGE. It is an important step beyond assigning one VIX observation to every equity option regardless of strike or expiry.

The module is intentionally limited to **static implied-volatility geometry and cross-sectional calibration**. It is not a stochastic process for future smile dynamics, a local-volatility model, a Heston pricer, or a production quote-calibration system.

## Market coordinates and conventions

For each expiry \(T\), define the forward log-moneyness

\[
k = \log(K/F_T),
\]

and the ATM total implied variance

\[
\theta_T = \sigma_{\mathrm{ATM}}(T)^2 T.
\]

The input \(\theta_T\) must be derived from the market's ATM Black implied volatility using the same expiry, forward, day-count and annualization conventions as the option quotes. Total variance is dimensionless when volatility is annualized and \(T\) is measured in years.

## SSVI parameterization

The surface total variance is

\[
w(k,\theta)=\frac{\theta}{2}
\left[
1+\rho\phi(\theta)k+
\sqrt{(\phi(\theta)k+\rho)^2+1-\rho^2}
\right],
\]

with

\[
\phi(\theta)=
\frac{\eta}{\theta^\gamma(1+\theta)^{1-\gamma}},
\quad
|\rho|<1,\quad \eta>0,\quad 0<\gamma<1.
\]

Black implied volatility is recovered as

\[
\sigma_{\mathrm{imp}}(k,T)=\sqrt{\frac{w(k,\theta_T)}{T}}.
\]

The ATM identity is exact: \(w(0,\theta)=\theta\). The parameter \(\rho\) controls skew direction and magnitude, \(\eta\) controls smile scale, and \(\gamma\) controls how the smile shape changes with ATM total variance.

## Static-arbitrage certificate

The implementation uses the sufficient global parameter region in Gatheral and Jacquier (2014), Remark 4.4:

\[
0<\gamma\leq\frac12,
\qquad
\eta(1+|\rho|)\leq2.
\]

For this power-law family, these restrictions are sufficient for the SSVI butterfly conditions over all \(\theta>0\). Calendar-spread absence additionally requires the ATM total-variance term structure to be non-decreasing with expiry.

The code also reports the underlying finite-grid inequalities from Theorems 4.1 and 4.2:

\[
0\leq
\frac{\partial_\theta[\theta\phi(\theta)]}{\phi(\theta)}
\leq
\frac{1+\sqrt{1-\rho^2}}{\rho^2},
\]

where the upper bound is infinite for \(\rho=0\), and

\[
\theta\phi(\theta)(1+|\rho|)<4,
\qquad
\theta\phi(\theta)^2(1+|\rho|)\leq4.
\]

A failed sufficient-region certificate means **not certified by this test**, not necessarily that an arbitrage exists. The module does not claim to test dynamic arbitrage.

## Calibration objective

The function fit_ssvi_power_law fits \((\rho,\eta,\gamma)\) to observed total implied variances. The expiry-specific \(\theta_T\) inputs are treated as observed, not estimated by this routine.

The weighted residual objective is

\[
\min_{\rho,\eta,\gamma}
\sum_i \omega_i
\left[
w(k_i,\theta_i;\rho,\eta,\gamma)-w_i^{\mathrm{mkt}}
\right]^2.
\]

The optimizer uses the constrained parameter \(\beta=\eta(1+|\rho|)\in(0,2]\), deterministic multi-start initializations and optional robust least-squares losses. Weights are normalized internally. Their statistical or economic interpretation must be specified by the caller.

This is a fit in **total-variance space**, not a price-space calibration. It does not include bid/ask constraints, vega weighting, quote cleaning, outlier filtering, or parameter uncertainty. Those are separate research tasks.

## Validation performed by the module

- Exact ATM total-variance identity.
- Smile symmetry when \(\rho=0\).
- Positive, finite total variance and implied-volatility conversion.
- NumPy broadcasting across strikes and expiries.
- Global sufficient-region certificate and rejection of uncertified inputs.
- Recovery of known parameters from a deterministic synthetic surface.
- Input validation for non-finite data, invalid parameters, weights and insufficient quote coverage.

The synthetic recovery test checks implementation and optimizer behavior under a correctly specified model. The calibration function rejects inconsistent ATM total variance within one expiry and a decreasing ATM total-variance term structure. It is not empirical evidence that SSVI fits real option markets, and it does not establish parameter identifiability under noisy or sparse quotes.

## Next research stages

1. Construct clean option quote panels with timestamp, bid/ask, forward, expiry, discounting and corporate-action conventions.
2. Add price-space calibration with bid/ask-aware and vega-aware objectives; quantify fit residuals by expiry and moneyness.
3. Benchmark against raw SVI slices and independent implementations; inspect density positivity and tail behavior.
4. Derive and validate implied-volatility and price Greeks, including cross effects.
5. Compare static SSVI with dynamic Heston and local-volatility models. Equal vanilla fits do not imply equal future smile dynamics or exotic prices.
6. Propagate calibration/model uncertainty into hedge P&L, HPL/RTPL, VaR and Expected Shortfall.

## Primary reference

Gatheral, J. and Jacquier, A. (2014), “Arbitrage-free SVI volatility surfaces,” *Quantitative Finance*, 14(1), 59–71. DOI: https://doi.org/10.1080/14697688.2013.819986. Open preprint: https://arxiv.org/abs/1204.0646.


## Black-76 price-space calibration

The project also includes Black-76 pricing and implied-volatility inversion for European options on forwards/futures. For forward \(F\), strike \(K\), discount factor \(D\), annualized volatility \(\sigma\), and maturity \(\tau>0\),

\[
d_1=\frac{\ln(F/K)+\frac12\sigma^2\tau}{\sigma\sqrt\tau},
\qquad d_2=d_1-\sigma\sqrt\tau,
\]
\[
C=D[F\Phi(d_1)-K\Phi(d_2)],\qquad
P=D[K\Phi(-d_2)-F\Phi(-d_1)].
\]

The implementation enforces discounted intrinsic lower bounds and discounted forward/strike upper bounds before inversion. It solves the monotone price equation using a bracketed Brent root solver rather than relying solely on Newton iteration, which can become unstable when Black vega is small. Analytic vega is available for diagnostics and finite-difference comparison.

The new price-space calibrator fits SSVI parameters directly to Black-76 prices using quote-aligned forwards, strikes, discount factors, maturities and ATM total variances. It supports weighted mid-price least squares and a bid/ask interval objective: a model price incurs no quote-fit residual while it remains inside the observed spread. The expiry ATM variance term structure is checked against actual maturities before calibration.

These routines are not yet connected to a live quote feed. Synthetic calibration recovery is an implementation test, not empirical evidence of market fit.

## Black (1976) reference

Black, F. (1976), “The Pricing of Commodity Contracts,” *Journal of Financial Economics*, 3(1–2), 167–179. DOI: https://doi.org/10.1016/0304-405X(76)90024-6.
