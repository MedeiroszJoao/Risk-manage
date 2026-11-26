# Methodology and formula map

## 1. Historical VaR and Expected Shortfall

For daily portfolio losses \(L_1,\ldots,L_n\), the empirical VaR uses the left-continuous inverse empirical CDF:

\[
VaR_c = F_n^{-1}(c).
\]

The implementation uses NumPy's \`inverted_cdf\` convention for both historical and rolling VaR. The finite-sample Expected Shortfall is the integrated empirical quantile:

\[
ES_c=\frac{1}{1-c}\int_c^1 VaR_u\,du.
\]

With \(q=VaR_c\) and \(m=n(1-c)\), the exact empirical formula is

\[
ES_c=
\frac{
\sum_{i:L_i>q}L_i+
\left[m-\#\{i:L_i>q\}\right]q
}{
m
}.
\]

The observations strictly above VaR receive full empirical mass. Only the fractional mass required at the VaR boundary is included. This avoids overweighting large groups of observations tied at VaR. If the tail probability cuts through a tied group, the required boundary mass is split fractionally; all tied observations are not averaged in full.

The code reports the inclusive number of observations at and above VaR for transparency, the number strictly above VaR, the number tied at VaR, the fractional boundary mass, and the total tail mass. Do not compare results across quantile conventions without identifying that choice.

Rolling one-day VaR at date \(t\) uses only the previous 250 daily observations. The forecast at \(t\) is not allowed to use HPL observed on day \(t\).

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
- KS distance is \(D=\sup_x|F_{HPL}(x)-F_{RTPL}(x)|\), the maximum absolute difference between empirical CDFs.

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
- Volatility exposure: a deliberately simplified variance-swap mark \(q\times(\sigma^2-K^2)\times\tau\). This proxy is used to test whether a model sees volatility risk; it is not a complete variance-swap fair-value model.

HPL fully revalues the constant-maturity hypothetical positions. The RTPL approximation includes first-order sensitivities; the expanded model additionally includes per-factor second-order curvature. Cross-gamma / vanna, volatility-surface dynamics and production curve conventions remain outside scope.

## 5. Liquidity-adjusted ES

Using a 10-day base horizon \(T\) and liquidity horizons 10, 20, 40, 60 and 120 days, the implemented structure is:

\[
ES_{LH}=\sqrt{ES_{10}(P)^2+
\sum_{j\ge2}
\left[ES_{10}(P_j)\sqrt{\frac{LH_j-LH_{j-1}}{10}}\right]^2}.
\]

\(P_j\) contains positions whose mapped liquidity horizon is at least \(LH_j\), while other factors are held constant. The current implementation uses historical 10-day overlapping factor P&L vectors and declared horizon mappings; it is an educational implementation of the scaling structure, not a production risk-factor mapping.

## 6. Stress and reduced-factor coverage

The engine selects the 250-day window with the highest 97.5% ES from available historical / synthetic factor P&L. A reduced factor set is selected in descending individual factor ES contribution. Its coverage is then checked at the latest 60 business-day endpoints, each using a 250-day estimation window; the average ratio \(ES_{reduced}/ES_{full}\) must be at least 75%.

The Basel stress calibration framework requires a sufficiently long historical stress data span (including the 2007 period). The synthetic reference sample begins in 2016; the public FRED SP500 daily series used here also does not supply a complete daily record back to 2007. Therefore the report explicitly marks stress-history sufficiency as **not met**.

## 7. Illustrative IMCC

The implementation follows the IMA aggregation shape with \(\rho=0.5\):

\[
IMCC=\rho\,IMCC(C)+(1-\rho)\sum_iIMCC(C_i)
\]

\[
IMCC(C)=ES_{R,S}\max\left(\frac{ES_{F,C}}{ES_{R,C}},1\right).
\]

Each partial risk class is calculated separately using its full-current, reduced-current and reduced-stress ES. All figures are research outputs with synthetic / public proxy inputs, not regulatory capital.

## 8. RFET and NMRF

The RFET diagnostic recognizes only explicitly flagged transaction, committed-quote or qualifying third-party quote records. Public closing-price rows are not treated as proof of those records. No modellability pass is issued when the evidence ledger is missing or contains only public closes.

The NMRF aggregation primitive is unit-tested separately. No NMRF charge is claimed from the reference case because no independently verified non-modellable risk-factor evidence / stress scenario file is available.


## Heston stochastic volatility and Monte Carlo valuation

The financial-engineering extension adds a risk-neutral Heston specification, its affine characteristic function, Lewis Fourier inversion for European options, and a separate QE-M Monte Carlo estimator. The model distinguishes the pricing measure from the physical-measure distribution used in VaR/ES forecasting. The Heston parameters supplied to this engine are risk-neutral pricing parameters; they must not be reused as real-world return-forecast parameters without an explicit risk-premium and measure-change model.

The Monte Carlo implementation matches the first two conditional moments of the CIR variance transition using the quadratic-exponential scheme. Its stock step applies the QE-M conditional exponential-moment correction to preserve the one-step risk-neutral stock martingale when the relevant moment exists. Antithetic variates are paired by path, and the discounted-stock control-variate coefficient is estimated on an independent pilot sample. Confidence intervals describe simulation sampling error conditional on the chosen time grid; they do not absorb QE discretization bias, parameter uncertainty, or model risk. Fourier quadrature error is reported separately.

Feller-condition status is diagnostic rather than a blanket parameter-rejection criterion. Numerical certification still requires positivity/no-arbitrage checks, characteristic-function identities, martingale tests, deterministic-variance recovery against Black–Scholes, Fourier/Monte Carlo comparison, and time-step convergence.

See [HESTON_MONTE_CARLO.md](HESTON_MONTE_CARLO.md) for the full equations, conventions, implementation details, tests, and limitations. The reproducible benchmark uses hypothetical parameters and is not empirical quote calibration.


## Risk-neutral Heston calibration

The Heston model's five pricing parameters are estimated from an option-price cross-section, while spot, discounting and carry conventions are treated as supplied market inputs. The optimizer uses logarithmic coordinates for positive parameters and Fisher-z coordinates for correlation, with explicit transformed bounds. It supports mid-price loss and signed distance outside bid–ask intervals, weights residuals by a half-spread floor, uses robust trust-region least squares and deterministic multi-start.

Quotes of a common maturity are priced together using vector-valued Lewis Fourier quadrature. The local weighted-residual Jacobian is decomposed by SVD; singular values, numerical rank and condition number flag weakly identified parameter combinations. This is not a confidence interval or proof of unique parameters. The calibration panel and its executable are synthetic, not actual option-market data.

See [Heston Fourier, QE-M and calibration](HESTON_MONTE_CARLO.md) for the full derivation, conventions and limitations.


## Monte Carlo sensitivity estimation under Heston

For European vanilla options, the QE-M engine now estimates pathwise Delta using the homogeneity relation \(\partial S_T/\partial S_0=S_T/S_0\). An independent Delta estimate and Gamma use central common-random-number bumps; initial-volatility Vega is \(\partial V/\partial\sqrt{v_0}\), not implied-volatility Vega. Antithetic path pairs are averaged before standard errors and Student-t intervals are calculated. Finite-difference truncation and QE-M time-grid bias are not included in those intervals.

The comparison runner reports the Monte Carlo estimators against Fourier finite-difference reference values. See [Heston Fourier, QE-M, calibration and Greeks](HESTON_MONTE_CARLO.md) for estimator derivations and interpretation limits.


The same coupled finite-difference stencil also reports Vanna \(\partial^2V/(\partial S_0\partial\sigma_0)\) and Volga \(\partial^2V/\partial\sigma_0^2\). These are Heston initial-volatility cross sensitivities, not implied-surface Vanna/Volga; bump truncation and time-grid bias remain outside their sampling intervals.


## Discrete Heston delta hedging under separate pricing and scenario measures

The new experiment values the short European option under risk-neutral Q parameters and simulates underlying paths under explicit physical/scenario total-return drift and variance parameters. On each hedge date, a vector-valued Lewis Fourier integral provides the Heston price and analytically differentiated spot Delta for every simulated spot/current-variance state. The benchmark strategy uses a Black–Scholes Delta with volatility frozen at the square root of Q long-run variance.

The self-financing hedge cash account accrues at the flat risk-free rate, receives trapezoidal dividend cashflows, pays for stock rebalancing and proportional transaction costs, and settles the option payoff at maturity. Gross/net P&L, turnover, transaction costs, empirical inverse-CDF VaR and integrated fractional-boundary ES are reported. The physical inputs are hypothetical scenario parameters. VaR/ES describe the finite simulated P&L sample and are not regulatory capital or a market forecast.

See [HESTON_MONTE_CARLO.md](HESTON_MONTE_CARLO.md) for the Fourier Delta derivation, cash-account equations, tail estimator and limitations.


## Randomized quasi-Monte Carlo for Heston valuation

A second simulation estimator uses scrambled Sobol nets of (2^m) points, inverse-normal mapping, Brownian-bridge path construction for both Gaussian drivers and the QE-M Heston transition. Discounted-stock control-variate beta is estimated on a separate scrambled pilot net. Integration error is estimated across independent randomizations of the Sobol net; Sobol payoff points are not treated as iid observations.

The benchmark compares Brownian-bridge RQMC, standard-order RQMC and antithetic pseudo-MC against the same Fourier price, with equal valuation and pilot path budgets. This is a path-count comparison, not a wall-clock benchmark. All standard errors are conditional on the QE time grid and exclude discretization bias, pilot-beta uncertainty, parameter uncertainty and model risk. The use of scrambled Sobol nets is an estimator option, not an assumption of superiority over pseudo-MC.

See HESTON_MONTE_CARLO.md for the bridge construction, randomized error estimator, fair comparison protocol and references.


## Heston calibration uncertainty

Heston calibration results are now accompanied by three uncertainty analyses: a parametric quote bootstrap using Gaussian quote noise with standard deviation derived from bid–ask width; one-dimensional profile likelihood that re-optimizes all nuisance parameters at each fixed parameter value; and propagation of bootstrap parameter vectors into European price and spot-Delta distributions. Covariance/correlation matrices are reported in natural and transformed coordinates. Profile intervals use an asymptotic chi-square(1) likelihood-ratio cutoff and are grid approximations.

This is conditional parameter uncertainty under an assumed quote-noise model and fixed Heston specification. It does not measure model risk, quote dependence, live market quality, curve uncertainty or physical-measure uncertainty. See [HESTON_UNCERTAINTY.md](HESTON_UNCERTAINTY.md) for the assumptions, equations and limitations.


## Calibration uncertainty propagated into hedge loss

The bootstrap parameter samples now flow through the Heston dynamic hedge. The same QE-M Gaussian innovations are reused for all risk-neutral bootstrap draws. The initial variance state (v_0) is aligned across Q and P for each draw, while all other P/scenario dynamics remain fixed. Thus path states differ through the initial variance, but shocks are common. For each Q draw, the experiment recalculates the initial Fourier premium, state-dependent Fourier Delta and frozen-volatility BS comparator, then executes both self-financing hedge ledgers with the same funding/dividend/transaction-cost assumptions. Each draw produces gross/net P&L, turnover, transaction costs, empirical VaR and fractional-boundary ES.

The distribution across bootstrap draws captures risk-neutral calibration uncertainty under common Gaussian innovations and fixed non-initial P dynamics; the initial P variance state follows each draw's (v_0). It does not include physical parameter uncertainty, scenario resampling uncertainty or model risk. Within-draw VaR/ES precision remains governed by the finite path count and tail depth. The comparison is not a claim of universal Heston hedge superiority.

See [HESTON_UNCERTAINTY.md](HESTON_UNCERTAINTY.md) for the nested design and interpretation limits.
