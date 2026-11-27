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

For a trading desk, Basel MAR32.18 requires separate HPL and APL backtests at 97.5% and 99% over at least one year of current observations. The overall exception count is the greater of the HPL and APL counts. MAR32.18(2) counts an unavailable P&L or daily risk measure as an outlier.

This repository has HPL only. It therefore:
- Counts an HPL exception when the HPL loss exceeds one-day VaR.
- Counts a missing HPL or VaR observation as an outlier; it does not drop the date.
- Evaluates the latest 250 observations against the desk-level MAR32.19 limits: more than 12 exceptions at 99%, or more than 30 at 97.5%.
- Reports a below-limit HPL count as `below_hpl_limit_apl_unknown`, not a pass. Without APL, it cannot calculate the overall APL/HPL exception count or establish desk eligibility.
- Does not apply separate bank-wide MAR32.9 green/amber/red zones or bank-wide capital multipliers to a desk.

Kupiec and Christoffersen likelihood-ratio statistics are asymptotic diagnostics. A constant or transition-degenerate exception sequence is marked as not estimable. This is not a complete supervisory backtesting process.

## 3. P&L attribution

For the latest 250 aligned observations:
- The code requires all 250 HPL/RTPL pairs to be present and finite; it does not silently drop missing rows.
- Spearman rank correlation measures whether RTPL ranks days like HPL.
- KS distance is \(D=\sup_x|F_{HPL}(x)-F_{RTPL}(x)|\), the maximum absolute difference between empirical CDFs.

The implemented Basel-style PLA zones are:
- Green: Spearman > 0.80 and KS < 0.09.
- Red: Spearman < 0.70 or KS > 0.12.
- Amber: all intermediate cases.

The code reports an insufficient-sample state unless 250 paired observations exist. If Spearman is undefined (for example, because HPL or RTPL is constant), it returns `not_classifiable` and no PLA zone; it does not replace an undefined correlation with zero.

## 4. Instrument valuation and RTPL

- Equity European options: Black–Scholes calls and puts.
- FX European calls: Garman–Kohlhagen.
- Rates: zero-coupon Treasury-yield proxies, with yield inputs expressed in percent.
- Commodities: long WTI Cushing and short Europe Brent linear spot proxies. The historical risk factors are daily EIA spot benchmarks, not listed futures settlements. WTI/Brent relative-price risk is represented, but no listed-contract basis, futures curve, expiry or roll model is implemented.
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

The exploratory selector chooses a contiguous 250-observation window by maximizing 10-observation ES from available historical / synthetic factor P&L. This is not the Basel MAR33.5/33.7 stress-period procedure: the MAR33.5 FAQ1 says to maximize aggregate IMCC, while MAR33.7 requires a 12-month period from a history that reaches back to and includes 2007, with equally weighted observations and at least quarterly updates. A reduced factor set is selected by descending individual-factor ES contribution using the current 250-observation sample. The reported ratio over the latest 60 endpoints, each using a 250-observation window, reuses observations from the factor-selection sample. It is therefore only a descriptive, overlapping **in-sample** diagnostic. The 75% threshold is only a research reference here. Basel MAR33.5 applies it to the aggregate of all IMA-approved desks at group level, averaged over the preceding 12 weeks. A single synthetic book with overlapping observations is not that calculation.

The Basel stress calibration framework requires a sufficiently long historical stress data span (including the 2007 period). The synthetic reference sample begins in 2016; the public FRED SP500 daily series used here also does not supply a complete daily record back to 2007. Therefore the report explicitly marks stress-history sufficiency as **not met**.

## 7. IMCC and NMRF formula primitives

The IMCC helper encodes the aggregation expression with the fixed \(\\rho=0.5\) weight and computes class components from supplied full-current, reduced-current and reduced-stress ES values. The NMRF helper uses the fixed \(\\rho=0.6\) aggregation weight. These functions are **formula primitives**, not a capital process: they do not implement instrument-level liquidity-horizon mapping, class-specific stress calibration, the group-level reduced-factor test, IMCC-based stress-period selection, complete NMRF stress scenarios or supervisory controls.

The reference book has no credit-spread risk factors. The build function therefore suppresses the IMCC output instead of calculating a partial four-class number. A populated five-class input would still not make the overall calculation regulatory or empirically validated.

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


## Crossed calibration/scenario hedge-risk uncertainty decomposition

FE-010 creates a balanced matrix with risk-neutral bootstrap draws in rows and independent physical/scenario seeds in columns. Within each seed, Gaussian QE-M innovations are common across calibration draws; the P initial variance follows each draw's (v_0), and other P dynamics remain fixed. Across seeds, the scenario innovations are independent.

Two-way random-effects method-of-moments estimates split the variance of each net VaR/ES metric into a Q-calibration component, a physical-scenario sampling component and a residual combining parameter-by-scenario interaction with within-cell Monte Carlo error. The decomposition is produced for Heston, frozen-BS and paired Heston-minus-BS metrics. Raw negative variance-component estimates are retained for diagnosis and truncated estimates/proportions are reported separately. This is not a confidence interval; small B/R and finite tail depth make estimates noisy. See [HESTON_UNCERTAINTY.md](HESTON_UNCERTAINTY.md) for the equations and limitations.


## Crossed decomposition of calibration and scenario hedge-risk uncertainty

A balanced crossed design evaluates each risk-neutral Heston bootstrap draw under every independent physical/scenario seed. Within a seed, the same Gaussian innovations are reused across Q draws; the initial variance state follows the draw's (v_0), while the remaining P dynamics are fixed. Different seeds generate independent scenario innovation sets.

For each risk metric, the two-way random-effects method of moments uses row/column means and residual sums of squares to estimate the calibration component, scenario component and interaction/within-cell Monte Carlo residual. It is applied to Heston ES/VaR, frozen-BS ES/VaR and paired differences. Raw variance components are retained; negative calibration/scenario estimates are reported and separately truncated at zero for non-negative proportions. The residual is not a pure interaction term.

This is not REML, a Bayesian hierarchical model or a confidence interval. Small numbers of bootstrap draws/seeds and finite tail depth produce noisy estimates; model risk and physical parameter uncertainty remain outside the decomposition. See [HESTON_UNCERTAINTY.md](HESTON_UNCERTAINTY.md) and Formula Register §72 for formulas and limitations.


## Extreme-value tail modelling of hedge P&L

In addition to empirical inverse-CDF VaR and fractional-boundary ES, the hedge P&L engine now exposes a Peaks-over-Threshold (POT) Generalized Pareto model. For each empirical loss threshold \(u\), it fits a zero-location GPD to strict exceedances \(L-u\) by maximum likelihood. It estimates tail VaR and finite-mean ES at confidence levels above the threshold, reports shape/scale, exceedance probability/count, mean-excess diagnostics and a descriptive probability-integral-transform KS distance. A conditional excess bootstrap holds the threshold/exceedance count fixed; a second IID bootstrap resamples the full loss sample and recalculates the numerical threshold, exceedances, GPD fit and VaR/ES in each replicate. Neither chooses the threshold quantile automatically.

Thresholds are presented as a stability grid; no automatic choice is made. The usual KS p-value is omitted because parameters are estimated from the same exceedances. If fitted shape \(\xi\geq1\), the fitted GPD has no finite mean and ES is reported as undefined/infinite. The bootstrap is conditional on the observed threshold and exceedance count and does not cover threshold selection, serial clustering, structural breaks or model risk.

See [TAIL_RISK_EVT.md](TAIL_RISK_EVT.md) for the derivation and reproducible Heston hedge tail-risk benchmark.


## Regulatory boundary: RFET, liquidity horizons and APL

The RFET module calculates only a numerical evidence-count screen. A count does not establish that a row is a valid real-price observation or that the factor mapping is representative. Third-party observations require vendor audit information and documented counts/dates and identifier mapping. The output is therefore a candidate for independent review or not evidenced; it never issues a regulatory modellability pass.

The 10/20/40/60/120-day liquidity-horizon scaling shape is implemented as a research primitive. Production FRTB mapping is instrument/risk-factor-specific under MAR33; the generic factor labels and manually supplied horizon values in this project do not implement that mapping.

The Basel desk backtest is not complete without Actual P&L. Public daily closes and hypothetical HPL cannot stand in for the bank's APL. See [MATURITY_REGISTER.md](MATURITY_REGISTER.md).


## Desk-level backtesting interpretation

The desk-level numeric limits above are distinct from bank-wide exception zones and multipliers. MAR32.18 requires separate comparisons against APL and HPL, and MAR32.19 sets the desk limits. This repository only has HPL, so it reports an HPL-only threshold result; it never reports a passing/approved desk. Missing HPL or VaR observations are counted as outliers, consistent with MAR32.18(2).


## Standards applied and scope

- [Basel MAR32 — backtesting and PLA](https://www.bis.org/basel_framework/chapter/MAR/32.htm): desk-level backtesting, separate HPL/APL counts and missing-data outlier treatment.
- [Basel MAR33 — IMA capital calculation](https://www.bis.org/basel_framework/chapter/MAR/33.htm): ES liquidity horizons, reduced-factor requirement, stress-period selection, five broad risk classes, IMCC and NMRF.

The code applies selected formulas and control principles as research primitives. It does not implement all requirements, data governance, model validation or supervisory approvals across MAR31–MAR33.


## Standards applied and scope

- [Basel MAR32 — backtesting and PLA](https://www.bis.org/basel_framework/chapter/MAR/32.htm): desk-level HPL/APL backtesting, missing-data outlier treatment and PLA.
- [Basel MAR33 — IMA capital calculation](https://www.bis.org/basel_framework/chapter/MAR/33.htm): liquidity horizons, reduced-factor criterion, stress-period selection, broad risk classes, IMCC and NMRF.

The code implements selected mathematical controls and research primitives. It does not implement all requirements, data controls, model validation or supervisory approvals across MAR31–MAR33.
