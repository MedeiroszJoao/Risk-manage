# Heston calibration uncertainty: bootstrap, profile likelihood and risk propagation

## Purpose and scope

A calibrated Heston parameter vector is an estimate, not an observed truth. A small price RMSE or a full-rank local Jacobian does not by itself establish that the five parameters are known precisely. FE-008 makes the uncertainty assumptions explicit and carries a distribution of calibrated parameter vectors into European option prices and Deltas.

This module has three distinct outputs:

1. Parametric bootstrap of the option quote panel.
2. One-dimensional profile-likelihood curves with nuisance-parameter re-optimization.
3. Propagation of bootstrap parameter samples into price and Delta distributions.

All outputs are conditional on a fixed Heston model, fixed market conventions and an explicit quote-noise approximation. No model-risk interval, live-market estimate, or physical-measure forecast is claimed.

## Quote-noise model

For quote \(i\), let \(Bid_i\), \(Ask_i\), and \(Mid_i\) be observed. Define the half-spread scale

\[
h_i=\max\left(\frac{Ask_i-Bid_i}{2},\varepsilon_p\right),
\]

where \(\varepsilon_p\) is a positive price-scale floor. The standard deviation used in this experiment is

\[
s_i=\frac{h_i}{\sqrt{3}}
=\frac{\max(Ask_i-Bid_i,2\varepsilon_p)}{\sqrt{12}}.
\]

The \(1/\sqrt{12}\) relationship is the standard deviation of a uniform random variable over an interval whose width is the bid–ask spread. The implementation uses a **Gaussian approximation with this same variance**:

\[
Mid_i=C_i(\vartheta)+s_i\epsilon_i,\qquad
\epsilon_i\overset{iid}{\sim}N(0,1).
\]

This is a transparent quote-noise model, not a claim that option quote errors are actually independent Gaussian variables. It ignores quote dependence, stale markets, crossed markets, discrete ticks, liquidity/size effects, and common surface-construction error. The floor prevents near-zero spreads from dominating the fit.

The base calibrator uses price residuals scaled by a half-spread floor. Multiplying every standardized residual by the same constant does not change the least-squares optimum. For profile likelihood, the stated Gaussian standard deviations are used directly.

## Parametric quote bootstrap

Let \(\hat\vartheta\) be the fitted risk-neutral parameter vector. Reprice the fixed quote panel under that vector:

\[
\hat C_i=C_i(\hat\vartheta).
\]

For bootstrap replicate \(b=1,\ldots,B\), generate

\[
Mid_i^{*(b)}=\max\left(
0,\hat C_i+s_i\epsilon_i^{(b)}
\right),
\qquad
\epsilon_i^{(b)}\overset{iid}{\sim}N(0,1),
\]

then refit all five Heston parameters under the same bounds, quote weights and optimizer conventions. The original bid–ask widths are retained and recentered around the simulated mids. The resulting estimates

\[
\hat\vartheta^{*(1)},\ldots,\hat\vartheta^{*(B)}
\]

form the conditional parametric-bootstrap distribution.

The output contains per-parameter means, standard deviations, percentile summaries, covariance and correlation matrices in both natural and transformed coordinates, convergence rate, optimizer costs, failures, seed, and the assumed quote-noise scales.

Positive parameters are analyzed in log coordinates and correlation in Fisher-z coordinates:

\[
z=(\log v_0,\log\kappa,\log\theta,\log\xi,\operatorname{atanh}\rho).
\]

The natural-scale summaries are intuitive for parameter reporting. The transformed covariance is useful for understanding uncertainty relative to the constrained optimization geometry. Correlations between bootstrap estimates reveal parameter compensation that marginal standard errors alone conceal.

### What the bootstrap does not measure

- Model misspecification or uncertainty between Heston and alternative models.
- Dependence across quotes, expiries, strikes or market-data vendors.
- Quote filtering, asynchronous timestamps, stale/crossed markets or hidden liquidity.
- Yield-curve, dividend-curve, borrow, funding or forward uncertainty.
- Uncertainty in the assumed spread-to-noise mapping.
- Physical-measure dynamics or the variance risk premium.
- Optimizer failure modes beyond the reported convergence/failure diagnostics.

Bootstrap percentiles are conditional confidence summaries under the assumed noise model; they are not guaranteed frequentist coverage intervals under arbitrary quote microstructure.

## Profile likelihood with nuisance re-optimization

For a quote-error model with known scales \(s_i\), define the Gaussian least-squares statistic

\[
\chi^2(\vartheta)=
\sum_{i=1}^{n}
\left[
\frac{C_i(\vartheta)-Mid_i}{s_i/w_i}
\right]^2,
\]

where \(w_i\) is the quote's explicit relative weight.

To profile parameter \(\psi\), fix it at a value \(\psi_0\) and re-optimize the remaining four parameters:

\[
\chi^2_{prof}(\psi_0)
=
\min_{\vartheta_{-\psi}}
\chi^2(\psi_0,\vartheta_{-\psi}).
\]

The profile likelihood-ratio statistic is

\[
\Lambda(\psi_0)=
\chi^2_{prof}(\psi_0)-\min_{\psi}\chi^2_{prof}(\psi).
\]

For one profiled parameter, the asymptotic 95% cutoff is

\[
\Lambda(\psi_0)\leq\chi^2_{1,0.95}\approx3.841.
\]

The code returns the profile curve, fitted nuisance parameters at every grid point, the accepted grid points, and a boundary-truncation flag. When the accepted set is bracketed by rejected grid points, it uses Brent root-finding to solve the likelihood-ratio cutoff on each boundary, re-optimizing nuisance parameters at every root evaluation. The continuous root-refined interval is reported separately from the coarse on-grid envelope. An accepted endpoint means the interval may extend beyond the evaluated grid; widen the grid before reporting a closed interval.

The chi-square cutoff is an asymptotic approximation. The Heston calibration can be nonlinear, bounded and weakly identified. Bracketing grid resolution, constraints, optimizer convergence and the assumed quote-noise model affect the result. The root-refined endpoint is numerically more precise than a grid-only estimate, but it does not remove asymptotic or model assumptions. The curve must still be checked under a wider/finer grid and alternative quote weights.

## Propagating parameter uncertainty into prices and Deltas

For each bootstrap parameter draw \(\vartheta^{*(b)}\), compute the European price

\[
C^{*(b)}(K,T)=C(K,T;\vartheta^{*(b)}).
\]

The corresponding spot Delta is estimated by a central spot bump using the same parameter draw:

\[
\Delta^{*(b)}(K,T)
=
\frac{
C(S_0+h,K,T;\vartheta^{*(b)})
-
C(S_0-h,K,T;\vartheta^{*(b)})
}{2h}.
\]

The report gives means, standard deviations, minima/maxima and 2.5%, 5%, 50%, 95% and 97.5% quantiles for both price and Delta, along with their bootstrap correlation.

These are parameter-uncertainty distributions under a fixed pricing model. They do not include Fourier quadrature error, spot-bump truncation, market-convention uncertainty or model risk. The Delta is a numerical spot derivative; it is not a full implied-surface recalibration Greek.

## Reproducibility

The deterministic synthetic experiment creates 15 noisy European quotes across three maturities and five strikes, calibrates the five risk-neutral parameters, bootstraps the quote panel, profiles correlation and volatility of variance, and propagates the parameter draws into six option prices and Deltas.

Run:

\`\`\`bash
python scripts/run_heston_uncertainty.py --replicates 250 --profile-points 9
\`\`\`

The default analysis uses 250 bootstrap refits and nine profile-grid points; this is more informative than the CI smoke configuration. For tail-stable percentile intervals, increase the bootstrap count further (for example 500–1,000) and rerun with multiple quote-noise seeds. The JSON report contains the exact synthetic quote panel, generating and fitted parameters, base calibration diagnostics, bootstrap samples and summaries, profile curves, propagated price/Delta distributions, and all uncertainty assumptions. The quote panel is synthetic by construction; it is an implementation and methodology demonstration, not live market calibration.

## Primary references

- Suárez-Taboada, M., Witteveen, J. A. S., Grzelak, L. A. and Oosterlee, C. W. (2018), “Uncertainty quantification and Heston model,” *Journal of Mathematics in Industry*, 8, article 5. https://doi.org/10.1186/s13362-018-0047-2. Studies propagation of uncertainty in Heston parameters to model outputs.
- Pfante, O. and Bertschinger, N. (2017), “Uncertainty of Volatility Estimates from Heston Greeks,” *Frontiers in Applied Mathematics and Statistics*, 3, article 27. https://doi.org/10.3389/fams.2017.00027. Relates Heston price gradients/Greeks to Fisher information under an additive Gaussian quote-noise assumption.
- Cohen, S. N. and Tegnér, M. (2018), “European Option Pricing with Stochastic Volatility Models under Parameter Uncertainty,” arXiv:1807.03882. Develops Heston pricing bounds under parameter uncertainty and distinguishes risk-neutral calibration uncertainty from physical-measure estimation.
- Guillaume, F. M. Y. and Schoutens, W. (2010), “Model and Calibration Risks for the Heston Model,” Eurandom Report 2010026. Discusses calibration risk from initial guesses, local minima, objective functions and calibration methodology.


## Propagating calibration uncertainty into hedge P&L, VaR and ES

The next risk layer takes the bootstrap parameter draws beyond static prices/Greeks and into a discrete delta-hedging experiment. This is important because hedge-loss tails depend on the calibrated risk-neutral premium and hedge ratios, not only on the point-estimate calibration fit.

### Common physical scenario design

Generate one full QE-M physical/scenario path set

\[
\{(S_{t_j}^{(p)},v_{t_j}^{(p)})\}_{p=1,\ldots,N;\,j=0,\ldots,M}
\]

using a fixed physical/scenario drift \(\mu\) and a separately supplied physical variance parameter set. Reuse these exact paths for every bootstrap risk-neutral parameter draw \(b\). This is a common-random-numbers design across parameter samples: it removes path-regeneration noise from between-draw differences.

For each bootstrap draw \(\vartheta_Q^{*(b)}\):

1. Reprice the initial option premium \(C_0^{*(b)}\) under the risk-neutral Heston model.
2. At each hedge date, recalculate state-dependent Fourier Delta using that Q parameter draw and the same physical path state \((S_{t_j}^{(p)},v_{t_j}^{(p)})\).
3. Calculate the comparator frozen-volatility Black–Scholes Delta using \(\sigma_{frozen}^{*(b)}=\sqrt{\theta_Q^{*(b)}}\).
4. Execute both strategies on the same spot paths, with the same cash-account funding, trapezoidal dividend accrual, rebalance dates and proportional transaction costs.
5. Compute gross/net hedge P&L, turnover, costs, empirical VaR and fractional-boundary ES for each draw.

The across-bootstrap distributions of the hedge risk metrics

\[
VaR_\alpha^{*(b)},\qquad ES_\alpha^{*(b)}
\]

then describe how risk-neutral calibration uncertainty changes the hedge outcome **conditional on the selected physical scenario sample**.

### What the distribution means

The bootstrap dimension is uncertainty in the five Q pricing parameters under the quote-noise model described above. The inner path dimension is held fixed. Thus, dispersion across bootstrap draws is not contaminated by regenerating a different physical scenario sample for every parameter set.

This does **not** quantify physical-measure parameter uncertainty, scenario sampling uncertainty, or model risk. Within each bootstrap draw, VaR/ES still depend on a finite synthetic path sample; tail depth is \(N(1-\alpha)\) observations. Increase path count when a stable tail estimate is required. For a more complete nested uncertainty study, combine independent outer quote bootstraps with multiple inner physical scenario seeds and estimate variance components separately.

The experiment also does not claim that Heston Delta dominates frozen-volatility Black–Scholes. It reports the distribution of the difference in ES/VaR across parameter draws and the fraction of bootstrap draws in which one strategy has lower tail risk. A strategy can have lower ES but higher VaR, turnover or costs.

### Reproducibility

Run:

\`\`\`bash
python scripts/run_heston_hedge_uncertainty.py --bootstrap-replicates 30 --paths 1000 --steps 8 --confidence 0.95
\`\`\`

The JSON report records the synthetic quote panel, base calibration, bootstrap parameter samples, physical scenario assumptions, the common-path design, per-draw hedge metrics, and the across-draw VaR/ES distributions.

