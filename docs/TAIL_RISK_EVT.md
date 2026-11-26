# Extreme-value tail risk for Heston hedge losses

## Purpose

Empirical Expected Shortfall is limited by the number of tail observations. The Peaks-over-Threshold (POT) approach fits a Generalized Pareto Distribution (GPD) to losses above a high threshold and extrapolates beyond the observed sample. This is useful as a model comparison, not as a replacement for empirical VaR/ES or a guarantee of accurate extreme-tail capital.

The implementation applies POT-GPD separately to:
- Heston state-dependent Delta hedge net losses.
- Heston state-dependent Delta hedge gross losses.
- Frozen-volatility Black–Scholes hedge net losses.
- Frozen-volatility Black–Scholes hedge gross losses.

It reports empirical VaR/ES next to the GPD tail extrapolation at 97.5% and 99%, across several thresholds.

## Peaks over Threshold

Let \(L_1,\ldots,L_N\) be losses, with larger positive values denoting worse outcomes. Choose a threshold \(u\) and retain exceedances

\[
Y_i=L_i-u,\qquad L_i>u.
\]

For a sufficiently high threshold, the Pickands–Balkema–de Haan limit motivates a GPD approximation to the conditional excess distribution:

\[
P(L-u\le y\mid L>u)
\approx
G_{\xi,\beta}(y)
=1-\left(1+\xi\frac{y}{\beta}\right)^{-1/\xi},
\]

where \(\beta>0\), and the support satisfies \(y\ge0\) and \(1+\xi y/\beta>0\). At \(\xi=0\), use the exponential limit

\[
G_{0,\beta}(y)=1-e^{-y/\beta}.
\]

The shape parameter \(\xi\) controls tail heaviness:
- \(\xi>0\): Pareto-type heavy tail.
- \(\xi=0\): exponential tail.
- \(\xi<0\): finite upper endpoint.

The implementation estimates \(\xi,\beta\) by maximum likelihood on excesses with the GPD location fixed at zero. The empirical threshold is an inverse-CDF quantile and exceedances are strictly greater than the threshold. This convention determines the empirical exceedance probability

\[
\hat p_u=\frac{N_u}{N}.
\]

## Extrapolated VaR

For target confidence \(\alpha\) above the selected threshold quantile, solve

\[
P(L>VaR_\alpha)=1-\alpha
\]

using the fitted GPD survival function:

\[
\boxed{
VaR_\alpha^{POT}
=u+\frac{\beta}{\xi}
\left[
\left(\frac{\hat p_u}{1-\alpha}\right)^\xi-1
\right]
}
\]

for \(\xi\ne0\). At \(\xi=0\),

\[
VaR_\alpha^{POT}
=u+\beta\log\left(\frac{\hat p_u}{1-\alpha}\right).
\]

The extrapolation ratio \(\hat p_u/(1-\alpha)\) is reported. Values close to one mean limited extrapolation; a large ratio means the target lies much farther beyond the threshold and is more sensitive to the fitted tail shape.

## Expected Shortfall

For \(\xi<1\), the GPD conditional mean excess exists. Expected Shortfall is

\[
\boxed{
ES_\alpha^{POT}
=VaR_\alpha^{POT}
+\frac{\beta+\xi(VaR_\alpha^{POT}-u)}{1-\xi}
}
\]

under the fitted GPD tail. If \(\xi\ge1\), the fitted tail has infinite mean and POT Expected Shortfall is reported as undefined/infinite under that fitted model. If \(\xi\ge 1/2\), the fitted excess distribution also has infinite variance. These are explicit diagnostics, not implementation errors.

## Threshold stability and diagnostics

No single threshold is automatically selected. The runner reports, for each threshold quantile:
- Threshold value and number of strict exceedances.
- Tail probability \(N_u/N\).
- MLE shape and scale.
- Descriptive Kolmogorov–Smirnov distance after the fitted probability-integral transform.
- Mean excess empirical versus fitted.
- Finite mean/variance conditions.
- POT VaR/ES and extrapolation ratio at each target confidence level.
- Conditional non-parametric bootstrap percentile intervals.

The KS distance is descriptive only. No p-value is reported because the GPD parameters were estimated on the same exceedance observations; the standard uniform KS null distribution is not valid without parameter-estimation adjustment.

Threshold stability matters more than a good fit at one chosen quantile. Shape estimates, scale estimates, VaR and ES should remain reasonably stable over a defensible range of high thresholds. The mean-excess plot should be approximately linear above a candidate threshold. If the estimates change materially across thresholds, the GPD extrapolation is not stable enough for an operational conclusion.

At least the configured minimum number of exceedances is required for an MLE fit. Fewer exceedances produce an explicit not-fitted status instead of a fragile estimate.

## Conditional exceedance bootstrap

For each fitted threshold, the code non-parametrically resamples the observed excesses with replacement, refits the zero-location GPD, and recomputes VaR/ES. Percentile intervals are conditional on:
- The observed threshold \(u\).
- The observed exceedance count \(N_u\).
- The selected GPD family.
- Independent resampling of excesses.

They do not include threshold-selection uncertainty, serial dependence, block-bootstrap uncertainty, structural breaks, or uncertainty in the GPD family itself. A block bootstrap or declustering is needed if losses are serially dependent or exceedances cluster.

## Empirical comparator and interpretation

The report retains empirical inverse-CDF VaR and integrated fractional-boundary ES for exactly the same loss samples. This prevents the EVT result from being reported without a non-parametric reference.

POT-GPD estimates are conditional tail-model extrapolations. A narrow bootstrap interval does not imply low model risk. Threshold choice, finite samples, dependence, structural breaks, regime changes and the assumed distribution family can dominate nominal parameter uncertainty.

The synthetic hedging scenario uses hypothetical physical drift/variance parameters. No live-market backtest, regulatory capital result or forecast is claimed.

## Reproducibility

\`\`\`bash
python scripts/run_heston_hedge_evt.py --paths 10000 --steps 12 --cost-bps 2 --confidence-levels 0.975 0.99 --thresholds 0.90 0.925 0.95 --min-exceedances 50 --bootstrap 200
\`\`\`

The output JSON includes the hedge experiment, raw pathwise loss samples summarized by empirical VaR/ES, and POT-GPD fits/bootstraps across thresholds for gross/net Heston and frozen-BS losses.

## Primary references

- Pickands, J. (1975), “Statistical Inference Using Extreme Order Statistics,” *The Annals of Statistics*, 3(1), 119–131. https://doi.org/10.1214/aos/1176343003
- Balkema, A. A. and de Haan, L. (1974), “Residual Life Time at Great Age,” *The Annals of Probability*, 2(5), 792–804. https://doi.org/10.1214/aop/1176996548
- McNeil, A. J. and Frey, R. (2000), “Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series: an Extreme Value Approach,” *Journal of Empirical Finance*, 7(3–4), 271–300. https://doi.org/10.1016/S0927-5398(00)00012-8
- McNeil, A. J., Frey, R. and Embrechts, P. (2015), *Quantitative Risk Management: Concepts, Techniques and Tools*, revised edition, Princeton University Press.
