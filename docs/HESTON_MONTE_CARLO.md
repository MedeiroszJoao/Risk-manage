# Heston stochastic volatility: Fourier pricing and QE-M Monte Carlo

## Scope

This module adds a second, structurally different valuation route to the derivatives stack:

1. Heston's affine characteristic function and Lewis Fourier inversion provide a semi-analytic European-option benchmark.
2. An Andersen-style quadratic-exponential (QE) variance transition with the QE-M stock martingale correction provides a simulation route.
3. Antithetic variates and an independently estimated stock-price control variate reduce Monte Carlo variance; confidence intervals and martingale diagnostics expose numerical uncertainty.

This is not a full production Heston calibration or a complete volatility-surface engine. Parameters in the reproducible benchmark are hypothetical. The reported Monte Carlo interval is conditional on the time grid and does not include discretization bias or parameter uncertainty.

## Risk-neutral model

Under the pricing measure $Q$, with continuous risk-free rate $r$, dividend yield $q$, spot $S_t$, instantaneous variance $v_t$, and correlated Brownian motions:

\[
\frac{dS_t}{S_t}=(r-q)dt+\sqrt{v_t}\,dW_t^S,
\qquad
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}\,dW_t^v,
\]
\[
d\langle W^S,W^v\rangle_t=\rho\,dt.
\]

Parameters are annualized: $\kappa$ has units year$^{-1}$, $\theta$ and $v_0$ are annualized variance, and $\xi$ is volatility of variance with units year$^{-1/2}$. The correlation $\rho$ lies in $(-1,1)$.

The Feller condition $2\kappa\theta\geq\xi^2$ is reported as a diagnostic. It ensures the CIR variance process does not hit zero under the standard parameterization; its violation does not make the model invalid and is not silently used as a parameter rejection rule. The QE transition remains non-negative.

## Affine characteristic function

For $X_T=\log S_T$, let $\varphi_T(u)=E^Q[e^{iuX_T}]$. For $\xi>0$, define

\[
\beta(u)=\kappa-\rho\xi iu,
\qquad
d(u)=\sqrt{\beta(u)^2+\xi^2(u^2+iu)},
\qquad
g(u)=\frac{\beta(u)-d(u)}{\beta(u)+d(u)}.
\]

The implementation uses the principal complex square root and the stable $g$ representation:

\[
C(u)=iu[\log S_0+(r-q)T]
+\frac{\kappa\theta}{\xi^2}
\left[(\beta-d)T-2\log\left(\frac{1-ge^{-dT}}{1-g}\right)\right],
\]
\[
D(u)=\frac{\beta-d}{\xi^2}
\frac{1-e^{-dT}}{1-ge^{-dT}},
\qquad
\varphi_T(u)=e^{C(u)+D(u)v_0}.
\]

The formula is checked through $\varphi_T(0)=1$, conjugate symmetry, and the risk-neutral forward moment

\[
\varphi_T(-i)=E^Q[S_T]=S_0e^{(r-q)T}.
\]

When $\xi=0$, variance follows the deterministic mean-reverting path. Its integrated variance is

\[
I_T=\int_0^T v_s ds
=\theta T+(v_0-\theta)\frac{1-e^{-\kappa T}}{\kappa}.
\]

The characteristic function then reduces to the Gaussian lognormal expression with integrated variance $I_T$. This limiting case is tested against an independently evaluated Black–Scholes formula.

## Lewis square-root Fourier inversion

For log-return $Y_T=\log(S_T/S_0)$, the characteristic function is

\[
\varphi_Y(z)=\varphi_T(z)e^{-iz\log S_0}.
\]

The European call price is

\[
C_0=S_0e^{-qT}
-\frac{e^{-rT}\sqrt{S_0K}}{\pi}
\int_0^\infty
\operatorname{Re}\left[
\frac{e^{iu\log(S_0/K)}\varphi_Y(u-i/2)}{u^2+1/4}
\right]du.
\]

The distinction between log-price and log-return characteristic functions matters: using $\varphi_T$ without removing $\log S_0$ shifts the Fourier integrand and gives an incorrect price. The code performs this conversion explicitly.

The integral is evaluated by adaptive improper quadrature. The returned absolute quadrature error estimate is scaled by the Lewis price prefactor and reported in option-price units. It is a numerical integration diagnostic; it is not a confidence interval for fair value, a calibration error, or model risk. Put prices are obtained by put-call parity. Results are checked against discounted no-arbitrage bounds.

## QE variance transition

At a step of size $\Delta t$, the exact first two conditional moments of the CIR variance transition are

\[
m=\theta+(v_t-\theta)e^{-\kappa\Delta t},
\]
\[
s^2=
\frac{v_t\xi^2e^{-\kappa\Delta t}(1-e^{-\kappa\Delta t})}{\kappa}
+\frac{\theta\xi^2(1-e^{-\kappa\Delta t})^2}{2\kappa},
\qquad
\psi=\frac{s^2}{m^2}.
\]

QE matches these moments with one of two non-negative distributions, using threshold $\psi_c$:

**Low-$\psi$ quadratic branch.** Define

\[
b^2=\frac{2}{\psi}-1+
\sqrt{\frac{2}{\psi}}\sqrt{\frac{2}{\psi}-1},
\qquad a=\frac{m}{1+b^2}.
\]

For $Z\sim N(0,1)$, set $v_{t+\Delta t}=a(b+Z)^2$.

**High-$\psi$ zero-mass/exponential branch.** Define

\[
p=\frac{\psi-1}{\psi+1},
\qquad \beta=\frac{1-p}{m}.
\]

Set $v_{t+\Delta t}=0$ with probability $p$; otherwise draw an exponential random variable with rate $\beta$. The inverse transform used in the code is

\[
v_{t+\Delta t}=\frac{\log[(1-p)/(1-U)]}{\lambda_{QE}},
\qquad U\sim U(0,1),\quad U>p.
\]

Both branches reproduce $m$ and $s^2$ by construction. The high-$\psi$ inverse transform is tested separately because a sign error produces negative or invalid draws.

## QE-M stock step and martingale correction

Let $\gamma_1=\gamma_2=1/2$ and set

\[
k_1=\gamma_1\Delta t\left(\frac{\kappa\rho}{\xi}-\frac12\right)-\frac{\rho}{\xi},
\quad
k_2=\gamma_2\Delta t\left(\frac{\kappa\rho}{\xi}-\frac12\right)+\frac{\rho}{\xi},
\]
\[
k_3=\gamma_1\Delta t(1-\rho^2),
\qquad
k_4=\gamma_2\Delta t(1-\rho^2).
\]

Conditional on current and next variance, QE-M evolves log spot as

\[
\log S_{t+\Delta t}=\log S_t+(r-q)\Delta t
+k_0+k_1v_t+k_2v_{t+\Delta t}
+\sqrt{k_3v_t+k_4v_{t+\Delta t}}\,Z_S,
\]

where $Z_S$ is independent standard normal and

\[
k_0=-\left(k_1+\frac{k_3}{2}\right)v_t
-\log E\left[e^{(k_2+k_4/2)v_{t+\Delta t}}\mid v_t\right].
\]

The conditional moment-generating function is evaluated analytically under the selected QE branch. Consequently,

\[
E[S_{t+\Delta t}\mid\mathcal F_t]
=S_te^{(r-q)\Delta t},
\]

provided the required conditional exponential moment exists. If it does not exist for the selected step size and parameter set, the routine raises an error instead of silently returning an unstable path. Decreasing $\Delta t$ is then required.

## Monte Carlo estimator and error accounting

For a discounted European payoff $Y=e^{-rT}H(S_T)$, the plain estimator is $\bar Y$. The control variate is discounted terminal stock $X=e^{-rT}S_T$, whose risk-neutral mean is known:

\[
E^Q[X]=S_0e^{-qT}.
\]

With pilot-estimated coefficient $\hat\beta$, the adjusted observations are

\[
Y_i^{CV}=Y_i-\hat\beta(X_i-S_0e^{-qT}),
\qquad
\widehat V=\frac1N\sum_{i=1}^N Y_i^{CV}.
\]

The coefficient is estimated using an independent pilot simulation, not the pricing sample. When antithetic sampling is enabled, each $Z$ path and its $-Z$ partner are averaged; these pair averages are the independent observations used for the standard error. A Student-$t$ interval is reported from their sample variance.

Keep these uncertainty sources separate:

- **Monte Carlo sampling error:** estimated by the standard error and confidence interval.
- **Time-discretization bias:** not included in that interval; study it by refining $\Delta t$ or compare against exact simulation.
- **Fourier quadrature error:** returned separately by adaptive integration.
- **Parameter/calibration uncertainty:** not included; this requires a calibrated quote panel and parameter uncertainty analysis.
- **Model risk:** not captured by numerical convergence or a narrow confidence interval.

Variance-reduction factor is reported as $(SE_{plain}/SE_{CV})^2$. It is an observed diagnostic for this simulation, not a guaranteed property for every payoff or parameter set.

## Validation implemented

- Characteristic-function normalization, conjugate symmetry and the risk-neutral forward moment.
- Deterministic-variance limit versus an independent Black–Scholes calculation.
- Put-call parity and discounted no-arbitrage bounds.
- QE-M Monte Carlo versus Fourier pricing with a stated allowance for time-discretization bias.
- Discounted-stock martingale diagnostic.
- Explicit high-$\psi$ branch coverage and Feller-condition reporting.
- Invalid inputs and incompatible antithetic path counts are rejected.

## Limits and next validation steps

- The implementation prices European options; it is not yet an American-option, barrier, callable, or path-dependent exotic engine.
- The Heston parameters are supplied, not calibrated to real bid/ask option quotes.
- No stochastic-rate, stochastic-dividend, jumps, local volatility, or rough-volatility extension is included.
- QE-M is an approximation to the variance transition. The reported simulation interval does not absorb its weak discretization bias.
- A production-grade follow-up should add quote governance, constrained calibration, parameter uncertainty, strike/expiry surface residuals, convergence tables over time steps, and an independent exact-simulation benchmark (Broadie–Kaya).

## Primary references

- Heston, S. L. (1993), “A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options,” *Review of Financial Studies*, 6(2), 327–343. https://doi.org/10.1093/rfs/6.2.327
- Andersen, L. (2008), “Simple and Efficient Simulation of the Heston Stochastic Volatility Model,” *Journal of Computational Finance*, 11(3). https://doi.org/10.21314/jcf.2008.189
- Broadie, M. and Kaya, Ö. (2006), “Exact Simulation of Stochastic Volatility and Other Affine Jump Diffusion Processes,” *Operations Research*, 54(2), 217–231. https://doi.org/10.1287/opre.1050.0247


## Cross-sectional Heston calibration to option quotes

### What is estimated

The Heston pricing parameters are risk-neutral quantities. For a cross-sectional option panel, hold the market conventions fixed and estimate

\[
\vartheta=(v_0,\kappa,\theta,\xi,\rho),
\]

where $v_0$ is initial variance, $\kappa$ mean reversion, $\theta$ long-run variance, $\xi$ volatility of variance, and $\rho$ spot/variance Brownian correlation. This is a pricing-measure calibration. It is not a physical-measure GARCH/volatility forecast and does not, by itself, estimate the variance risk premium.

The current implementation treats spot, flat continuously compounded risk-free rate and flat dividend yield as supplied market inputs. A production-quality calibration must replace those flat inputs with timestamp-aligned forwards and discount/dividend curves, and must apply instrument calendars, exercise conventions and corporate-action rules consistently.

### Parameter transformations and constraints

Optimize in unconstrained coordinates

\[
z=(\log v_0,\log\kappa,\log\theta,\log\xi,\operatorname{atanh}\rho),
\]

then map back by exponentials and $\rho=\tanh(z_5)$. Box bounds in transformed coordinates imply explicit physical bounds on each parameter. This avoids negative variance, negative mean reversion, negative vol-of-variance and invalid correlation during optimization. The Feller condition is reported as a model diagnostic; it is not imposed as an arbitrary calibration constraint.

### Quote objectives

For mid-price fitting, quote $i$ contributes

\[
e_i(\vartheta)=C_i^{model}(\vartheta)-C_i^{mid}.
\]

For bid–ask fitting, the residual is the signed distance to the quoted interval:

\[
e_i^{BA}(\vartheta)=
\begin{cases}
C_i^{model}-Bid_i,&C_i^{model}<Bid_i,\\
0,&Bid_i\le C_i^{model}\le Ask_i,\\
C_i^{model}-Ask_i,&C_i^{model}>Ask_i.
\end{cases}
\]

Weighted residuals divide by $\max((Ask_i-Bid_i)/2,\varepsilon)$ and multiply by the quote's explicit user weight. The floor $\varepsilon$ prevents zero-width or very tight spreads from producing unbounded weights. SciPy trust-region reflective least squares is used with a selectable robust loss (default soft-$L_1$) and a deterministic multi-start design.

Quotes are grouped by maturity. A vector-valued Lewis integration prices all strikes for one maturity with a single characteristic-function evaluation per quadrature node. This cuts repeated work during finite-difference optimization while preserving one numerical pricing route.

### Identifiability and numerical diagnostics

The calibrated weighted residual Jacobian $J=\partial e/\partial z$ is decomposed as

\[
J=U\Sigma V^\top,\qquad
\Sigma=\operatorname{diag}(s_1,\ldots,s_5).
\]

The singular values, numerical rank and condition number are reported in transformed coordinates. A small singular value indicates a locally weakly identified parameter combination. A good price fit does not imply unique or stable parameters: the Heston objective can contain long, flat valleys where parameters compensate for one another. Do not convert this Jacobian SVD into a parameter confidence interval without a justified statistical model. Profile likelihood, quote/bootstrap resampling, and parameter stability across quote subsets are separate tasks.

The implementation currently uses central finite-difference Jacobians. This is transparent but expensive and sensitive to calibration tolerances. Analytic or semi-analytic gradients, parameter scaling studies, profile likelihood and independent calibration-library comparisons remain future upgrades. Cui, del Baño Rollin and Germano (2017) analyze the Heston calibration objective's flat valleys, bid–ask weighting and the value of analytic gradients.

### Reproducible synthetic calibration

Run:

```bash
python scripts/run_heston_calibration.py --starts 3 --max-nfev 60
```

The runner generates a deterministic synthetic panel with 15 calls/puts spanning three expiries and five strikes, then attempts to recover the Heston parameters from a perturbed starting point. The JSON report records true generating parameters, estimates, price errors, bid–ask hit ratio, optimizer starts and Jacobian diagnostics. Synthetic recovery is a code test, not evidence of real-market fit or stable identification.

## Additional calibration limitations

- No live or licensed exchange/OTC option quote set has been calibrated.
- Quotes are assumed European and clean; quote timestamps, sizes, stale/crossed-market filters and contract calendars are not yet governed.
- Discounting and carry are flat constants, not bootstrapped curves.
- The residual Jacobian uses finite differences; analytic gradients and independent optimizer/library benchmarks remain open.
- Parameter uncertainty is not estimated. Local SVD is only a warning signal about identifiability.
- No dynamic parameter evolution, variance risk premium estimation or physical-measure transition is calibrated.



## A caution from calibration geometry: bid–ask fit is not parameter identification

An interval objective deliberately assigns zero residual while every model quote lies within its market spread. This is useful when the market does not support a meaningful ranking of prices inside a spread, but it creates a flat region in parameter space. If the entire synthetic panel is inside its spreads, the local interval-loss Jacobian can have rank zero even though the price fit is perfect.

Therefore the reproducible recovery runner defaults to weighted mid-price fitting so that its local Jacobian contains information about all five parameters. Bid–ask fitting remains available as a complementary quote-consistency objective. Its hit ratio must be read alongside Jacobian rank and singular values; a 100% hit ratio alone is not evidence of parameter identification.

This distinction is consistent with the Heston calibration literature: the inverse problem can have flat valleys and parameter compensation, so objective choice, spread width, weighting and conditioning matter. See Cui, del Baño Rollin and Germano (2017), DOI https://doi.org/10.1016/j.ejor.2017.05.018.


## Monte Carlo Greeks under Heston

The Heston simulation now reports five sensitivities using distinct estimators. They must not be conflated:

- **Delta, pathwise:** a direct payoff derivative.
- **Delta and Gamma, CRN finite differences:** central bump-and-revalue using common random numbers.
- **Vega, CRN finite difference:** sensitivity to the initial Heston volatility \(\sigma_0=\sqrt{v_0}\), not to a Black–Scholes implied-volatility surface quote.

### Pathwise Delta

For a European call,

\[
V(S_0)=e^{-rT}E^Q[(S_T-K)^+].
\]

In the Heston model the variance process does not depend on the initial spot. Conditional on the same variance/ Brownian path, the stock terminal value scales linearly with initial spot, so

\[
\frac{\partial S_T}{\partial S_0}=\frac{S_T}{S_0}.
\]

For a call, the pathwise estimator is therefore

\[
\widehat\Delta_{PW}
=\frac{e^{-rT}}{N}\sum_{i=1}^N
\mathbf 1_{\{S_T^{(i)}>K\}}\frac{S_T^{(i)}}{S_0}.
\]

For a put, replace the indicator with \(-\mathbf 1_{\{S_T<K\}}\). This derivative is valid for the European vanilla payoff because its kink is a probability-zero event under a continuous terminal distribution; the same argument does not automatically extend to every discontinuous/path-dependent payoff.

### Common-random-number Delta and Gamma

For bumped initial spots, the same generated path multipliers are reused:

\[
\widehat\Delta_{CRN}
=\frac{\hat V(S_0+h;Z)-\hat V(S_0-h;Z)}{2h},
\]
\[
\widehat\Gamma_{CRN}
=\frac{\hat V(S_0+h;Z)-2\hat V(S_0;Z)+\hat V(S_0-h;Z)}{h^2}.
\]

The common path set induces positive covariance between nearby valuations in many settings and can reduce the variance of the difference. The benefit is measured by the resulting standard error; it is not assumed to occur automatically. Gamma uses finite differences because the payoff's second pathwise derivative is distributional at the strike and is not an ordinary pathwise estimator.

### Initial-volatility Vega

Let \(\sigma_0=\sqrt{v_0}\). The code perturbs the initial volatility while holding \(\kappa,\theta,\xi,\rho\) and market inputs fixed, and reuses identical Gaussian draws:

\[
\widehat{\mathcal V}_{\sigma_0,CRN}
=\frac{\hat V((\sigma_0+\varepsilon)^2;Z)
-\hat V((\sigma_0-\varepsilon)^2;Z)}{2\varepsilon}.
\]

This is \(\partial V/\partial\sigma_0\), a model-parameter sensitivity. It is **not** the usual implied-volatility Vega \(\partial V/\partial\sigma_{BS}\), nor a parallel shift of an SSVI surface. The parameter convention is explicitly included in every report.

### Cross-Greeks: Vanna and Volga

The same coupled spot/initial-volatility simulations also estimate

\[
\operatorname{Vanna}=\frac{\partial^2 V}{\partial S_0\,\partial\sigma_0},
\qquad
\operatorname{Volga}=\frac{\partial^2 V}{\partial\sigma_0^2}.
\]

With spot bump h and initial-volatility bump epsilon, the mixed estimator is

\[
\widehat{\operatorname{Vanna}}=
\frac{
V(S_0+h,\sigma_0+\epsilon)-V(S_0+h,\sigma_0-\epsilon)
-V(S_0-h,\sigma_0+\epsilon)+V(S_0-h,\sigma_0-\epsilon)
}{4h\epsilon},
\]

and Volga is

\[
\widehat{\operatorname{Volga}}=
\frac{V(S_0,\sigma_0+\epsilon)-2V(S_0,\sigma_0)
+V(S_0,\sigma_0-\epsilon)}{\epsilon^2}.
\]

Both use common random numbers across the full bump stencil. They are numerical sensitivities to the initial Heston volatility parameter; they are not the cross-Greeks of a calibrated implied-volatility surface. The estimates have finite-difference truncation bias and may be noisy when the chosen bumps are too small. Report bump stability, sampling errors and time-grid sensitivity before using them in a P&L approximation.

### Sampling uncertainty and limits

Antithetic paths are pair-averaged before the standard error is computed. Each reported Greek includes its estimate, standard error and Student-t confidence interval over the independent pair averages. The interval is conditional on the QE-M discretization and the chosen finite-difference bump. It does not include time-discretization bias, finite-difference truncation bias, parameter uncertainty or model risk.

The executable comparison uses Fourier prices as a deterministic finite-difference reference for the same model parameters. These are numerical reference Greeks, not closed-form derivative formulae. The benchmark reports discrepancies in units of the Monte Carlo standard error, but it does not convert a noisy agreement into a proof of unbiasedness or convergence.

Run:

\`\`\`bash
python scripts/run_heston_greeks.py --paths 50000 --steps 96
\`\`\`

The JSON report includes pathwise Delta, CRN Delta, Gamma, initial-volatility Vega, Vanna and Volga, sampling errors, bumps and Fourier finite-difference reference values.

### Primary reference

Broadie, M. and Glasserman, P. (1996), “Estimating Security Price Derivatives Using Simulation,” *Management Science*, 42(2), 269–285. https://doi.org/10.1287/mnsc.42.2.269. Their pathwise and likelihood-ratio estimators provide the classical foundation for simulation Greeks, including stochastic-volatility settings. The current module implements the pathwise Delta and CRN finite-difference Gamma/Vega slice; likelihood-ratio and adjoint estimators remain future work.
