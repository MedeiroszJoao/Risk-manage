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

The integral is evaluated by adaptive improper quadrature. The returned quadrature error estimate is a numerical integration diagnostic; it is not a confidence interval for fair value, a calibration error, or model risk. Put prices are obtained by put-call parity. Results are checked against discounted no-arbitrage bounds.

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
