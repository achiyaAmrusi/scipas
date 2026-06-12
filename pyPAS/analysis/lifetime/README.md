# Lifetime Analysis Module

Analysis of positron annihilation lifetime spectroscopy (PALS) spectra:
synthetic spectrum generation, discrete multi-exponential fitting, and
continuous lifetime-distribution inversion.

## The problem

A measured PALS spectrum is counts vs. time,

```
y(t) = N · ∫ R(t; τ) f(τ) dτ + bg,        R(t; τ) = IRF(t) ⊗ (1/τ)e^{-t/τ}Θ(t)
```

where `f(τ)` is the lifetime distribution, `R(t; τ)` is the detector response
to a single exponential component (the instrument response function convolved
with the decay), `N` the total counts, and `bg` a flat background. The module
answers the two standard questions:

1. **Discrete fitting** — the sample contains a few known components; find
   their lifetimes τ_i and intensities I_i (`fit/`).
2. **Inversion** — no parametric assumption; recover the continuous
   distribution f(τ) (`inversion/`). This is a Fredholm integral equation of
   the first kind (an exponential / Laplace-type inversion): it is ill-posed,
   so each method differs only in *how it regularizes*.

All times are in **nanoseconds**; intensities are dimensionless and sum to 1.

## Module map

```
analysis/lifetime/
├── generator.py            synthetic spectra (analytical + Poisson-sampled)
├── fit/
│   └── lifetime_fit.py     LifetimeFitter, FitParameter, FitResult
└── inversion/
    ├── base.py             LifetimeInvert — common interface
    ├── tikhonov.py         TikhonovRegularization
    ├── maximum_entropy.py  MaximalEntropyInversion (MELT, Bryan 1990)
    ├── gp_regression.py    GPRegression — Gaussian-process inversion
    └── utils.py            _response_matrix, _svd_truncate, t0_scan
```

## Generating spectra (`generator.py`)

```python
from pyPAS.analysis.lifetime import generate_analytical_lt_spectrum, generate_random_lt_spectrum
from pyPAS.model.lifetime import LifetimeModel
from pyPAS.core.time_resolution import MultiGaussianRF

model = LifetimeModel("sample", lifetimes=np.array([0.4, 2.0]), intensities=[0.7, 0.3])
irf = MultiGaussianRF(sigma=np.array([0.098]), intensities=np.array([1.0]), centers=np.array([0.0]))

analytical = generate_analytical_lt_spectrum(time, model, irf)        # noiseless, integral = 1
random     = generate_random_lt_spectrum(time, model, irf, num_events=1_000_000)  # Poisson
```

Both return a `PASLifetime` (spectrum + resolution function). `LifetimeModel`
also accepts a *distribution*: a fine τ grid with intensities per grid point.

## Discrete fitting (`fit/`)

For samples with a known number of components. Each parameter is a
`FitParameter(value, fixed=False, lower=-inf, upper=inf)`; fixing parameters
is also how source corrections are applied (fix the source τ and I).

```python
from pyPAS.analysis.lifetime import LifetimeFitter, FitParameter

fitter = LifetimeFitter()
result = fitter.fit(
    pals,
    lifetimes=[FitParameter(0.3), FitParameter(1.5)],
    intensities=[FitParameter(0.5), FitParameter(0.5)],
    t0=FitParameter(0.0),
    background=FitParameter(10.0, lower=0.0),
)
result.model        # fitted LifetimeModel
result.errors       # 1σ from the least-squares covariance
result.chi_squared  # reduced χ²
```

The model is `N·IRF ⊗ Σ_i (I_i/τ_i)e^{-(t-t0)/τ_i}` with the last non-fixed
intensity computed as `1 − Σ(others)` (PALSfit convention). The engine is
`scipy.optimize.least_squares`; errors come from the Jacobian at the optimum.

## Inversion (`inversion/`)

All inverters share the interface

```python
inv = SomeInversion(time_grid, characteristic_time_grid)   # τ grid to reconstruct on
...  = inv.invert(pals, bg_est=..., t0_shift=...)
```

The response matrix is built internally: column j is the IRF-convolved decay
for τ_j, evaluated on the (t0-shifted) time grid.

### Tikhonov (`TikhonovRegularization`)

Minimizes `‖R f − y‖² + α‖f‖²` subject to f ≥ 0 (NNLS), with the
regularization weight α chosen by the L-curve/χ² criterion. Fast and simple;
tends to over-smooth narrow features.

```python
f, opt = tik.invert(pals, bg_est=bg)        # f in counts/ns over the τ grid
```

### MELT (`MaximalEntropyInversion`)

Maximum-entropy inversion in the truncated-SVD subspace (Bryan 1990).
Maximizes `αS(f) − ½χ²` where S is the Shannon entropy relative to a flat
prior; α is selected by Bryan's criterion. Excellent peak localization;
provides no uncertainty estimate.

```python
alpha_opt, f = melt.invert(pals, bg_est=bg, noise_level=1e-3)
```

### Gaussian process (`GPRegression`)

Models the **log** of the distribution as a GP, which makes positivity
structural rather than a constraint:

```
g(τ) ~ GP(m, K),    K = exp(log_amplitude)·exp(−(Δ log τ)²/2ℓ²),    c = e^g
y(t) = N · (RM @ c) + noise,         noise_k ~ N(0, counts_k)
```

- The kernel lives on **log τ**, matching the natural resolution of
  exponential analysis (components are distinguishable by their *ratio*).
- `RM` is the column-sum-normalized response matrix, so `c` are probability
  weights and the free amplitude `N` carries the total counts — shape and
  scale are decoupled.
- The prior mean `m = log(1/n_τ)` pulls toward a flat distribution where the
  data carry no information.

**Posterior.** The MAP of `[log N, g]` is found by L-BFGS with analytic
gradients; the posterior is approximated by Laplace,
`cov(g) = (W + K⁻¹)⁻¹` with W the Gauss-Newton Hessian of the data term.
Pointwise uncertainty on f follows from the delta method, `σ_f = f·σ_g` —
the only method here with calibrated error bars.

**Hyperparameters.** ℓ (smoothness) and the kernel amplitude are selected by
maximizing the Laplace-approximated marginal likelihood (evidence),

```
−log Z ≈ ½χ² + ½(g−m)ᵀK⁻¹(g−m) + ½ log det(I + K·W)
```

The log-det term is the Occam penalty: flexibility is only awarded when the
data pay for it. The grid of (ℓ, amplitude) is traversed smooth→flexible with
warm starts. Because the evidence plateaus below the resolution limit of the
data, models within 1 nat of the best are considered ties and the smoothest
is chosen. **The effective resolution therefore adapts to the statistics**:
with more counts the evidence supports smaller ℓ and previously merged
components separate.

```python
gp = GPRegression(pals.lifetime.energy.values, tau_grid)
f, meta = gp.invert(pals, bg_est=bg)        # f is a density on the τ grid

meta['posterior_std']     # 1σ pointwise uncertainty on f
meta['length_scale']      # selected ℓ  (resolution the data support)
meta['evidence_grid']     # (ℓ, amplitude, −logZ) for every grid point
```

Fixed hyperparameters: `gp.invert(pals, bg_est=bg, optimize_hyperparams=False,
length_scale=0.3, log_amplitude=3.0)`.

### Choosing a method

| | Tikhonov | MELT | GP |
|---|---|---|---|
| Peak localization | fair | good | good |
| Narrow features | over-smoothed | good | good |
| Uncertainty estimate | – | – | ±σ (Laplace) |
| Resolution adapts to statistics | – | – | yes (evidence) |
| Cost | seconds | seconds | ~minute (hyperparameter grid) |

### t0 handling

All inverters accept `t0_shift`. To find the best time-zero, scan it:

```python
from pyPAS.analysis.lifetime.inversion.utils import t0_scan
result = t0_scan(inverter, pals, t0_values=np.linspace(-0.05, 0.05, 11), bg_est=bg)
result['best_t0'], result['best_result'], result['chi_squared']
```

In discrete fitting, t0 is simply a free `FitParameter`.

## Examples

See `examples/lifetime/`:
- `generate_lifetime_spectrum.ipynb` — synthetic spectrum generation
- `lifetime_inversion.ipynb` — Tikhonov and MELT inversion
- `gp_inversion.ipynb` — GP inversion: usage, evidence diagnostics, and the
  resolution-vs-statistics behaviour
