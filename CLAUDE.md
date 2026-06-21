# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**PyPAS** is a Python library for Positron Annihilation Spectroscopy (PAS) analysis. It covers Doppler broadening (DB) and coincidence Doppler broadening (CDB) spectrum analysis, positron implantation profiling, finite-difference transport simulation, and variable-energy Doppler broadening (VEDB) diffusion-length fitting.

**Project name:** PyPAS  
**PyPI / import name:** `pypas` (the package directory is `pypas/`)  
**GitHub:** `achiyaAmrusi/pypas`  
**Author:** Achiya Yosef Amrusi  
**Email:** ahia.amrosi@mail.huji.ac.il (Hebrew University of Jerusalem)  
**License:** MIT  
**Python:** 3.11+ (SPEC 0 policy; tested through 3.13)  
**Build:** `setuptools.build_meta` via `pyproject.toml` (no `setup.py`)  
**Status:** Active development. DB/CDB analysis, transport, and VEDB fitting are stable. Lifetime analysis lives on the `lifetime` branch only.

---

## Install and Develop

```bash
pip install -e "/home/owner/gitProjects/pyPAS[dev]"
```

The companion spectrum library **SciSpectrum** (`scispectrum>=0.3` on PyPI) is listed in `pyproject.toml` dependencies and installed automatically.

---

## Running Tests

```bash
cd /home/owner/gitProjects/pyPAS
pytest tests/
```

All test files carry the `test_` prefix and are discovered by pytest. The full suite (105 tests on main, 144 on lifetime) passes on Python 3.11–3.13. `tests/transport/diffusion/scipy_positron_profile_solver.py` is a reference/validation helper, not a test file.

CI: `.github/workflows/tests.yml` runs pytest on Python 3.11, 3.12, 3.13 on every push and PR.

---

## Architecture

### Full PAS workflow

```
Raw detector files (list-mode .txt)
  └─ scispectrum TimeChannelParser → Spectrum

DB pipeline:
  Spectrum
    └─ DB.from_spectrum()  → DB (Domain slice around 511 keV peak)
         ├─ .s_parameter_calculation() → ufloat
         └─ .w_parameter_calculation() → ufloat

CDB pipeline:
  PasCoincidenceFilter.time_coincidence_filter()   → coincident pairs
  PasCoincidenceFilter.energy_coincidence_filter() → energy-validated pairs
  CDB(pairs, energy_min, energy_max, mesh_interval)
    ├─ .doppler_broadening() → DB
    └─ .resolution()         → Domain

Transport + VEDB fitting:
  makhov_profile / ghosh_profile  (one per beam energy) [nm-depth, positrons/nm]
    └─ multilayer_implantation_profile → list of xr.DataArray
         └─ profile_solver(profile, sample, electric_field) → annihilation profile [positrons/nm]
              └─ compute_annihilation_fractions(profile, sample) → fractions per layer
                   └─ DiffusionLengthOptimization(profiles, s_measurement, initial_sample)
                        └─ .optimize_diffusion_length() → (best_fit [nm], covariance)
```

### Module map

```
pypas/
├── __init__.py              public API exports
├── core/
│   ├── db.py               DB — extends Domain; S/W parameter extraction
│   ├── cdb.py              CDB — 2D coincidence histogram; DB/resolution projections
│   ├── lifetime.py         PASLifetime (LIFETIME BRANCH ONLY)
│   ├── time_resolution.py  TimeResolution, MeasuredRF, MultiGaussianRF (LIFETIME BRANCH ONLY)
│   └── const.py            ELECTRON_REST_MASS_KEV (computed from scipy.constants)
├── filter/
│   ├── pas_coincidence.py  PasCoincidenceFilter — time + energy coincidence for CDB
│   └── pals_coincidence.py PALSCoincidenceFilter — STUB
├── libs/
│   └── positron_profile/   Ghosh & Makhov parameter tables (.txt CSV files)
├── transport/
│   ├── implantation/
│   │   ├── profiles.py           makhov_profile, ghosh_profile
│   │   ├── multilayer.py         multilayer_implantation_profile
│   │   └── material_parameters.py loads tables via importlib.resources from pypas.libs.positron_profile
│   └── diffusion/
│       └── positron_profile_solver.py  profile_solver — 1D FD diffusion-drift-annihilation solver
├── model/
│   ├── material.py         Material dataclass (diffusion, mobility, bulk_annihilation_rate, defects)
│   ├── layer.py            Layer dataclass (material, start, width)
│   ├── sample.py           Sample dataclass (layers, absorption_length)
│   └── lifetime.py         LifetimeModel dataclass (lifetimes, intensities)
└── analysis/
    ├── vedb/
    │   ├── annihilation_fractions.py  compute_annihilation_fractions
    │   ├── diffusion_length.py        DiffusionLengthOptimization
    │   ├── lineshape.py               compute_s_lineshape, compute_w_lineshape
    │   └── ve_implanation.py          variable_energy_implantation_profiles
    └── lifetime/                      LIFETIME BRANCH ONLY — not in main
        ├── generator.py
        ├── fit/  (LifetimeFitter, FitParameter, FitResult)
        └── inversion/  (TikhonovRegularization, MaximalEntropyInversion, GPRegression)
```

---

## Critical Dependency — SciSpectrum

SciSpectrum (on PyPI as `scispectrum`, source at `/home/owner/gitProjects/scispectrum`) provides the foundational types:

| Type | Role in PyPAS |
|---|---|
| `Spectrum` | Universal 1D count array with calibrated axis and Poisson errors |
| `Domain` | Contiguous slice of a `Spectrum` — **`DB` inherits from it** |
| `AxisCalibration` | `channel → energy` callable |
| `ResolutionCalibration` | Models detector FWHM vs energy; required by `SNRFinder` |
| `SNRFinder`, `Convolution`, `gaussian_2_dev` | Automatic 511 keV peak detection inside `DB.from_spectrum` |
| `center_estimator`, `sum_under` | Peak centroid finding and windowed integration for S/W |

Import with `from scispectrum import ...` (lowercase package name).

Key SciSpectrum invariants:
- `Domain` uses lazy background subtraction — background is stored and applied only when `.data` is accessed.
- All `Spectrum` / `Domain` arithmetic propagates uncertainties via the `uncertainties` library.
- `SNRFinder` requires `ResolutionCalibration` attached to the `Spectrum` before calling.

---

## Non-Obvious Design Rules

### Units — these are strict throughout
- Depth / position: **nm**
- Diffusion coefficient: **nm²/ps**
- Mobility: **nm²/(ps·V)**
- Annihilation rate (λ): **1/ps**
- Electric field: **V/nm**
- Implantation profiles: **positrons/nm** (must be normalised so integral = 1)
- Implantation energy for profiles: **keV**
- Material density (for Ghosh/Makhov profiles): **g/cm³**

### Governing PDE (`profile_solver`)
```
d/dz[ D(z) dc/dz ] − μ(z) E(z) dc/dz − λ(z) c(z) = −g(z)
```
Boundary conditions are **radiative** at both surfaces:
- `dc/dz|_{z=0}  =  c(0) / L_a`  (surface absorption length from `Sample.absorption_length`)
- `dc/dz|_{z=L}  = −c(L) / L_bulk`  (bulk diffusion length `sqrt(D/λ)`)

The FD solver handles discontinuous material coefficients at layer interfaces correctly. The scipy BVP solver (`tests/transport/diffusion/scipy_positron_profile_solver.py`, used only for validation) does not handle discontinuous interfaces well.

### Sample / Layer construction
`Sample.__post_init__` auto-computes `Layer.start` from widths in list order — never set `start` manually when building a `Sample`. The last layer must be thick enough that `c(z) ≈ 0` at its far end.

### DB and Domain
`DB` inherits `spectrum`, `start`, `stop`, `background`, and `data` from `Domain`. `DB` itself only adds `s_parameter_calculation`, `w_parameter_calculation`, `from_spectrum`, `from_domain`, and `recenter`. When modifying `DB`, be aware that `self.data` is background-subtracted lazily.

### `ELECTRON_REST_MASS_KEV`
Computed from `scipy.constants`, not hardcoded as 511.

### xarray coordinate name
All depth-dependent `xr.DataArray` objects (implantation profiles, electric field, solver output) use coordinate name **`'x'`** in nm. `profile_solver` interpolates the input profile onto its mesh via `.interp(x=mesh_points)`.

### DiffusionLengthOptimization — normalized trial samples
Inside the optimizer, trial samples are constructed with `D=1`, `λ = 1/L²` so that `L_eff = sqrt(D/λ) = L`. Absolute D and λ are not independently identifiable from VEDB data.

### `annihilation_fractions` layer coordinate convention
`compute_annihilation_fractions` returns an `xr.DataArray` with coordinate `'layer'`:
- `layer = -1` : surface annihilation
- `layer = 0, 1, 2, …` : bulk layers in depth order

---

## Branch Strategy

- **`main`** — paper submission branch. No lifetime module.
- **`lifetime`** — active development for positron lifetime spectrum analysis. All `analysis/lifetime/` work happens here. Do not merge back to `main` without discussion.

---

## Known Issues (publication deadline 2026-07-03)

All reviewer code issues for the CPC revision are resolved. Remaining items are future work.

### Stub files needing implementation
- `filter/pals_coincidence.py` — PALS timing coincidence filter (out of scope for CPC revision)

---

## Notes

- Do not read example data files (`.txt`, `.nc`, `.csv` in `examples/`).
- Data files in `pypas/libs/` are package data loaded via `importlib.resources`. They are included in the distribution via `[tool.setuptools.package-data]` in `pyproject.toml`. The `__init__.py` files in `libs/` subdirectories are required for `importlib.resources.files()` to treat them as packages.
- `tests/transport/diffusion/scipy_positron_profile_solver.py` is a reference/validation implementation, not a test file. It is used by `test_fd_vs_scipy.py`.
- `tests/transport/diffusion/test_fd_analytical.py` tests the FD solver against a closed-form analytical solution (< 0.01% error threshold).
