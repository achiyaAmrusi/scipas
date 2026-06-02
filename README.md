# PyPAS: Positron Annihilation Spectroscopy in Python

A Python package for **Doppler Broadening (DB)** and **Coincidence Doppler Broadening (CDB)** analysis, positron implantation profiling, transport simulation, and variable-energy Doppler broadening (VEDB) diffusion-length fitting.

PyPAS provides a unified, modular workflow — from raw detector data to material parameters — built on standard scientific Python.

---

## Features

- **DB spectrum analysis** — S and W parameter extraction with Poisson uncertainty propagation; automatic 511 keV peak identification and axis centralization
- **CDB analysis** — 2D coincidence histogram; DB and resolution projections from detector-pair data
- **Event filtering** — time and energy coincidence filtering for synchronized detector pairs
- **Implantation profiles** — Makhovian and Ghosh positron stopping profiles; multilayer cumulative stitching; support for external (MC-simulated) profiles
- **Positron transport solver** — 1D finite-difference solver for the diffusion–drift–annihilation equation with electric fields and radiative boundary conditions
- **Layered sample model** — `Sample` / `Layer` / `Material` descriptors with depth-dependent diffusion, mobility, and annihilation rates
- **VEDB fitting** — diffusion-length optimization with covariance estimation; single- and multi-layer support; weighted nonlinear least squares
- **xarray throughout** — labeled, sliceable data with named coordinates
- **Uncertainty propagation** via the `uncertainties` library

---

## Installation

PyPAS requires [pySpectrum](https://github.com/achiyaAmrusi/pySpectrum) as a companion package. Install both from GitHub:

```bash
git clone https://github.com/achiyaAmrusi/pySpectrum
cd pySpectrum
pip install -e .
cd ..

git clone https://github.com/achiyaAmrusi/pyPAS
cd pyPAS
pip install -e .
```

For a development install (includes test dependencies):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### Doppler Broadening — S and W parameters

```python
import pandas as pd
from pyspectrum.core import Spectrum
from pyspectrum.calibration import AxisCalibration, ResolutionCalibration
from pyPAS.core import PASdb

calib = AxisCalibration(lambda ch: 0.5 * ch + 1.0, name="energy_keV")
res   = ResolutionCalibration(lambda e: 1.8)   # constant FWHM in keV
spec  = Spectrum.from_dataframe(df, channel_col="channel", counts_col="counts",
                                axis_calib=calib, resolution_calib=res)

# Identify the 511 keV peak automatically
db = PASdb.from_spectrum(spec)

# Extract S and W parameters (energy in keV, relative to 511 keV)
s = db.s_parameter_calculation(energy_domain_total=(-8, 8),
                                energy_domain_s=(-0.8, 0.8))
w = db.w_parameter_calculation(energy_domain_total=(-8, 8),
                                energy_domain_w_left=(-8, -2),
                                energy_domain_w_right=(2, 8))
print(s, w)  # ufloat values with propagated uncertainties
```

### Coincidence Doppler Broadening

```python
from pyPAS.filter import PasCoincidenceFilter
from pyPAS.core import PAScdb

# Step 1: find time-coincident events
pairs = PasCoincidenceFilter.time_coincidence_filter(
    det_1_df, det_2_df, max_time_interval=10)

# Step 2: apply energy-conservation window (keeps E1 + E2 ≈ 1022 keV)
energy_pairs = PasCoincidenceFilter.energy_coincidence_filter(
    pairs,
    axis_calibration_1=calib_1,
    axis_calibration_2=calib_2,
    local_fwhm_1=1.2,
    local_fwhm_2=1.2)

# Step 3: build CDB object (histogram computed once at construction)
cdb = PAScdb(energy_pairs, energy_min=-4, energy_max=4, mesh_interval=0.05)

db  = cdb.doppler_broadening()   # PASdb ready for S/W analysis
res = cdb.resolution()           # 1D resolution spectrum
```

### Implantation profiles

```python
import numpy as np
from pyPAS.transport import makhov_profile, makhov_material_parameters

depth  = np.arange(0, 5000, 1)   # nm
params = makhov_material_parameters()
si     = params[params["Material"] == "Si"].iloc[0]

profile = makhov_profile(positron_energy=10, depth_vector=depth,
                         density=si.density, makhov_parms=si)
```

### Multilayer implantation profile

```python
from pyPAS.transport import (multilayer_implantation_profile,
                              makhov_profile, makhov_material_parameters)

params = makhov_material_parameters()
cu     = params[params["Material"] == "Cu"].iloc[0]
si     = params[params["Material"] == "Si"].iloc[0]

profile = multilayer_implantation_profile(
    positron_energy=10,
    depth_vector=np.arange(0, 5000, 1),
    widths=[500],                          # 500 nm Cu film on Si substrate
    materials_parameters=[cu, si],
    densities=[cu.density, si.density],
    implantation_profile_function=makhov_profile)
```

### Positron transport — diffusion solver

```python
from pyPAS.model import Sample, Layer, Material
from pyPAS.transport import profile_solver

silicon = Material(name="Si", diffusion=1.0, mobility=0.0,
                   bulk_annihilation_rate=2.0)
layer   = Layer(width=10000.0, material=silicon)
sample  = Sample(layers=[layer], absorption_length=0.5)

# Returns xr.DataArray of c(z) on a uniform mesh
positron_profile = profile_solver(implantation_profile, sample)
```

### VEDB diffusion-length fitting

```python
from pyPAS.analysis import DiffusionLengthOptimization

optimizer = DiffusionLengthOptimization(
    positron_implantation_profiles=profiles,   # list of xr.DataArray, one per energy
    s_measurement=s_series,                    # pd.Series of ufloat
    initial_guess=initial_sample)

best_fit, covariance = optimizer.optimize_diffusion_length(bounds=(0, 1000))
sigma = np.sqrt(np.diag(covariance))
print(f"L+ = {best_fit} ± {sigma} nm")
```

---

## Examples

Full worked examples are in the [`examples/`](./examples) directory:

**DB / CDB Analysis**
- [DB spectrum analysis](./examples/core/pas_db.ipynb) — load, calibrate, and extract S/W from a DB spectrum
- [CDB analysis](./examples/core/pas_cdb.ipynb) — process coincidence data into a DB spectrum and S/W parameters

**VEDB Analysis**
- [S(E) and W(E) lineshape extraction](./examples/vedb%20analysis/vedb_lineshape.ipynb) — load multi-energy DB spectra; compute S(E) and W(E) curves with errorbars and S–W parametric plot
- [Diffusion-length fitting — measurement](./examples/vedb%20analysis/vedb_diffusion_length_measurement.ipynb) — fit L₊ from measured S(E) using the transport model; plot fit vs data and annihilation fractions per channel
- [Diffusion-length fitting — two-layer simulation](./examples/vedb%20analysis/vedb_diffusion_length_2layer_simulation.ipynb) — simulate a damaged surface layer over bulk, fit both L₀ and L₁ simultaneously, and visualise the 2D χ² joint confidence region

**Implantation Profiles and Transport**
- [Positron profile in Si](./examples/positron%20profile/positron%20profile%20in%20Si.ipynb) — Makhov and Ghosh profiles, multilayer stitching, transport solver, annihilation fractions
- [Transport solver benchmark](./examples/transport%20benchmark/solver_benchmark.ipynb) — validates `profile_solver` against two analytical results: exact surface fraction formula and closed-form full profile; confirms O(N⁻²) convergence

---

## Requirements

Following the [SPEC 0](https://scientific-python.org/specs/spec-0000/) support policy:

| Package | Version |
|---|---|
| Python | ≥ 3.11 |
| numpy | ≥ 2.0, < 3 |
| pandas | ≥ 2.3, < 4 |
| scipy | ≥ 1.14 |
| xarray | ≥ 2024.6 |
| uncertainties | ≥ 3.1 |
| pyspectrum | ≥ 0.3 |

---

## Project Status

PyPAS is under active development. The DB/CDB analysis, implantation profiles, diffusion solver, and VEDB fitting are stable. Planned additions include positron lifetime spectrum analysis and extended Bayesian workflows for model comparison and uncertainty quantification.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

Achiya Yosef Amrusi — [GitHub](https://github.com/achiyaAmrusi)

Contributions and issues are welcome. Please include a minimal reproducible example when reporting a bug.
