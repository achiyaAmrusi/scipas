# PyPAS: Positron annihilation spectroscopy in Python

Modern, modular **Doppler Broadening (DB)** and **Coincidence Doppler Broadening (CDB)** analysis for **Positron Annihilation Spectroscopy (PAS)** — with implantation profiles, a finite‑difference diffusion solver, and VEDB fitting.

---

## **Overview**

PyPAS unifies PAS spectroscopy and depth profiling in a single Python workflow.  
It handles:

- **DB/CDB spectrum processing** and S/W parameter extraction via direct summation with robust background subtraction  
- **Thermal implantation profiles** (Makhovian and Ghosh)  
- **1D positron diffusion equation solving** (including drift in electric fields)  
- **Diffusion length fitting** for VEDB measurements  

Modular, open‑source, and built on standard scientific Python libraries for transparent, reproducible analysis.

---

## **Install**

- **Requirements:** Python 3.9+, and the rest are listed in `requirements.txt`.
To install dependencies, run:

```bash
pip install -r requirements.txt
```
- **Latest (GitHub):**
```
pip install "git+https://github.com/achiyaAmrusi/pyPAS"
```
- **Dev install (editable):**
```
git clone https://github.com/achiyaAmrusi/pyPAS cd pyPAS pip install -e ".[dev]"
```
---

## **Features**

- **DB/CDB analysis:** Direct S/W extraction with robust background; 2D CDB histograms; DB and resolution projections
- **Event filtering:** Flexible time‑energy filtering for synchronized detector pairs with pluggable strategies
- **Implantation profiles:** Makhovian and Ghosh models; multilayer cumulative stitching; support for custom/external profiles
- **Diffusion solver:** Finite‑difference 1D solver with drift \( v(z) = μ(z)E(z) \); radiative boundary conditions; cell‑centered scheme
- **Layered models:** Sample/Layer descriptors with depth‑dependent transport coefficients, annihilation rates, and electric fields
- **VEDB fitting:** Diffusion‑length estimation with confidence intervals; model comparison (e.g., AIC) to guard against overfitting
- **Ecosystem:** NumPy/SciPy/xarray core; integrates with PySpectrum for spectrum I/O, peak metrics, and uncertainty propagation

---

## **Examples and documentation**

- **Examples:** See the `examples/` directory for notebooks covering DB/CDB analysis, implantation profiles, diffusion solutions, and VEDB fitting
- **API docs:** Docstrings throughout the codebase; browse in your IDE or with Python’s `help()`

---

## **Project status and roadmap**

- **Validation:** Benchmarked against analytical solutions and SciPy solvers, showing sub‑percent relative errors under appropriate resolution
- **Roadmap:** Planned extensions include positron lifetime analysis and additional Bayesian workflows for model comparison and uncertainty quantification

---

## **License, contributions, and citation**

- **License:** MIT license
- **Contributions:** Issues and PRs are welcome. Please include minimal reproducible examples.
- **Citation:** If PyPAS contributes to your work, please mention the package https://doi.org/10.48550/arXiv.2509.08023. 
