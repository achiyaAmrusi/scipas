import numpy as np
import xarray as xr
from pypas.model.sample import Sample
import scipy.sparse as sp

def profile_solver(positron_implantation_profile: xr.DataArray,
                   sample: Sample,
                   electric_field: xr.DataArray = None,
                   mesh_size: int =10000):
    """
    Solve for the steady-state positron annihilation profile c(z) in a multilayer sample.

    Uses a finite differences discretization of the 1D positron diffusion-drift-annihilation
    equation with radiative boundary conditions at both surfaces. The sparse linear system is
    solved directly via scipy.linlag solver.
    A detailed derivation will be published in a future paper.

    The governing equation solved is:

        d/dz [ D(z) dc/dz ] - μ(z) E(z) dc/dz - λ(z) c(z) = -g(z)

    where:
        - D(z)  : spatially resolved diffusion coefficient [nm²/ps]
        - μ(z)  : positron mobility [nm²/(ps·V)]
        - E(z)  : electric field [V/nm]
        - λ(z)  : effective annihilation rate [1/ps]
        - g(z)  : positron implantation source profile [1/nm]
        - c(z)  : positron density (solved quantity)

    Boundary conditions:
        - Surface (z=0)  : radiative condition with absorption length from sample
        - Back surface   : radiative condition using bulk diffusion length L = sqrt(D/λ)

    Parameters
    ----------
    positron_implantation_profile : xr.DataArray
        Thermal positron implantation profile as a function of depth [positrons/nm],
        with coordinate 'x' in nm. Typically computed via makhov_profile or ghosh_profile.
        Linearly interpolated onto the mesh before solving.
    sample : Sample
        Multilayer sample defining geometry, material properties (diffusion, mobility,
        annihilation rate) and surface absorption length. The last layer should be
        sufficiently thick that c(z) ≈ 0 at its far end.
    electric_field : xr.DataArray, optional
        Electric field as a function of depth [V/nm], with coordinate 'x' in nm.
        If None, the field is taken to be zero everywhere.
    mesh_size : int, optional
        Number of uniformly spaced mesh points spanning [0, sample_length()].
        Default is 10000. Higher values improve accuracy at the cost of speed.

    Returns
    -------
    xr.DataArray
        Positron annihilation profile c(z) [annihilations/nm/s] on the mesh grid,
        with coordinate 'x' in nm.
    """
    # The mesh array
    if mesh_size is not None:
        mesh_points = np.linspace(start=0, stop=sample.sample_length(), num=int(mesh_size))
    else:
        raise ValueError("mesh_size must be an integer greater than 0.")
    # If the field is not defined it is taken to be 0
    if electric_field is None:
        electric_field = xr.DataArray(np.zeros(mesh_points.size), coords={'x': mesh_points})

    # define the diffusion equation operator matrix
    # Step 1: generate material property vectors
    ds = sample_to_material_vectors(sample, mesh_points, electric_field)
    # Step 2: build operator
    diff_operator = diffusion_operator(
        mesh_points=mesh_points,
        diffusion=ds["diffusion"].values,
        drift=ds["drift"].values,
        annihilation_rate=ds["annihilation_rate"].values,
        absorption_length=sample.absorption_length)

    # define the thermal positron profile
    positron_implantation = positron_implantation_profile.interp(x=mesh_points)
    positron_implantation = np.nan_to_num(positron_implantation)
    # solve for the positron distribution
    final_positron_distribution = sp.linalg.spsolve(A=diff_operator.tocsr(), b=-positron_implantation)

    return xr.DataArray(final_positron_distribution, coords={'x': mesh_points})

def sample_to_material_vectors(sample: Sample,
                               mesh_points: np.ndarray,
                               electric_field: xr.DataArray = None) -> xr.Dataset:
    """
    Map a layered Sample onto spatially resolved material property vectors on a 1D mesh.

    Each mesh point is assigned the material properties of the layer it falls in.
    Layer boundaries are rounded to the nearest mesh point. The drift velocity
    μ·E(z) is computed only when an electric field is provided.

    Parameters
    ----------
    sample : Sample
        Multilayer sample whose layers define material properties at each depth.
    mesh_points : np.ndarray
        1D array of uniformly spaced depth coordinates [nm] spanning [0, sample_length()].
    electric_field : xr.DataArray, optional
        Electric field as a function of depth [V/nm], with coordinate 'x' in nm.
        Must be a xr.DataArray if provided. If None, drift is set to zero everywhere.

    Returns
    -------
    xr.Dataset
        A Dataset with coordinate 'x' (nm) and three variables:
        - 'diffusion'        : positron diffusion coefficient D(z) [nm²/ps]
        - 'annihilation_rate': effective annihilation rate λ(z) [1/ps],
                               including bulk and all defect contributions
        - 'drift'            : drift velocity μ(z)·E(z) [nm/ps],
                               zero everywhere if no field is supplied

    Raises
    ------
    ValueError
        If electric_field is provided but is not type xr.DataArray.
    """
    if electric_field is None:
        electric_field = xr.DataArray(np.zeros_like(mesh_points), coords={'x': mesh_points})

    if electric_field is not None:
        if not isinstance(electric_field, xr.DataArray):
            raise ValueError("electric field needs to be an xarray.DataArray")

    # Allocate arrays
    diffusion = np.zeros_like(mesh_points)
    annihilation_rate = np.zeros_like(mesh_points)
    drift = np.zeros_like(mesh_points)

    # Layer indices in mesh
    for i, layer in enumerate(sample.layers):
        start = int(np.round(layer.start / sample.sample_length() * mesh_points.size))
        end = int(np.round((layer.start + layer.width) / sample.sample_length() * mesh_points.size))

        # Fill material properties into mesh segment
        diffusion[start:end] = layer.material.diffusion
        annihilation_rate[start:end] = layer.material.effective_annihilation_rate()

        # Add drift term if electric field is present
        if electric_field is not None:
            drift[start:end] = layer.material.mobility * electric_field.interp(x=mesh_points[start:end])

    # Package into Dataset
    ds = xr.Dataset(
        {
            "diffusion": ("x", diffusion),
            "annihilation_rate": ("x", annihilation_rate),
            "drift": ("x", drift),
        },
        coords={"x": mesh_points}
    )

    return ds

def diffusion_operator(mesh_points: np.ndarray,
                       diffusion: np.ndarray,
                       drift: np.ndarray,
                       annihilation_rate: np.ndarray,
                       absorption_length: float):
    """
    Build the sparse tridiagonal finite-difference operator for the positron
    diffusion-drift-annihilation equation.

    Discretizes the operator:

        L[c]_i = -[ D_{i+½}(c_{i+1} - c_i) - D_{i-½}(c_i - c_{i-1}) ] / dz²
                 + μE_i (c_{i+1} - c_{i-1}) / (2 dz)
                 + λ_i c_i

    using central finite differences (second-order accurate in dz), with
    interface diffusion coefficients averaged as D_{i+½} = (D_i + D_{i+1}) / 2.

    Boundary conditions (radiative, applied at cell centers for stability):
        - z = 0  : -D dc/dz = (D/absorption_length) · c
                   enforced as: diag[0] accounts for the surface loss term
        - z = L  : -D dc/dz = (D/L_bulk) · c,  L_bulk = sqrt(D[-1] / λ[-1])
                   If λ[-1] = 0 (no annihilation in last layer), L_bulk → ∞
                   and the back boundary becomes a zero-flux (Neumann) condition.

    Parameters
    ----------
    mesh_points : np.ndarray
        Uniformly spaced 1D mesh [nm] of length N. Spacing dz = mesh_points[1] - mesh_points[0].
    diffusion : np.ndarray
        Diffusion coefficient D(z) at each mesh point [nm²/ps], length N.
    drift : np.ndarray
        Drift velocity μ(z)·E(z) at each mesh point [nm/ps], length N.
    annihilation_rate : np.ndarray
        Effective annihilation rate λ(z) at each mesh point [1/ps], length N.
    absorption_length : float
        Positron surface absorption length [nm]. Controls the radiative boundary condition
        at z = 0. A value of 0 would imply perfect surface absorption.

    Returns
    -------
    sp.dia_matrix
        Sparse N×N tridiagonal matrix in DIA format representing the discretized
        operator. Convert to CSR before solving: operator.tocsr().
    """
    # definitions
    dx = mesh_points[1] - mesh_points[0]

    diag = np.zeros_like(mesh_points)
    diag_upper = np.zeros(mesh_points.size - 1)
    diag_lower = np.zeros(mesh_points.size - 1)

    # Calculate the 3 diagonals of the differential equation operator
    # (note: We use central finite differences o(dz**2))

    diag[1:-1] = -((diffusion[2:] + 2*diffusion[1:-1]+ diffusion[:-2]) / 2 / dx ** 2 + annihilation_rate[1:-1])
    diag_upper[1:] = (diffusion[2:] + diffusion[1:-1]) / 2 / dx ** 2 - drift[2:] / 2 / dx
    diag_lower[:-1] = (diffusion[1:-1] + diffusion[:-2]) / 2 / dx ** 2 + drift[:-2] / 2 / dx

    if annihilation_rate[-1] == 0:
        l_bulk = np.inf
    else:
        l_bulk = (diffusion[-1] / annihilation_rate[-1]) ** 0.5
    # boundary conditions are taken on the centers of the cells and not the edges for stability
    diag[0] = -(2 * diffusion[0] / dx ** 2 + annihilation_rate[0] + (diffusion[0] / dx ** 2 + drift[0] / 2 / dx) * 2 * dx / absorption_length)
    diag_upper[0] = 2*diffusion[0] / dx ** 2


    diag[-1] = -(2 * diffusion[-1] / dx ** 2 + annihilation_rate[-1] + (diffusion[-1] / dx ** 2 - drift[-1] / 2 / dx) * (2 * dx / l_bulk))
    diag_lower[-1] = 2*diffusion[-1] / dx ** 2

    return sp.diags([diag, diag_upper, diag_lower],
                    [0, 1, -1], shape=(mesh_points.size, mesh_points.size))
