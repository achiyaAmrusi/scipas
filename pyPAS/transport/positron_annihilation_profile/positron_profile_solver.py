import numpy as np
import xarray as xr
from pyPAS.materials.sample import Sample
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def profile_solver(positron_implantation_profile: xr.DataArray,
                   sample: Sample,
                   electric_field: xr.DataArray = None,
                   mesh_size=10000):
    """
    Finite differences solver for the positron annihilation profile in a materials, c(z).
    detailed description of the solver will be published in future paper [].

    Parameters
    ----------
    - positron_implantation_profile: xarray.DataArray
    Thermal positron implantation profile [positron/length] in the materials.
    Note that in the code the profile is linearly interpolated into the mesh points
     See ghosh_profile, makhov_profile
    - materials: Sample
    The materials for which c(z) is calculated,
      it is advised to define the last layer to be very large such that c(z) is expected to be negligible at the end of the materials
    - electric_field: xarray.DataArray
    The electric field in the materials
    If None, taken to be 0
    - mesh_size: int (default 10000)
    Specifies the number of cells used to discretize the 1D domain in the finite element method.

    Returns
    -------
    The positron annihilation profile [annihilation/micron/s]
    """
    # The mesh array
    mesh_points = np.linspace(0, sample.sample_length(), mesh_size)

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
        absorbtion_length=sample.absorbtion_length)

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
    Convert a layered Sample object into spatially resolved material property vectors
    (diffusion, annihilation rate, drift).

    Parameters
    ----------
    - materials: Sample
        The layered materials defining material properties.
    - mesh_points: np.ndarray
        Evenly spaced mesh points spanning the materials depth.
    - electric_field: xr.DataArray, optional
        Electric field defined on the mesh coordinates.
        If None, assumed to be zero everywhere.

    Returns
    -------
    xr.Dataset
        Dataset containing the material property vectors:
        - diffusion [cm^2/s]
        - annihilation_rate [1/s]
        - drift [cm/s] (zero if no field)
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
                       absorbtion_length: float):
    # definitions
    dx = mesh_points[1] - mesh_points[0]

    diag = np.zeros_like(mesh_points)
    diag_upper = np.zeros(mesh_points.size - 1)
    diag_lower = np.zeros(mesh_points.size - 1)

    # Calculate the 3 diagonals of the differential equation operator
    # (note: We use central central finite differences o(dz**2))

    diag[1:-1] = -((diffusion[2:] + 2*diffusion[1:-1]+ diffusion[:-2]) / 2 / dx ** 2 + annihilation_rate[1:-1])
    diag_upper[1:] = (diffusion[2:] + diffusion[1:-1]) / 2 / dx ** 2 - drift[2:] / 2 / dx
    diag_lower[:-1] = (diffusion[1:-1] + diffusion[:-2]) / 2 / dx ** 2 + drift[:-2] / 2 / dx

    if annihilation_rate[-1] == 0:
        l_bulk = np.inf
    else:
        l_bulk = (diffusion[-1] / annihilation_rate[-1]) ** 0.5
    # boundary conditions are taken on the centers of the cells and not the edges for stability
    diag[0] = -(2 * diffusion[0] / dx ** 2 + annihilation_rate[0] + (diffusion[0] / dx ** 2 + drift[0] / 2 / dx) * 2 * dx / absorbtion_length)
    diag_upper[0] = 2*diffusion[0] / dx ** 2


    diag[-1] = -(2 * diffusion[-1] / dx ** 2 + annihilation_rate[-1] + (diffusion[-1] / dx ** 2 - drift[-1] / 2 / dx) * (2 * dx / l_bulk))
    diag_lower[-1] = 2*diffusion[-1] / dx ** 2

    return sp.diags([diag, diag_upper, diag_lower],
                    [0, 1, -1], shape=(mesh_points.size, mesh_points.size))
