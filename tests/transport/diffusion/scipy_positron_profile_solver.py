import numpy as np
import xarray as xr
from pypas.model.sample import Sample
from scipy.integrate import solve_bvp


def scipy_profile_solver(positron_implantation_profile: xr.DataArray,
                         sample: Sample,
                         electric_field: xr.DataArray = None,
                         num_of_mesh_cells=1000,
                         initial_guess: xr.DataArray = None,
                         max_nodes: int = None):
    """
    A solver for positron diffusion problem for given model and positron energy.
    The solution method here uses scipy solve_bvp function in order to solve the self-consistent problem.
    because of its self-consisting nature this method is slow comparable with the matrix solution.

    Parameters
    ----------
    - implantation: xarray.DataArray
    A function of the thermal positrons per micron in the model. see also ghosh_profile, makhov_profile
    - model: Sample
    The model for which the implantation is calculated,
      it is advised to define the last layer to be
        at least 3 mean free path from the end of the implantation (or where the implantation is negligible).
    - electric_field: xarray.DataArray
    The electric field value if it exists,
    If None, taken to be 0
    - num_of_mesh_cells: int
    The number of mesh cells for the system
    The function takes the total size of he model, divide by num_of_mesh_cells,
     and this is the size of ever mesh cell
     default is 1000
    Returns
    -------
    The thermalized positron distribution in units of positrons per micron
    """

    # define the diffusion ODE system for scipy solver
    def ode_system(location, density):
        """
        The ode for the positron diffusion-trapping-annihilation ode[]:
        D*(d**2c(x)/dx**2 )-d/dx(v_d(x)*c(x))+I(x)-sum(labda_defects)*c(x)-lambda_bulk*c(x) = 0
        The diffusion and and the annihilation rate depends on the material which depend on the location.
        density[0] is the positron density
        density[1] is the positron density derivative
        Parameters
        ----------
        - location: list
        location in which the ode is calculated
        - density: list
        the density and the density derivitive

        Returns
        -------
        np.ndarray
        The density derivative and the density second derivative in compliance with the equation above.
        """

        # check in which material the point is, and get the annhilation rates
        materials = [sample.get_layer_at(x).material for x in location]

        eff_annihilation_rate = np.array([material.effective_annihilation_rate() for material in materials])

        # positron influx in the locations
        I = positron_implantation_profile.interp(x=location)
        I = I.fillna(0)
        I[0] = 0
        # electric field in the location
        if electric_field is None:
            E = np.zeros_like(location)
            dE_dx = np.zeros_like(location)
        else:
            electric_field_deriv = electric_field.differentiate('x')
            E = electric_field.interp(x=location)
            dE_dx = electric_field_deriv.interp(x=location)
        # Derivitives calculation
        # First derivitive
        dc_dx = density[1]
        # second derivitive
        d_2_c_dx_2 = [(1 / material.diffusion) * ( \
                    + material.mobility * (density[1][i] * E[i] + density[0][i] * dE_dx[i]) \
                    + (eff_annihilation_rate[i]) * density[0][i] \
                    - I[i].values) for i, material in enumerate(materials)]
        return np.vstack((dc_dx, d_2_c_dx_2))

    # define the boundary condition for the ode system
    def boundary_conditions(density_in_surface, density_in_deep_bulk):
        """
        The boundary condition definition for the beginning and end of the model/
       The boundary conditions given are
        (bc 1) dc(x_f)/dx = c(x_f)/L_plus
        (bc 2) Ddc(0)/dx = $\alpha_s$*c(0) - > radiative condition
        Note: I'd like to make from both vacuum condition and see if it is more compatible
        """
        L_a = sample.absorption_length
        L_p = sample.layers[-1].material.effective_diffusion_length()
        return np.array([density_in_surface[1] - density_in_surface[0] / L_a,
                         density_in_deep_bulk[1]+density_in_deep_bulk[0]/L_p])  # Boundary conditions

    # The mesh array
    mesh = np.linspace(0, sample.sample_length(), num_of_mesh_cells)

    if initial_guess is not None:
        y0_vals = np.clip(initial_guess.interp(x=mesh).values, 0.0, None)
    else:
        y0_vals = np.array([float(positron_implantation_profile.interp(x=x)) for x in mesh])
    y0_vals = np.nan_to_num(y0_vals)
    y0_da = xr.DataArray(y0_vals, coords={'x': mesh})

    effective_max_nodes = max_nodes if max_nodes is not None else max(num_of_mesh_cells + 1, 1000)
    sol = solve_bvp(fun=ode_system, bc=boundary_conditions, x=mesh,
                    y=np.vstack((y0_da, y0_da.differentiate('x'))),
                    max_nodes=effective_max_nodes, tol=1e-5)
    return sol
