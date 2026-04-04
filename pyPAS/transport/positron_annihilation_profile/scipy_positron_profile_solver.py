import numpy as np
import xarray as xr
from pyPAS.materials.sample import Sample
from scipy.integrate import solve_bvp


def scipy_profile_solver(positron_implantation_profile: xr.DataArray,
                         sample: Sample,
                         electric_field: xr.DataArray = None,
                         num_of_mesh_cells=1000):
    """
    The solution for positron positron_implantation_profile boundary problem for given materials and positron energy.
    The solution method here uses scipy solve_bvp function in order to solve the self-consistent problem.
    because of its self-consisting nature this method is slow comparable with the matrix solution.

    Parameters
    ----------
    - positron_implantation_profile: xarray.DataArray
    A function of the thermal positrons per micron in the materials. see also ghosh_profile, makhov_profile
    - materials: Sample
    The materials for which the positron_implantation_profile is calculated,
      it is advised to define the last layer to be
        at least 3 mean free path from the end of the positron_implantation_profile (or where the implantation is negligible).
    - electric_field: xarray.DataArray
    The electric field value if it exists,
    If None, taken to be 0
    - num_of_mesh_cells: int
    The number of mesh cells for the system
    The function takes the total size of he materials, divide by num_of_mesh_cells,
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
                    - material.mobility * (density[1][i] * E[i] + density[0][i] * dE_dx[i]) \
                    + (eff_annihilation_rate[i]) * density[0][i] \
                    - I[i].values) for i, material in enumerate(materials)]
        return np.vstack((dc_dx, d_2_c_dx_2))

    # define the boundary condition for the ode system
    def boundary_conditions(density_in_surface, density_in_deep_bulk):
        """
        The boundary condition definition for the beginning and end of the materials/
       The boundary conditions given are
        (bc 1) dc(x_f)/dx = c(x_f)/L_plus
        (bc 2) Ddc(0)/dx = $\alpha_s$*c(0) - > radiative condition
        Note: I'd like to make from both vacuum condition and see if it is more compatible
        """
        L_a = sample.absorbtion_length
        L_p = sample.layers[-1].material.effective_diffusion_length()
        return np.array([density_in_surface[1] - density_in_surface[0] / L_a,
                         density_in_deep_bulk[1]+density_in_deep_bulk[0]/L_p])  # Boundary conditions

    # The mesh array
    mesh = np.linspace(0, sample.sample_length(), num_of_mesh_cells)

    # The initial guess of the positron positron_implantation_profile is the solution from the fast solve,
    # we can see in scipy if it converge
    initial_guess = np.array([positron_implantation_profile.interp(x=x) for x in mesh])
    initial_guess = xr.DataArray(np.nan_to_num(initial_guess), coords={'x': mesh})

    sol = solve_bvp(fun=ode_system, bc=boundary_conditions, x=mesh,
                    y=np.vstack((initial_guess, initial_guess.differentiate('x'))),
                    max_nodes=max(num_of_mesh_cells + 1, 1000), tol=1e-5)
    return sol
