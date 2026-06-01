""" positron annihilation diffusion solver.
The module include a fast finite difference solver and a slow scipy-bvp solver with iterative method for comparison"""
from .positron_profile_solver import profile_solver, diffusion_operator, sample_to_material_vectors
