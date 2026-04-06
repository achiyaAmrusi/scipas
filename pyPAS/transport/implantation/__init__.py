"""Package for the implantation profile in different materials
TODO: add a function to make complex implementation profile from a Sample"""
from .implantation_profile import ghosh_profile, makhov_profile, multilayer_implantation_profile
from .profile_material_parameters import ghosh_material_parameters, makhov_material_parameters
