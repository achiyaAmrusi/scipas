from pathlib import Path
import pandas as pd


def ghosh_material_parameters():
    """
    Function to get the parameters for the positron implantation positron_implantation_profile fit in various materials.
     The fit is Gosh fit which is described in [1]
    Returns
    -------
    pd.DataFrame
    The parameters of Gosh fit for the positrons implantation positron_implantation_profile in various materials

    Reference
    ---------
    [1] V.J. Ghosh et al. https://doi.org/10.1016/0169-4332(94)00331-9.
    """
    # Navigate to the 'lib' directory within the current directory
    parameters_file_path = Path(__file__).parents[2] / 'libs/positron_profile/gosh_profile_parameters.txt'

    # Read the contents of the file
    try:
        parm = pd.read_csv(parameters_file_path)
    except FileNotFoundError:
        print(f"Error: File '{parameters_file_path}' not found.")
        parm = None
    return parm


def makhov_material_parameters():
    """
    Function to get the parameters for the positron implantation positron_implantation_profile fit in various materials.
     The fit is Makhov fit which is described in [1]
    Returns
    -------
    pd.DataFrame
    The parameters of Makhov fit for the positrons implantation positron_implantation_profile in various materials

    Reference
    ---------
    [2] Jerzy Dryzek et al. https://doi.org/10.1016/j.nimb.2008.06.033.
    """
    # Get the path to the current script
    current_dir = Path(__file__).resolve().parent.parent.parent

    # Navigate to the 'lib' directory within the current directory
    parameters_file_path = Path(__file__).parents[2] / 'libs/positron_profile/makhov_profile_parameters.txt'

    # Read the contents of the file
    try:
        parm = pd.read_csv(parameters_file_path)
    except FileNotFoundError:
        print(f"Error: File '{parameters_file_path}' not found.")
        parm = None
    return parm
