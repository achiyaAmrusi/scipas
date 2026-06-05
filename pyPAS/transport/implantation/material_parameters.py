from pathlib import Path
import pandas as pd


def ghosh_material_parameters():
    """
    Function to get the parameters for the positron implantation fit in various model.
     The fit is Gosh fit which is described in [1]
    Returns
    -------
    pd.DataFrame
    The parameters of Gosh fit for the positrons implantation in various model

    Reference
    ---------
    [1] V.J. Ghosh et al. https://doi.org/10.1016/0169-4332(94)00331-9.
    >>> params = ghosh_material_parameters()
    >>> params is not None
    True
    >>> len(params) > 0
    True
    >>> 'Material' in params.columns  # or whatever the actual column name is
    True
    >>> params.iloc[0]['Material']
    'Be'
    """
    # Navigate to the 'lib' directory within the current directory
    parameters_file_path = Path(__file__).parents[3] / 'libs/positron_profile/gosh_profile_parameters.txt'

    # Read the contents of the file
    try:
        parm = pd.read_csv(parameters_file_path)
    except FileNotFoundError:
        print(f"Error: File '{parameters_file_path}' not found.")
        parm = None
    return parm


def makhov_material_parameters():
    """
    Function to get the parameters for the positron implantation fit in various model.
     The fit is Makhov fit which is described in [1]
    Returns
    -------
    pd.DataFrame
    The parameters of Makhov fit for the positrons implantation in various model

    Reference
    ---------
    [2] Jerzy Dryzek et al. https://doi.org/10.1016/j.nimb.2008.06.033.

    Examples
    --------
    >>> params = makhov_material_parameters()
    >>> params is not None
    True
    >>> len(params) > 0
    True
    >>> 'Material' in params.columns  # or whatever the actual column name is
    True
    >>> params.iloc[0]['Material']
    'Be'
    """

    # Navigate to the 'lib' directory within the current directory
    parameters_file_path = Path(__file__).parents[3] / 'libs/positron_profile/makhov_profile_parameters.txt'

    # Read the contents of the file
    try:
        parm = pd.read_csv(parameters_file_path)
    except FileNotFoundError:
        print(f"Error: File '{parameters_file_path}' not found.")
        parm = None
    return parm
