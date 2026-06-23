from importlib.resources import files
import pandas as pd

_DATA = files('scipas.libs.positron_profile')


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
    with _DATA.joinpath('gosh_profile_parameters.txt').open() as f:
        return pd.read_csv(f)


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
    with _DATA.joinpath('makhov_profile_parameters.txt').open() as f:
        return pd.read_csv(f)
