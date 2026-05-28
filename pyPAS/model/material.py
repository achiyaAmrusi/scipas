from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class Defect:
    """
    Represents a specific type of defect that can trap or annihilate positrons.

    Parameters
    ----------
    - name: str
        Identifier for the defect type.
    - annihilation_rate: float
        Annihilation rate of trapped positrons in the defect [1/ps].

    Examples
    --------
    >>> from pyPAS.model import Defect
    >>> vacancy = Defect(name='Copper-vacancy', annihilation_rate=0.1)
    """
    name: str
    annihilation_rate: float  # [1/ps]


@dataclass
class Material:
    """
    Describes a material through which positrons diffuse.

    This includes parameters for positron diffusion and annihilation in the bulk,
    as well as an optional list of defect types that can capture or annihilate positrons.

    Parameters
    ----------
    - name: str
        Name of the material.
    - diffusion_coefficient: float
        Positron diffusion constant [nm²/ps].
        A value of 1 can be used for normalized unit studies.
    - bulk_annihilation_rate: float
        Effective annihilation rate in the bulk (λ_b) [1/ps].
        Always present in the material.
    - defects: list of Defect, optional
        Defect types with their respective capture and annihilation rates.
        These represent additional positron loss channels.

    Notes
    -----
    - Defect capture rates can be interpreted as proportional to the defect density.
    - The total loss rate from the free positron population is (λ_b + κ_d), where κ_d is the
      sum of all defect capture rates.
    - In standard diffusion length analysis, defect escape terms (e_d * n_d) are often neglected,
      assuming n_d ∝ n_b.

    Attributes
    ----------
    - name: str
    - diffusion_coefficient: float
    - bulk_annihilation_rate: float
    - defects: List[Defect]
    Examples
    --------
    >>> from pyPAS.model import Material
    >>> divacancy = Defect(name='Copper-divacancy', annihilation_rate=0.15)
    >>> silicon = Material(name="Silicon",
    ...                    diffusion=0.1,
    ...                    mobility=0.1,
    ...                    bulk_annihilation_rate=0.1,
    ...                    defects=[divacancy])
    """
    name: str ='temporary'
    diffusion: float  = 0.0 # [nm²/ps]
    mobility: float  = 0.0 # [nm²/ps*V]
    bulk_annihilation_rate: float = 0.0  # [1/ps] – always present
    defects: List[Defect] = field(default_factory=list)

    def effective_annihilation_rate(self):
        effective_annihilation_rate = self.bulk_annihilation_rate
        if self.defects:
            for defect in self.defects:
                effective_annihilation_rate += defect.annihilation_rate
        return effective_annihilation_rate

    def effective_diffusion_length(self):
        return np.sqrt(self.diffusion / self.effective_annihilation_rate())
