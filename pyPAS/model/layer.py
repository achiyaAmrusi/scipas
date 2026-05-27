from dataclasses import dataclass, field
from pyPAS.model.material import Material

@dataclass
class Layer:
    """
    Represents a single continuous layer within a model.

    Each layer is defined by its spatial extent (start and width) and
    the material it consists of. Layers are used to build up a complete
    model structure for positron diffusion simulations.

    Parameters
    ----------
    start : float
        Starting position of the layer in the model [nm].
    width : float
        Width (thickness) of the layer [nm].
    material : Material
        The material that fills the layer uniformly.

    Attributes
    ----------
    start : float
        Starting coordinate of the layer [nm].
    width : float
        Thickness of the layer [nm].
    material : Material
        Material description of the layer (including diffusion and annihilation properties).
    """
    material: Material
    start: float = 0.0  # [nm]
    width: float  = 0.0 # [nm]
