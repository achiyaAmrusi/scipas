from dataclasses import dataclass
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

    >>> from pyPAS.model.layer import Layer
    >>> from pyPAS.model import Material
    >>> copper = Material(name="copper",
    ...                    diffusion=0.1,
    ...                    mobility=0.1,
    ...                    bulk_annihilation_rate=0.1)
    >>> layer = Layer(start=0.0, width=1.0, material=copper)
    >>> layer.width
    1.0
    """
    material: Material
    start: float = 0.0  # [nm]
    width: float  = 0.0 # [nm]
