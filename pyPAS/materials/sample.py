from pyPAS.materials.layer import Layer
from typing import List
from dataclasses import dataclass, field


@dataclass
class Sample:
    """
    Represents a multilayer materials composed of sequential layers, each made of a single material.

    This class is used to describe the full structure of a materials through which positrons diffuse,
    annihilate, or get trapped. Each `Layer` defines a material and its spatial extent, and together
    the materials defines a 1D geometry for the simulation or analysis.

    Parameters
    ----------
    layers : List[Layer]
        A list of layers in the materials. The order of the list corresponds to increasing depth
        along the materials axis (e.g. z-axis in 1D).

    Attributes
    ----------
    layers : List[Layer]
        The list of materials layers from surface to back.

    Methods
    -------
    total_thickness() -> float
        Returns the total thickness of the materials [nm].

    get_layer_at(position: float) -> Layer
        Returns the layer object in which a given depth (position) lies [nm].
    """
    layers: List[Layer] = field(default_factory=list)
    absorbtion_length: float = 0.0 # [nm]

    def __post_init__(self):
        """Automatically set start positions of layers based on widths."""
        start = 0.0
        for layer in self.layers:
            layer.start = start
            start += layer.width

    def sample_length(self) -> float:
        """Compute the total thickness of the materials in nanometers."""
        if not self.layers:
            return 0.0
        return self.layers[-1].start + self.layers[-1].width

    def get_layer_at(self, position: float) -> Layer:
        """
        Return the layer at a specific position in the materials.

        Parameters
        ----------
        position : float
            Depth in nanometers from the surface (0 is the start of the first layer).

        Returns
        -------
        Layer
            The layer containing the given position.

        Raises
        ------
        ValueError
            If the position is outside the bounds of the materials.
        """
        positon_layer = self.layers[-1]
        for layer in self.layers:
            if layer.start <= position < (layer.start + layer.width):
                positon_layer = layer
                break
        return positon_layer

