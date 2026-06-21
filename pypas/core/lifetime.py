from scispectrum import Spectrum
from pypas.core.time_resolution import  TimeResolution

class PASLifetime:
    """"
    Represents a lifetime spectrum with a resolution instance.
    """

    def __init__(self, lifetime: Spectrum, resolution: TimeResolution):
        self.lifetime = lifetime
        self.resolution = resolution