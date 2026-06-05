import numpy as np
from collections import namedtuple

LayerIndex = namedtuple("LayerIndex", ["layer_number", "ind_start", "ind_end", "depth_start", "depth_end"])

def get_layer_indices(depth_vector: np.ndarray, widths: list) -> list:
    """
    Compute the index ranges in the depth_vector corresponding to each material layer.

    Returns a list of LayerIndex(start, end, depth_start, depth_end) with index intervals
    [start:end] for each layer, where `start` is inclusive and `end` is exclusive.
    """
    cumulative_depths = np.cumsum(widths)
    layer_bounds = np.concatenate([[0], cumulative_depths])
    indices = []
    for i in range(len(widths)):
        depth_start = layer_bounds[i]
        depth_end = layer_bounds[i + 1]

        start_idx = np.searchsorted(depth_vector, depth_start, side='left')
        end_idx = np.searchsorted(depth_vector, depth_end, side='left')

        indices.append(LayerIndex(layer_number=i,
                                  ind_start=start_idx,
                                  ind_end=end_idx,
                                  depth_start=depth_start,
                                  depth_end=depth_end))

    # If depth_vector extends beyond final layer, add a tail "overflow" region using the last layer
    if depth_vector[-1] > layer_bounds[-1]:
        start_idx = indices[-1].ind_end
        end_idx = len(depth_vector)
        indices.append(LayerIndex(layer_number=i+1,
                                  ind_start=start_idx,
                                  ind_end=end_idx,
                                  depth_start=layer_bounds[-1],
                                  depth_end=depth_vector[-1]))

    return indices