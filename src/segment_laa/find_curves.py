# Author: Fanli Zhou
# Date: 2022-01-20

import sys
import numpy as np
from skimage import measure

sys.path.append('..')
from segment_laa.help_functions import get_resample, get_length

def get_ring(data, z):
    """
    Get points of the target ring.
    Parameters:
    -----------
    data: np.array
        points on the object surface
    z: int
        the z axis of the target ring
    Returns:
    --------
    np.ndarray
        points on the target ring
    """
    # (x, y, z) -> (z, x, y) -> (x, y)
    rings = measure.find_contours((data.transpose(2, 0, 1))[z] // 255, 0.9)
    # remain the largest contour
    ring = sorted(rings, key=lambda x: len(x))[-1]
    # (N, 2)
    return ring

def get_curves(data, start, end, num):
    """
    Get curves on the transition region.
    Parameters:
    -----------
    data: np.ndarray
        points on the object surface
    start: int
        the z axis of the start ring
    end: int
        the z axis of the end ring
    num: int
        the number of points to resample 
    Returns:
    --------
    np.ndarray
        curves on the transition region
    """
    curves = []
    for i in range(start, end):
        try:
            # (Nc, 2)
            ring = get_ring(data, i)
            # (3, Nc)
            ring_resample = get_resample(ring.transpose(), num, i)
            # (Nr, 3, Nc)
            curves.append(ring_resample)
        except:
            print('Failed to get any contour')
            pass
    # (Nc, 3, Nr)
    return np.array(curves).transpose()

def decide_curves(rotate_data, rotate_ind, spacing, RESOLUTION):
    """
    Get curves on the transition region.
    Parameters:
    -----------
    rotate_data: np.ndarray
        rotated image data 
    rotate_ind: np.ndarray
        rotated point index
    spacing: list
        the 3D image spacing
    RESOLUTION: float
        the resolution to resample points
    Returns:
    --------
    np.ndarray
        curves on the transition region
    """
    start = int(np.round(np.min(rotate_ind[2]))) + 5
    end = int(np.round(np.quantile(rotate_ind[2], 0.4)))

    max_length = get_length(get_ring(rotate_data, start), spacing)
    i = 1
    while i <= end - start:
        if get_length(get_ring(rotate_data, start + i), spacing) > max_length:
            break
        i += 1
    end = start + i
    end = max(end, start + 4)
    
    center_ring = get_ring(rotate_data, start + (end - start) // 2)
    num_curves = min(600, int(np.round(get_length(center_ring, spacing) / RESOLUTION)))

    # (num_curves, 3, num_slices)
    curves = get_curves(rotate_data, start, end, num_curves)
    return curves