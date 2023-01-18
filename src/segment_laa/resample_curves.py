# Author: Fanli Zhou
# Date: 2022-01-22

import sys
import numpy as np

sys.path.append('../src')
from segment_laa.help_functions import get_norm, get_length, get_resample

def get_curvature(p1, p2, p3):
    """
    Get the curvature of the input curve.
    Parameters:
    -----------
    p1: np.ndarray
        the first point 
    p2: np.ndarray
        the second point 
    p3: np.ndarray
        the third point 
    Returns:
    --------
    float
        curvature
    """
    x = (p3 - p2) - (p2 - p1)
    return get_norm(x)

def decide_resamples(curves, rotate_data, spacing):
    """
    Get resampled curves.
    Parameters:
    -----------
    curves: np.ndarray
        the first point 
    rotate_data: np.ndarray
        rotated image data 
    spacing: list
        the 3D image spacing
    Returns:
    --------
    np.ndarray, np.ndarray
        resampled curves, curvatures
    """
    length = []
        
    for i in range(curves.shape[0]):
        points = curves[i].transpose()
        dist = get_length(points, spacing)
        if dist > 0:
            length.append(dist) 

    num_resample = max(min(600, int(np.round(np.min(length) / RESOLUTION))), 100)
    resamples = []
    size = []
    for i in range(curves.shape[0]):
        # (3, num_points) -> (num_points, 3)
        curve = get_resample(curves[i], num_resample).transpose()
        resample = []
        for point in curve:
            int_point = np.round(point).astype('int')
            for ind in (int_point, 
                        int_point + [1, 0, 0], int_point + [-1, 0, 0], 
                        int_point + [0, 1, 0], int_point + [0, -1, 0],
                        int_point + [0, 0, 1], int_point + [0, 0, -1]):
                try:
                    if rotate_data[ind[0]][ind[1]][ind[2]] != 0:
                        resample.append(point)
                        break
                except:
                    pass
        if len(resample) > 0:
            size.append(len(resample))
            resamples.append(np.array(resample))  

    num_points = min(size)
    curves_resample = []
    curvatures = []
    for curve in resamples:
        # (num_points, 3)
        curves_resample.append(curve[len(curve) - num_points:])

        curvature = []
        for j in range(1, num_points - 1):
            curvature.append(
                get_curvature(curves_resample[-1][j - 1], 
                              curves_resample[-1][j], 
                              curves_resample[-1][j + 1])
            )
        curvatures.append(curvature)

    # (num_curves, num_points, 3)
    curves_resample = np.array(curves_resample)
    # (num_curves, num_points - 2)
    curvatures = np.array(curvatures)

    print(f'curves_resample: {curves_resample.shape}')
    print(f'curvatures: {curvatures.shape}')
    
    return curves_resample, curvatures