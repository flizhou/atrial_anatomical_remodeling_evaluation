# Author: Fanli Zhou
# Date: 2022-01-23

import sys, math
import numpy as np

sys.path.append('../src')
from segment_laa.help_functions import get_norm, get_distance

def get_val(curves_resample, curvatures, valmax, argmax, val, points, i, j, k, spacing):
    """
    Get resampled curves.
    Parameters:
    -----------
    curves_resample: np.ndarray
        resampled curves
    curvatures: np.ndarray
        curvatures 
    valmax: np.ndarray
        the max value for each position
    argmax:
        the index producing the max value for each position
    val: list
        stores the calculated values of next points to consider 
    points: list
        stores next points to consider  
    i: int
        the row index of the current point
    j: int
        the column index of the current point
    k: int
        the column index of the next point
    spacing: list
        the 3D image spacing
    Returns:
    --------
    list, list
        points, val
    """
    N, M = curvatures.shape
    w1 = - 0.1
    w2 = 2
    w3 = 0.01
    w4 = 0.01
    N0 = 100
    epsilon = 0.001
    points.append(k)

    j0 = curves_resample[argmax[i - 1][k][0][0]][argmax[i - 1][k][0][1]]

    val.append(valmax[i - 1][k] + curvatures[i - 1][j - 1] +\
               w1 * get_norm(np.array([0, 0, 1]).dot(curves_resample[i - 1][j] - j0)) +\
               w2 * math.exp((i - N) / N0) / (get_distance(curves_resample[i - 1][j], j0, spacing) + epsilon) +\
               w3 / (get_distance(curves_resample[i - 1][j], curves_resample[i - 2][k], spacing) + epsilon) +\
               w4 * (M - j) / M )

    return points, val

def get_boundary_points(curves_resample, curvatures, rotation, translate, spacing):
    """
    Get boundary points.
    Parameters:
    -----------
    curves_resample: np.ndarray
        resampled curves
    curvatures: np.ndarray
        curvatures 
    rotation: np.ndarray 
        rotation matrix
    translate: np.ndarray
        translate vector
    spacing: list
        the 3D image spacing
    Returns:
    --------
    np.ndarray
        boundary points
    """
    N, M = curvatures.shape
    # curves_resample.shape = (N, M + 2, 3)
    # argmax.shape = (N + 1, M + 1)
    # argmax (i, j) (i, j >= 1) 
    # -> curvatures (i - 1, j - 1)
    # -> curves_resample (i - 1, j) ( 1 <= j <= M + 1)

    argmax = [[0 for x in range(M + 1)] for x in range(N + 1)]
    valmax = np.zeros((N + 1, M + 1))
    for i in range(N + 1):
        for j in range(M + 1):
            if i == 0 or j == 0:
                pass
            elif i == 1:
                argmax[i][j] = [(i - 1, j), (i - 1, j)]
                valmax[i][j] = curvatures[i - 1, j - 1]
            else:
                points = []
                val = []
                dist = []
                ind = []
                for k in range(1, M + 1):
                    ind.append(k)
                    dist.append(get_distance(curves_resample[i - 2][k],
                                             curves_resample[i - 1][j], 
                                             spacing))
                for p in np.argsort(np.array(dist))[:10]:
                    points, val = get_val(curves_resample, curvatures, valmax, 
                                          argmax, val, points, i, j, ind[p], spacing)

                if len(val) == 0:
                    print(dist)
                ind = points[np.argmax(val)]

                argmax[i][j] = (argmax[i - 1][ind][0], (i - 2, ind))
                valmax[i][j] = max(val)
    
    boundary_ind = [np.argmax(valmax[-1])]
    boundary = []
    for i in range(1, N + 1):

        boundary.append(curves_resample[-i][boundary_ind[-1]])
        boundary_ind.append(argmax[-i][boundary_ind[-1]][1][1])

    points = np.array(boundary).transpose()

    translate_back = np.array([points[0] - translate[0], 
                               points[1] - translate[1], 
                               points[2] - translate[2]])
    back_points = []
    for point in translate_back.transpose():
        back_points.append(point.dot(rotation))
    points = np.array(back_points).transpose()
    
    return points