# Author: Fanli Zhou
# Date: 2021-07-15

import sys
import SimpleITK as sitk
import numpy as np
from scipy import interpolate

sys.path.append('..')
from utilities import save_nrrd

def get_cross(v1, v2):
    """
    Get cross product of input vectors
    Parameters:
    -----------
    v1: np.ndarray
        the first vector
    v2: np.ndarray
        the second vector

    Returns:
    --------
    np.ndarray
        the cross product
    """
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])

def get_distance(p1, p2, spacing):
    """
    Get the distance between two points
    Parameters:
    -----------
    p1: np.ndarray
        the first point
    p2: np.ndarray
        the second point
    spacing: list
        the 3D image spacing

    Returns:
    --------
    float
        the distance
    """
    if len(p1) == 2:
        # 2D distance
        return np.sqrt((spacing[1] * (p1[1] - p2[1])) ** 2 + (spacing[2] * (p1[0] - p2[0])) ** 2)
    # 3D distance
    return np.sqrt((spacing[0] * (p1[2] - p2[2])) ** 2 + (spacing[1] * (p1[1] - p2[1])) ** 2 + (spacing[2] * (p1[0] - p2[0])) ** 2)

def get_length(ring, spacing):
    """
    Get the length of the input ring
    Parameters:
    -----------
    ring: np.ndarray
        a collection of points on the ring
    spacing: list
        the 3D image spacing

    Returns:
    --------
    float
        the length
    """
    length = 0
    for i in range(len(ring) - 1):
        length += get_distance(ring[i], ring[i + 1], spacing)
    length += get_distance(ring[0], ring[-1], spacing)

    return length

def get_resample(ring, num, z=None):
    """
    Interpolate the ring with cubic spline and densely resample points on the ring
    Parameters:
    -----------
    ring: np.ndarray
        a collection of points on the ring
    spacing: list
        the 3D image spacing

    Returns:
    --------
    float
        the length
    """
    # input (3, N) 
    tck, u = interpolate.splprep(ring, s=0)
    unew = np.linspace(0, 1.0, num=num, endpoint=False)
    ring_resample = np.array(interpolate.splev(unew, tck))   
    if z is not None:
        ring_resample = np.stack([*ring_resample, np.full(len(ring_resample[0]), z)])
    # (3, N)
    return ring_resample

def get_norm(p):
    """
    Get the vector norm
    Parameters:
    -----------
    p: np.ndarray
        the vector
        
    Returns:
    --------
    float
        the norm
    """
    return np.sqrt(np.sum(p * p))

def save_to_img(points, filename, shape, img, path):
    """
    Save points to image
    Parameters:
    -----------
    points: np.ndarray
        a collection of points
    filename: str
        name of the output file
    shape: list
        the output data shape
    img: SimpleITK Image
        referral image
    path: str
        the output path
    """
    # input (N, 3)
    data = np.zeros(shape)
    for i in range(len(points)):
        try:
            data[int(points[i, 0])][int(points[i, 1])][int(points[i, 2])] = 1
        except: pass
    print(np.sum(data))
    save_nrrd(data.astype('uint8'), img, path, filename)
