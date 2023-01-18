# Author: Fanli Zhou
# Date: 2022-02-01

import math
import numpy as np
import alphashape
import SimpleITK as sitk
from skimage import measure
from sklearn.decomposition import PCA

def get_volume(ob, spacing):
    """
    Get volume of the object.
    Parameters:
    -----------
    ob: np.ndarray
        points of the targeted object
    spacing: list
        the 3D image spacing

    Returns:
    --------
    float
        volume
    """
    return np.sum(ob) // 255 * np.prod(spacing)

def get_edge(ob1, ob2):
    """
    Get the boundary of two objects.
    Parameters:
    -----------
    ob1: np.ndarray
        the first object
    ob2: np.ndarray
        the second object

    Returns:
    --------
    np.ndarray
        points on the boundary
    """
    half_ob2 = ob2 // 2
    remain_1 = ob1[:, :, :-1] - half_ob2[:, :, 1:]
    edge_1 = np.array(np.where(remain_1 == 128))
    remain_2 = ob1[:, :, 1:] - half_ob2[:, :, :-1]
    edge_2 = np.array(np.where(remain_2 == 128))
#     save_to_img(edge.transpose(), 'edge', shape)
    if len(edge_1.transpose()) > len(edge_2.transpose()):
        return edge_1
    else:
        return edge_2

def pca_trans(points, spacing):
    """
    Get PCA transformation of input points.
    Parameters:
    -----------
    points: np.ndarray
        input points
    spacing: list
        the 3D image spacing

    Returns:
    --------
    np.ndarray
        transformed points
    """
    points = np.array([points[i] * spacing[2 - i] for i in range(3)]).transpose()
    pca = PCA(n_components=2).fit(points)
    trans = pca.transform(points) 
    return trans

def get_eccentricity(diameters):
    """
    Get eccentricity.
    Parameters:
    -----------
    diameters: np.ndarray
        input diameters

    Returns:
    --------
    float
        eccentricity
    """
    temp = diameters[1] ** 2 / diameters[0] ** 2
    if temp <= 1:
        eccentricity = math.sqrt(1 - temp)
    else:
        eccentricity = math.sqrt(1 - 1 / temp)
        
    return eccentricity

def get_diameter(points):
    """
    Get the length and width of the given ostium.
    Parameters:
    -----------
    points: np.ndarray
        points on the given ostium

    Returns:
    --------
    np.ndarray
        diameters
    """
    diameters = np.max(points, axis=0) - np.min(points, axis=0)
    return diameters

def get_area(points, alpha=1):
    """
    Get the best alpha shape of the given ostium.
    Parameters:
    -----------
    points: np.ndarray
        points on the given ostium
    alpha: float (default: 1)
        the alpha value
    Returns:
    --------
    float
        area
    """
    area = alphashape.alphashape(points, alpha=alpha).area
    return area

def get_surface_area(ob, spacing):
    """
    Get surface area of the object.
    Parameters:
    -----------
    ob: np.ndarray
        points of the targeted object
    spacing: list
        the 3D image spacing

    Returns:
    --------
    float
        surface area
    """
    verts, faces, _, _ = measure.marching_cubes(ob, spacing=spacing)
    surface_area = measure.mesh_surface_area(verts, faces)
    return surface_area

def sort_pv(labels, nums):
    """
    Sort and label PVs.
    Parameters:
    -----------
    labels: np.ndarray
        labeled points
    nums: np.ndarray
        labels of labeled points

    Returns:
    --------
    np.ndarray
        sorted labels
    """
    pv = []
    for num in nums + 1:
        pv.append(np.mean(np.nonzero(labels * (labels == num)), axis=1))
    pv = np.array(pv)
    sort_1 = np.argsort(pv[:, 2])
    sort_2 = np.argsort(pv[:, 0])
    if np.where(sort_2 == sort_1[0]) > np.where(sort_2 == sort_1[1]):
        sort_1[0], sort_1[1] = sort_1[1], sort_1[0]
    if np.where(sort_2 == sort_1[2]) > np.where(sort_2 == sort_1[3]):
        sort_1[2], sort_1[3] = sort_1[3], sort_1[2]
    return nums[sort_1]