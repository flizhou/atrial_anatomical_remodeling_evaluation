# Author: Fanli Zhou
# Date: 2022-01-15

import sys
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression

sys.path.append('..')
from utilities import save_nrrd
from segment_laa.help_functions import get_cross

def normalize_vec(v):
    """
    Normalize the input vector.
    Parameters:
    -----------
    v: np.ndarray
        the input vector

    Returns:
    --------
    np.ndarray
        the normalized vector
    """
    return v / np.sqrt(v.dot(v))

def get_normal_vec(plane):
    """
    Get the normal vector of the input plane.
    Parameters:
    -----------
    plane: list
        a collection of points on the input plane

    Returns:
    --------
    np.ndarray
        the normal vector
    """
    lr = LinearRegression().fit(np.array(plane)[:, :-1], np.array(plane)[:, -1])
    points = np.array([[0, 0], [1, 2], [2, 1]])
    points = np.concatenate((points.transpose(), lr.predict(points).reshape(1, 3))).transpose()
    
    v = normalize_vec(get_cross(points[1] - points[0], points[2] - points[0]))
    return v

def get_start_point(ind, z_start=True):
    """
    Get the starting point to find the plane to rotate
    Parameters:
    -----------
    ind: np.ndarray
        points on the object surface
    z_start: bool (default: True)
        whether to decide z axis first
    Returns:
    --------
    tuple
        the starting point
    """
    if z_start:
        z0 = np.min(ind[2]) + 2
        y0 = int(np.round(np.mean(ind.transpose()[ind[2, :] == z0].transpose()[1])))
    else:
        y0 = np.max(ind[1]) - 2
        z0 = int(np.round(np.mean(ind.transpose()[ind[1, :] == y0].transpose()[2])))
        
    return (y0, z0)

def get_plane_points(ind, y0, z0, z_start=True, n=5):
    """
    Get points on a plane 
    Parameters:
    -----------
    ind: np.ndarray
        points on the object surface.
    y0: int
        the y axis of the starting point
    z0: int
        the z axis of the starting point
    z_start: bool (default: True)
        whether z axis was decided first
    n: int (default: 5)
        helps calculate points on the plane
    Returns:
    --------
    matrix
        the plane to rotate
    """
    # find the plane to rotate
    plane = []
    for i in range(2 * n):
        for j in range(-n, n):
            try:
                if z_start:
                    plane.append([np.max(ind.transpose()[((ind[1, :] == y0 + j)* 1 * (ind[2, :] == z0 + i)) == True].transpose()[0]),
                                  y0 + j, 
                                  z0 + i])
                else:
                    plane.append([np.max(ind.transpose()[((ind[1, :] == y0 - i)* 1 * (ind[2, :] == z0 + j)) == True].transpose()[0]),
                                  y0 - i, 
                                  z0 + j])
                    
            except:
                pass
    return np.mat(plane)

def get_rotation_matrix(v):
    """
    Get the rotation matrix for a vector to [0, 0, -1].
    Parameters:
    -----------
    v: np.array
        the input vector
    Returns:
    --------
    np.array
        the rotation matrix
    """
    z0 = np.array([0, 0, -1])
    # the angel between v and z0
    theta = np.arccos(z0.dot(v))
    # rotation axis
    axis = normalize_vec(get_cross(z0, v))
    # rotation vector
    axis_theta = - theta * axis
    # rotation matrix
    rotation, _ = cv2.Rodrigues(axis_theta)
    
    return rotation

def ind_to_array(ind, shape):
    """
    Estimate 3D float points to 3D nt points.
    Parameters:
    -----------
    ind: np.array
        3D float points
    shape: list
        data shape
    Returns:
    --------
    np.array
        the estimated 3D int points
    """
    # input (3, N) 
    data = np.zeros(shape)
    for point in np.round(ind).astype('int').transpose():
        data[point[0]][point[1]][point[2]] = 255
    return data.astype('uint8')

def rotate_img(img_ind, shape, z_start=True, img=None, path=None):
    """
    Rotate the target to make the bottom plane parallel to the P-S plane.
    Parameters:
    -----------
    img_ind: np.array
        3D float points
    shape: list
        data shape
    z_start: bool (default: True)
        whether to decide z axis first
    img: SimpleITK Image (default: None)
        referral image
    path: str (default: None)
        the output path
    Returns:
    --------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
        rotated image data, rotated point index, rotation matrix, translate vector
    """
    # find the starting point
    y0, z0 = get_start_point(img_ind, z_start=z_start)
    
    # rotate laa so that the bottom is parallel to XOY
    plane = get_plane_points(img_ind, y0, z0, z_start=z_start)

    # get normal vector of the plane
    v = get_normal_vec(plane)
    print(f'The normal vector of the origin plane is {v}')

    rotation = get_rotation_matrix(plane, v)
    
    # rotate data
    rotate_ind = []
    for point in img_ind.transpose():
        rotate_ind.append(rotation.dot(point))
    rotate_ind = np.array(rotate_ind).transpose()
    
    # translate data
    translate = np.mean(img_ind, axis=1) - np.mean(rotate_ind, axis=1)
    rotate_ind[0] += translate[0]
    rotate_ind[1] += translate[1]
    rotate_ind[2] += translate[2]

    # convert index to image data array
    rotate_data = ind_to_array(rotate_ind, shape)
    
    if img is not None:
        save_nrrd(rotate_data.astype('uint8'), img, path, 'rotate')
    return rotate_data, rotate_ind, rotation, translate