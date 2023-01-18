# Author: Fanli Zhou
# Date: 2022-01-23

import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import skimage 

sys.path.append('../src')
from utilities import load_data, save_nrrd
from segment_laa.rotate_target import rotate_img
from segment_laa.find_curves import decide_curves
from segment_laa.resample_curves import decide_resamples
from segment_laa.decide_boundary import get_boundary_points

def plot_results(points, lr, img_ind):
    """
    Plot boundary points.
    Parameters:
    -----------
    points: np.ndarray
        boundary points
    lr: np.ndarray
        the boundary plane 
    img_ind: np.ndarray
        points on the object surface
    """
    x = np.linspace(min(points[0]) - 30, max(points[0]) + 30, num=1000)
    y = np.linspace(min(points[1]) - 30, max(points[1]) + 30, num=1000)
    x, y = np.array(np.meshgrid(x, y))
    x = x.flatten()
    y = y.flatten()
    z = lr.predict(np.array([x, y]).transpose())

    plt.plot(x, y, z, c='y', alpha=0.5)
    plt.plot(*img_ind, alpha=0.5)
    plt.plot(points[0], points[1], points[2], c='r', alpha=0.5)

    plt.show()

def get_laa(data, origin_data=None):
    """
    Divide the object with the plane to decide which part to remove
    when origin_data is given or decide which part in the leftover to 
    keep when origin_data is None.
    Parameters:
    -----------
    data: np.ndarray
        the input object 
    origin_data: np.ndarray (default: None)
        the image data 
    Returns:
    --------
    np.ndarray
        the points to keep
    """
    labels = skimage.measure.label(data.astype('uint8'), background=0)
    regions = skimage.measure.regionprops(labels)
    num = np.argmax(np.array([region.area for region in regions])) + 1
    laa_data = 255 * labels * (labels == num) // num
    if origin_data is not None:
        laa_data = origin_data - laa_data
        print(f'Found {np.unique(labels)[-1]} groups and label {num} is removed.')
    else:
        print(f'Found {np.unique(labels)[-1]} groups and label {num} is kept.')
        
    return laa_data
    
def save_laa(points, img_ind, img, img_data, shape, path, plot=True):
    """
    Decide LAA and save LAA.
    Parameters:
    -----------
    points: np.ndarray
        the input object 
    img_ind: np.ndarray
        points on the object surface
    img: SimpleITK Image
        referral image
    img_data: np.ndarray
        the image data
    shape: list
        data shape
    path: str
        the output path
    plot: bool (default: True)
        whether to plot results
    """    
    lr = LinearRegression().fit(points[:-1].transpose(), points[-1])
    
    if plot:
        plot_results(points, lr, img_ind)
    
    non_laa = []
    for point in img_ind.transpose():
        if point[-1] < lr.predict(point[: -1].reshape(1, 2)):
            non_laa.append(point)

    non_laa = np.round(np.array(non_laa)).astype('int')
    non_laa_data = np.zeros(shape)
    for point in non_laa:
        non_laa_data[point[0]][point[1]][point[2]] = 1
        
    print(f'Divide the object with the plane to decide which part to remove ...')
    laa_data = get_laa(non_laa_data, img_data)
    
    print(f'Decide which part in the leftover to keep ...')
    laa_data = get_laa(laa_data)

    print(f'Saving {path}/laa.nrrd')
    save_nrrd(laa_data.astype('uint8'), img, path, 'laa')

def segment_pre_laa(path_to_root, raw, folder, y_start=set()):
    """
    Segment pre_LAA to get LAA.
    Parameters:
    -----------
    path_to_root: str
        path to the root folder
    raw: str
        path to the raw data folder
    folder: str
        path to the image folder
    y_start: set (default: set())
        the set of y_start samples
    """
    df = pd.read_excel(f'{path_to_root}/{raw}/{folder}.xlsx')
    samples = np.array(df.iloc[:, 0].dropna())
    for sample in samples:
        files = os.listdir(f'{path_to_root}/{raw}/{folder}/{sample}')
        if 'pre_laa.nrrd' in files and 'laa.nrrd' not in files:
            path = f'{path_to_root}/{raw}/{folder}/{sample}'
            img, img_data = load_data(path, 'pre_laa')
            img_ind = np.array(np.nonzero(img_data))
            spacing = img.GetSpacing()
            shape = img_data.shape
            print(f'Segmenting {folder}/{sample}')

            # rotate the object so that the bottom is parallel to XOY
            z_start = True
            if sample in y_start:
                z_start = False
            rotate_data, rotate_ind, rotation, translate = \
            rotate_img(img_data, img_ind, shape, z_start=z_start)

            # decide curves to resample
            curves = decide_curves(rotate_data, rotate_ind, spacing)

            # resample curves and get curvatures
            curves_resample, curvatures = decide_resamples(curves, rotate_data, spacing)

            # use dynamic programing to decide boundary points
            points = get_boundary_points(curves_resample, curvatures, rotation, translate, spacing)

            # decide the cutting plane to segment the object and save the target
            save_laa(points, img_ind, img, img_data, shape, path, False)

            print('===============================================')    
