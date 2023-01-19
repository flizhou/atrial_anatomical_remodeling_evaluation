# Author: Fanli Zhou
# Date: 2022-02-10

import sys
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure

sys.path.append('../src')
from utilities import results_to_df, load_data, hms_string

# algorithm based on http://paulbourke.net/fractals/cubecount/#offsets
def box_count(img_ind_trans, dim, box_max, box_min=4, box_fact=1.5, max_offset=10, mute=True):
    """
    Calculate FD with box counting method.
    Parameters:
    -----------
    img_ind_trans: np.ndarray
        object data
    dim: int
        dimension
    box_max: int
        maximun box size
    box_min: int (default: 4)
        minimun box size
    box_fact: float (default: 1.5)
        box size increase rate
    max_offset: int (default: 10)
        maximun offset
    mute: bool (default: True) 
        whether to plot log (1/s) vs. log (N(s))
    Returns:
    --------
    float
        box counting FD
    """
    box_size = box_min
    results = []
    while box_size < box_max:

        offsets = np.random.randint(0, box_size, size=[min(max_offset, int(box_size ** dim)), dim])
        min_box_count = np.inf
        
        for i in range(len(offsets)):
            # add offset
            img_ind_offset = np.array([img_ind_trans[j] + offsets[i][j] for j in range(dim)])
            # find the box edge for each point and caculate the number of unique box
            box_count = np.unique(
                (img_ind_offset // box_size).astype('int').transpose(), 
                axis=0
            ).shape[0]
            if box_count < min_box_count:
                min_box_count = box_count
                if not mute:
                    print(f'Computing box size: {box_size}, offset: {i} / {len(offsets)}, minimum box count: {min_box_count}')
        
        results.append([box_size, min_box_count, np.log(1 / box_size), np.log(min_box_count)])        
        box_size = int(box_size * box_fact)
        
    results = np.array(results).transpose()
    coeff = np.polyfit(results[2], results[3], 1)

    if not mute:
        plot_fd(results[2], results[3], coeff)
        
    return coeff[0]

def get_contours(data):
    """
    Get contours of each slide.
    Parameters:
    -----------
    data: np.ndarray
        object data

    Returns:
    --------
    np.ndarray
        contours
    """
    contours = []
    for z in range(data.shape[0]):
        edge = measure.find_contours(data[z])
        if edge != []:
            edge = np.concatenate(edge).transpose()
            edge = np.stack([np.full(len(edge[0]), z), edge[0], edge[1]]).transpose()
            contours.append(edge)
    # output (3, N)
    return np.concatenate(contours).transpose()

def convert_contours_to_ts(contours, center, spacing):
    """
    Convert contours data to time series data.
    Parameters:
    -----------
    contours: np.ndarray
        contours data
    center: np.ndarray
        center data of each contour
    spacing: list
        the 3D image spacing

    Returns:
    --------
    np.ndarray
        time series data
    """
    ts = np.sqrt([
        ((contours[0][i] - center[0]) * spacing[2]) ** 2 +\
        ((contours[1][i] - center[1]) * spacing[0]) ** 2 +\
        ((contours[2][i] - center[2]) * spacing[1]) ** 2\
        for i in range(len(contours[0]))])
    return np.stack([np.arange(len(ts)), ts])

def plot_fd(x, y, coeff):
    """
    Plot log (1/s) vs. log (N(s)).
    Parameters:
    -----------
    x: np.ndarray
        log (1/s)
    y: np.ndarray
        log (N(s))
    coeff: np.ndarray
        fitted coefficients
    """
    _, ax = plt.subplots(figsize = (8, 6))
    ax.plot(x, y, label = "Measured", marker='o')
    ax.set_ylabel("$\log (N(s))$")
    ax.set_xlabel("$\log (1/s)$")
    fitted_y_vals = np.polyval(coeff, x)
    ax.plot(x, fitted_y_vals, "k--", label = f"Fit: {np.round(coeff[0], 3)}x + {np.round(coeff[1], 3)}")
    ax.legend()
    plt.show()
    
def get_fd_bc(path_to_root, raw, folder, filename, box_max, box_min=4, box_fact=1.5, max_offset=10, mute=True):
    """
    Get 1D and 2D box counting FDs.
    Parameters:
    -----------
    path_to_root: str
        path to the root folder
    raw: str
        path to the raw data folder
    folder: str
        path to the image folder
    filename: str
        filename of the image
    box_max: int
        maximun box size
    box_min: int (default: 4)
        minimun box size
    box_fact: float (default: 1.5)
        box size increase rate
    max_offset: int (default: 10)
        maximun offset
    mute: bool (default: True) 
        whether to plot log (1/s) vs. log (N(s))

    Returns:
    --------
    pandas.DataFrame
        resulted data frame
    """        
    df = pd.read_excel(f'{path_to_root}/{raw}/{folder}.xlsx')
    samples = np.array(df.iloc[:, 0].dropna())
    fd = []
    start = time()
    for sample in tqdm(samples):

        path = f'{path_to_root}/{raw}/{folder}/{sample}'
        if filename != 'pre_la-laa':
            img, img_data = load_data(path, filename)
        else:
            img, pre_la_data = load_data(path, 'pre_la')
            _, laa_data = load_data(path, 'laa')
            img_data = pre_la_data - laa_data
        img_ind = np.array(np.nonzero(img_data))
        spacing = img.GetSpacing()
        
        center = np.mean(img_ind, axis=1)
        contours = get_contours(img_data)
        line_trans = convert_contours_to_ts(contours, center, spacing)
        vol_trans = [img_ind[i] * spacing[2 - i] * 10 for i in range(3)]
            
        fd.append([
            f'{folder}/{sample}/{filename}',
            box_count(line_trans, 2, box_max, box_min=box_min,
                      box_fact=box_fact, max_offset=max_offset, mute=mute),
            box_count(vol_trans, 3, box_max, box_min=box_min, 
                      box_fact=box_fact, max_offset=max_offset, mute=mute)
        ])
        if not mute:
            print(f'\n{fd[-1][0]}:\n1D FD: {fd[-1][1]}\n2D FD: {fd[-1][2]}')
    
    print(f"Preprocessing took: {hms_string(time() - start)}")

    return results_to_df(fd, folder, ['sample', 'fd_bc_1d', 'fd_bc_2d'])    
#     return results_to_df(fd, folder, ['sample', 'fd_bc_1d'])