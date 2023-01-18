# Author: Fanli Zhou
# Date: 2022-02-01

import sys
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import skimage
from tqdm import tqdm

sys.path.append('../src')
from utilities import results_to_df, load_data
from measure_geo_features.get_geo_features import get_volume, get_radius, \
    get_surface_area, get_edge, pca_trans, get_diameter, get_eccentricity, \
        get_area, sort_pv

def get_la_info(pre_la, laa, spacing):
    """
    Get geometric features of LA.
    Parameters:
    -----------
    pre_la: np.ndarray
        points of pre_LA
    laa: np.ndarray
        points of LAA
    spacing: list
        the 3D image spacing

    Returns:
    --------
    list
        measured geometric features
    """
    laa_vol = get_volume(laa, spacing)
    la_vol = get_volume(pre_la, spacing)
    radius = get_radius(la_vol - laa_vol)
    la_laa_surface_area = get_surface_area(pre_la - laa, spacing)
    laa_surface_area = get_surface_area(laa, spacing)
    return [
        la_vol,
        laa_vol,
        la_vol - laa_vol,
        la_laa_surface_area,
        laa_surface_area,
        radius
    ]    

def get_la_laa_info(pre_la, laa, spacing):
    """
    Get geometric features of LAA.
    Parameters:
    -----------
    pre_la: np.ndarray
        points of pre_LA
    laa: np.ndarray
        points of LAA
    spacing: list
        the 3D image spacing

    Returns:
    --------
    list
        measured geometric features
    """
    edge = get_edge(pre_la - laa, laa)
    trans = pca_trans(edge, spacing)
    diameters = get_diameter(trans)
    eccentricity = get_eccentricity(diameters)
    return [
        * diameters,
        eccentricity,
        get_area(trans)
    ]  

def get_la_pv_info(mask, pre_la, spacing, pv_num='all', alpha=1):
    """
    Get geometric features of PVs.
    Parameters:
    -----------
    mask: np.ndarray
        points of pre_LA+PVs
    pre_la: np.ndarray
        points of pre_LA
    spacing: list
        the 3D image spacing
    pv_num: str (default: 'all')
        which PV to process
    alpha: float (default: 1)
        the alpha value

    Returns:
    --------
    list
        measured geometric features
    """    
    labels = skimage.measure.label(mask - pre_la, background=0)
    regions = skimage.measure.regionprops(labels)
    result = []
    nums = np.argsort(np.array([region.area for region in regions]))[-4:]
    nums = sort_pv(labels, nums)
    for num in nums + 1:
        if pv_num == 'all' or num == nums[pv_num - 1] + 1:
            pv = (255 * labels * (labels == num) // num).astype('uint8')
            edge = get_edge(pre_la, pv)
            trans = pca_trans(edge, spacing)            
            diameters = get_diameter(trans)
            eccentricity = get_eccentricity(diameters)
            result.extend([
                * diameters,
                eccentricity,
                get_area(trans, alpha=alpha)
            ])

    return result

def get_ob_info(path_to_root, raw, folder, col, ob='la'):
    """
    Get geometric features of targets.
    Parameters:
    -----------
    path_to_root: str
        path to the root folder
    raw: str
        path to the raw data folder
    folder: str
        path to the image folder
    col: list
        column names
    ob: str (default: 'la')
        the target object

    Returns:
    --------
    pandas.DataFrame
        resulted data frame
    """        
    df = pd.read_excel(f'{path_to_root}/{raw}/{folder}.xlsx')
    samples = np.array(df.iloc[:, 0].dropna())
    results = []
    for sample in tqdm(samples):
        path = f'{path_to_root}/{raw}/{folder}/{sample}'
        img, pre_la = load_data(path, 'pre_la')
        spacing = img.GetSpacing()
        results.append([f'{folder}/{sample}'])
        if ob == 'la':
            _, laa = load_data(path, 'laa')
            results[-1].extend(get_la_info(pre_la, laa, spacing))
        elif ob == 'la_laa':
            _, laa = load_data(path, 'laa')
            results[-1].extend(get_la_laa_info(pre_la, laa, spacing))
        elif ob == 'la_pa':
            img, mask = load_data(path, 'mask')
            results[-1].extend(get_la_pv_info(mask, pre_la, spacing))
    
    results = results_to_df(results, folder, col)
    return results
