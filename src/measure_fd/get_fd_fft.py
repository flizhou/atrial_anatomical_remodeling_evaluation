# Author: Fanli Zhou
# Date: 2022-02-10
# This code is revised from https://uk.mathworks.com/matlabcentral/fileexchange/6964-calculation-of-fractal-dimension-of-a-3d-volume-using-fft?s_tid=mwa_osa_a

import sys
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('../src')
from utilities import results_to_df, load_data, hms_string
from measure_fd.get_fd_bc import plot_fd

def convert_index(ind, M, P):
    """
    Convert index.
    Parameters:
    -----------
    ind: np.ndarray
        index
    M: int
        the last dimension of data shape
    P: int
        the first dimension of data shape

    Returns:
    --------
    list
        converted index
    """
    return [list(map(lambda x: x % (M * P) % P, ind)),
            list(map(lambda x: x % (M * P) // P, ind)),
            list(map(lambda x: x // (M * P), ind))]

def count_helper(grouped, sum_val, count, psd, M, P):
    """
    Help function for get_fd_fft to update sum_val and count.
    Parameters:
    -----------
    grouped: GroupBy object
        grouped pd.DataFrame
    sum_val: np.ndarray
        accumulation PSD for all directions
    count: np.ndarray
        number of PSD for all directions
    psd: np.ndarray
        a 2D array of size 24x12 which stores the FDs in 24x12 directions
    M: int
        the last dimension of data shape
    P: int
        the first dimension of data shape
    """
    for name in grouped.groups.keys():
        ind = np.array(grouped.get_group(name).index)

        for ind_slice in np.array_split(ind, 6):
            sum_val[name] += np.sum(psd[tuple(convert_index(ind_slice, M, P))])
            count[name] += len(ind_slice)          

def get_fd_fft(path_to_root, raw, folder, filename, mute=True):
    """
    Get FFT FDs.
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
    mute: bool (default: True) 
        whether to plot log (1/s) vs. log (N(s))

    Returns:
    --------
    pandas.DataFrame
        resulted data frame
    """    
    # number of points that the radial line is evenly divided
    NUM_RAD = 30
    
    df = pd.read_excel(f'{path_to_root}/{raw}/{folder}.xlsx')
    samples = np.array(df.iloc[:, 0].dropna())
    fd = []
    start = time()
    for sample in tqdm(samples):

        path = f'{path_to_root}/{raw}/{folder}/{sample}'
        if filename != 'pre_la-laa':
            _, img_data = load_data(path, filename)
        else:
            _, pre_la_data = load_data(path, 'pre_la')
            _, laa_data = load_data(path, 'laa')
            img_data = pre_la_data - laa_data
        P, N, M = img_data.shape
        # x coordinate of center point
        xctr = 1 + N >> 1
        # y coordinate of center point
        yctr = 1 + M >> 1
        # z coordinate of center point
        zctr = 1 + P >> 1
        
        fim = np.fft.fftshift(np.fft.fftn(np.double(img_data) - np.mean(img_data)))
        psd = np.real(np.log(fim * np.conj(fim) + 10 ** (-6)))
        
        z = zctr - np.arange(P)
        y = yctr - np.arange(M)
        x = np.arange(N) - xctr

        z, y = np.meshgrid(z, y)
        z, y = z.flatten(), y.flatten()

        zyx = np.concatenate(
            (np.tile(np.stack((z, y)), N), 
             np.repeat(x, P * M).reshape(1, P * M * N))
        )[::-1]

        rmax = np.log(np.min([M, N, P]) / 2) 
        r = np.sqrt(np.sum(zyx * zyx, axis=0))
        rho = np.log(r)
        
        radSN = (2 * NUM_RAD * rho / rmax).astype(int)
        radSN[radSN > 2 * NUM_RAD - 1] = 2 * (NUM_RAD - 1) - 1
        radSN = pd.DataFrame({'radSN': radSN})
        grouped = radSN[radSN['radSN'] >= 5].groupby(by=['radSN'])
        
        # accumulation PSD for all directions
        radius = np.zeros((2 * NUM_RAD, 1))
        # number of PSD for all directions
        radCount = np.zeros((2 * NUM_RAD, 1))
        count_helper(grouped, radius, radCount, psd, M, P)

        # compute average slope over all directions and radius
        yval = []
        tempr = []

        for i in range(5, 2 * NUM_RAD):
            if radCount[i] > 0:
                yval.append(radius[i] / radCount[i])
                tempr.append((i - 1) * rmax / (2 * NUM_RAD))

        coeff = np.polyfit(tempr, yval, 1)
        if not mute:
            plot_fd(tempr, yval, coeff)
        fd.append([f'{folder}/{sample}/{filename}',
                   (11 + coeff[0][0]) / 2])
    print(f"Preprocessing took: {hms_string(time() - start)}")
    return results_to_df(fd, folder, ['sample', 'fd_ftt'])