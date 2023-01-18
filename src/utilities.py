# Author: Fanli Zhou
# Date: 2021-06-01

import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import pickle


def get_dataset(data):
    """
    Get train, validation and test datasets.
    Parameters:
    -----------
    data: dict
        the whole dataset

    Returns:
    --------
    np.ndarray, np.ndarray, np.ndarray
        trainset, validset, testset
    """
    trainset, validset, testset = data['train'], data['valid'], data['test']
    print(f'trainset size: {len(trainset)}')
    print(f'validset size: {len(validset)}')
    print(f'testset size: {len(testset)}')
    return trainset, validset, testset

# read images
def load_data(path, filename):
    """
    Read nrrd images.
    Parameters:
    -----------
    path: str
        input file path 
    filename: str
        name of the input image

    Returns:
    --------
    SimpleITK Image, np.ndarray 
        image, image data
    """
    img = sitk.ReadImage(f'{path}/{filename}.nrrd')
    img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
    # N*H*W
    img_data = sitk.GetArrayFromImage(img)
    return img, img_data

def results_to_df(data, tag, columns):
    """
    Convert results to pandas.DataFrame.
    Parameters:
    -----------
    data: np.ndarray
        The image data
    tag: str
        tag of the data
    columns: list
        list of column names

    Returns:
    --------
    pandas.DataFrame
        resulted data frame
    """
    data = pd.DataFrame(data) 
    data.columns = columns
    data['tag'] = tag
    return data
    
# crop and zero_pad mri image to 96 * 576 * 576
# crop and zero_pad ct image to 256 * 512 * 512
def resize_data(data, target_size, ct='False'):
    
    if ct != 'True':
        h_crop = (data.shape[1] - target_size[1]) // 2
        w_crop = (data.shape[2] - target_size[2]) // 2
        data = data[:, h_crop : h_crop + target_size[1], w_crop : w_crop + target_size[2]]

    if data.shape[0] < target_size[0]:
        padding = np.zeros([target_size[0] - data.shape[0], *target_size[1:]], dtype='uint8')
    #     print(data.shape, padding.shape)
        data = np.concatenate((data, padding), axis=0)        
    elif data.shape[0] > target_size[0]:
        data = data[: target_size[0], :, :]
    
    return data

# # tests
# data = np.ones([3, 10, 10])
# target = resize_data(data, np.array([4, 2, 2]))
# assert np.alltrue(target.shape == np.array([4, 2, 2]))
# assert np.alltrue(target == np.array([[[1., 1.],[1., 1.]], [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]]]))

def save_nrrd(data, ref_image, path, filename):
    """
    Save nrrd images.
    Parameters:
    -----------
    data: np.ndarray
        The image data
    ref_image: SimpleITK Image
        the referral image
    path: str
        output file path 
    filename: str
        name of the output image
    """
    image = sitk.GetImageFromArray(data)
    image.SetDirection(ref_image.GetDirection())
    image.SetOrigin(ref_image.GetOrigin())
    image.SetSpacing(ref_image.GetSpacing())
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    sitk.WriteImage(image, f'{path}/{filename}.nrrd')
    
def save_pickle(data, path, filename):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(f'{path}/{filename}.pkl', "wb") as f:
        pickle.dump(data, f)
    assert os.path.isfile(f'{path}/{filename}.pkl'), "Image data are not saved."


def normalize_data(data):
    mean = np.mean(data, axis=(1, 2), keepdims=True)
    std = np.std(data, axis=(1, 2), keepdims=True)
    data_norm = (data - mean) / (std + 1e-10) 
    return data_norm

    
def hms_string(sec_elapsed):
    """
    Returns the formatted time 

    Parameters:
    -----------
    sec_elapsed: int
        second elapsed

    Return:
    --------
    str
        the formatted time
    """

    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"
