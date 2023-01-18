# Author: Fanli Zhou
# Date: 2021-06-01

import os
import SimpleITK as sitk
import pandas as pd

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
