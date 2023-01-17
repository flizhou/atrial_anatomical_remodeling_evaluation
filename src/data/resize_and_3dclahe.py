import sys
import numpy as np
from time import time
from tqdm import tqdm
from skimage import exposure, util

sys.path.append('..')
from utilities import load_data, resize_data, save_nrrd, hms_string

def apply_3dclahe(data, kernel_div=[2, 5, 5], clip_limit=0.9, clip=False):
# code modified from https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_adapt_hist_eq_3d.html#sphx-glr-auto-examples-color-exposure-plot-adapt-hist-eq-3d-py
    
    if clip:
        data = np.clip(data,
                       np.percentile(data, 5),
                       np.percentile(data, 95)).astype('uint8')

    kernel_size = np.array((
        data.shape[0] // kernel_div[0],
        data.shape[1] // kernel_div[1],
        data.shape[2] // kernel_div[2]))
    data_ahe = \
    exposure.equalize_adapthist(data, kernel_size=kernel_size, clip_limit=clip_limit)
    data_ahe = (data_ahe * 255).astype('uint8')
    return data_ahe

def preprocess_image(input_path, output_path, dataset, filename, target_size, ct='False'):

    start = time()
    for i in tqdm(range(len(dataset))):

        # N*H*W
        image, data = load_data(f'{input_path}/{dataset[i]}', filename)
        
        # crop and zero_pad
        data = resize_data(data, target_size, ct)
        # apply 3D CLAHE
        data_ahe = apply_3dclahe(data, [5, 5, 5])

        save_nrrd(data_ahe, image, f'{output_path}/{dataset[i]}', filename)
 
    print(f"Preprocessing took: {hms_string(time() - start)}")
