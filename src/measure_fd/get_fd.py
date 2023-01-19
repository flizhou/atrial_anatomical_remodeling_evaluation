# Author: Fanli Zhou
# Date: 2022-02-10

import sys
from measure_fd.get_fd_bc import get_fd_bc
from measure_fd.get_fd_fft import get_fd_fft

def get_fd(path_to_root, raw, samples, label, func_name, box_max=50):
    """
    Get FFT FDs.
    Parameters:
    -----------
    path_to_root: str
        path to the root folder
    raw: str
        path to the raw data folder
    samples: list
        image folders
    label: str
        filename of the image
    func_name: str
        function name 'fd_bc' or 'fd_fft'
    box_max: int (default: 50)
        maximun box size
    """    
    for sample in samples:
        if func_name == 'fd_bc':
            result = get_fd_bc(path_to_root, raw, sample, label, box_max)
        elif func_name == 'fd_fft':             
            result = get_fd_fft(path_to_root, raw, sample, label)
        result.to_csv(f'{path_to_root}/results/{label}_{func_name}_{sample}.csv', index=False)