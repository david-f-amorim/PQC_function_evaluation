"""
Collection of functions relating to handling resources provided as part of the *pqcprep* package. 
"""
import numpy as np 
from importlib.resources import as_file, files
import os 

def load_data_H23(name):
    """
    Load data corresponding to key results from [Hayes 2023](https://arxiv.org/pdf/2306.11073). 

    This can be useful to compare QCNN performance to the techniques used in the paper. 

    Arguments:
    ---
    - **name** : *str*

        Name of the file to be loaded. Options are:

        - `'mismatch_QGAN_12'` : mismatch after each epoch when training the QGAN with 12 layers 

        - `'mismatch_QGAN_20'` : mismatch after each epoch when training the QGAN with 20 layers 

        - `'amp_state_GR'` : statevector after amplitude preparation with the Grover-Rudolph algorithm 

        - `'amp_state_QGAN'` : statevector after amplitdude preparation with the QGAN 

        - `'psi_LPF_processed'` : phase function encoded via linear piecewise functions 

        -  `'full_state_GR'` : full waveform (amplitude and phase) encoded via the Grover-Rudolh algorithm and linear piecewise functions 

        - `'full_state_QGAN'` :  full waveform (amplitude and phase) encoded via the QGAN and linear piecewise functions 

    Returns:
    ---
    - **arr** : *array_like* 

        Array corresponding to the data specified in `name`.         

    """
    package="pqcprep"
    resource="resources"
    name_list =["mismatch_QGAN_12","mismatch_QGAN_20", "amp_state_GR", "amp_state_QGAN", "full_state_GR","full_state_QGAN", "psi_LPF_processed"] 
  
    if name in name_list:
        with as_file(files(package).joinpath(os.path.join(resource, name +".npy"))) as path:
            arr = np.load(path)
        return arr 
    else:
        raise FileNotFoundError('No such file. Options are "mismatch_QGAN_12","mismatch_QGAN_20", "amp_state_GR", "amp_state_QGAN", "full_state_GR","full_state_QGAN", "psi_LPF_processed".')

