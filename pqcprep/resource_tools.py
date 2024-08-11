"""
DOCUMENT THIS ...
"""
import numpy as np 
from importlib.resources import as_file, files
import os 

def load_resources(name):
    """
    ....
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

    
## TES THIS !!