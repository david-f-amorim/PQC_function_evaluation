"""
Collection of functions regarding file handling. 
"""

import os 
from fractions import Fraction

def compress_args(n,m,L, seed, epochs,func_str,loss_str,meta,nint, mint, phase_reduce, train_superpos, real, repeat_params, WILL_p, WILL_q):
    """
    Compress a set of variable values used in network training into a dictionary . 

    Arguments:
    ---- 

    Arguments are the same as those passed to `.train_PLACEHOLDER.train_QNN()`. See the description there.


    Returns:
    ---

    - **arg_dict** : *dict* 

        Dictionary containing variable values 

    """
    arg_dict = {}
    arg_dict["n"]=n 
    arg_dict["m"]=m
    arg_dict["L"]=L
    arg_dict["seed"]=seed
    arg_dict["epochs"]=epochs 
    arg_dict["func_str"]=func_str
    arg_dict["loss_str"]=loss_str
    arg_dict["meta"]=meta
    arg_dict["nint"]=nint
    arg_dict["mint"]=mint
    arg_dict["phase_reduce"]=phase_reduce
    arg_dict["train_superpos"]=train_superpos
    arg_dict["real"]=real
    arg_dict["repeat_params"]=repeat_params
    arg_dict["WILL_p"]=WILL_q
    arg_dict["WILL_q"]=WILL_p

    return arg_dict

def compress_args_ampl(n,L,x_min,x_max,seed, epochs,func_str,loss_str,meta, nint, repeat_params):
    """
    Compress a set of variable values in training the amplitude encoding network into a dictionary . 

    Arguments:
    ---- 

    Arguments are the same as those passed to `.train_PLACEHOLDER.ampl_train_QNN()`. See the description there. 


    Returns:
    ---

    - **arg_dict** : *dict* 

        Dictionary containing variable values 

    """
    arg_dict = {}
    arg_dict["n"]=n 
    arg_dict["L"]=L
    arg_dict["seed"]=seed
    arg_dict["epochs"]=epochs 
    arg_dict["func_str"]=func_str
    arg_dict["loss_str"]=loss_str
    arg_dict["meta"]=meta
    arg_dict["nint"]=nint
    arg_dict["x_min"]=x_min
    arg_dict["x_max"]=x_max
    arg_dict["repeat_params"]=repeat_params
    
    return arg_dict

def vars_to_name_str(arg_dict):
    """
    Generate a name string from a set of variable values used in network training.

    Arguments:
    ---- 

    - **arg_dict** : *dict* 

        Dictionary created from variable values using `compress_args()`. 


    Returns:
    ---
    - **name_str** : *str* 

        Name string in the appropriate file naming convention. 
    
    """
    # set precision information 
    if arg_dict["nint"]==None or arg_dict["nint"]==arg_dict["n"]:
        arg_dict["nis"]=""
    else:
        arg_dict["nis"]=f'({arg_dict["nint"]})'
    if arg_dict["mint"]==None or arg_dict["mint"]==arg_dict["m"]:
        arg_dict["mis"]=""
    else:
        arg_dict["mis"]=f'({arg_dict["mint"]})' 

    # add to meta string
    if arg_dict["train_superpos"]:
        arg_dict["meta"]+='(S)'
    if arg_dict["phase_reduce"]:
        arg_dict["mis"]="(0)"
        arg_dict["meta"]+='(PR)' 
    if arg_dict["real"]:
        arg_dict["meta"]+='(r)' 
    if arg_dict["repeat_params"] != None:
        arg_dict["meta"]+=f'({arg_dict["repeat_params"]})' 

    # set WILL information     
    if arg_dict["loss_str"]=="WILL":
            arg_dict["meta"] +=f'_{Fraction(arg_dict["WILL_p"]).numerator}-{Fraction(arg_dict["WILL_p"]).denominator}_{Fraction(arg_dict["WILL_q"]).numerator}-{Fraction(arg_dict["WILL_q"]).denominator}'

    # set name string 
    name_str = f'_{arg_dict["n"]}{arg_dict["nis"]}_{arg_dict["m"]}{arg_dict["mis"]}_{arg_dict["L"]}_{arg_dict["epochs"]}_{arg_dict["func_str"]}_{arg_dict["loss_str"]}_{arg_dict["meta"]}_{arg_dict["seed"]}' 

    return name_str       

def vars_to_name_str_ampl(arg_dict):
    """
    Generate a name string from a set of variable values used in training the amplitude encoding network. 

    Arguments:
    ---- 
    - **arg_dict** : *dict* 

        Dictionary created from variable values using `compress_args()`. 


    Returns:
    ----
    - **name_str** : *str* 

        Name string in the appropriate file naming convention. 
    
    """
    # set precision information 
    if arg_dict["nint"]==None or  arg_dict["nint"]== arg_dict["n"]:
        arg_dict["nis"]=""
    else:
        arg_dict["nis"]=f'({arg_dict["nint"]})'
     
    # add to meta string
    if  arg_dict["repeat_params"]:
        arg_dict["meta"]+='(RP)' 

    # set name string 
    name_str = f'_{arg_dict["n"]}{arg_dict["nis"]}_{arg_dict["L"]}_{arg_dict["epochs"]}_{arg_dict["func_str"]}_{arg_dict["loss_str"]}_{arg_dict["x_min"]}_{arg_dict["x_max"]}_{arg_dict["meta"]}_{arg_dict["seed"]}'

    return name_str 

def check_duplicates(arg_dict, ampl=False):
    """
    For a given set of input parameters, check if training and testing results already exist. 

    Arguments:
    ---- 

    - **arg_dict** : *dict* 

        Dictionary created from variable values using `compress_args()` or `compress_args_ampl()`. 

    - **ampl** : *boolean* 

        If True, check if results exist for the amplitude-only network. Default is False     
        

    Returns:
    ---
    - **exist** : *boolean* 

        Returns True if results already exists and False otherwise. 
    
    """
    if ampl:
        name=vars_to_name_str_ampl(arg_dict)
        out_dir="ampl_outputs"
        labels=["mismatch", "loss", "weights", "statevec"]   
    else:                 
        name=vars_to_name_str(arg_dict)
        out_dir="outputs"
        labels=["mismatch", "loss", "weights", "grad", "vargrad"]  

    count=0 

    for i in range(len(labels)):
        if os.path.isfile(os.path.join(out_dir, f"{labels[i]}{name}.npy")):
            count +=1

    return count==len(labels)
    
def check_temp(arg_dict, ampl=False): 
    """
    For a given set of input parameters, check if temp files already exist. 

    Arguments:
    ---- 

    - **arg_dict** : *dict* 

        Dictionary created from variable values using `compress_args()` or `compress_args_ampl()`. 

    - **ampl** : *boolean* 

        If True, check if results exist for the amplitude-only network. Default is False     
        

    Returns:
    ---
    - **exist** : *boolean* 

        Returns True if results already exists and False otherwise. 
    
    """  
    
    # set precision strings 
    if ampl:
        name=vars_to_name_str_ampl(arg_dict)
        out_dir="ampl_outputs"
        labels=["mismatch", "loss", "weights", "statevec"]   
    else:                 
        name=vars_to_name_str(arg_dict)
        out_dir="outputs"
        labels=["mismatch", "loss", "weights", "grad", "vargrad"]  

    count=0 

    for k in range(100,arg_dict["epochs"], step=100):    
        for i in range(len(labels)):
            if os.path.isfile(os.path.join(out_dir, f"__TEMP{k}_{labels[i]}{name}.npy")):
                count +=1

    return count==len(labels)


def check_plots(arg_dict, log=True):  
    """
    For a given set of input parameters, check if plots already exist (excluding compare plots). 

    Arguments:
    ---- 

    - **arg_dict** : *dict* 

        Dictionary created from variable values using `compress_args()` or `compress_args_ampl()`. 

    - **log** : *boolean* 

        If True, check for log plots. Default is True.     
        

    Returns:
    ---
    - **exist** : *boolean* 

        Returns True if results already exists and False otherwise. 
    
    """
    name=vars_to_name_str(arg_dict)  
    out_dir="plots"          
    labels=["mismatch", "loss", "bar_mismatch"]
    log_str= ("" if log==False else "log_")

    count=0 

    for i in range(len(labels)):
        if os.path.isfile(os.path.join(out_dir, f"{labels[i]}{name}.npy")):
            count +=1

    return count==len(labels)
