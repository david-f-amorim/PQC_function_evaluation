"""
Collection of functions regarding plotting and visualisation. 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import os
from .psi_tools import x_trans_arr, get_phase_target, psi, A 
from .file_tools import vars_to_name_str, vars_to_name_str_ampl

# general settings
rcParams['mathtext.fontset'] = 'stix' 
rcParams['font.family'] = 'STIXGeneral' 
width=0.75 
""" @private """
color='black' 
""" @private """
fontsize=28 
""" @private """
titlesize=32
""" @private """
ticksize=22
""" @private """
figsize=(10,10)
""" @private """

def benchmark_plots(arg_dict,DIR, show=False, pdf=False):
        
    """
    ...

    Arguments:
    ---- 
    - **arg_dict** : *dict* 

        Dictionary containing varialbe information created using `pqcprep.file_tools.compress_args()`. 

    - **show** : *boolean* 

        If True, display plots. Default is False. 

    - **pdf** : *boolean* 

        If True, save plots in pdf format. Default is False. 

    - **DIR** : *str*

        Directory for output files.     

    Returns:
    ---

    ....    

    """

    name_str = vars_to_name_str(arg_dict)
    pdf_str = ".pdf" if pdf else ".png"

    # data to plot 
    arrs =["loss", "mismatch", "grad", "vargrad"]
    labels=["Loss", "Mismatch", r"$|\nabla_\theta W|^2$",r"Var($\partial_\theta W$)" ]

    for i in np.arange(len(arrs)):
        arr = np.load(os.path.join(DIR, "outputs", f"{arrs[i]}{name_str}.npy"))

        plt.figure(figsize=figsize)
        plt.xlabel("Epoch", fontsize=fontsize)
        plt.ylabel(labels[i], fontsize=fontsize)
        plt.yscale('log')
        plt.tick_params(axis="both", labelsize=ticksize)
        plt.scatter(np.arange(len(arr))+1,arr,color="red")

        plt.tight_layout()
        plt.savefig(os.path.join(DIR, "plots", f"{arrs[i]}{name_str}{pdf_str}"), bbox_inches='tight', dpi=500)

        if show:
            plt.show()
        plt.close()      

    # plot mismatch by state 
    dic = np.load(os.path.join(DIR, "outputs", f"mismatch_by_state{name_str}.npy"),allow_pickle='TRUE').item()
    mismatch = list(dic.values())
    x_arr = x_trans_arr(arg_dict["n"])

    plt.figure(figsize=figsize)
    plt.xlabel(r"$f$ (Hz)", fontsize=fontsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.yscale('log')
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.scatter(x_arr,mismatch,color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(DIR, "plots", f"mismatch_by_state{name_str}{pdf_str}"), bbox_inches='tight', dpi=500)

    if show:
        plt.show()
    plt.close()      

    # plot extracted phase function   
    mint = arg_dict["mint"] if arg_dict["mint"] != None else arg_dict["m"] 
    if arg_dict["phase_reduce"]:
        mint = 0      
    phase = np.load(os.path.join(DIR, "outputs", f"phase{name_str}.npy"))
    phase_target_rounded = get_phase_target(m=arg_dict["m"], psi_mode=arg_dict["func_str"], phase_reduce=arg_dict["phase_reduce"], mint=mint)
    phase_target = psi(np.arange(2**arg_dict["n"]),mode=arg_dict["func_str"])

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})

    ax[0].plot(x_arr,phase_target, color="black")
    ax[0].plot(x_arr,phase_target_rounded, color="gray", ls="--")
    ax[0].scatter(x_arr,phase, color="red")

    ax[0].set_ylabel(r"$\Psi (f)$", fontsize=fontsize)
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xticks([])

    ax[1].scatter(x_arr,phase_target_rounded-phase, color="red")
    ax[1].set_ylabel(r"$\Delta \Psi(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "plots", f"phase{name_str}{pdf_str}"), bbox_inches='tight', dpi=500)

    if show:
        plt.show()
    plt.close()      


    return 0


def benchmark_plots_ampl(arg_dict,DIR, show=False, pdf=False):
        
    """
    ...

     Arguments:
    ---- 
    - **arg_dict** : *dict* 

        Dictionary containing varialbe information created using `pqcprep.file_tools.compress_args_ampl()`. 

    - **show** : *boolean* 

        If True, display plots. Default is False. 

    - **pdf** : *boolean* 

        If True, save plots in pdf format. Default is False. 

    - **DIR** : *str*

        Directory for output files.     

    Returns:
    ---

    ....
    """
    name_str = vars_to_name_str_ampl(arg_dict)
    pdf_str = ".pdf" if pdf else ".png"

    # data to plot 
    arrs =["loss", "mismatch"]
    labels=["Loss", "Mismatch"]

    for i in np.arange(len(arrs)):
        arr = np.load(os.path.join(DIR, "ampl_outputs", f"{arrs[i]}{name_str}.npy"))

        plt.figure(figsize=figsize)
        plt.xlabel("Epoch", fontsize=fontsize)
        plt.ylabel(labels[i], fontsize=fontsize)
        plt.yscale('log')
        plt.tick_params(axis="both", labelsize=ticksize)
        plt.scatter(np.arange(len(arr))+1,arr,color="red")

        plt.tight_layout()
        plt.savefig(os.path.join(DIR, "ampl_plots", f"{arrs[i]}{name_str}{pdf_str}"), bbox_inches='tight', dpi=500)

        if show:
            plt.show()
        plt.close()      

    # plot amplitude 
    x_arr = x_trans_arr(arg_dict["n"])

    ampl_vec = np.abs(np.load(os.path.join(DIR, "ampl_outputs", f"statevec{name_str}.npy")))
    ampl_target = np.array([A(i, mode=arg_dict["func_str"]) for i in x_arr])
    ampl_target /= np.sqrt(np.sum(ampl_target**2))

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})
    ax[0].plot(x_arr,ampl_target, color="black")
    ax[0].scatter(x_arr,ampl_vec,color="red")
    ax[0].set_ylabel(r"$\tilde A(f)$", fontsize=fontsize)
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xticks([])
    ax[1].scatter(x_arr,ampl_target-ampl_vec,label="QCNN", color="red")
    ax[1].set_ylabel(r"$\Delta \tilde A(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "ampl_plots", f"amplitude{name_str}{pdf_str}"), bbox_inches='tight', dpi=500)

    if show:
        plt.show()  
    plt.close()          

    return 0

