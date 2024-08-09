"""
Collection of functions regarding plotting and visualisation. 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from .__input__ import DIR
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

def benchmark_plots(arg_dict, show=False, pdf=False):
        
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

    Returns:
    ---

    ....    

    """

    name_str = vars_to_name_str(arg_dict)
    pdf_str = ".pdf" if pdf else ".png"

    # data to plot 
    arrs =["loss", "mismatch", "grad", "vargrad"]
    labels=["Loss", "Mismatch", r"$|\nabla_\boldsymbol{\theta} W|^2$",r"Var($\partial_\theta W$)" ]

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


    return 0


def benchmark_plots_ampl(arg_dict, show=False, pdf=False):
        
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

    return 0


####
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='', description="Generate plots of  QCNN training and testing.")   
    parser.add_argument('-n','--n', help="Number of input qubits.", default=2, type=int)
    parser.add_argument('-m','--m', help="Number of target qubits.", default=2, type=int)
    parser.add_argument('-L','--L', help="Number of network layers.", default=[6],type=int, nargs="+")
    parser.add_argument('-l','--loss', help="Loss function.", default=["MM"], nargs="+")
    parser.add_argument('-fs','--f_str', help="String describing function.",nargs="+", default=["x"])
    parser.add_argument('-e','--epochs', help="Number of epochs.", default=[600],nargs="+", type=int)
    parser.add_argument('-ni','--nint', help="Number of integer input qubits.", default=None, type=int)
    parser.add_argument('-mi','--mint', help="Number of integer target qubits.", default=None, type=int)

    parser.add_argument('-RP','--repeat_params', help="Use the same parameter values for different layers", default=None ,choices=["CL", "IL", "both"])
    parser.add_argument('-r','--real', help="Output states with real amplitudes only.", action='store_true')
    parser.add_argument('-PR','--phase_reduce', help="Reduce function values to a phase between 0 and 1.", action='store_true')
    parser.add_argument('-TS','--train_superpos', help="Train circuit in superposition. (Automatically activates --phase_reduce).", action='store_true')

    parser.add_argument('-p','--WILL_p', help="WILL p parameter.", default=1,type=float)
    parser.add_argument('-q','--WILL_q', help="WILL q parameter.", default=1,type=float)

    parser.add_argument('-cL','--compL', help="Compare different L values (pass multiple).", action='store_true')
    parser.add_argument('-cf','--compf', help="Compare different f_str values (pass multiple).", action='store_true')
    parser.add_argument('-cM','--compM', help="Compare different meta values (pass multiple).", action='store_true')
    parser.add_argument('-cl','--compl', help="Compare different loss values (pass multiple).", action='store_true')
    parser.add_argument('-ce','--compe', help="Compare different epoch values (pass multiple).", action='store_true')
    parser.add_argument('-ceL','--compeL', help="Compare different epoch and L values (pass multiple).", action='store_true')

    parser.add_argument('-lg','--log', help="Take logarithm of values.", action='store_true')
    parser.add_argument('-s','--show', help="Display plots in terminal.", action='store_true')
    parser.add_argument('-M','--meta', help="String with meta data.",nargs="+", default=[""])
    parser.add_argument('-I','--ignore_duplicates', help="Ignore and overwrite duplicate files.", action='store_true')

    parser.add_argument('-H','--hayes', help="Train circuit to reproduce Hayes 2023. -n 6 -PR -f psi. Still set own m", action='store_true')

    opt = parser.parse_args()

    if opt.hayes:
        opt.n=6 
        opt.phase_reduce=True 
        opt.train_superpos=True 
        opt.real=True 
        opt.f_str=["psi"]  

    if int(opt.compL)+int(opt.compf)+int(opt.compM)+int(opt.compl)+int(opt.compe)+int(opt.compeL) > 1:
        raise ValueError("Cannot do two comparisons at once.")

    if opt.compL:
        comp_L(n=opt.n,m=opt.m,L_arr=opt.L,epochs=opt.epochs[0], func_str=opt.f_str[0],loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p, WILL_q=opt.WILL_q)
    elif opt.compf:
        comp_f(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs[0], func_str_arr=opt.f_str,loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p, WILL_q=opt.WILL_q)
    elif opt.compM:
        comp_meta(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs[0], func_str=opt.f_str[0],loss_str=opt.loss[0], meta_arr=opt.meta, show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p, WILL_q=opt.WILL_q) 
    elif opt.compl:
        comp_loss_funcs(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs[0], func_str=opt.f_str[0],loss_str_arr=opt.loss, meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p, WILL_q=opt.WILL_q)
    elif opt.compe:
        comp_epochs(n=opt.n,m=opt.m,L=opt.L[0],epochs_arr=opt.epochs, func_str=opt.f_str[0],loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p, WILL_q=opt.WILL_q)
    elif opt.compeL:
        comp_mean_mismatch(n=opt.n,m=opt.m,L_arr=opt.L,epochs_arr=opt.epochs, func_str=opt.f_str[0],loss=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p, WILL_q=opt.WILL_q)
    else:
        dupl_files = check_plots(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs[0],func_str=opt.f_str[0],loss_str=opt.loss[0],meta=opt.meta[0], log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p, WILL_q=opt.WILL_q)

        if dupl_files and opt.ignore_duplicates==False:
            print("\nThe required plots already exist and will not be recomputed. Use '-I' or '--ignore_duplicates' to override this.\n")
        else: 
            standard(n=opt.n, m=opt.m, L=opt.L[0], epochs=opt.epochs[0], loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, func_str=opt.f_str[0], log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p, WILL_q=opt.WILL_q)
            standard_bar(n=opt.n, m=opt.m, L=opt.L[0], epochs=opt.epochs[0], loss=opt.loss[0], meta=opt.meta[0], show=opt.show, func_str=opt.f_str[0], log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p, WILL_q=opt.WILL_q)

"""



