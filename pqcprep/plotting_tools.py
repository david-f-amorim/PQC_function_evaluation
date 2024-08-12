"""
Collection of functions relating to plotting and visualisation. 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import os
from .psi_tools import x_trans_arr, get_phase_target, psi, A 
from .file_tools import vars_to_name_str, vars_to_name_str_ampl
from .resource_tools import load_data_H23
from .phase_tools import full_encode, phase_from_state

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
    Generates plots visualising the outputs produced by `pqcprep.training_tools.train_QNN()` and `pqcprep.training_tools.test_QNN()`. 

    Arguments:
    ---- 
    - **arg_dict** : *dict* 

        Dictionary containing varialbe information created using `pqcprep.file_tools.compress_args()`. 

    - **show** : *boolean* 

        If True, display plots. Default is False. 

    - **pdf** : *boolean* 

        If True, save plots in pdf format. If False, save plots in png format. Default is False. 

    - **DIR** : *str*

        Parent directory for output files.     

    Returns:
    ---

    Saves plots corresponding to each of the files produced by `pqcprep.training_tools.train_QNN()` and `pqcprep.training_tools.test_QNN()` (apart from `metrics_<NAME_STR>.npy`)
    in the directory `DIR/plots`. 
   
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
    Generates plots visualising the outputs produced by `pqcprep.training_tools.ampl_train_QNN()`.

     Arguments:
    ---- 
    - **arg_dict** : *dict* 

        Dictionary containing varialbe information created using `pqcprep.file_tools.compress_args_ampl()`. 

    - **show** : *boolean* 

        If True, display plots. Default is False. 

    - **pdf** : *boolean* 

        If True, save plots in pdf format. Default is False. 

    - **DIR** : *str*

        Parent directory for output files.     

    Returns:
    ---

    Saves plots corresponding to each of the files produced by `pqcprep.training_tools.ampl_train_QNN()` 
    in the directory `DIR/ampl_plots`.
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

def waveform_plots(name_str_phase, name_str_ampl, in_dir, out_dir,comp, no_UA=False,show=False, pdf=False):
    """
    Plot the amplitude, phase, and full waveform of a state prepared using QCNNs. 

    Arguments:
    ----

    - **name_str_phase** : *str* 

        Name string (produced using `pqcprep.file_tools.vars_to_name_str()`) corresponding to the weights of the QCNN used for phase encoding. 

    - **name_str_ampl** : *str* 

        Name string (produced using `pqcprep.file_tools.vars_to_name_str_ampl()`) corresponding to the weights of the QCNN used for amplitude preparation. This is ignored if `no_UA` is True.      

    - **in_dir** : *str*

        Directory containing the input files (as specified by `name_str_phase` and `name_str_ampl`). The directory is expected to contain a `weights` file for both cases. 

    - **out_dir** : *str*

        Directory for output files. 

    - **comp** : *str*, *optional*

        Compare outputs to results from Hayes 2023: if `'GR'` compare to results obtained using the Grover-Rudolph algorithm and if `'QGAN'` compare to results 
        obtained using the QGAN.  

    - **no_UA** : *boolean*

        If True, a Hadamard transform is applied to the input register instead of preparing the amplitude via a PQC. Default is False.      
    
    - **show** : *boolean* 

        If True, display plots. Default is False. 

    - **pdf** : *boolean* 

        If True, save plots in pdf format. Default is False. 


    Returns:
    ----

    Plots of the amplitude, phase, and full waveform saved in the directory `out_dir`.
    """

    pdf_str = ".pdf" if pdf else ".png"

    # read in given files 
    weights_phase = os.path.join(in_dir, f"weights{name_str_phase}.npy") 
    weights_ampl = os.path.join(in_dir, f"weights{name_str_ampl}.npy") 

    # extract information from name strings 
    phase_reduce = '(PR)' in name_str_phase
    real_p = '(r)' in name_str_phase
    if '(CL)' in name_str_phase:
        repeat_params="CL"
    elif '(IL)' in name_str_phase:
        repeat_params="IL" 
    elif '(both)' in name_str_phase:
        repeat_params="both"       
    else:
        repeat_params=None 

    psi_mode=None 
    func_A=None 
    for i in ["psi", "linear", "quadratic", "sine"]:
        if name_str_phase.count(i)==1:
            psi_mode=i 
            break 
    if psi_mode==None:
        raise ValueError("Name string could not be interpreted.")    
    for i in ["x76", "linear", "uniform"]:
        if name_str_ampl.count(i)==1:
            func_A=i 
            break     
    if func_A==None:
        raise ValueError("Name string could not be interpreted.")   
    
    n = name_str_ampl[1] if name_str_ampl[2]=="_" else name_str_ampl[1:3]
    if int(n) < 10:
        L_A= name_str_ampl[3] if name_str_ampl[4]=="_" else name_str_ampl[3:5] # this assumes nint=None 
        m = name_str_phase[3] if (name_str_phase[4]=="_" or name_str_phase[4]=="(") else name_str_phase[3:5]
    else: 
        L_A= name_str_ampl[4] if name_str_ampl[5]=="_" else name_str_ampl[4:6] # this assumes nint=None 
        m = name_str_phase[4] if (name_str_phase[5]=="_" or name_str_phase[5]=="(") else name_str_phase[4:6]   

    mint=None 
    for i in np.arange(int(m)+1):
        if name_str_phase.count(f"({i})")==1:
            mint=i 
            break 
    s = f"_{n}_{m}({mint})_"
    if name_str_phase.count(s) != 1 :
        raise ValueError("Name string could not be interpreted.")   
    else:
        L_phase = name_str_phase[len(s)] if name_str_phase[len(s)+1]=="_" else name_str_phase[len(s)+2]
    s = f"_{n}_{m}({mint})_{L_phase}_"
    if name_str_phase.count(s) != 1 :
        raise ValueError("Name string could not be interpreted.")

    n= int(n)
    m=int(m)
    L_phase=int(L_phase)
    L_A=int(L_A)
    mint=int(mint)    

    # generate x array 
    x_arr = x_trans_arr(n)

    # calculate target outputs 
    ampl_target = np.array([A(i, mode=func_A) for i in x_arr])
    ampl_target = ampl_target / np.sqrt(np.sum(ampl_target**2))

    phase_rounded = get_phase_target(m=m, psi_mode=psi_mode, phase_reduce=phase_reduce, mint=mint)
    phase_target = psi(np.arange(2**n),mode=psi_mode)

    h_target = ampl_target * np.exp(2*1.j*np.pi* phase_target)
    wave_real_target = np.real(h_target)
    wave_im_target = np.imag(h_target)

    h_target_rounded = ampl_target * np.exp(2*1.j*np.pi* phase_rounded)
    wave_real_target_rounded = np.real(h_target_rounded)
    wave_im_target_rounded = np.imag(h_target_rounded)

    # load Hayes 2023 data for comparison  
    if comp=="GR":
        ampl_vec_comp=np.abs(load_data_H23("amp_state_GR"))
        h_comp= load_data_H23("full_state_GR")
        psi_LPF = load_data_H23("psi_LPF_processed")  
        comp_label="GR"
        comp=True
    elif comp=="QGAN":
        ampl_vec_comp=np.abs(load_data_H23("amp_state_QGAN"))
        h_comp= load_data_H23("full_state_QGAN")
        psi_LPF = load_data_H23("psi_LPF_processed")  
        comp_label="QGAN"
        comp=True
    else:
        comp=False 

    if comp and no_UA:
        comp=False    
        print("Comparison to Hayes 2023 not shown due to incompatible amplitude function.") 
    if comp and not (psi_mode=="psi" and func_A=="x76"):
        comp=False
        print("Comparison to Hayes 2023 not shown due to incompatible amplitude or phase function.")    

    # get PQC state
    print(n,m, weights_ampl, weights_phase, L_A, L_phase,real_p,repeat_params,no_UA)

    state_vec = full_encode(n,m, weights_ampl, weights_phase, L_A, L_phase,real_p=real_p,repeat_params=repeat_params,no_UA=no_UA)
    phase = phase_from_state(state_vec)

    print(phase)

    ampl_vec = np.abs(state_vec)
    real_wave =np.real(state_vec)
    im_wave = np.imag(state_vec) 

    # plot amplitude 
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})

    ax[0].plot(x_arr,ampl_target, color="black")
    if comp:
        ax[0].scatter(x_arr,ampl_vec_comp,label=comp_label, color="blue")
    ax[0].scatter(x_arr,ampl_vec,label="QCNN", color="red")

    ax[0].set_ylabel(r"$\tilde A(f)$", fontsize=fontsize)
    if comp: ax[0].legend(fontsize=fontsize, loc='upper right')
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xticks([])

    if comp:
        ax[1].scatter(x_arr,ampl_target-ampl_vec_comp,label=comp_label, color="blue")
    ax[1].scatter(x_arr,ampl_target-ampl_vec,label="QCNN", color="red")

    ax[1].set_ylabel(r"$\Delta \tilde A(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,f"amplitude{pdf_str}"), bbox_inches='tight', dpi=500)

    if show:
        plt.show()
    plt.close() 

    # plot phase 
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})

    ax[0].plot(x_arr,phase_target, color="black")
    ax[0].plot(x_arr,phase_rounded, color="gray", ls="--")
    if comp:
        ax[0].scatter(x_arr,psi_LPF,label="LPF", color="blue")
    ax[0].scatter(x_arr,phase,label="QCNN", color="red")

    ax[0].set_ylabel(r"$\Psi (f)$", fontsize=fontsize)
    if comp: ax[0].legend(fontsize=fontsize, loc='upper right')
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xticks([])

    if comp:
        ax[1].scatter(x_arr,phase_target-psi_LPF,label="LPF + ", color="blue")
    ax[1].scatter(x_arr,phase_rounded-phase,label="QCNN", color="red")

    ax[1].set_ylabel(r"$\Delta \Psi(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,f"phase{pdf_str}"), bbox_inches='tight', dpi=500)

    if show:
        plt.show()
    plt.close() 

    # plot full waveform 
    fig, ax = plt.subplots(2, 2, figsize=(2*figsize[0],figsize[1]), gridspec_kw={'height_ratios': [1.5, 1]})
    fig.subplots_adjust()

    ax[0,0].plot(x_arr,wave_real_target, color="black")
    ax[0,0].plot(x_arr,wave_real_target_rounded, color="gray", ls="--")
    if comp:
        ax[0,0].scatter(x_arr,np.real(h_comp),label="LPF + "+comp_label, color="blue")
    ax[0,0].scatter(x_arr,real_wave,label="QCNN", color="red")

    ax[0,0].set_ylabel(r"$\Re[\tilde h(f)]$", fontsize=fontsize)
    if comp: ax[0,0].legend(fontsize=fontsize, loc='upper right')
    ax[0,0].tick_params(axis="both", labelsize=ticksize)
    ax[0,0].set_xticks([])

    if comp:
        ax[1,0].scatter(x_arr,wave_real_target -np.real(h_comp),label="LPF"+comp_label, color="blue")
    ax[1,0].scatter(x_arr,wave_real_target_rounded -real_wave,label="QCNN", color="red")

    ax[1,0].set_ylabel(r"$\Delta \Re[\tilde h(f)]$", fontsize=fontsize)
    ax[1,0].tick_params(axis="both", labelsize=ticksize)
    ax[1,0].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    ax[0,1].plot(x_arr,wave_im_target, color="black")
    ax[0,1].plot(x_arr,wave_im_target_rounded, color="gray", ls="--")
    if comp:
        ax[0,1].scatter(x_arr,np.imag(h_comp),label="LPF + "+comp_label, color="blue")
    ax[0,1].scatter(x_arr,im_wave,label="QCNN", color="red")

    ax[0,1].set_ylabel(r"$\Im[\tilde h(f)]$", fontsize=fontsize)
    if comp: ax[0,1].legend(fontsize=fontsize, loc='upper right')
    ax[0,1].tick_params(axis="both", labelsize=ticksize)
    ax[0,1].set_xticks([])

    if comp:
        ax[1,1].scatter(x_arr,wave_im_target -np.imag(h_comp),label="LPF + "+comp_label, color="blue")
    ax[1,1].scatter(x_arr,wave_im_target_rounded -im_wave,label="QCNN", color="red")

    ax[1,1].set_ylabel(r"$\Delta \Im[\tilde h(f)]$", fontsize=fontsize)
    ax[1,1].tick_params(axis="both", labelsize=ticksize)
    ax[1,1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,f"waveform{pdf_str}"), bbox_inches='tight', dpi=500)

    if show:
        plt.show()
    plt.close()

    return 0 