import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
from pqcprep.phase_tools import full_encode, phase_from_state
from pqcprep.binary_tools import bin_to_dec, dec_to_bin
from pqcprep.psi_tools import psi, get_phase_target 

# config 
L_phase = 6
real_p = True 
m = 3
psi_mode="psi"
weights_phase =f"full_encode/weights_6_{m}(0)_{L_phase}_600_{psi_mode}_SAM_0.4_(S)(PR)(r)_1680458526.npy" #"outputs/weights_6_3(0)_6_600_psi_MM_(S)(PR)(r).npy" 

repeat_params=None
operators="QRQ"
no_UA=True

n = 6
weights_ampl = "full_encode/weights_6_3_600_x76_MM_40_168_zeros.npy" 
ampl_vec = np.load("full_encode/statevec_6_3_600_x76_MM_40_168_zeros.npy")
L_ampl =3

mint = 0
phase_reduce = True

# plot settings
comp =True # compare to Hayes 2023  
show = True # show plots
pdf = False # save outputs as pdf 
delta_round =True #calculate difference from rounded version 

no_A = False # don't produce amplitude plot 
no_p = True# don't produce phase plot 
no_h = True # don't produce h plot

no_full_A =True # don't produce full amplitude plot
no_full_p = True # don't produce full phase plot
no_full_3D= True # don't produce full 3D plot

comp_full = False

# additional plots 
A_L_comp = False 
QGAN_comp = False
phase_round_comp = False
phase_L_comp = False
phase_loss_comp = False
phase_shift_comp = False 
phase_RP_comp=False 

#------------------------------------------------------------------------------
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
if pdf:
    rcParams["text.usetex"] = True 
    pdf_str=".pdf"
else:
    pdf_str=""    
width=0.75
color='black'
fontsize=28
titlesize=32
ticksize=22
figsize=(10,10)
 
# define x array 
x_min = 40
x_max = 168 
dx = (x_max-x_min)/(2**n) 
x_arr = np.arange(x_min, x_max, dx) 

# calculate target output for amplitude 
ampl_target = x_arr**(-7./6)
ampl_target = ampl_target / np.sqrt(np.sum(ampl_target**2))

# calculate target output for phase 
phase_target = psi(np.arange(2**n),mode=psi_mode)

# calculate target output for wavefunc 
h_target = ampl_target * np.exp(2*1.j*np.pi* phase_target)
wave_real_target = np.real(h_target)
wave_im_target = np.imag(h_target)

# calculate target for phase taking into account rounding 
phase_rounded = get_phase_target(m, psi_mode=psi_mode, phase_reduce=phase_reduce, mint=mint)

# calculate target output for wavefunc taking into account rounding 
h_target_rounded = ampl_target * np.exp(2*1.j*np.pi* phase_rounded)
wave_real_target_rounded = np.real(h_target_rounded)
wave_im_target_rounded = np.imag(h_target_rounded)

# load amplitude-only statevector (for QCNN, QGAN, GR)
ampl_vec_QGAN = np.abs(np.load("ampl_outputs/amp_state_QGAN.npy"))
ampl_vec_GR = np.abs(np.load("ampl_outputs/amp_state_GR.npy")) 

# load LPF phase and waveform
psi_LPF = np.load("full_encode/psi_LPF_processed.npy")
h_QGAN = np.load("full_encode/full_state_QGAN.npy")
h_GR = np.load("full_encode/full_state_GR.npy")

# calculate state vector from QCNNs 
state_vec, state_vec_full = full_encode(n,m, weights_ampl, weights_phase, L_ampl, L_phase,real_p=real_p,repeat_params=repeat_params,full_state_vec=True, no_UA=no_UA, operators=operators)
phase = phase_from_state(state_vec)

# get full wavefunction 
real_wave =np.real(state_vec)
im_wave = np.imag(state_vec)
"""
# print info
metrics =np.load("pqcprep/outputs/metrics"+weights_phase[23:],allow_pickle='TRUE')
print("-----------------------------------")
print(f'Mu: \t{metrics.item().get("mu"):.3e}') 
print(f'Sigma: \t{metrics.item().get("sigma"):.3e}') 
print(f'Eps: \t{metrics.item().get("eps"):.3e}')
print(f'Chi: \t{metrics.item().get("chi"):.3e}') 
print(f'Omega: \t{metrics.item().get("omega"):.3f}')
print("-----------------------------------")
"""
#------------------------------------------------------------------------------
if no_A==False:
    """
    PLOT QCNN AMPLITUDE VERSUS TARGET AND GR/QGAN 
    """
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})

    ax[0].plot(x_arr,ampl_target, color="black")
    if comp:
        #ax[0].scatter(x_arr,ampl_vec_QGAN,label="QGAN", color="green")
        ax[0].scatter(x_arr,ampl_vec_GR,label="GR", color="blue")
    ax[0].scatter(x_arr,ampl_vec,label="QCNN", color="red")

    ax[0].set_ylabel(r"$\tilde A(f)$", fontsize=fontsize)
    ax[0].legend(fontsize=fontsize, loc='upper right')
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)
    #ax[0].set_xticks([])
    """
    if comp:
        ax[1].scatter(x_arr,ampl_target-ampl_vec_QGAN,label="QGAN", color="green")
        ax[1].scatter(x_arr,ampl_target-ampl_vec_GR,label="GR", color="blue")
    ax[1].scatter(x_arr,ampl_target-ampl_vec,label="QCNN", color="red")

    ax[1].set_ylabel(r"$\Delta \tilde A(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)
    """
    ax[1].set_visible(False)  
    fig.tight_layout()
    fig.savefig(f"full_encode/amplitude_comp{pdf_str}", bbox_inches='tight', dpi=500)

    if show:
        plt.show()

if no_p==False:
    """
    PLOT QCNN PHASE VERSUS TARGET AND LPF 
    """
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})

    ax[0].plot(x_arr,phase_target, color="black")
    ax[0].plot(x_arr,phase_rounded, color="gray", ls="--")
    if comp:
        ax[0].scatter(x_arr,psi_LPF,label="LPF", color="blue")
    ax[0].scatter(x_arr,phase,label="QCNN", color="red")

    ax[0].set_ylabel(r"$\Psi (f)$", fontsize=fontsize)
    ax[0].legend(fontsize=fontsize, loc='upper right')
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xticks([])

    if comp:
        ax[1].scatter(x_arr,phase_target-psi_LPF,label="GR", color="blue")
    if delta_round:
        phase_target = phase_rounded
    ax[1].scatter(x_arr,phase_target-phase,label="QCNN", color="red")

    ax[1].set_ylabel(r"$\Delta \Psi(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(f"full_encode/phase_comp{pdf_str}", bbox_inches='tight', dpi=500)

    if show:
        plt.show()

if no_h==False:
    """
    PLOT WAVEFUNC VERSUS TARGET AND LPF 
    """

    fig, ax = plt.subplots(2, 2, figsize=(2*figsize[0],figsize[1]), gridspec_kw={'height_ratios': [1.5, 1]})
    fig.subplots_adjust()

    ax[0,0].plot(x_arr,wave_real_target, color="black")
    ax[0,0].plot(x_arr,wave_real_target_rounded, color="gray", ls="--")
    if comp:
        ax[0,0].scatter(x_arr,np.real(h_QGAN),label="LPF + QGAN", color="green")
        ax[0,0].scatter(x_arr,np.real(h_GR),label="LPF + GR", color="blue")
    ax[0,0].scatter(x_arr,real_wave,label="QCNN", color="red")

    ax[0,0].set_ylabel(r"$\Re[\tilde h(f)]$", fontsize=fontsize)
    ax[0,0].legend(fontsize=fontsize, loc='upper right')
    ax[0,0].tick_params(axis="both", labelsize=ticksize)
    ax[0,0].set_xticks([])

    if comp:
        ax[1,0].scatter(x_arr,wave_real_target -np.real(h_QGAN),label="LPF + QGAN", color="green")
        ax[1,0].scatter(x_arr,wave_real_target -np.real(h_GR),label="LPF + GR", color="blue")
    if delta_round:
        wave_real_target = wave_real_target_rounded
    ax[1,0].scatter(x_arr,wave_real_target -real_wave,label="QCNN", color="red")

    ax[1,0].set_ylabel(r"$\Delta \Re[\tilde h(f)]$", fontsize=fontsize)
    ax[1,0].tick_params(axis="both", labelsize=ticksize)
    ax[1,0].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    ax[0,1].plot(x_arr,wave_im_target, color="black")
    ax[0,1].plot(x_arr,wave_im_target_rounded, color="gray", ls="--")
    if comp:
        ax[0,1].scatter(x_arr,np.imag(h_QGAN),label="LPF + QGAN", color="green")
        ax[0,1].scatter(x_arr,np.imag(h_GR),label="LPF + GR", color="blue")
    ax[0,1].scatter(x_arr,im_wave,label="QCNN", color="red")

    ax[0,1].set_ylabel(r"$\Im[\tilde h(f)]$", fontsize=fontsize)
    ax[0,1].legend(fontsize=fontsize, loc='upper right')
    ax[0,1].tick_params(axis="both", labelsize=ticksize)
    ax[0,1].set_xticks([])

    if comp:
        ax[1,1].scatter(x_arr,wave_im_target -np.imag(h_QGAN),label="LPF + QGAN", color="green")
        ax[1,1].scatter(x_arr,wave_im_target -np.imag(h_GR),label="LPF + GR", color="blue")
    if delta_round:
        wave_im_target = wave_im_target_rounded
    ax[1,1].scatter(x_arr,wave_im_target -im_wave,label="QCNN", color="red")

    ax[1,1].set_ylabel(r"$\Delta \Im[\tilde h(f)]$", fontsize=fontsize)
    ax[1,1].tick_params(axis="both", labelsize=ticksize)
    ax[1,1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(f"full_encode/h_comp{pdf_str}", bbox_inches='tight', dpi=500)

    if show:
        plt.show()

if no_full_A==False:
    """
    PLOT STATEVECTOR AMPLITUDES FOR VARIOUS TARGET REGISTER STATES
    """
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2**m, 1]},constrained_layout=True)
    ax[0].scatter(x_arr,phase,label="QCNN", color="red")

    ax[0].set_ylabel(r"$\tilde{\Psi}(f)$", fontsize=fontsize)
    ax[0].tick_params(axis="both", labelsize=ticksize)
    im=ax[0].pcolormesh(x_arr,np.arange(2**m), np.abs(state_vec_full))
    im.set_clim(np.min(np.abs(state_vec_full)),np.max(np.abs(state_vec_full)))
    locs = ax[0].get_yticks() 
    ax[0].set_yticks(locs[1:-1], [np.round(bin_to_dec(dec_to_bin(i,m),nint=0)*2*np.pi,2) for i in np.arange(len(locs)-2)])
    ax[0].set_xticks([]) if comp_full else ax[0].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    if comp_full:
        ax[1].tick_params(axis="both", labelsize=ticksize)
        ax[1].set_yticks([])
        ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

        tar_arr = ampl_target if no_UA==False else np.ones(len(x_arr)) / np.sqrt(2**n)
        
        im2 =ax[1].pcolormesh(x_arr,np.arange(2**m)[0:2],tar_arr*np.ones((2,len(x_arr))))
        im2.set_clim(np.min(np.abs(state_vec_full)),np.max(np.abs(state_vec_full))) 
    else:
        ax[1].set_visible(False)    
        
    cb=fig.colorbar(im,ax=ax.ravel().tolist(), location='top', orientation='horizontal', pad=0.05)
    cb.ax.tick_params(labelsize=ticksize)

    if show:
        plt.show()
    
    fig.savefig(f"full_encode/{operators}_amp_{psi_mode}_{'H' if no_UA else 'UA'}{pdf_str}", bbox_inches='tight', dpi=500)

if no_full_p==False:
    """
    PLOT STATEVECTOR PHASES FOR VARIOUS TARGET REGISTER STATES
    """
    
    full_phase = np.angle(state_vec_full) + 2* np.pi * (np.angle(state_vec_full) < 0).astype(int)
    full_phase *= (np.abs(state_vec_full) > 1e-15).astype(float) 
    
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2**m, 1]},constrained_layout=True)
    ax[0].scatter(x_arr,phase,label="QCNN", color="red")

    ax[0].set_ylabel(r"$\tilde{\Psi}(f)$", fontsize=fontsize)
    ax[0].tick_params(axis="both", labelsize=ticksize)
    im=ax[0].pcolormesh(x_arr,np.arange(2**m), full_phase, cmap="hsv")
    im.set_clim(0, 2*np.pi)
    locs = ax[0].get_yticks() 
    ax[0].set_yticks(locs[1:-1], [np.round(bin_to_dec(dec_to_bin(i,m),nint=0)*2*np.pi,2) for i in np.arange(len(locs)-2)])
    ax[0].set_xticks([]) if comp_full else ax[0].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    if comp_full:
        ax[1].tick_params(axis="both", labelsize=ticksize)
        ax[1].set_yticks([])
        ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

        tar_arr = phase_rounded
        
        im2 =ax[1].pcolormesh(x_arr,np.arange(2**m)[0:2],tar_arr*np.ones((2,len(x_arr))), cmap="hsv")
        im2.set_clim(0, 2*np.pi) 
    else:
        ax[1].set_visible(False)    
    
    cb=fig.colorbar(im,ax=ax.ravel().tolist(), location='top', orientation='horizontal', pad=0.05)
    cb.ax.tick_params(labelsize=ticksize)
   
    if show:
        plt.show()
    
    fig.savefig(f"full_encode/{operators}_phase_{psi_mode}_{'H' if no_UA else 'UA'}{pdf_str}", bbox_inches='tight', dpi=500)
    
if no_full_3D==False:
    """
    PLOT STATEVECTOR AMPLITUDE AND PHASES FOR DIFFERENT TARGET REGISTER STATES
    """    
    from matplotlib import cm 
    
    full_phase = np.angle(state_vec_full) + 2* np.pi * (np.angle(state_vec_full) < 0).astype(int)
    full_phase *= (np.abs(state_vec_full) > 1e-15).astype(float) 
   
    X = x_arr
    Y = np.arange(2**m)
    X, Y = np.meshgrid(X, Y)
    Z = np.abs(state_vec_full)
    P = full_phase
    P_norm = (full_phase) / (2* np.pi)  # normalise to value in [0,1]

    # Plot the surface
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"}, constrained_layout=False)
    surf = ax.plot_surface(X, Y, Z,facecolors=cm.hsv(P_norm),linewidth=0, antialiased=True, shade=False)
    ax.set_xlabel(r"$f$ (Hz)", fontsize=fontsize, labelpad=20)
    ax.set_ylabel(r"$\tilde{\Psi}(f)$", fontsize=fontsize, labelpad=20)
    locs = ax.get_yticks() 
    ax.set_yticks(locs[1:-1], [np.round(bin_to_dec(dec_to_bin(i,m),nint=0)*2*np.pi,2) for i in np.arange(len(locs)-2)])
    ax.tick_params(axis="both", labelsize=ticksize)

    cb=fig.colorbar(cm.ScalarMappable(cmap=cm.hsv, norm=surf.norm), ax = ax , shrink = 0.6, aspect = 5)
    cb.ax.tick_params(labelsize=ticksize)
    locs = cb.ax.get_yticks()
    cb.ax.set_yticks( [0, 0.25,0.5,0.75,1], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{2}$' , r'$2\pi$'])
    
    if show:
        plt.show()
    
    fig.savefig(f"full_encode/{operators}_3D_{psi_mode}_{'H' if no_UA else 'UA'}{pdf_str}", bbox_inches='tight', dpi=500)
    
#------------------------------------------------------------------------------------------------------

if A_L_comp:

    arr_1 = np.load("pqcprep/ampl_outputs/mismatch_6_1_600_x76_MM_40_168_.npy")
    arr_2 = np.load("pqcprep/ampl_outputs/mismatch_6_3_600_x76_MM_40_168_.npy")
    arr_3 = np.load("pqcprep/ampl_outputs/mismatch_6_6_600_x76_MM_40_168_.npy")
    arr_4 = np.load("pqcprep/ampl_outputs/mismatch_6_9_600_x76_MM_40_168_.npy")
    arr_5 = np.load("pqcprep/ampl_outputs/mismatch_6_11_300_x76_MM_40_168_.npy")

    M = np.array([arr_1,arr_2,arr_3,arr_4,arr_5], dtype="object")
    L = np.array([1,3,6,9,11])

    N = len(M)

    plt.figure(figsize=figsize)
    cmap = plt.get_cmap('jet', N)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.yscale('log')         

    for i in range(N): 
        plt.scatter(np.arange(len(M[i]))+1, M[i], color=cmap(i), label=r"$L=$"+f"{L[i]}")

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(f"full_encode/A_L_comp{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()

if QGAN_comp:

    arr_1 = np.load("pqcprep/ampl_outputs/mismatch_QGAN_12.npy")
    arr_2 = np.load("pqcprep/ampl_outputs/mismatch_QGAN_20.npy")
    arr_3 = np.load("pqcprep/ampl_outputs/mismatch_6_3_600_x76_MM_40_168_.npy")
    
    M = np.array([arr_1,arr_2,arr_3], dtype="object")
    labels = [r"QGAN ($L=12$)",r"QGAN ($L=20$)",r"QCNN ($L=3$)"]
    colours=["green", "blue", "red"]

    N = len(M)

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.yscale('log')         

    for i in range(N): 
        plt.scatter(np.arange(len(M[i]))+1, M[i], color=colours[i], label=labels[i])

    plt.legend(fontsize=fontsize, loc='lower right')
    plt.savefig(f"full_encode/QGAN_comp{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()        

if phase_round_comp:

    if delta_round:
        phase_target = psi(np.linspace(0, 2**n, len(x_arr)),mode=psi_mode) # set back to previous value for the following

    def phi_round(p):
        phase_reduced = np.modf(phase_target / (2* np.pi))[0] 
        phase_reduced_bin = [dec_to_bin(i,p, "unsigned mag", 0) for i in phase_reduced]
        phase_reduced_dec =  np.array([bin_to_dec(i,"unsigned mag", 0) for i in phase_reduced_bin])
        return 2 * np.pi * phase_reduced_dec
    
    P =np.array([phi_round(3),phi_round(4),phi_round(5),phi_round(6),phi_round(7),phi_round(8) ], dtype="object")

    labels= [r"$m=3$",r"$m=4$",r"$m=5$", r"$m=6$", r"$m=7$" ,r"$m=8$"]

    N = len(P)

    plt.figure(figsize=figsize)
    cmap = plt.get_cmap('jet', N)
    plt.ylabel(r"$\Psi (f)$", fontsize=fontsize)
    plt.xlabel(r"$f$ (Hz)", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
  
    for i in range(N): 
        plt.plot(x_arr, P[i], color=cmap(i), label=labels[i], linewidth=2.5)

    plt.plot(x_arr,phase_target,color="black",ls="--", linewidth=2, label=r"$m \to \infty$")     

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(f"full_encode/phase_round_comp{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()    

if phase_L_comp==True:
    """
    PLOT QCNN PHASE VERSUS TARGET FOR DIFFERENT L
    """
    L_arr = np.array([3,6,9, 12])
    arr_1 = "pqcprep/outputs/weights_6_4(0)_3_600_psi_MM_(S)(PR)(r).npy"
    arr_2 = "pqcprep/outputs/weights_6_4(0)_6_600_psi_MM_(S)(PR)(r).npy"
    arr_3 = "pqcprep/outputs/weights_6_4(0)_9_600_psi_MM_(S)(PR)(r).npy"
    arr_4 = "pqcprep/outputs/weights_6_4(0)_12_600_psi_MM_(S)(PR)(r).npy"

    weights_arr =np.array([arr_1, arr_2, arr_3, arr_4])
    colours = ["green", "blue", "red", "purple"]
    N = len(weights_arr)

    phase_arr = np.empty(N, dtype="object")

    for i in np.arange(N):

        state_vec = full_encode(n,m, weights_ampl, weights_arr[i], L_ampl, L_arr[i],real_p=real_p,repeat_params=repeat_params)
        amplitude = np.abs(state_vec)
        phase = np.angle(state_vec) + 2* np.pi * (np.angle(state_vec) < -np.pi).astype(int)
        phase *= (amplitude > 1e-15).astype(float) 
        phase_arr[i]=phase 
        print("Norm: ",np.sum(amplitude**2))

    if delta_round:
        phase_target = psi(np.linspace(0, 2**n, len(x_arr)),mode=psi_mode) # set back to previous value for the following

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})
    cmap = plt.get_cmap('Dark2', N)

    ax[0].plot(x_arr,phase_target, color="black")
    ax[0].plot(x_arr,phase_rounded, color="gray", ls="--")
    
    for i in np.arange(N):
        ax[0].scatter(x_arr,phase_arr[i],label=r"$L=$"+f"{L_arr[i]}", color=colours[i])

    ax[0].set_ylabel(r"$\Psi (f)$", fontsize=fontsize)
    ax[0].legend(fontsize=fontsize, loc='upper left')
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xticks([])

    if delta_round:
        phase_target = phase_rounded

    for i in np.arange(N):
        ax[1].scatter(x_arr,phase_target-phase_arr[i],label=r"$L=$"+f"{L_arr[i]}", color=colours[i])
    
    ax[1].set_ylabel(r"$\Delta \Psi(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(f"full_encode/phase_L_comp{pdf_str}", bbox_inches='tight', dpi=500)

    if show:
        plt.show()   

if phase_loss_comp==True:
    """
    PLOT QCNN PHASE VERSUS TARGET FOR DIFFERENT loss
    """
    loss_arr =np.array(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    arr_1 =f"pqcprep/outputs/weights_6_{m}(0)_{L_phase}_600_{psi_mode}_SAM_0.0_(S)(PR)(r)_1680458526.npy"
    arr_2 =f"pqcprep/outputs/weights_6_{m}(0)_{L_phase}_600_{psi_mode}_SAM_0.2_(S)(PR)(r)_1680458526.npy"
    arr_3 =f"pqcprep/outputs/weights_6_{m}(0)_{L_phase}_600_{psi_mode}_SAM_0.4_(S)(PR)(r)_1680458526.npy"
    arr_4 =f"pqcprep/outputs/weights_6_{m}(0)_{L_phase}_600_{psi_mode}_SAM_0.6_(S)(PR)(r)_1680458526.npy"
    arr_5 =f"pqcprep/outputs/weights_6_{m}(0)_{L_phase}_600_{psi_mode}_SAM_0.8_(S)(PR)(r)_1680458526.npy"
    arr_6 =f"pqcprep/outputs/weights_6_{m}(0)_{L_phase}_600_{psi_mode}_SAM_0.8_(S)(PR)(r)_1680458526.npy"
    
    weights_arr =np.array([arr_1, arr_2, arr_3, arr_4, arr_5, arr_6])
    colours = ["red", "blue", "green", "purple", "orange", "cyan"]
    N = len(weights_arr)
    cmap = plt.get_cmap('viridis', N)

    phase_arr = np.empty(N, dtype="object")

    for i in np.arange(N):

        state_vec = full_encode(n,m, weights_ampl, str(weights_arr[i]), L_ampl, L_phase,real_p=real_p,repeat_params=repeat_params)
        amplitude = np.abs(state_vec)
        phase = np.angle(state_vec) + 2* np.pi * (np.angle(state_vec) < -np.pi).astype(int)
        phase *= (amplitude > 1e-15).astype(float) 
        phase_arr[i]=phase 

        bar =np.array(list(np.load("pqcprep/outputs/mismatch_by_state"+weights_arr[i][23:],allow_pickle='TRUE').item().values()))
        mu = np.mean(bar) 
        sigma = np.std(bar)
        norm = np.sum(amplitude**2)
        eps = 1 - norm 
        chi = np.mean(np.abs(phase - phase_rounded))
        omega= 1/(mu+sigma+eps+chi)

        print(f"[{loss_arr[i]}] norm: {norm:.5f}; epsilon: {eps:.3e}; chi: {chi:.3e}; mu: {mu:.3e}; sigma: {sigma:.3e}; omega: {omega:.3f}") 

    if delta_round:
        phase_target = psi(np.linspace(0, 2**n, len(x_arr)),mode=psi_mode) # set back to previous value for the following

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})
    
    ax[0].plot(x_arr,phase_target, color="black")
    ax[0].plot(x_arr,phase_rounded, color="gray", ls="--")
    
    for i in np.arange(N):
        ax[0].scatter(x_arr,phase_arr[i],label=f"{loss_arr[i]}", color=cmap(i))

    ax[0].set_ylabel(r"$\Psi (f)$", fontsize=fontsize)
    ax[0].legend(fontsize=fontsize, loc='upper left')
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xticks([])

    if delta_round:
        phase_target = phase_rounded

    for i in np.arange(N):
        ax[1].scatter(x_arr,phase_target-phase_arr[i],label=f"{loss_arr[i]}", color=cmap(i))
    
    ax[1].set_ylabel(r"$\Delta \Psi(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(f"full_encode/phase_loss_comp{pdf_str}", bbox_inches='tight', dpi=500)

    if show:
        plt.show()                

if phase_shift_comp==True:
    """
    PLOT QCNN PHASE VERSUS TARGET FOR IL SHIFT OR NOT
    """
    loss_arr =np.array(["shifts", "no shifts"])
    arr_1 ="pqcprep/outputs/weights_6_3(0)_6_600_psi_MM_(S)(PR)(r).npy"
    arr_2 ="pqcprep/outputs/weights_6_3(0)_6_600_psi_MM_noshift(S)(PR)(r).npy"#"OLD/outputs/weights_6_3(0)_6_600_psi_MM_linear(S)(PR)(r).npy"  #
    
    weights_arr =np.array([arr_1, arr_2])
    colours = ["red", "blue"]
    N = len(weights_arr)

    phase_arr = np.empty(N, dtype="object")

    for i in np.arange(N):

        state_vec = full_encode(n,m, weights_ampl, weights_arr[i], L_ampl, L_phase,real_p=real_p,repeat_params=repeat_params)
        amplitude = np.abs(state_vec)
        phase = np.angle(state_vec) + 2* np.pi * (np.angle(state_vec) < -np.pi).astype(int)
        phase *= (amplitude > 1e-15).astype(float) 
        phase_arr[i]=phase 

        bar =np.array(list(np.load("outputs/bar"+weights_arr[i][15:],allow_pickle='TRUE').item().values()))
        mu = np.mean(bar) 
        sigma = np.std(bar)
        norm = np.sum(amplitude**2)
        eps = 1 - norm 
        chi = np.mean(np.abs(phase - phase_rounded))
        omega= 1/(mu+sigma+eps+chi)

        print(f"[{loss_arr[i]}] norm: {norm:.5f}; epsilon: {eps:.3e}; chi: {chi:.3e}; mu: {mu:.3e}; sigma: {sigma:.3e}; omega: {omega:.3f}  ")

    if delta_round:
        phase_target = psi(np.linspace(0, 2**n, len(x_arr)), mode=psi_mode) # set back to previous value for the following

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})
    cmap = plt.get_cmap('Dark2', N)

    ax[0].plot(x_arr,phase_target, color="black")
    ax[0].plot(x_arr,phase_rounded, color="gray", ls="--")
    
    for i in np.arange(N):
        ax[0].scatter(x_arr,phase_arr[i],label=f"{loss_arr[i]}", color=colours[i])

    ax[0].set_ylabel(r"$\Psi (f)$", fontsize=fontsize)
    ax[0].legend(fontsize=fontsize, loc='upper left')
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xticks([])

    if delta_round:
        phase_target = phase_rounded

    for i in np.arange(N):
        ax[1].scatter(x_arr,phase_target-phase_arr[i],label=f"{loss_arr[i]}", color=colours[i])
    
    ax[1].set_ylabel(r"$\Delta \Psi(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(f"full_encode/phase_shift_comp{pdf_str}", bbox_inches='tight', dpi=500)

    if show:
        plt.show()  


if phase_RP_comp==True:
    """
    PLOT QCNN PHASE VERSUS TARGET FOR DIFFERENT RP SETTINGS
    """
    RP_arr =np.array([None,"CL","IL", "both"])
    label_arr =np.array(["none","CL","IL", "both"])
    arr_1 ="pqcprep/outputs/weights_6_4(0)_6_600_psi_MM_quadratic(S)(PR)(r).npy"
    arr_2 ="pqcprep/outputs/weights_6_4(0)_6_600_psi_MM_quadratic(S)(PR)(r)(CL).npy"
    arr_3 ="pqcprep/outputs/weights_6_4(0)_6_600_psi_MM_quadratic(S)(PR)(r)(IL).npy"
    arr_4 ="pqcprep/outputs/weights_6_4(0)_6_600_psi_MM_quadratic(S)(PR)(r)(both).npy"
    
    weights_arr =np.array([arr_1, arr_2, arr_3, arr_4])
    colours = ["red", "blue", "green", "purple"]
    N = len(weights_arr)

    phase_arr = np.empty(N, dtype="object")

    for i in np.arange(N):

        state_vec = full_encode(n,m, weights_ampl, weights_arr[i], L_ampl, L_phase,real_p=real_p,repeat_params=RP_arr[i])
        amplitude = np.abs(state_vec)
        phase = np.angle(state_vec) + 2* np.pi * (np.angle(state_vec) < -np.pi).astype(int)
        phase *= (amplitude > 1e-15).astype(float) 
        phase_arr[i]=phase 

        bar =np.array(list(np.load("outputs/bar"+weights_arr[i][15:],allow_pickle='TRUE').item().values()))
        mu = np.mean(bar) 
        sigma = np.std(bar)
        norm = np.sum(amplitude**2)
        eps = 1 - norm 
        chi = np.mean(np.abs(phase - phase_rounded))
        omega= 1/(mu+sigma+eps+chi)

        print(f"[{label_arr[i]}] norm: {norm:.5f}; epsilon: {eps:.3e}; chi: {chi:.3e}; mu: {mu:.3e}; sigma: {sigma:.3e}; omega: {omega:.3f}  ")

    if delta_round:
        phase_target = psi(np.linspace(0, 2**n, len(x_arr)),mode=psi_mode) # set back to previous value for the following

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})
    cmap = plt.get_cmap('Dark2', N)

    ax[0].plot(x_arr,phase_target, color="black")
    ax[0].plot(x_arr,phase_rounded, color="gray", ls="--")
    
    for i in np.arange(N):
        ax[0].scatter(x_arr,phase_arr[i],label=f"{label_arr[i]}", color=colours[i])

    ax[0].set_ylabel(r"$\Psi (f)$", fontsize=fontsize)
    ax[0].legend(fontsize=fontsize, loc='upper left')
    ax[0].tick_params(axis="both", labelsize=ticksize)
    ax[0].set_xticks([])

    if delta_round:
        phase_target = phase_rounded

    for i in np.arange(N):
        ax[1].scatter(x_arr,phase_target-phase_arr[i],label=f"{label_arr[i]}", color=colours[i])
    
    ax[1].set_ylabel(r"$\Delta \Psi(f)$", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=ticksize)
    ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(f"full_encode/phase_RP_comp{pdf_str}", bbox_inches='tight', dpi=500)

    if show:
        plt.show()        