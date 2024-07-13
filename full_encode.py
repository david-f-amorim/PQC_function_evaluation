import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
from tools import psi, bin_to_dec, dec_to_bin, full_encode  

# general settings
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=28
titlesize=32
ticksize=22
figsize=(10,10)

# number of qubits 
n = 6
m = 4

# define x array 
x_min = 40
x_max = 168 
dx = (x_max-x_min)/(2**n) 
x_arr = np.arange(x_min, x_max, dx) 

# calculate target output for amplitude 
ampl_target = x_arr**(-7./6)
ampl_target = ampl_target / np.sqrt(np.sum(ampl_target**2))

# calculate target output for phase 
phase_target = psi(np.linspace(0, 2**n, len(x_arr)))

# calculate target output for wavefunc 
h_target = ampl_target * np.exp(2*1.j*np.pi* phase_target)
wave_real_target = np.real(h_target)
wave_im_target = np.imag(h_target)

# calculate target for phase taking into account rounding 
phase_reduced = np.modf(phase_target / (2* np.pi))[0] 
phase_reduced_bin = [dec_to_bin(i,m, "unsigned mag", 0) for i in phase_reduced]
phase_reduced_dec =  np.array([bin_to_dec(i,"unsigned mag", 0) for i in phase_reduced_bin])
phase_rounded = 2 * np.pi * phase_reduced_dec

# calculate target output for wavefunc taking into account rounding 
h_target_rounded = ampl_target * np.exp(2*1.j*np.pi* phase_rounded)
wave_real_target_rounded = np.real(h_target_rounded)
wave_im_target_rounded = np.imag(h_target_rounded)

# load amplitude-only statevector (for QCNN, QGAN, GR)
ampl_vec = np.load("ampl_outputs/statevec_6_8_800_x76_MSE_40_168_.npy")
ampl_vec_QGAN = np.abs(np.load("ampl_outputs/amp_state_QGAN.npy"))
ampl_vec_GR = np.abs(np.load("ampl_outputs/amp_state_QGAN.npy")) # GET GR !!

# load LPF phase 
#LPF_phase = np.load("...") GET DATA !!

# calculate state vector from QCNNs 
weights_ampl = "ampl_outputs/weights_6_8_800_x76_MSE_40_168_.npy"
weights_phase = "outputs/weights_6_4(0)_3_1000_x_L1_(S)(PR)(r).npy"
L_ampl = 8 
L_phase = 3
state_vec = full_encode(n,m, weights_ampl, weights_phase, L_ampl, L_phase,real_p=True)

amplitude = np.abs(state_vec)
phase = np.angle(state_vec) + 2* np.pi * (np.angle(state_vec) < 0).astype(int)
phase *= (amplitude > 1e-15).astype(float) 

# get full wavefunction 
real_wave =np.real(state_vec)
im_wave = np.imag(state_vec)
 
##################################################################################
"""
PLOT QCNN AMPLITUDE VERSUS TARGET AND GR/QGAN 
"""
fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})

ax[0].plot(x_arr,ampl_target, color="black")
ax[0].scatter(x_arr,ampl_vec_QGAN,label="QGAN", color="green")
ax[0].scatter(x_arr,ampl_vec_QGAN,label="GR", color="blue")
ax[0].scatter(x_arr,ampl_vec,label="QCNN", color="red")

ax[0].set_ylabel(r"$\tilde A(f)$", fontsize=fontsize)
ax[0].legend(fontsize=fontsize, loc='upper right')
ax[0].tick_params(axis="both", labelsize=ticksize)
ax[0].set_xticks([])

ax[1].scatter(x_arr,ampl_target-ampl_vec_QGAN,label="QGAN", color="green")
ax[1].scatter(x_arr,ampl_target-ampl_vec_QGAN,label="GR", color="blue")
ax[1].scatter(x_arr,ampl_target-ampl_vec,label="QCNN", color="red")

ax[1].set_ylabel(r"$\Delta \tilde A(f)$", fontsize=fontsize)
ax[1].tick_params(axis="both", labelsize=ticksize)
ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

fig.tight_layout()
fig.savefig("full_encode/amplitude_comp", bbox_inches='tight', dpi=500)

plt.show()

"""
PLOT QCNN PHASE VERSUS TARGET AND LPF 
"""
fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})

ax[0].plot(x_arr,phase_target, color="black")
ax[0].plot(x_arr,phase_rounded, color="gray", ls="--")
ax[0].scatter(x_arr,phase,label="LPF", color="blue")
ax[0].scatter(x_arr,phase,label="QCNN", color="red")

ax[0].set_ylabel(r"$\Psi (f)$", fontsize=fontsize)
ax[0].legend(fontsize=fontsize, loc='upper right')
ax[0].tick_params(axis="both", labelsize=ticksize)
ax[0].set_xticks([])

ax[1].scatter(x_arr,phase_target-phase,label="GR", color="blue")
ax[1].scatter(x_arr,phase_target-phase,label="QCNN", color="red")

ax[1].set_ylabel(r"$\Delta \Psi(f)$", fontsize=fontsize)
ax[1].tick_params(axis="both", labelsize=ticksize)
ax[1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

fig.tight_layout()
fig.savefig("full_encode/phase_comp", bbox_inches='tight', dpi=500)

plt.show()

"""
PLOT WAVEFUNC VERSUS TARGET AND LPF 
"""

fig, ax = plt.subplots(2, 2, figsize=(2*figsize[0],figsize[1]), gridspec_kw={'height_ratios': [1.5, 1]})
fig.subplots_adjust()

ax[0,0].plot(x_arr,wave_real_target, color="black")
ax[0,0].plot(x_arr,wave_real_target_rounded, color="gray", ls="--")
ax[0,0].scatter(x_arr,real_wave,label="LPF & QGAN", color="green")
ax[0,0].scatter(x_arr,real_wave,label="LPF & GR", color="blue")
ax[0,0].scatter(x_arr,real_wave,label="QCNN", color="red")

ax[0,0].set_ylabel(r"$\Re[\tilde h(f)]$", fontsize=fontsize)
ax[0,0].legend(fontsize=fontsize, loc='upper right')
ax[0,0].tick_params(axis="both", labelsize=ticksize)
ax[0,0].set_xticks([])

ax[1,0].scatter(x_arr,wave_real_target -real_wave,label="LPF & QGAN", color="green")
ax[1,0].scatter(x_arr,wave_real_target -real_wave,label="LPF & GR", color="blue")
ax[1,0].scatter(x_arr,wave_real_target -real_wave,label="QCNN", color="red")

ax[1,0].set_ylabel(r"$\Delta \Re[\tilde h(f)]$", fontsize=fontsize)
ax[1,0].tick_params(axis="both", labelsize=ticksize)
ax[1,0].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

ax[0,1].plot(x_arr,wave_im_target, color="black")
ax[0,1].plot(x_arr,wave_im_target_rounded, color="gray", ls="--")
ax[0,1].scatter(x_arr,im_wave,label="LPF & QGAN", color="green")
ax[0,1].scatter(x_arr,im_wave,label="LPF & GR", color="blue")
ax[0,1].scatter(x_arr,im_wave,label="QCNN", color="red")

ax[0,1].set_ylabel(r"$\Im[\tilde h(f)]$", fontsize=fontsize)
ax[0,1].legend(fontsize=fontsize, loc='upper right')
ax[0,1].tick_params(axis="both", labelsize=ticksize)
ax[0,1].set_xticks([])

ax[1,1].scatter(x_arr,wave_im_target -im_wave,label="LPF & QGAN", color="green")
ax[1,1].scatter(x_arr,wave_im_target -im_wave,label="LPF & GR", color="blue")
ax[1,1].scatter(x_arr,wave_im_target -im_wave,label="QCNN", color="red")

ax[1,1].set_ylabel(r"$\Delta \Im[\tilde h(f)]$", fontsize=fontsize)
ax[1,1].tick_params(axis="both", labelsize=ticksize)
ax[1,1].set_xlabel(r"$f$ (Hz)", fontsize=fontsize)

fig.tight_layout()
fig.savefig("full_encode/h_comp", bbox_inches='tight', dpi=500)

plt.show()


