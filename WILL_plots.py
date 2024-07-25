import numpy as np 
import os
from fractions import Fraction
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib import rcParams
from tools import psi, dec_to_bin, bin_to_dec, full_encode

m=3 
L=6 
e=600  

p_arr =np.array([0.25, 0.5,0.75,1,1.25,1.5,1.75,2]) 
q_arr =np.array([0,0.5,1, 1.5, 2, 2.5, 3]) 

pdf=True
verbose=True
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
 
def get_data(m,L,e,p,q):
    # assuming "default conditions"
    file=os.path.join("outputs",f"bar_6_{m}(0)_{L}_{e}_psi_WILL_(S)(PR)(r)_{Fraction(p).numerator}-{Fraction(p).denominator}_{Fraction(q).numerator}-{Fraction(q).denominator}.npy") 
    alt_file=os.path.join("outputs",f"bar_6_{m}(0)_{L}_{e}_psi_WILL_(S)(PR)(r)_{p if p != 1 else int(p)}_{q}.npy")
    if os.path.isfile(file):
        dic = np.load(file,allow_pickle='TRUE').item()
    elif os.path.isfile(alt_file):
        dic = np.load(alt_file,allow_pickle='TRUE').item()    
    else:
        raise FileNotFoundError(f"File '{file}' or '{alt_file}' not found.")    
    vals=np.array(list(dic.values()))

    return np.mean(vals), np.std(vals) 

def get_phase_target(m):

    # define x array 
    n = 6
    x_min = 40
    x_max = 168 
    dx = (x_max-x_min)/(2**n) 
    x_arr = np.arange(x_min, x_max, dx) 

    # calculate target output for phase 
    phase_target = psi(np.linspace(0, 2**n, len(x_arr)))

    # calculate target for phase taking into account rounding 
    phase_reduced = np.modf(phase_target / (2* np.pi))[0] 
    phase_reduced_bin = [dec_to_bin(i,m, "unsigned mag", 0) for i in phase_reduced]
    phase_reduced_dec =  np.array([bin_to_dec(i,"unsigned mag", 0) for i in phase_reduced_bin])
    phase_rounded = 2 * np.pi * phase_reduced_dec

    return phase_rounded

def get_eps_chi(m,L,e,p,q):
    # assume "default conditions"

    file=os.path.join("outputs",f"weights_6_{m}(0)_{L}_{e}_psi_WILL_(S)(PR)(r)_{Fraction(p).numerator}-{Fraction(p).denominator}_{Fraction(q).numerator}-{Fraction(q).denominator}.npy") 
    alt_file=os.path.join("outputs",f"weights_6_{m}(0)_{L}_{e}_psi_WILL_(S)(PR)(r)_{p if p != 1 else int(p)}_{q}.npy")
    if os.path.isfile(file):
        weights_phase=file 
    elif os.path.isfile(alt_file):
        weights_phase=alt_file    
    else:
        raise FileNotFoundError(f"File '{file}' or '{alt_file}' not found.")
     
    weights_ampl = "ampl_outputs/weights_6_3_600_x76_MM_40_168_zeros.npy" 
    state_vec = full_encode(6,m, weights_ampl, weights_phase, 3, L,real_p=True,repeat_params=None)

    amplitude = np.abs(state_vec)
    phase = np.angle(state_vec) + 2* np.pi * (np.angle(state_vec) < -np.pi).astype(int)
    phase *= (amplitude > 1e-15).astype(float) 

    phase_rounded=get_phase_target(m)

    epsilon = 1 - np.sum(amplitude**2)
    chi = np.mean(np.abs(phase - phase_rounded))

    return epsilon, chi 

#------------------------------------------------------------------------------
p_arr = np.append(p_arr[0],p_arr)
q_arr = np.append(q_arr[0],q_arr)

mean = np.empty((len(q_arr)-1,len(p_arr)-1))
std = np.empty((len(q_arr)-1,len(p_arr)-1))
chi = np.empty((len(q_arr)-1,len(p_arr)-1))
eps = np.empty((len(q_arr)-1,len(p_arr)-1))
omega =np.empty((len(q_arr)-1,len(p_arr)-1)) 

for i in np.arange(len(p_arr)-1):
    for j in np.arange(len(q_arr)-1):
        mean[j,i], std[j,i] = get_data(m,L,e, p_arr[i+1], q_arr[j+1])
        eps[j,i], chi[j,i] = get_eps_chi(m,L,e, p_arr[i+1], q_arr[j+1])

        omega[j,i] = 1/(mean[j,i] + std[j,i] + eps[j,i] + chi[j,i])

        if verbose:
            print(f"[p={p_arr[i+1]:.2f}; q={q_arr[j+1]:.2f}] mean: {mean[j,i]:.2e}; std: {std[j,i]:.2e}; eps: {eps[j,i]:.2e}; chi: {chi[j,i]:.2e}; omega: {omega[j,i]:.2e}")
        
#------------------
mean_ind=np.unravel_index(np.argmin(mean, axis=None), mean.shape) 
std_ind=np.unravel_index(np.argmin(std, axis=None), mean.shape) 
eps_ind=np.unravel_index(np.argmin(eps, axis=None), mean.shape) 
chi_ind=np.unravel_index(np.argmin(chi, axis=None), mean.shape) 
omega_ind=np.unravel_index(np.argmax(omega, axis=None), mean.shape) 

print("--------------")
print(f"Minimum mu: p={p_arr[mean_ind[1]+1]:.2f}, q={q_arr[mean_ind[0]+1]:.2f}")
print(f"Minimum sigma: p={p_arr[std_ind[1]+1]:.2f}, q={q_arr[std_ind[0]+1]:.2f}")
print(f"Minimum epsilon: p={p_arr[eps_ind[1]+1]:.2f}, q={q_arr[eps_ind[0]+1]:.2f}")
print(f"Minimum chi: p={p_arr[chi_ind[1]+1]:.2f}, q={q_arr[chi_ind[0]+1]:.2f}")
print(f"Maximum omega: p={p_arr[omega_ind[1]+1]:.2f}, q={q_arr[omega_ind[0]+1]:.2f}")
#------------------

arrays = np.array([mean, std, eps, chi,omega])   
labels = np.array(["mean", "std", "eps", "chi", "omega"])   

for i in np.arange(len(arrays)):
    plt.figure(figsize=figsize)
    plt.ylabel("q", fontsize=fontsize)
    plt.xlabel("p", fontsize=fontsize)
    plt.xticks(p_arr[1:], labels=p_arr[1:])
    plt.yticks(q_arr[1:], labels=q_arr[1:])
    if labels[i] !="omega": 
        plt.pcolormesh(p_arr, q_arr, arrays[i],norm=colors.LogNorm(vmin=arrays[i].min(), vmax=arrays[i].max()), cmap="turbo")   
        cb=plt.colorbar(location='top', orientation='horizontal', pad=0.05) # format="{x:.1e}",
    else: 
        plt.pcolormesh(p_arr, q_arr, arrays[i],cmap="turbo_r")   
        cb=plt.colorbar(location='top', orientation='horizontal', pad=0.05) # format="{x:.1e}"    
    cb.ax.tick_params(which='both',labelsize=ticksize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.tight_layout()
    plt.savefig(f"WILL/{labels[i]}_{m}_{L}_{e}{pdf_str}", bbox_inches='tight', dpi=500)
    plt.show()