import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
from itertools import combinations
from scipy.stats import pearsonr 

# -------------------- AUX FUNCTIONS 
def compile_arrs():

    from tools import psi, dec_to_bin, bin_to_dec, full_encode
    import glob 

    #---------------
    # extract information from name 
    def get_info(file):

        m=int(file[3])
        L=int(file[8])

        if "(both)" in file:
            RP="both"
        elif "(IL)" in file:
            RP="IL"
        elif "(CL)" in file:
            RP="CL"
        else:
            RP=None   

        if "linear" in file:
            psi_mode="linear"
        if "quadratric" in file:
            psi_mode="quadratic"
        else:
            psi_mode="psi"                     

        return m,L,RP, psi_mode  

    # target phase function
    def get_phase_target(m, psi_mode):

        # define x array 
        n = 6
        x_min = 40
        x_max = 168 
        dx = (x_max-x_min)/(2**n) 
        x_arr = np.arange(x_min, x_max, dx) 

        # calculate target output for phase 
        phase_target = psi(np.linspace(0, 2**n, len(x_arr)),mode=psi_mode)

        # calculate target for phase taking into account rounding 
        phase_reduced = np.modf(phase_target / (2* np.pi))[0] 
        phase_reduced_bin = [dec_to_bin(i,m, "unsigned mag", 0) for i in phase_reduced]
        phase_reduced_dec =  np.array([bin_to_dec(i,"unsigned mag", 0) for i in phase_reduced_bin])
        phase_rounded = 2 * np.pi * phase_reduced_dec

        return phase_rounded 

    def get_eps_chi(weights,m,L,RP,psi_mode):
        
        weights_ampl = "ampl_outputs/weights_6_3_600_x76_MM_40_168_zeros.npy" 
        state_vec = full_encode(6,m, weights_ampl, weights, 3, L,real_p=True,repeat_params=RP)

        amplitude = np.abs(state_vec)
        phase = np.angle(state_vec) + 2* np.pi * (np.angle(state_vec) < -np.pi).astype(int)
        phase *= (amplitude > 1e-15).astype(float) 

        phase_rounded=get_phase_target(m,psi_mode)

        epsilon = 1 - np.sum(amplitude**2)
        chi = np.mean(np.abs(phase - phase_rounded))

        return epsilon, chi 

    #--------------

    # collect files 
    N = len(glob.glob("outputs/weights*"))
    files=np.empty(N, dtype=object)
    for i in np.arange(N):
        files[i]=glob.glob("outputs/weights*")[i][15:]

    weights="outputs/weights"+files
    bar="outputs/bar"+files

    # get mu and sigma 
    mu = np.empty(N)
    sigma = np.empty(N) 
    for i in np.arange(N):
        arr=np.array(list(np.load(bar[i],allow_pickle='TRUE').item().values())) 
        mu[i]=np.mean(arr)
        sigma[i]=np.std(arr)

    # get eps and chi
    eps = np.empty(N)
    chi = np.empty(N) 
    for i in np.arange(N):
        m,L,RP, psi_mode= get_info(files[i])
        eps[i], chi[i]=get_eps_chi(weights[i],m,L,RP,psi_mode) 

    # save outputs 
    np.save("correlations/mu", mu)  
    np.save("correlations/sigma", sigma) 
    np.save("correlations/eps", eps) 
    np.save("correlations/chi", chi)    

#--------------- 
mu = np.load("correlations/mu.npy")
sigma = np.load("correlations/sigma.npy")
eps = np.load("correlations/eps.npy")
chi = np.load("correlations/chi.npy")

arr=np.array([mu, sigma, eps, chi]) 
label_arr=["mu", "sigma", "eps", "chi"]
label_arr_greek=[r"$\mu$", r"$\sigma$", r"$\epsilon$", r"$\chi$"]
c=list(combinations(range(4),2))

pdf=True
show=False 

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

for i in np.arange(len(c)):
    x=c[i][0]
    y=c[i][1]

    plt.figure(figsize=figsize)
    plt.ylabel(label_arr_greek[y], fontsize=fontsize)
    plt.xlabel(label_arr_greek[x], fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.scatter(arr[x], arr[y], color="red", label=f"R={pearsonr(arr[x], arr[y])[0]:.3f}, N={len(arr[x])}")
    #z=np.linspace(arr[x].min(), arr[x].max(), 1000)
    #m,b = np.polyfit(arr[x], arr[y],1)
    #plt.plot(z,m*z+b, color="black", ls="--" )
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"correlations/{label_arr[x]}_v_{label_arr[y]}_{pdf_str}", bbox_inches='tight', dpi=500)
    if show:
        plt.show()