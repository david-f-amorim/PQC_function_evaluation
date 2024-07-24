import numpy as np 
import os
from fractions import Fraction
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib import rcParams

m=3 
L=9 
e=600  

p_arr =np.array([0.25, 0.5,0.75,1,1.25,1.5,1.75,2]) 
q_arr =np.array([0,0.5,1, 1.5, 2, 2.5, 3]) 

pdf=False
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

    return np.mean(vals), np.std(vals), np.median(vals)  

#------------------------------------------------------------------------------
p_arr = np.append(p_arr[0],p_arr)
q_arr = np.append(q_arr[0],q_arr)

mean = np.empty((len(q_arr)-1,len(p_arr)-1))
std = np.empty((len(q_arr)-1,len(p_arr)-1))
median = np.empty((len(q_arr)-1,len(p_arr)-1))

for i in np.arange(len(p_arr)-1):
    for j in np.arange(len(q_arr)-1):
        mean[j,i], std[j,i], median[j,i] = get_data(m,L,e, p_arr[i+1], q_arr[j+1])
        
arrays = np.array([mean, std])   
labels = np.array(["mean", "std"])   

for i in np.arange(len(arrays)):
    plt.figure(figsize=figsize)
    plt.ylabel("q", fontsize=fontsize)
    plt.xlabel("p", fontsize=fontsize)
    plt.xticks(p_arr[1:], labels=p_arr[1:])
    plt.yticks(q_arr[1:], labels=q_arr[1:])
    plt.pcolormesh(p_arr, q_arr, arrays[i],norm=colors.LogNorm(vmin=arrays[i].min(), vmax=arrays[i].max()), cmap="turbo")# TURBO !! or nipy_spectral
    cb=plt.colorbar(location='top', orientation='horizontal', pad=0.05) # format="{x:.1e}",
    cb.ax.tick_params(which='both',labelsize=ticksize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.tight_layout()
    plt.savefig(f"WILL/{labels[i]}_{m}_{L}_{e}{pdf_str}", bbox_inches='tight', dpi=500)
    plt.show()