import numpy as np 
import matplotlib.pyplot as plt 
from tools import test_QNN, f
from matplotlib import rcParams
import os 

# general settings
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=28
titlesize=32
ticksize=22
figsize=(10,10)


"""
Plot performance for single-input use 
"""

def single_input(n,m,L,epochs,func_str, comp=False): 

    arr = range(2**n)
    
    str_arr = [f"_{n}_{m}_{L}_{epochs}_{func_str}_s{i}" for i in arr]

    mis_arr = [np.load(os.path.join("outputs", f"mismatch{str_arr[i]}.npy")) for i in arr]
    epoch_arr = np.arange(len(mis_arr[0]))+1 

    if comp:
        mix_arr = np.load(os.path.join("outputs", f"mismatch_{n}_{m}_{L}_{epochs}.npy"))

    plt.figure(figsize=figsize)
    for i in arr:
        plt.scatter(epoch_arr, mis_arr[i], label=f"x={bin(i)[2:]}")

    if comp:
        plt.scatter(epoch_arr, mix_arr, label="all x")

    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}", fontsize=titlesize)

    plt.savefig(os.path.join("plots", f"_{n}_{m}_{L}_{epochs}_{func_str}_s"), dpi=500)
    plt.show()

    return 0

"""
Show standard result (mismatch as function of epoch)
"""

def standard(n,m,L,epochs,func_str):

    mismatch = np.load(os.path.join("outputs", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}.npy"))

    plt.figure(figsize=figsize)
    
    plt.scatter(np.arange(len(mismatch))+1, mismatch, label=f"Final: {mismatch[-1]:.2e}", color="blue")

    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}", fontsize=titlesize)

    plt.savefig(os.path.join("plots", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}"), dpi=500)
    plt.show()

    return 0

"""
Show mismatch for various input states after training  
"""

def standard_hist(n,m,L,epochs,func_str, func=f):

    dict = test_QNN(n,m,L,epochs,f, func_str)
    xaxis = list(dict.keys())
    yaxis = list(dict.values())

    label_arr = [f"{np.binary_repr(i)}" for i in xaxis]

    plt.figure(figsize=figsize)
    plt.bar(xaxis, yaxis, color="blue")
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(xaxis, labels=label_arr)

    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}", fontsize=titlesize)
    plt.savefig(os.path.join("plots", f"hist_mismatch_{n}_{m}_{L}_{epochs}_{func_str}"), dpi=500)
    plt.show()
    
    return 0

####

#single_input(2,3,6,100,"2x+1", comp=True)

standard_hist(2,2,6,100,"x")