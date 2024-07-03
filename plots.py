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

DEPRECATED

"""

def single_input(n,m,L,epochs,func_str, comp=False): 

    raise DeprecationWarning("Function needs to be updated")

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
Show standard result (mismatch and loss as functions of epoch)
"""

def standard(n,m,L,epochs,func_str, loss_str, meta):

    # mismatch 
    mismatch = np.load(os.path.join("outputs", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))

    plt.figure(figsize=figsize)
    
    plt.scatter(np.arange(len(mismatch))+1, mismatch, label=f"Final: {np.mean(mismatch[-5:]):.2e}", color="blue")

    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    plt.savefig(os.path.join("plots", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"), dpi=500)
    plt.show()

    # loss
    loss = np.load(os.path.join("outputs", f"loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))

    plt.figure(figsize=figsize)
    
    plt.scatter(np.arange(len(loss))+1, loss, label=f"Final: {np.mean(loss[-5:]):.2e}", color="blue")

    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    plt.savefig(os.path.join("plots", f"loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"), dpi=500)
    plt.show()

    return 0

"""
Show mismatch for various input states after training  
"""

def standard_bar(n,m,L,epochs,func_str,loss,meta):

    dic = np.load(os.path.join("outputs",f"bar_{n}_{m}_{L}_{epochs}_{func_str}_{loss}_{meta}.npy"),allow_pickle='TRUE').item()
    xaxis = list(dic.keys())
    yaxis = list(dic.values())

    label_arr = [f"{np.binary_repr(i,n)}" for i in xaxis]
    
    plt.figure(figsize=figsize)
    plt.bar(xaxis, yaxis, color="blue",align='center')
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(xaxis, labels=label_arr)

    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {loss}, {meta}", fontsize=titlesize)
    plt.savefig(os.path.join("plots", f"bar_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss}_{meta}"), dpi=500)
    plt.show()
    
    return 0

"""
Compare mismatch for different loss functions
"""

def comp_loss_funcs(n,m,L,epochs, func_str,loss_str_arr, meta):

    mismatch_arr = np.empty(len(loss_str_arr), dtype=object)
    bar_arr = np.empty(len(loss_str_arr), dtype=object)

    for i in np.arange(len(loss_str_arr)):
        mismatch = np.load(os.path.join("outputs", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str_arr[i]}_{meta}.npy")) 
        bar = np.load(os.path.join("outputs",f"bar_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str_arr[i]}_{meta}.npy"),allow_pickle='TRUE').item()

        mismatch_arr[i]=mismatch 
        bar_arr[i]=np.array(list(bar.values()))

    bar_labels = [f"{np.binary_repr(i,n)}" for i in list(bar.keys())]  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(loss_str_arr)):
        plt.scatter(np.arange(len(mismatch_arr[i]))+1, mismatch_arr[i], label=loss_str_arr[i])

    plt.legend(fontsize=fontsize)
    plt.savefig(os.path.join("plots", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}_lfcomp_{meta}"), dpi=500)
    plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {meta}", fontsize=titlesize)

    width=1/(len(loss_str_arr)+1) 
   
    for i in np.arange(len(loss_str_arr)):
        plt.bar(list(bar.keys())+width*i,bar_arr[i], width=width, label=loss_str_arr[i],align='edge')

    plt.legend(fontsize=fontsize)
    plt.savefig(os.path.join("plots", f"bar_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_lfcomp_{meta}"), dpi=500)
    plt.show()

    return 0

"""
Compare results for QCNNs with different numbers of layers 
(expecting everything else to be identical)
"""

def comp_L(n,m,L_arr,epochs, func_str,loss_str, meta):

    mismatch_arr = np.empty(len(L_arr), dtype=object)
    loss_arr = np.empty(len(L_arr), dtype=object)
    bar_arr = np.empty(len(L_arr), dtype=object)

    for i in np.arange(len(L_arr)):
        mismatch = np.load(os.path.join("outputs", f"mismatch_{n}_{m}_{L_arr[i]}_{epochs}_{func_str}_{loss_str}_{meta}.npy")) 
        loss= np.load(os.path.join("outputs", f"loss_{n}_{m}_{L_arr[i]}_{epochs}_{func_str}_{loss_str}_{meta}.npy")) 
        bar = np.load(os.path.join("outputs",f"bar_{n}_{m}_{L_arr[i]}_{epochs}_{func_str}_{loss_str}_{meta}.npy"),allow_pickle='TRUE').item()

        mismatch_arr[i]=mismatch 
        loss_arr[i]=loss 
        bar_arr[i]=np.array(list(bar.values()))

    bar_labels = [f"{np.binary_repr(i,n)}" for i in list(bar.keys())]  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(L_arr)):
        plt.scatter(np.arange(len(mismatch_arr[i]))+1, mismatch_arr[i], label=f"L={L_arr[i]}")

    plt.legend(fontsize=fontsize)
    plt.savefig(os.path.join("plots", f"mismatch_{n}_{m}_{epochs}_{func_str}_{loss_str}_Lcomp_{meta}"), dpi=500)
    plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(L_arr)):
        plt.scatter(np.arange(len(loss_arr[i]))+1, mismatch_arr[i], label=f"L={L_arr[i]}")

    plt.legend(fontsize=fontsize)
    plt.savefig(os.path.join("plots", f"loss_{n}_{m}_{epochs}_{func_str}_{loss_str}_Lcomp_{meta}"), dpi=500)
    plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}, m={m}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    width=1/(len(L_arr)+1) 
   
    for i in np.arange(len(L_arr)):
        plt.bar(list(bar.keys())+width*i,bar_arr[i], width=width, label=f"L={L_arr[i]}",align='edge')

    plt.legend(fontsize=fontsize)
    plt.savefig(os.path.join("plots", f"bar_mismatch_{n}_{m}_{epochs}_{func_str}_{loss_str}_Lcomp_{meta}"), dpi=500)
    plt.show()

    return 0

####

#standard(2,2,8,300,"x", "CE", meta="")
#standard_bar(2,2,8,300,"x", "CE", meta="")

comp_L(n=2,m=2,L_arr=[6,9,12],epochs=300, func_str="x",loss_str="CE", meta="")

#comp_loss_funcs(2,2,9,300, "x", ["MSE", "L1", "KLD","CE"])
