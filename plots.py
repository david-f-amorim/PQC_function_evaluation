import numpy as np 
import matplotlib.pyplot as plt 
from tools import check_plots
from matplotlib import rcParams
import os, argparse 

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
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}", fontsize=titlesize)

    plt.savefig(os.path.join("plots", f"_{n}_{m}_{L}_{epochs}_{func_str}_s"), dpi=500)
    plt.show()

    return 0

"""
Show standard result (mismatch and loss as functions of epoch)
"""

def standard(n,m,L,epochs,func_str, loss_str, meta, show, log):

    log_str= ("" if log==False else "log_")

    # mismatch 
    mismatch = np.load(os.path.join("outputs", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))

    plt.figure(figsize=figsize)
    
    plt.scatter(np.arange(len(mismatch))+1, mismatch, label=f"Final: {np.mean(mismatch[-5:]):.2e}", color="blue")

    if log:
        plt.yscale('log') 

    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"), dpi=500)
    
    if show:
        plt.show()

    # loss
    loss = np.load(os.path.join("outputs", f"loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))

    plt.figure(figsize=figsize)
    
    plt.scatter(np.arange(len(loss))+1, loss, label=f"Final: {np.mean(loss[-5:]):.2e}", color="blue")

    if log:
        plt.yscale('log') 

    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    plt.savefig(os.path.join("plots", f"{log_str}loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"), dpi=500)
    if show:
        plt.show()

    return 0

"""
Show mismatch for various input states after training  
"""

def standard_bar(n,m,L,epochs,func_str,loss,meta, show, log):

    log_str= ("" if log==False else "log_")

    dic = np.load(os.path.join("outputs",f"bar_{n}_{m}_{L}_{epochs}_{func_str}_{loss}_{meta}.npy"),allow_pickle='TRUE').item()
    xaxis = list(dic.keys())
    yaxis = list(dic.values())

    label_arr = [f"{np.binary_repr(i,n)}" for i in xaxis]
    
    plt.figure(figsize=figsize)
    
    plt.bar(xaxis, yaxis, color="blue",align='center')
   
    if log:
        plt.yscale('log')
        ticks = 10**np.arange(np.floor(np.log10(np.min(yaxis))), np.ceil(np.log10(np.max(yaxis)))+1)
        plt.yticks(ticks=ticks)
 
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(xaxis, labels=label_arr)

    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {loss}, {meta}", fontsize=titlesize)
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss}_{meta}"), dpi=500)
    if show:
        plt.show()
    
    return 0

"""
Compare mismatch for different loss functions
"""

def comp_loss_funcs(n,m,L,epochs, func_str,loss_str_arr, meta, show, log):

    log_str= ("" if log==False else "log_")

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
    
    if log:
        plt.yscale('log') 

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}_{m}_{L}_{epochs}_{func_str}_lfcomp_{meta}"), dpi=500)
    if show:
        plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {meta}", fontsize=titlesize)

    width=1/(len(loss_str_arr)+1) 
    bar_min = 1
    bar_max = 0
   
    for i in np.arange(len(loss_str_arr)):
        plt.bar(list(bar.keys())+width*i,bar_arr[i], width=width, label=loss_str_arr[i],align='center')

        if log:
            bar_min = (np.min(bar_arr[i]) if np.min(bar_arr[i])<bar_min else bar_min)
            bar_max = (np.max(bar_arr[i]) if np.max(bar_arr[i])>bar_max else bar_max)
    
    if log:
        plt.yscale('log')   
        ticks = 10**np.arange(np.floor(np.log10(bar_min)), np.ceil(np.log10(bar_max))+1)
        plt.yticks(ticks=ticks) 

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_lfcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    return 0

"""
Compare results for QCNNs with different numbers of layers 
(expecting everything else to be identical)
"""

def comp_L(n,m,L_arr,epochs, func_str,loss_str, meta, show, log):

    log_str= ("" if log==False else "log_")

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
        
    if log:
        plt.yscale('log')         

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}_{m}_{epochs}_{func_str}_{loss_str}_Lcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    plt.figure(figsize=figsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(L_arr)):
        plt.scatter(np.arange(len(loss_arr[i]))+1, mismatch_arr[i], label=f"L={L_arr[i]}")    

    if log:
        plt.yscale('log') 

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}loss_{n}_{m}_{epochs}_{func_str}_{loss_str}_Lcomp_{meta}"), dpi=500)
    if show:
        plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}, m={m}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    width=1/(len(L_arr)+1) 
    bar_min =1
    bar_max =0
   
    for i in np.arange(len(L_arr)):
        plt.bar(list(bar.keys())+width*i,bar_arr[i], width=width, label=f"L={L_arr[i]}",align='center')

        if log:
            bar_min = (np.min(bar_arr[i]) if np.min(bar_arr[i])<bar_min else bar_min)
            bar_max = (np.max(bar_arr[i]) if np.max(bar_arr[i])>bar_max else bar_max)
    
    if log:
        plt.yscale('log')   
        ticks = 10**np.arange(np.floor(np.log10(bar_min)), np.ceil(np.log10(bar_max))+1)
        plt.yticks(ticks=ticks) 

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}_{m}_{epochs}_{func_str}_{loss_str}_Lcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    return 0

"""
Compare results for QCNNs evaluating different functions 
(expecting everything else to be identical)
"""

def comp_f(n,m,L,epochs, func_str_arr,loss_str, meta, show, log):

    log_str= ("" if log==False else "log_")

    mismatch_arr = np.empty(len(func_str_arr), dtype=object)
    loss_arr = np.empty(len(func_str_arr), dtype=object)
    bar_arr = np.empty(len(func_str_arr), dtype=object)

    for i in np.arange(len(func_str_arr)):
        mismatch = np.load(os.path.join("outputs", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str_arr[i]}_{loss_str}_{meta}.npy")) 
        loss= np.load(os.path.join("outputs", f"loss_{n}_{m}_{L}_{epochs}_{func_str_arr[i]}_{loss_str}_{meta}.npy")) 
        bar = np.load(os.path.join("outputs",f"bar_{n}_{m}_{L}_{epochs}_{func_str_arr[i]}_{loss_str}_{meta}.npy"),allow_pickle='TRUE').item()

        mismatch_arr[i]=mismatch 
        loss_arr[i]=loss 
        bar_arr[i]=np.array(list(bar.values()))

    bar_labels = [f"{np.binary_repr(i,n)}" for i in list(bar.keys())]  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, {loss_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(func_str_arr)):
        plt.scatter(np.arange(len(mismatch_arr[i]))+1, mismatch_arr[i], label=f"f(x)={func_str_arr[i]}")

    if log:
        plt.yscale('log')

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}_{m}_{L}_{epochs}_{loss_str}_fcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    plt.figure(figsize=figsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, {loss_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(func_str_arr)):
        plt.scatter(np.arange(len(loss_arr[i]))+1, mismatch_arr[i],  label=f"f(x)={func_str_arr[i]}")

    if log:
        plt.yscale('log')

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}loss_{n}_{m}_{L}_{epochs}_{loss_str}_fcomp_{meta}"), dpi=500)
    if show:
        plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, {loss_str}, {meta}", fontsize=titlesize)

    width=1/(len(func_str_arr)+1) 
    bar_max =0
    bar_min =1 
   
    for i in np.arange(len(func_str_arr)):
        plt.bar(list(bar.keys())+width*i,bar_arr[i], width=width, label=f"f(x)={func_str_arr[i]}",align='center')

        if log:
            bar_min = (np.min(bar_arr[i]) if np.min(bar_arr[i])<bar_min else bar_min)
            bar_max = (np.max(bar_arr[i]) if np.max(bar_arr[i])>bar_max else bar_max)
    
    if log:
        plt.yscale('log')   
        ticks = 10**np.arange(np.floor(np.log10(bar_min)), np.ceil(np.log10(bar_max))+1)
        plt.yticks(ticks=ticks) 

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}_{m}_{L}_{epochs}_{loss_str}_fcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    return 0


"""
Compare results for QCNNs with different meta strings. 
(expecting everything else to be identical)
"""

def comp_meta(n,m,L,epochs, func_str,loss_str, meta_arr, show, log):

    log_str= ("" if log==False else "log_")

    mismatch_arr = np.empty(len(meta_arr), dtype=object)
    loss_arr = np.empty(len(meta_arr), dtype=object)
    bar_arr = np.empty(len(meta_arr), dtype=object)

    for i in np.arange(len(meta_arr)):
        mismatch = np.load(os.path.join("outputs", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta_arr[i]}.npy")) 
        loss= np.load(os.path.join("outputs", f"loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta_arr[i]}.npy")) 
        bar = np.load(os.path.join("outputs",f"bar_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta_arr[i]}.npy"),allow_pickle='TRUE').item()

        mismatch_arr[i]=mismatch 
        loss_arr[i]=loss 
        bar_arr[i]=np.array(list(bar.values()))

    bar_labels = [f"{np.binary_repr(i,n)}" for i in list(bar.keys())]  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, ", fontsize=titlesize)

    for i in np.arange(len(meta_arr)):
        plt.scatter(np.arange(len(mismatch_arr[i]))+1, mismatch_arr[i], label=f"{meta_arr[i]}")

    if log:
        plt.yscale('log')

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_Mcomp"), dpi=500)
    if show:
        plt.show()

    plt.figure(figsize=figsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, ", fontsize=titlesize)

    for i in np.arange(len(meta_arr)):
        plt.scatter(np.arange(len(loss_arr[i]))+1, mismatch_arr[i],  label=f"{meta_arr[i]}")

    if log:
        plt.yscale('log')

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_Mcomp"), dpi=500)
    if show:
        plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}, m={m}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, ", fontsize=titlesize)

    width=1/(len(meta_arr)+1) 
    bar_max =0
    bar_min =1 
   
    for i in np.arange(len(meta_arr)):
        plt.bar(list(bar.keys())+width*i,bar_arr[i], width=width, label=f"{meta_arr[i]}",align='center')

        if log:
            bar_min = (np.min(bar_arr[i]) if np.min(bar_arr[i])<bar_min else bar_min)
            bar_max = (np.max(bar_arr[i]) if np.max(bar_arr[i])>bar_max else bar_max)
    
    if log:
        plt.yscale('log')   
        ticks = 10**np.arange(np.floor(np.log10(bar_min)), np.ceil(np.log10(bar_max))+1)
        plt.yticks(ticks=ticks) 

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_Mcomp"), dpi=500)
    if show:
        plt.show()

    return 0

####

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='', description="Generate plots of  QCNN training and testing.")   
    parser.add_argument('-n','--n', help="Number of input qubits.", default=2, type=int)
    parser.add_argument('-m','--m', help="Number of target qubits.", default=2, type=int)
    parser.add_argument('-L','--L', help="Number of network layers.", default=[6],type=int, nargs="+")
    parser.add_argument('-l','--loss', help="Loss function.", default=["CE"], nargs="+")
    parser.add_argument('-fs','--f_str', help="String describing function.",nargs="+", default=["x"])
    parser.add_argument('-e','--epochs', help="Number of epochs.", default=300,type=int)
    parser.add_argument('-cL','--compL', help="Compare different L values (pass multiple).", action='store_true')
    parser.add_argument('-cf','--compf', help="Compare different f_str values (pass multiple).", action='store_true')
    parser.add_argument('-cM','--compM', help="Compare different meta values (pass multiple).", action='store_true')
    parser.add_argument('-cl','--compl', help="Compare different loss values (pass multiple).", action='store_true')
    parser.add_argument('-lg','--log', help="Take logarithm of values.", action='store_true')
    parser.add_argument('-s','--show', help="Display plots in terminal.", action='store_true')
    parser.add_argument('-M','--meta', help="String with meta data.",nargs="+", default=[""])
    parser.add_argument('-I','--ignore_duplicates', help="Ignore and overwrite duplicate files.", action='store_true')

    opt = parser.parse_args()

    if int(opt.compL)+int(opt.compf)+int(opt.compM)+int(opt.compl) > 1:
        raise ValueError("Cannot do two comparisons at once.")

    if opt.compL:
        comp_L(n=opt.n,m=opt.m,L_arr=opt.L,epochs=opt.epochs, func_str=opt.f_str[0],loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log)
    elif opt.compf:
        comp_f(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs, func_str_arr=opt.f_str,loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log)
    elif opt.compM:
        comp_meta(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs, func_str=opt.f_str[0],loss_str=opt.loss[0], meta_arr=opt.meta, show=opt.show, log=opt.log) 
    elif opt.compl:
        comp_loss_funcs(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs, func_str=opt.f_str[0],loss_str_arr=opt.loss, meta=opt.meta[0], show=opt.show, log=opt.log)       
    else:
        dupl_files = check_plots(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs,func_str=opt.f_str,loss_str=opt.loss[0],meta=opt.meta, log=opt.log)

        if dupl_files and opt.ignore_duplicates==False:
            print("\nThe required plots already exist and will not be recomputed. Use '-I' or '--ignore_duplicates' to override this.\n")
        else: 
            standard(n=opt.n, m=opt.m, L=opt.L[0], epochs=opt.epochs, loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, func_str=opt.f_str[0], log=opt.log)
            standard_bar(n=opt.n, m=opt.m, L=opt.L[0], epochs=opt.epochs, loss=opt.loss[0], meta=opt.meta[0], show=opt.show, func_str=opt.f_str[0], log=opt.log)





