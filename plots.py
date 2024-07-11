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
Show standard result (mismatch and loss as functions of epoch)
"""

def standard(n,m,L,epochs,func_str, loss_str, meta, show, log, nint, mint, phase_reduce):

    # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)'    

    log_str= ("" if log==False else "log_")

    # mismatch 
    mismatch = np.load(os.path.join("outputs", f"mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))

    plt.figure(figsize=figsize)
    
    plt.scatter(np.arange(len(mismatch))+1, mismatch, label=f"Final: {np.mean(mismatch[-5:]):.2e}", color="blue")

    if log:
        plt.yscale('log') 

    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"), dpi=500)
    
    if show:
        plt.show()

    # loss
    loss = np.load(os.path.join("outputs", f"loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))

    plt.figure(figsize=figsize)
    
    plt.scatter(np.arange(len(loss))+1, loss, label=f"Final: {np.mean(loss[-5:]):.2e}", color="blue")

    if log:
        plt.yscale('log') 

    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    plt.savefig(os.path.join("plots", f"{log_str}loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"), dpi=500)
    if show:
        plt.show()

    return 0

"""
Show mismatch for various input states after training  
"""

def standard_bar(n,m,L,epochs,func_str,loss,meta, show,  log, nint, mint, phase_reduce):

     # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)' 

    log_str= ("" if log==False else "log_")

    dic = np.load(os.path.join("outputs",f"bar_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss}_{meta}.npy"),allow_pickle='TRUE').item()
    xaxis = list(dic.keys())
    yaxis = list(dic.values())

    label_arr = [f"{np.binary_repr(i,n)}" for i in xaxis]
    
    plt.figure(figsize=figsize)
    
    plt.bar(xaxis, yaxis, color="blue",align='center', label=f"Mean: {np.mean(yaxis):.2e}\nSTDEV: {np.std(yaxis):.2e}")
   
    if log:
        plt.yscale('log')
        ticks = 10**np.arange(np.floor(np.log10(np.min(yaxis))), np.ceil(np.log10(np.max(yaxis)))+1)
        plt.yticks(ticks=ticks)
 
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(xaxis, labels=label_arr)

    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, f(x)={func_str}, {loss}, {meta}", fontsize=titlesize)
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss}_{meta}"), dpi=500)
    if show:
        plt.show()
    
    return 0

"""
Compare mismatch for different loss functions
"""

def comp_loss_funcs(n,m,L,epochs, func_str,loss_str_arr, meta, show,  log, nint, mint, phase_reduce):

     # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)' 

    log_str= ("" if log==False else "log_")

    mismatch_arr = np.empty(len(loss_str_arr), dtype=object)
    bar_arr = np.empty(len(loss_str_arr), dtype=object)

    for i in np.arange(len(loss_str_arr)):
        mismatch = np.load(os.path.join("outputs", f"mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str_arr[i]}_{meta}.npy")) 
        bar = np.load(os.path.join("outputs",f"bar_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str_arr[i]}_{meta}.npy"),allow_pickle='TRUE').item()

        mismatch_arr[i]=mismatch 
        bar_arr[i]=np.array(list(bar.values()))

    bar_labels = [f"{np.binary_repr(i,n)}" for i in list(bar.keys())]  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, f(x)={func_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(loss_str_arr)):
        plt.scatter(np.arange(len(mismatch_arr[i]))+1, mismatch_arr[i], label=loss_str_arr[i])
    
    if log:
        plt.yscale('log') 

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_lfcomp_{meta}"), dpi=500)
    if show:
        plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, f(x)={func_str}, {meta}", fontsize=titlesize)

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
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_lfcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    return 0

"""
Compare results for QCNNs with different numbers of layers 
(expecting everything else to be identical)
"""

def comp_L(n,m,L_arr,epochs, func_str,loss_str, meta, show,  log, nint, mint, phase_reduce):

     # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)' 

    log_str= ("" if log==False else "log_")

    mismatch_arr = np.empty(len(L_arr), dtype=object)
    loss_arr = np.empty(len(L_arr), dtype=object)
    bar_arr = np.empty(len(L_arr), dtype=object)

    for i in np.arange(len(L_arr)):
        mismatch = np.load(os.path.join("outputs", f"mismatch_{n}{nis}_{m}{mis}_{L_arr[i]}_{epochs}_{func_str}_{loss_str}_{meta}.npy")) 
        loss= np.load(os.path.join("outputs", f"loss_{n}{nis}_{m}{mis}_{L_arr[i]}_{epochs}_{func_str}_{loss_str}_{meta}.npy")) 
        bar = np.load(os.path.join("outputs",f"bar_{n}{nis}_{m}{mis}_{L_arr[i]}_{epochs}_{func_str}_{loss_str}_{meta}.npy"),allow_pickle='TRUE').item()

        mismatch_arr[i]=mismatch 
        loss_arr[i]=loss 
        bar_arr[i]=np.array(list(bar.values()))

    bar_labels = [f"{np.binary_repr(i,n)}" for i in list(bar.keys())]  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}{nis}, m={m}{mis}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(L_arr)):
        plt.scatter(np.arange(len(mismatch_arr[i]))+1, mismatch_arr[i], label=f"L={L_arr[i]}")
        
    if log:
        plt.yscale('log')         

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}{nis}_{m}{mis}_{epochs}_{func_str}_{loss_str}_Lcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    plt.figure(figsize=figsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}{nis}, m={m}{mis}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(L_arr)):
        plt.scatter(np.arange(len(loss_arr[i]))+1, mismatch_arr[i], label=f"L={L_arr[i]}")    

    if log:
        plt.yscale('log') 

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}loss_{n}{nis}_{m}{mis}_{epochs}_{func_str}_{loss_str}_Lcomp_{meta}"), dpi=500)
    if show:
        plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}{nis}, m={m}{mis}, epochs={epochs}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

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
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}{nis}_{m}{mis}_{epochs}_{func_str}_{loss_str}_Lcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    return 0

"""
Compare results for QCNNs evaluating different functions 
(expecting everything else to be identical)
"""

def comp_f(n,m,L,epochs, func_str_arr,loss_str, meta, show,  log, nint, mint, phase_reduce):

     # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)' 

    log_str= ("" if log==False else "log_")

    mismatch_arr = np.empty(len(func_str_arr), dtype=object)
    loss_arr = np.empty(len(func_str_arr), dtype=object)
    bar_arr = np.empty(len(func_str_arr), dtype=object)

    for i in np.arange(len(func_str_arr)):
        mismatch = np.load(os.path.join("outputs", f"mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str_arr[i]}_{loss_str}_{meta}.npy")) 
        loss= np.load(os.path.join("outputs", f"loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str_arr[i]}_{loss_str}_{meta}.npy")) 
        bar = np.load(os.path.join("outputs",f"bar_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str_arr[i]}_{loss_str}_{meta}.npy"),allow_pickle='TRUE').item()

        mismatch_arr[i]=mismatch 
        loss_arr[i]=loss 
        bar_arr[i]=np.array(list(bar.values()))

    bar_labels = [f"{np.binary_repr(i,n)}" for i in list(bar.keys())]  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, {loss_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(func_str_arr)):
        plt.scatter(np.arange(len(mismatch_arr[i]))+1, mismatch_arr[i], label=f"f(x)={func_str_arr[i]}")

    if log:
        plt.yscale('log')

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{loss_str}_fcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    plt.figure(figsize=figsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, {loss_str}, {meta}", fontsize=titlesize)

    for i in np.arange(len(func_str_arr)):
        plt.scatter(np.arange(len(loss_arr[i]))+1, mismatch_arr[i],  label=f"f(x)={func_str_arr[i]}")

    if log:
        plt.yscale('log')

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{loss_str}_fcomp_{meta}"), dpi=500)
    if show:
        plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, {loss_str}, {meta}", fontsize=titlesize)

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
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{loss_str}_fcomp_{meta}"), dpi=500)
    if show:
        plt.show()

    return 0


"""
Compare results for QCNNs with different meta strings. 
(expecting everything else to be identical)
"""

def comp_meta(n,m,L,epochs, func_str,loss_str, meta_arr, show,  log, nint, mint, phase_reduce):

     # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)' 

    log_str= ("" if log==False else "log_")

    mismatch_arr = np.empty(len(meta_arr), dtype=object)
    loss_arr = np.empty(len(meta_arr), dtype=object)
    bar_arr = np.empty(len(meta_arr), dtype=object)

    for i in np.arange(len(meta_arr)):
        mismatch = np.load(os.path.join("outputs", f"mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta_arr[i]}.npy")) 
        loss= np.load(os.path.join("outputs", f"loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta_arr[i]}.npy")) 
        bar = np.load(os.path.join("outputs",f"bar_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta_arr[i]}.npy"),allow_pickle='TRUE').item()

        mismatch_arr[i]=mismatch 
        loss_arr[i]=loss 
        bar_arr[i]=np.array(list(bar.values()))

    bar_labels = [f"{np.binary_repr(i,n)}" for i in list(bar.keys())]  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, ", fontsize=titlesize)

    for i in np.arange(len(meta_arr)):
        plt.scatter(np.arange(len(mismatch_arr[i]))+1, mismatch_arr[i], label=f"{meta_arr[i]}")

    if log:
        plt.yscale('log')

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_Mcomp"), dpi=500)
    if show:
        plt.show()

    plt.figure(figsize=figsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, ", fontsize=titlesize)

    for i in np.arange(len(meta_arr)):
        plt.scatter(np.arange(len(loss_arr[i]))+1, mismatch_arr[i],  label=f"{meta_arr[i]}")

    if log:
        plt.yscale('log')

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_Mcomp"), dpi=500)
    if show:
        plt.show()  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, epochs={epochs}, f(x)={func_str}, {loss_str}, ", fontsize=titlesize)

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
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_Mcomp"), dpi=500)
    if show:
        plt.show()

    return 0


"""
Compare results for QCNNs with different epochs. 
(expecting everything else to be identical)
"""

def comp_epochs(n,m,L,epochs_arr, func_str,loss_str, meta, show,  log, nint, mint, phase_reduce):

     # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)' 

    log_str= ("" if log==False else "log_")

    bar_arr = np.empty(len(epochs_arr), dtype=object)

    for i in np.arange(len(epochs_arr)):
        
        bar = np.load(os.path.join("outputs",f"bar_{n}{nis}_{m}{mis}_{L}_{epochs_arr[i]}_{func_str}_{loss_str}_{meta}.npy"),allow_pickle='TRUE').item()
        bar_arr[i]=np.array(list(bar.values()))

    bar_labels = [f"{np.binary_repr(i,n)}" for i in list(bar.keys())]  

    plt.figure(figsize=figsize)
    plt.ylabel("Mismatch", fontsize=fontsize)
    plt.xlabel("Input State", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.xticks(list(bar.keys()), labels=bar_labels)
    plt.title(f"n={n}{nis}, m={m}{mis}, L={L}, f(x)={func_str}, {loss_str}, {meta}", fontsize=titlesize)

    width=1/(len(epochs_arr)+1) 
    bar_max =0
    bar_min =1 
   
    for i in np.arange(len(epochs_arr)):
        plt.bar(list(bar.keys())+width*i,bar_arr[i], width=width, label=f"epochs={epochs_arr[i]}",align='center')

        if log:
            bar_min = (np.min(bar_arr[i]) if np.min(bar_arr[i])<bar_min else bar_min)
            bar_max = (np.max(bar_arr[i]) if np.max(bar_arr[i])>bar_max else bar_max)
    
    if log:
        plt.yscale('log')   
        ticks = 10**np.arange(np.floor(np.log10(bar_min)), np.ceil(np.log10(bar_max))+1)
        plt.yticks(ticks=ticks) 

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}bar_mismatch_{n}{nis}_{m}{mis}_{L}_{func_str}_{loss_str}_{meta}_ecomp"), dpi=500)
    if show:
        plt.show()

    return 0


"""
Compare average mismatch for QCNNs with different epochs and L values 
(expecting everything else to be identical)
"""

def comp_mean_mismatch(n,m,L_arr,epochs_arr, func_str,loss, meta, show,  log, nint, mint, phase_reduce):

     # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)' 

    log_str= ("" if log==False else "log_") 

    mean_arr = np.empty(shape=(len(L_arr), len(epochs_arr)))
    stdev_arr= np.empty(shape=(len(L_arr), len(epochs_arr)))

    for i in range(len(L_arr)):
        for j in range(len(epochs_arr)):
            dict = np.load(os.path.join("outputs",f"bar_{n}{nis}_{m}{mis}_{L_arr[i]}_{epochs_arr[j]}_{func_str}_{loss}_{meta}.npy"),allow_pickle='TRUE').item()
            vals = list(dict.values())

            mean_arr[i,j] = np.mean(vals)
            stdev_arr[i,j] = np.std(vals) 
    
    plt.figure(figsize=figsize)
    plt.ylabel("Mean Mismatch", fontsize=fontsize)
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)

    plt.title(f"n={n}, m={m} f(x)={func_str}, {loss}, {meta}", fontsize=titlesize)

    for i in range(len(L_arr)):
        plt.errorbar(epochs_arr, mean_arr[i,:], yerr=None, label=f"L={L_arr[i]}", fmt='o', linestyle='--', capsize=4, markersize=10)

    if log:
        plt.yscale('log')
        ticks = 10**np.arange(np.floor(np.log10(np.min(mean_arr))), np.ceil(np.log10(np.max(mean_arr)))+1)
        plt.yticks(ticks=ticks) 
       
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.savefig(os.path.join("plots", f"{log_str}mean_mismatch_{n}{nis}_{m}{mis}_{func_str}_{loss}_{meta}"), dpi=500)
    if show:
        plt.show()
    
    plt.figure(figsize=figsize)

    for i in range(len(L_arr)):
        plt.errorbar(epochs_arr, stdev_arr[i,:], yerr=None, label=f"L={L_arr[i]}", fmt='o', linestyle='--', capsize=4, markersize=10)
    
    if log:
        plt.yscale('log')
        ticks = 10**np.arange(np.floor(np.log10(np.min(stdev_arr))), np.ceil(np.log10(np.max(stdev_arr)))+1)
        plt.yticks(ticks=ticks) 
        
    plt.ylabel("STDEV Mismatch", fontsize=fontsize)
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.title(f"n={n}, m={m} f(x)={func_str}, {loss}, {meta}", fontsize=titlesize)
    plt.savefig(os.path.join("plots", f"{log_str}stdev_mismatch_{n}{nis}_{m}{mis}_{func_str}_{loss}_{meta}"), dpi=500)
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
    parser.add_argument('-e','--epochs', help="Number of epochs.", default=[800],nargs="+", type=int)
    parser.add_argument('-ni','--nint', help="Number of integer input qubits.", default=None, type=int)
    parser.add_argument('-mi','--mint', help="Number of integer target qubits.", default=None, type=int)

    parser.add_argument('-PR','--phase_reduce', help="Reduce function values to a phase between 0 and 1.", action='store_true')

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

    opt = parser.parse_args()

    if int(opt.compL)+int(opt.compf)+int(opt.compM)+int(opt.compl)+int(opt.compe)+int(opt.compeL) > 1:
        raise ValueError("Cannot do two comparisons at once.")

    if opt.compL:
        comp_L(n=opt.n,m=opt.m,L_arr=opt.L,epochs=opt.epochs[0], func_str=opt.f_str[0],loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce)
    elif opt.compf:
        comp_f(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs[0], func_str_arr=opt.f_str,loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce)
    elif opt.compM:
        comp_meta(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs[0], func_str=opt.f_str[0],loss_str=opt.loss[0], meta_arr=opt.meta, show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce) 
    elif opt.compl:
        comp_loss_funcs(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs[0], func_str=opt.f_str[0],loss_str_arr=opt.loss, meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce)
    elif opt.compe:
        comp_epochs(n=opt.n,m=opt.m,L=opt.L[0],epochs_arr=opt.epochs, func_str=opt.f_str[0],loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce)
    elif opt.compeL:
        comp_mean_mismatch(n=opt.n,m=opt.m,L_arr=opt.L,epochs_arr=opt.epochs, func_str=opt.f_str[0],loss=opt.loss[0], meta=opt.meta[0], show=opt.show, log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce)
    else:
        dupl_files = check_plots(n=opt.n,m=opt.m,L=opt.L[0],epochs=opt.epochs[0],func_str=opt.f_str[0],loss_str=opt.loss[0],meta=opt.meta[0], log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce)

        if dupl_files and opt.ignore_duplicates==False:
            print("\nThe required plots already exist and will not be recomputed. Use '-I' or '--ignore_duplicates' to override this.\n")
        else: 
            standard(n=opt.n, m=opt.m, L=opt.L[0], epochs=opt.epochs[0], loss_str=opt.loss[0], meta=opt.meta[0], show=opt.show, func_str=opt.f_str[0], log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce)
            standard_bar(n=opt.n, m=opt.m, L=opt.L[0], epochs=opt.epochs[0], loss=opt.loss[0], meta=opt.meta[0], show=opt.show, func_str=opt.f_str[0], log=opt.log, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce)





