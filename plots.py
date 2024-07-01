import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import os 

# general settings
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=16
ticksize=22
figsize=(10,10)

## specifier string
s ="_2_2_8_200"

# import data
loss_file = os.path.join("outputs","loss"+s+".npy")
mismatch_file = os.path.join("outputs","mismatch"+s+".npy")

# configure arrays
loss_arr = np.load(loss_file)
mismatch_arr = np.load(mismatch_file)
fidelity_arr = 1- mismatch_arr
epochs = np.arange(len(loss_arr))+1

# set up plots 
fig, axs = plt.subplots(1,3,figsize = [15,10])
fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, hspace=.65)
axs = axs.ravel()

arrs= [loss_arr, mismatch_arr, fidelity_arr]
labels = ["Loss", "Mismatch", "Fidelity"]

for i in np.arange(len(axs)):

    axs[i].scatter(epochs, arrs[i], color="black", s=10)
    axs[i].set_ylabel(labels[i], fontsize=fontsize)
    axs[i].set_xlabel("Epochs", fontsize=fontsize)

plt.savefig(os.path.join("plots","plots"+s+".png"), dpi=500)
plt.show()

# get indices of three regions 
seed = 1680458526
rng = np.random.default_rng(seed=seed)

n =2 
epochs = 2000 

x_min = 0
x_max = 2**n 
x_arr = rng.integers(x_min, x_max, size=epochs)


low_mismatch_cluster = np.where(mismatch_arr < 0.45)
mid_mismatch_cluster = np.where((mismatch_arr > 0.45) * (mismatch_arr < 0.8))
high_mismatch_cluster = np.where(mismatch_arr > 0.8)

low_x = [] 
mid_x = []
high_x = [] 

for i in low_mismatch_cluster:

    low_x.append(x_arr[i])

for i in mid_mismatch_cluster:

    mid_x.append(x_arr[i])

for i in high_mismatch_cluster:

    high_x.append(x_arr[i]) 

unique, counts = np.unique(low_x, return_counts=True)    
low_dict = dict(zip(unique, counts))

unique, counts = np.unique(mid_x, return_counts=True)    
mid_dict = dict(zip(unique, counts))

unique, counts = np.unique(high_x, return_counts=True)    
high_dict = dict(zip(unique, counts))
         
print("low",low_dict)
print("mid",mid_dict)
print("high",high_dict)

