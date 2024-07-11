import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams

# general settings
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=28
titlesize=32
ticksize=22
figsize=(10,10)

n = 6
x_min = 40
x_max = 168 

func = lambda x : x**(-7./6) 

dx = (x_max-x_min)/(2**n)
target_arr = np.array([func(i) for i in np.arange(x_min,x_max, dx)])**2
target_arr = target_arr / np.sum(target_arr)

target_arr= np.sqrt(target_arr)

x = np.arange(x_min,x_max, dx)
y = np.load("ampl_outputs/statevec_6_8_800_x76_MSE_40_168_.npy")
yF = np.abs(np.load("ampl_outputs/amp_state_QGAN.npy"))

plt.figure(figsize=figsize)
plt.plot(x, target_arr, label="Target", color="black")
plt.scatter(x,y,label="QCNN", color="red")
plt.scatter(x,yF,label="QGAN", color="blue")
plt.ylabel(r"$\tilde A(f)$", fontsize=fontsize)
plt.xlabel(r"$f$", fontsize=fontsize)
plt.legend(fontsize=fontsize, loc='upper right')
plt.tick_params(axis="both", labelsize=ticksize)

plt.savefig("ampl_outputs/A_QCNN_v_QGAN", dpi=500)

plt.show()

plt.figure(figsize=figsize)
plt.scatter(x,target_arr -y ,label="QCNN", color="red")
plt.scatter(x,target_arr -yF,label="QGAN", color="blue")
plt.ylabel(r"$\Delta \tilde A(f)$", fontsize=fontsize)
plt.xlabel(r"$f$", fontsize=fontsize)
plt.tick_params(axis="both", labelsize=ticksize)
plt.legend(fontsize=fontsize, loc='upper right')


plt.savefig("ampl_outputs/DeltaA_QCNN_v_QGAN", dpi=500)

plt.show()