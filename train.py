import argparse 
from pqcprep.file_tools import check_duplicates, compress_args
from pqcprep.psi_tools import psi_H, psi_linear, psi_quadratic, psi_sine 

parser = argparse.ArgumentParser(usage='', description="Train and test the QCNN.")   
parser.add_argument('-n','--n', help="Number of input qubits.", default=2, type=int)
parser.add_argument('-m','--m', help="Number of target qubits.", default=2, type=int)
parser.add_argument('-L','--L', help="Number of network layers. If multiple values given will execute sequentially.", default=[6],type=int, nargs="+")
parser.add_argument('-f','--f', help="Phase function to evaluate.", default=None ,choices=["psi", "linear", "quadratic", "sine"])
parser.add_argument('-l','--loss', help="Loss function.", default="SAM", choices=["CE", "MSE", "L1", "KLD", "SAM", "WIM", "WILL"])
parser.add_argument('-e','--epochs', help="Number of epochs.", default=600,type=int)
parser.add_argument('-M','--meta', help="String with meta data.", default="")
parser.add_argument('-d','--delta', help="Value of delta parameter.", default=0., type=float)

parser.add_argument('-ni','--nint', help="Number of integer input qubits.", default=None, type=int)
parser.add_argument('-mi','--mint', help="Number of integer target qubits.", default=None, type=int)

parser.add_argument('-RP','--repeat_params', help="Use the same parameter values for different layers", default=None ,choices=["CL", "IL", "both"])


parser.add_argument('-r','--real', help="Output states with real amplitudes only.", action='store_true')
parser.add_argument('-PR','--phase_reduce', help="Reduce function values to a phase between 0 and 1.", action='store_true')
parser.add_argument('-TS','--train_superpos', help="Train circuit in superposition. (Automatically activates --phase_reduce).", action='store_true')

parser.add_argument('-H','--hayes', help="Train circuit to reproduce Hayes 2023. Sets -TS -r -n 6 -PR -f psi. Still set own m.", action='store_true')

parser.add_argument('-p','--WILL_p', help="WILL p parameter.", default=[1],type=float, nargs="+")
parser.add_argument('-q','--WILL_q', help="WILL q parameter.", default=[1],type=float, nargs="+")

parser.add_argument('--seed', help="Seed for random number generation.", default=1680458526,type=int)
parser.add_argument('-gs','--gen_seed', help="Generate seed from timestamp (Overrides value given with '--seed').", action='store_true')

parser.add_argument('-I','--ignore_duplicates', help="Ignore and overwrite duplicate files.", action='store_true')
parser.add_argument('-R','--recover', help="Continue training from existing TEMP files.", action='store_true')

opt = parser.parse_args()

# configure arguments
if opt.f==None:
    opt.f="psi" 
f_str=opt.f

if opt.f=="psi":
    opt.f=psi_H 
elif opt.f=="linear": 
    opt.f=psi_linear 
elif opt.f=="quadratic": 
    opt.f=psi_quadratic
elif opt.f=="sine": 
    opt.f=psi_sine 

if opt.gen_seed:
    import time 
    opt.seed = int(time.time())  

if opt.hayes:
    opt.n=6 
    opt.phase_reduce=True 
    opt.train_superpos=True 
    opt.real=True 

if opt.delta < 0 or opt.delta > 1:
    raise ValueError("Delta parameter must be between 0 and 1.")    
         
# check for duplicates
from pqcprep.tools import train_QNN, test_QNN

for j in range(len(opt.WILL_p)):
    for k in range(len(opt.WILL_q)):

        for i in range(len(opt.L)):

            args=compress_args(n=opt.n,m=opt.m,L=opt.L[i],seed=int(opt.seed),epochs=opt.epochs,func_str=f_str,loss_str=opt.loss,meta=opt.meta, nint=opt.nint, mint=opt.mint, phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p[j], WILL_q=opt.WILL_q[k], delta=opt.delta)
            
            dupl_files = check_duplicates(args, ampl=False)

            if dupl_files and opt.ignore_duplicates==False:
                print("\nThe required data already exists and will not be recomputed. Use '-I' or '--ignore_duplicates' to override this.\n")
            else: 
                train_QNN(n=int(opt.n),m=int(opt.m),L=int(opt.L[i]), seed=int(opt.seed), epochs=int(opt.epochs), func=opt.f, func_str=f_str, loss_str=opt.loss, meta=opt.meta, recover_temp=opt.recover, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos,real=opt.real, repeat_params=opt.repeat_params,WILL_p=opt.WILL_p[j], WILL_q=opt.WILL_q[k],delta=opt.delta)
                test_QNN(n=int(opt.n),m=int(opt.m),L=int(opt.L[i]),seed=int(opt.seed),epochs=int(opt.epochs), func=opt.f, func_str=f_str, loss_str=opt.loss, meta=opt.meta,nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real,repeat_params=opt.repeat_params, WILL_p=opt.WILL_p[j], WILL_q=opt.WILL_q[k], delta=opt.delta)    










