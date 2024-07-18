import argparse 
from tools import check_duplicates, generate_seed, psi 

parser = argparse.ArgumentParser(usage='', description="Train and test the QCNN.")   
parser.add_argument('-n','--n', help="Number of input qubits.", default=2, type=int)
parser.add_argument('-m','--m', help="Number of target qubits.", default=2, type=int)
parser.add_argument('-L','--L', help="Number of network layers. If multiple values given will execute sequentially.", default=[6],type=int, nargs="+")
parser.add_argument('-l','--loss', help="Loss function.", default="MM", choices=["CE", "MSE", "L1", "KLD", "MM", "WIM"])
parser.add_argument('-f','--f', help="Function to evaluate (variable: x).", default="x")
parser.add_argument('-fs','--f_str', help="String describing function.")
parser.add_argument('-e','--epochs', help="Number of epochs.", default=600,type=int)
parser.add_argument('-M','--meta', help="String with meta data.", default="")
parser.add_argument('-ni','--nint', help="Number of integer input qubits.", default=None, type=int)
parser.add_argument('-mi','--mint', help="Number of integer target qubits.", default=None, type=int)

parser.add_argument('-r','--real', help="Output states with real amplitudes only.", action='store_true')
parser.add_argument('-PR','--phase_reduce', help="Reduce function values to a phase between 0 and 1.", action='store_true')
parser.add_argument('-TS','--train_superpos', help="Train circuit in superposition. (Automatically activates --phase_reduce).", action='store_true')

parser.add_argument('-H','--hayes', help="Train circuit to reproduce Hayes 2023. Sets -TS -r -n 6 -PR -f psi. Still set own m.", action='store_true')

parser.add_argument('--tau1', help="WIM tau1 parameter.", default=1,type=float)
parser.add_argument('--tau2', help="WIM tau2 parameter.", default=10,type=float)
parser.add_argument('--tau3', help="WIM tau3 parameter.", default=100,type=float)

parser.add_argument('--seed', help="Seed for random number generation.", default=1680458526,type=int)
parser.add_argument('-gs','--gen_seed', help="Generate seed from timestamp (Overrides value given with '--seed').", action='store_true')
parser.add_argument('--lr', help="Learning rate.", default=0.01,type=float)
parser.add_argument('--b1', help="Adam optimizer b1 parameter.", default=0.7,type=float)
parser.add_argument('--b2', help="Adam optimizer b2 parameter.", default=0.999,type=float)
parser.add_argument('--shots', help="Number of shots used by sampler.", default=10000,type=int)

parser.add_argument('-I','--ignore_duplicates', help="Ignore and overwrite duplicate files.", action='store_true')
parser.add_argument('-R','--recover', help="Continue training from existing TEMP files.", action='store_true')

opt = parser.parse_args()

# configure arguments
if opt.f_str==None:
    opt.f_str=opt.f 

if opt.f=="psi":
    opt.f=psi   
else:
    f = str(opt.f)
    opt.f=lambda x: eval(f)      

if opt.gen_seed:
    opt.seed = generate_seed()   

if opt.hayes:
    opt.n=6 
    opt.phase_reduce=True 
    opt.train_superpos=True 
    opt.real=True 
    opt.f=psi 
    opt.f_str="psi"     

# check for duplicates
from tools import train_QNN, test_QNN

for i in range(len(opt.L)):
    
    dupl_files = check_duplicates(n=opt.n,m=opt.m,L=opt.L[i],epochs=opt.epochs,func_str=opt.f_str,loss_str=opt.loss,meta=opt.meta, nint=opt.nint, mint=opt.mint, phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real)

    if dupl_files and opt.ignore_duplicates==False:
        print("\nThe required data already exists and will not be recomputed. Use '-I' or '--ignore_duplicates' to override this.\n")
    else: 
        train_QNN(n=int(opt.n),m=int(opt.m),L=int(opt.L[i]), seed=int(opt.seed), shots=int(opt.shots), lr=float(opt.lr), b1=float(opt.b1), b2=float(opt.b2), epochs=int(opt.epochs), func=opt.f, func_str=opt.f_str, loss_str=opt.loss, meta=opt.meta, recover_temp=opt.recover, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos,real=opt.real, tau_1=opt.tau1,tau_2=opt.tau2,tau_3=opt.tau3)
        test_QNN(n=int(opt.n),m=int(opt.m),L=int(opt.L[i]),epochs=int(opt.epochs), func=opt.f, func_str=opt.f_str, loss_str=opt.loss, meta=opt.meta,nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real)    










