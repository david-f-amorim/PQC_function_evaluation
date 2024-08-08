import argparse 
from pqcprep.file_tools import check_duplicates, compress_args, compress_args_ampl

parser = argparse.ArgumentParser(usage='', description="Train and test the QCNN.")   
parser.add_argument('-n','--n', help="Number of input qubits.", default=6, type=int)
parser.add_argument('-m','--m', help="Number of target qubits.", default=3, type=int)
parser.add_argument('-L','--L', help="Number of network layers. If multiple values given will execute sequentially.", default=[6],type=int, nargs="+")
parser.add_argument('-f','--f', help="Phase function to evaluate.", default="psi")
parser.add_argument('-l','--loss', help="Loss function.", default="SAM")
parser.add_argument('-e','--epochs', help="Number of epochs.", default=600,type=int)
parser.add_argument('-M','--meta', help="String with meta data.", default="")

parser.add_argument('-d','--delta', help="Value of delta parameter.", default=[0.], type=float, nargs="+")

parser.add_argument('-ni','--nint', help="Number of integer input qubits.", default=None, type=int)
parser.add_argument('-mi','--mint', help="Number of integer target qubits.", default=None, type=int)

parser.add_argument('-RP','--repeat_params', help="Use the same parameter values for different layers", default=None)

parser.add_argument('-fA','--f_ampl', help="Amplitude function to learn.", default="x76")
parser.add_argument('--xmin', help="Minimum value of amplitude function range.", default=40, type=int)
parser.add_argument('--xmax', help="Maximum value of amplitude function range.", default=168, type=int)
parser.add_argument('-RPA','--repeat_params_ampl', help="Use the same parameter values for different layers", action='store_true')

parser.add_argument('-A','--ampl', help="Train network to prepare amplitude distribution instead of function evaluation.", action='store_true')

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
if opt.gen_seed:
    import time 
    opt.seed = int(time.time())  

if opt.hayes:
    opt.n=6 
    opt.phase_reduce=True 
    opt.train_superpos=True 
    opt.real=True 

for l in range(len(opt.delta)):
    if opt.delta[l] < 0 or opt.delta[l] > 1:
        raise ValueError("Delta parameter must be between 0 and 1.")    
         
# check for duplicates

if opt.ampl==False:

    from pqcprep.training_tools import train_QNN, test_QNN #type: ignore 

    for j in range(len(opt.WILL_p)):
        for k in range(len(opt.WILL_q)):
            for i in range(len(opt.L)):
                for l in range(len(opt.delta)):

                    args=compress_args(n=opt.n,m=opt.m,L=opt.L[i],seed=int(opt.seed),epochs=opt.epochs,func_str=opt.f,loss_str=opt.loss,meta=opt.meta, nint=opt.nint, mint=opt.mint, phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real, repeat_params=opt.repeat_params, WILL_p=opt.WILL_p[j], WILL_q=opt.WILL_q[k], delta=opt.delta[l])
                    
                    dupl_files = check_duplicates(args, ampl=False)

                    if dupl_files and opt.ignore_duplicates==False:
                        print("\nThe required data already exists and will not be recomputed. Use '-I' or '--ignore_duplicates' to override this.\n")
                    else: 
                        train_QNN(n=int(opt.n),m=int(opt.m),L=int(opt.L[i]), seed=int(opt.seed), epochs=int(opt.epochs), func_str=opt.f, loss_str=opt.loss, meta=opt.meta, recover_temp=opt.recover, nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos,real=opt.real, repeat_params=opt.repeat_params,WILL_p=opt.WILL_p[j], WILL_q=opt.WILL_q[k],delta=opt.delta[l])
                        test_QNN(n=int(opt.n),m=int(opt.m),L=int(opt.L[i]),seed=int(opt.seed),epochs=int(opt.epochs), func_str=opt.f, loss_str=opt.loss, meta=opt.meta,nint=opt.nint, mint=opt.mint,phase_reduce=opt.phase_reduce, train_superpos=opt.train_superpos, real=opt.real,repeat_params=opt.repeat_params, WILL_p=opt.WILL_p[j], WILL_q=opt.WILL_q[k], delta=opt.delta[l])    

else:
    
    from pqcprep.training_tools import  ampl_train_QNN #type: ignore 

    for i in range(len(opt.L)):

        args=compress_args_ampl(n=opt.n,L=opt.L[i],x_min=int(opt.xmin),x_max=int(opt.xmax), seed=int(opt.seed),epochs=opt.epochs,func_str=opt.f_ampl,loss_str=opt.loss,meta=opt.meta, nint=opt.nint, repeat_params=opt.repeat_params_ampl)
                
        dupl_files = check_duplicates(args, ampl=True)
        
        if dupl_files and opt.ignore_duplicates==False:
            print("\nThe required data already exists and will not be recomputed. Use '-I' or '--ignore_duplicates' to override this.\n")
        else: 
            ampl_train_QNN(n=int(opt.n),x_min=int(opt.xmin),x_max=int(opt.xmax),L=int(opt.L[i]), seed=int(opt.seed), epochs=int(opt.epochs), func_str=opt.f_ampl, loss_str=opt.loss, meta=opt.meta, recover_temp=opt.recover, nint=opt.nint, repeat_params=opt.repeat_params_ampl)
 










