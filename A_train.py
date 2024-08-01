from pqcprep.tools import ampl_train_QNN
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='', description="Train and test the QCNN.")   
    parser.add_argument('-n','--n', help="Number of input qubits.", default=6, type=int)
    parser.add_argument('-L','--L', help="Number of network layers. If multiple values given will execute sequentially.", default=[3],type=int, nargs="+")
    parser.add_argument('-l','--loss', help="Loss function.", default="MM", choices=["CE", "MSE", "L1", "KLD", "MM"])
    parser.add_argument('-f','--f', help="Function to evaluate (variable: x).", default=None)
    parser.add_argument('-fs','--f_str', help="String describing function.")
    parser.add_argument('-e','--epochs', help="Number of epochs.", default=600,type=int)
    parser.add_argument('--xmin', help="Minimum value of function range.", default=40, type=int)
    parser.add_argument('--xmax', help="Maximum value of function range.", default=168, type=int)
    parser.add_argument('-M','--meta', help="String with meta data.", default="")
    parser.add_argument('-ni','--nint', help="Number of integer input qubits.", default=None, type=int)

    parser.add_argument('-RP','--repeat_params', help="Use the same parameter values for different layers", action='store_true')
    
    parser.add_argument('--seed', help="Seed for random number generation.", default=1680458526,type=int)
    parser.add_argument('-gs','--gen_seed', help="Generate seed from timestamp (Overrides value given with '--seed').", action='store_true')
    
    parser.add_argument('-R','--recover', help="Continue training from existing TEMP files.", action='store_true')

    opt = parser.parse_args()

    # configure arguments
    if opt.f==None:
        opt.f = lambda x: x**(-7./6)
        opt.f_str = "x76"
    else:
        opt.f = lambda x: eval(opt.f)

    if opt.f_str==None:
        opt.f_str=opt.f 
    
    if opt.gen_seed:
        import time 
        opt.seed = int(time.time())  

    for i in range(len(opt.L)):
        
        ampl_train_QNN(n=int(opt.n),x_min=int(opt.xmin),x_max=int(opt.xmax),L=int(opt.L[i]), seed=int(opt.seed), epochs=int(opt.epochs), func=opt.f, func_str=opt.f_str, loss_str=opt.loss, meta=opt.meta, recover_temp=opt.recover, nint=opt.nint, repeat_params=opt.repeat_params)
 

