import numpy as np 
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute 
from qiskit.circuit import ParameterVector
from itertools import combinations
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss, KLDivLoss
from torch import Tensor, no_grad 
import sys, time, os 
import argparse 
from tools import check_duplicates, generate_seed 
import torch
import warnings

from tools import A_generate_network

def R_train_QNN(n,L,x_min,x_max,seed, shots, lr, b1, b2, epochs, func,func_str,loss_str,meta, recover_temp, nint):
    """
    Initialise circuit as QNN for training purposes.
    """

    # set seed for PRNG 
    algorithm_globals.random_seed= seed
    
    # generate circuit and set up as QNN 
    qc = A_generate_network(n,L)
    qnn = SamplerQNN(
            circuit=qc.decompose(),            # decompose to avoid data copying (?)
            sampler=Sampler(options={"shots": shots, "seed": algorithm_globals.random_seed}),
            weight_params=qc.parameters, 
            input_params=[],  
            input_gradients=False 
        )
    
    # set precision strings 
    if nint==None:
        nint=n   
    if nint==n:
        nis=""
    else:
        nis=f"({nint})"
    
    # choose initial weights
    recovered_k =0

    if recover_temp:    
        recovered_weights=None
        recovered_mismatch=None 
        recovered_loss=None 
        for k in np.arange(100,epochs, step=100):
            if os.path.isfile(os.path.join("ampl_outputs", f"__TEMP{k}_weights_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
                recovered_weights=os.path.join("ampl_outputs", f"__TEMP{k}_weights_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")
                recovered_k=k+1
            if os.path.isfile(os.path.join("ampl_outputs", f"__TEMP{k}_mismatch_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
                recovered_mismatch=os.path.join("ampl_outputs", f"__TEMP{k}_mismatch_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")
            if os.path.isfile(os.path.join("ampl_outputs", f"__TEMP{k}_loss_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
                recovered_loss=os.path.join("ampl_outputs", f"__TEMP{k}_loss_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")        
        
        if recovered_weights != None and recovered_mismatch != None and recovered_loss != None:
            initial_weights=np.load(recovered_weights)
        else:
            initial_weights = algorithm_globals.random.random(len(qc.parameters))    
    
    else:
        initial_weights = algorithm_globals.random.random(len(qc.parameters))
    
    # initialise TorchConnector
    model = TorchConnector(qnn, initial_weights)

    # choose optimiser and loss function 
    optimizer = Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005) # Adam optimizer 

    if loss_str=="MSE":
        criterion=MSELoss() 
    elif loss_str=="L1":
        criterion=L1Loss() 
    elif loss_str=="KLD":
        criterion=KLDivLoss()
    elif loss_str=="CE":
        criterion=CrossEntropyLoss()
                    
    # set up arrays to store training outputs 
    if recover_temp and recovered_weights != None and recovered_mismatch != None and recovered_loss != None:
        mismatch_vals=np.load(recovered_mismatch)
        loss_vals=np.load(recovered_loss)
    else:    
        mismatch_vals = np.empty(epochs)
        loss_vals = np.empty(epochs)

    # calculate target and normalise 
    dx = (x_max-x_min)/(2**n)
    target_arr = np.array([func(i) for i in np.arange(x_min,x_max, dx)])**2
    target_arr = target_arr / np.sum(target_arr)
    
    # start training 
    print(f"\n\nTraining started. Epochs: {epochs}. Input qubits: {n}. Function range: [{x_min},{x_max}]. QCNN layers: {L}. \n")
    start = time.time() 

    warnings.filterwarnings("ignore", category=UserWarning)

    for i in np.arange(epochs)[recovered_k:]:

        # get input data
        input = Tensor([]) 

        # get target data 
        target=Tensor(target_arr)

        # set sign_tensor 
        if i==recovered_k:
            sign_tensor=Tensor(np.ones(2**n))
        else:
            sign_tensor=Tensor(sign_arr)
       
        # train model  
        optimizer.zero_grad()
        loss = criterion(torch.mul(torch.sqrt(torch.abs(model(input))+1e-10), sign_tensor), torch.sqrt(target)) # adding 1e-10 to prevent taking sqrt(0) ??!!
        loss.backward()
        optimizer.step()

        # save loss for plotting 
        loss_vals[i]=loss.item()

        # set up circuit with calculated weights
        circ = A_generate_network(n,L)

        with no_grad():
            generated_weights = model.weight.detach().numpy()
        
        circ = circ.assign_parameters(generated_weights)    

        # get statevector 
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend)
        result = job.result()
        state_vector = np.asarray(result.get_statevector()) 

        # get signs of states:
        sign_arr= np.ones(2**n, dtype=float) * np.sign(state_vector) 
        
        # calculate fidelity and mismatch 
        fidelity = np.abs(np.dot(np.sqrt(target_arr),np.conjugate(state_vector)))**2
        mismatch = 1. - np.sqrt(fidelity)

        # save mismatch for plotting 
        mismatch_vals[i]=mismatch

        # temporarily save outputs every hundred iterations
        temp_ind = epochs - 100 
        
        if recover_temp:
            temp_ind = recovered_k -1

        if (i % 100 ==0) and (i != 0) and (i != epochs-1): 
            np.save(os.path.join("ampl_outputs", f"__TEMP{i}_weights_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}"),generated_weights)
            np.save(os.path.join("ampl_outputs", f"__TEMP{i}_mismatch_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}"),mismatch_vals)
            np.save(os.path.join("ampl_outputs", f"__TEMP{i}_loss_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}__{x_min}_{x_max}{meta}"),loss_vals)
            
            # delete previous temp files
            prev_weights=f"__TEMP{i-100}_weights_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}.npy"
            prev_mismatch=f"__TEMP{i-100}_mismatch_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}.npy"
            prev_loss=f"__TEMP{i-100}_loss_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}.npy"

            if os.path.isfile(os.path.join("ampl_outputs", prev_weights)):
                os.remove(os.path.join("ampl_outputs", prev_weights))

            if os.path.isfile(os.path.join("ampl_outputs", prev_mismatch)):
                os.remove(os.path.join("ampl_outputs", prev_mismatch))

            if os.path.isfile(os.path.join("ampl_outputs", prev_loss)):
                os.remove(os.path.join("ampl_outputs", prev_loss))        

            # make note of last created temp files
            temp_ind = i   
        
        # print status
        a = int(20*(i+1)/epochs)

        if i==recovered_k:
            time_str="--:--:--.--"
        elif i==epochs-1:
            time_str="00:00:00.00"    
        else:
            if recover_temp:
                    remaining = ((time.time() - start) / (i-recovered_k)) * (epochs - i)
            else:
                remaining = ((time.time() - start) / i) * (epochs - i)
            mins, sec = divmod(remaining, 60)
            hours, mins = divmod(mins, 60)
            time_str = f"{int(hours):02}:{int(mins):02}:{sec:05.2f}"

        prefix="\t" 
        print(f"{prefix}[{u'█'*a}{('.'*(20-a))}] {100.*((i+1)/epochs):.2f}% ; Loss {loss_vals[i]:.2e} ; Mismatch {mismatch:.2e} ; ETA {time_str}", end='\r', file=sys.stdout, flush=True)
        
    warnings.filterwarnings("default", category=UserWarning)

    print(" ", flush=True, file=sys.stdout)

    elapsed = time.time()-start
    mins, sec = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    time_str = f"{int(hours):02}:{int(mins):02}:{sec:05.2f}" 

    # decompose circuit for gate count 
    num_CX = dict(circ.decompose(reps=4).count_ops())["cx"]
    num_gates = num_CX + dict(circ.decompose(reps=4).count_ops())["u"]

    print(f"\nTraining completed in {time_str}. Number of weights: {len(generated_weights)}. Number of gates: {num_gates} (of which CX gates: {num_CX}). \n\n")

    # delete temp files 
    temp_weights=f"__TEMP{temp_ind}_weights_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}.npy"
    temp_mismatch=f"__TEMP{temp_ind}_mismatch_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}.npy"
    temp_loss=f"__TEMP{temp_ind}_loss_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}.npy"

    if os.path.isfile(os.path.join("ampl_outputs", temp_weights)):
            os.remove(os.path.join("ampl_outputs", temp_weights))
    if os.path.isfile(os.path.join("ampl_outputs", temp_mismatch)):
            os.remove(os.path.join("ampl_outputs", temp_mismatch))
    if os.path.isfile(os.path.join("ampl_outputs", temp_loss)):
            os.remove(os.path.join("ampl_outputs", temp_loss))               

    # save outputs 
    with no_grad():
            generated_weights = model.weight.detach().numpy()

    np.save(os.path.join("ampl_outputs", f"weights_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}"),generated_weights)
    np.save(os.path.join("ampl_outputs", f"mismatch_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}"),mismatch_vals)
    np.save(os.path.join("ampl_outputs", f"loss_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}"),loss_vals)
    np.save(os.path.join("ampl_outputs", f"statevec_{n}{nis}_{L}_{epochs}_{func_str}_{loss_str}_{x_min}_{x_max}_{meta}"),np.abs(state_vector))

    return 0

####

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='', description="Train and test the QCNN.")   
    parser.add_argument('-n','--n', help="Number of input qubits.", default=2, type=int)
    parser.add_argument('-L','--L', help="Number of network layers. If multiple values given will execute sequentially.", default=[15],type=int, nargs="+")
    parser.add_argument('-l','--loss', help="Loss function.", default="MSE", choices=["CE", "MSE", "L1", "KLD"])
    parser.add_argument('-f','--f', help="Function to evaluate (variable: x).", default="x")
    parser.add_argument('-fs','--f_str', help="String describing function.")
    parser.add_argument('-e','--epochs', help="Number of epochs.", default=800,type=int)
    parser.add_argument('--xmin', help="Minimum value of function range.", default=40, type=int)
    parser.add_argument('--xmax', help="Maximum value of function range.", default=168, type=int)
    parser.add_argument('-M','--meta', help="String with meta data.", default="")
    parser.add_argument('-ni','--nint', help="Number of integer input qubits.", default=None, type=int)
    

    parser.add_argument('--seed', help="Seed for random number generation.", default=1680458526,type=int)
    parser.add_argument('-gs','--gen_seed', help="Generate seed from timestamp (Overrides value given with '--seed').", action='store_true')
    parser.add_argument('--lr', help="Learning rate.", default=0.01,type=float)
    parser.add_argument('--b1', help="Adam optimizer b1 parameter.", default=0.7,type=float)
    parser.add_argument('--b2', help="Adam optimizer b2 parameter.", default=0.999,type=float)
    parser.add_argument('--shots', help="Number of shots used by sampler.", default=10000,type=int)

    parser.add_argument('-R','--recover', help="Continue training from existing TEMP files.", action='store_true')

    opt = parser.parse_args()

    # configure arguments
    if opt.f_str==None:
        opt.f_str=opt.f 

    if opt.gen_seed:
        opt.seed = generate_seed()    

    for i in range(len(opt.L)):
        
        R_train_QNN(n=int(opt.n),x_min=int(opt.xmin),x_max=int(opt.xmax),L=int(opt.L[i]), seed=int(opt.seed), shots=int(opt.shots), lr=float(opt.lr), b1=float(opt.b1), b2=float(opt.b2), epochs=int(opt.epochs), func=lambda x: eval(opt.f), func_str=opt.f_str, loss_str=opt.loss, meta=opt.meta, recover_temp=opt.recover, nint=opt.nint)
 

