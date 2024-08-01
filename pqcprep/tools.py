"""
Collection of useful functions for a variety of purposes.
"""

import numpy as np 
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute 
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss, KLDivLoss
from torch import Tensor, no_grad
import sys, time, os 
import torch 
import warnings
from .binary_tools import bin_to_dec, dec_to_bin  
from .pqc_tools import generate_network, binary_to_encode_param, A_generate_network  
from .file_tools import compress_args,compress_args_ampl, vars_to_name_str, vars_to_name_str_ampl 

#---------------------------------------------------------------------------------------------------

def train_QNN(n,m,L, seed, epochs, func,func_str,loss_str,meta, recover_temp, nint, mint, phase_reduce, train_superpos, real, repeat_params, WILL_p, WILL_q):
    """
    Initialise circuit as QNN for training purposes.
    """
    
    # compress arguments into dictionary 
    args =compress_args(n,m,L, seed, epochs,func_str,loss_str,meta,nint, mint, phase_reduce, train_superpos, real, repeat_params, WILL_p, WILL_q)

    # set precision strings 
    if nint==None: nint=n
    if mint==None: mint=m  
    if phase_reduce: mint=0

    # set seed for PRNG 
    algorithm_globals.random_seed= seed
    rng = np.random.default_rng(seed=seed)

    # check conditions for QRQ loss:
    if loss_str=="QRQ":
        if train_superpos==False:
            raise ValueError("This loss function requires training in superposition.")
        if n != 6:
            raise ValueError("This loss function requires n=6.")
        if phase_reduce !=True:
            raise ValueError("This loss function required phase reduction.")

    # generate circuit and set up as QNN 
    if train_superpos: 
        qc = generate_network(n,m,L, encode=False, toggle_IL=True, initial_IL=True, input_H=True, real=real, repeat_params=repeat_params)

        if loss_str=="QRQ":
            input_register = QuantumRegister(n, "input")
            target_register = QuantumRegister(m, "target")
            circuit = QuantumCircuit(input_register, target_register) 

            # load weights 
            weights_A = np.load("ampl_outputs/weights_6_3_600_x76_MM_40_168_zeros.npy") 
            
            # encode amplitudes 
            circuit.compose(A_generate_network(n, 3), input_register, inplace=True)
            circuit = circuit.assign_parameters(weights_A)
            
            # evaluate function
            qc = generate_network(n,m, L, real=real,repeat_params=repeat_params)
            inv_qc = qc.inverse()
            circuit.compose(qc, [*input_register,*target_register], inplace=True) 
            
            # extract phases 
            circuit.compose(extract_phase(m),target_register, inplace=True) 

            # clear ancilla register 
            circuit.compose(inv_qc, [*input_register,*target_register], inplace=True) 

            # export as qc 
            qc=circuit 

        qnn = SamplerQNN(
                circuit=qc.decompose(),            # decompose to avoid data copying (?)
                sampler=Sampler(options={"shots": 10000, "seed": algorithm_globals.random_seed}),
                input_params=[],    # no input params
                weight_params=qc.parameters,   # weights
                input_gradients=False # ?? 
            ) 
    else:     
        qc = generate_network(n,m,L, encode=True, toggle_IL=True, initial_IL=True, real=real,repeat_params=repeat_params)
        qnn = SamplerQNN(
                circuit=qc.decompose(),            # decompose to avoid data copying (?)
                sampler=Sampler(options={"shots": 10000, "seed": algorithm_globals.random_seed}),
                input_params=qc.parameters[:n],    # encoding params treated as input params
                weight_params=qc.parameters[n:],   # encoding params not selected as weights
                input_gradients=True # ?? 
            )        

    # choose initial weights
    recovered_k =0

    if recover_temp:    
        recover_labels=["weights", "mismatch", "loss", "grad", "vargrad"]
        recover_paths={}
        for k in np.arange(100,epochs, step=100):
            for e in np.arange(len(recover_labels)):
                file=os.path.join("outputs", f"__TEMP{k}_{recover_labels[e]}{vars_to_name_str(args)}.npy")
                recover_paths[recover_labels[e]]= (file if os.path.isfile(file) else None)

                if recover_labels[e]=="weights" and os.path.isfile(file):
                    recovered_k=k+1
                     
        if not None in list(recover_paths.values):
            initial_weights=np.load(recover_paths["weights"])
        else:
            if train_superpos: 
                initial_weights =rng.normal(0,1/np.sqrt(n+m),len(qc.parameters)) #np.zeros(len(qc.parameters))  
            else:    
                initial_weights =rng.normal(0,1/np.sqrt(n+m),len(qc.parameters[n:])) #np.zeros(len(qc.parameters)[n:]) #algorithm_globals.random.random(len(qc.parameters[n:]))    
    else:
        if train_superpos: 
            initial_weights =rng.normal(0,1/np.sqrt(n+m),len(qc.parameters))   #[LECUN NORMAL] 
            # -1/np.sqrt(n+m) +(2/np.sqrt(n+m)) *rng.random(len(qc.parameters)) #[LECUN UNIFORM]
            # -a*np.pi +(2*a*np.pi) *rng.random(len(qc.parameters)) #[A-UNIFORM] (0 < a < 1)
            # -np.pi +(2*np.pi) *rng.random(len(qc.parameters)) #[UNIFORM]
            #rng.normal(0,1/np.sqrt(n+m),len(qc.parameters))   #[LECUN NORMAL] 
            #np.zeros(len(qc.parameters)) #[ZERO INIT] CHANGE BACK AFTER TESTING!!
        else:    
            initial_weights =rng.normal(0,1/np.sqrt(n+m),len(qc.parameters[n:]))# np.zeros(len(qc.parameters[n:])) #algorithm_globals.random.random(len(qc.parameters[n:]))
    
    # initialise TorchConnector
    model = TorchConnector(qnn, initial_weights)

    # choose optimiser 
    optimizer = Adam(model.parameters(), lr=0.01, betas=(0.7, 0.999), weight_decay=0.005) # Adam optimizer 
                
    # set up arrays to store training outputs 
    if recover_temp and not None in list(recover_paths.values):
        mismatch_vals=np.load(recover_paths["mismatch"])
        loss_vals=np.load(recover_paths["loss"])
        grad_vals=np.load(recover_paths["grad"])
        var_grad_vals=np.load(recover_paths["var_grad"])
    else:    
        mismatch_vals = np.empty(epochs)
        loss_vals = np.empty(epochs)
        grad_vals = np.empty(epochs)
        var_grad_vals = np.empty(epochs)

    # generate x and f(x) values
    pn =n - nint
    pm =m - mint

    if train_superpos:

        # sample all basis states of input register and convert to binary 
        x_arr = np.arange(0, 2**n)
        x_arr_bin =[dec_to_bin(i,n,encoding="unsigned mag") for i in x_arr]

        # apply function and reduce to phase value between 0 and 1 
        # (run with phase_reduce=True !)
        fx_arr = [func(i) for i in x_arr]

        if phase_reduce:
            fx_arr = [np.modf(i/ (2* np.pi))[0] for i in fx_arr]

        # convert fx_arr to binary at available target register precision 
        fx_arr_bin = [dec_to_bin(i,m,nint=mint,encoding="unsigned mag") for i in fx_arr]

        if np.max(fx_arr)> 2.**mint - 2.**(-pm) and mint != 0:
            raise ValueError(f"Insufficient number of target (integer) qubits.")
        
        # get bit strings corresponding to target arrays and convert to indices
        target_bin = [fx_arr_bin[i]+x_arr_bin[i] for i in x_arr]
        target_ind = [bin_to_dec(i, encoding='unsigned mag') for i in target_bin]

        # prepare target array and normalise 
        target_arr = np.zeros(2**(n+m))
        for k in target_ind:
            target_arr[int(k)]=1 
        target_arr = target_arr / (2.**n) 

        # define input and target tensors 
        input = Tensor([]) 
        target=Tensor(target_arr)

    else:        
        x_min = 0
        x_max = 2.**nint - 2.**(-pn) 
        x_arr = np.array(x_min + (x_max - x_min) *rng.random(size=epochs))
        fx_arr = [func(i) for i in x_arr]

        # reduce to phase value between 0 and 1:
        if phase_reduce: 
            fx_arr = [np.modf(i/ (2* np.pi))[0] for i in fx_arr]
        
        if np.max(fx_arr)> 2.**mint - 2.**(-pm) and mint != 0:
            raise ValueError(f"Insufficient number of target (integer) qubits.")

    # choose loss function 
    if loss_str=="MSE":
        criterion=MSELoss() 
    elif loss_str=="L1":
        criterion=L1Loss() 
    elif loss_str=="KLD":
        criterion=KLDivLoss()
    elif loss_str=="CE":
        criterion=CrossEntropyLoss()
    elif loss_str=="MM":
        def criterion(output, target):
            return  torch.abs(1. -torch.sum(torch.mul(output, target)))  # redefine to punish sign errors (moved abs outwards) 
    elif loss_str=="WIM":  

        # set tau parameters 
        tau_1 = 0.8
        tau_2 = 10
        tau_3 = 1

        if train_superpos==False:
            raise ValueError("This loss function requires training in superposition.")   
        def criterion(output, target, weights):
            output = torch.mul(output, weights) # apply weights 
            output = output / torch.sum(torch.mul(output, output)) # normalise
            return  torch.abs(1. -torch.sum(torch.mul(output, target)))  # redefine to punish sign errors (moved abs outwards)
        WIM_weights_arr=np.ones(2**(n+m)) # initially set all weights to one
    
    elif loss_str=="WILL":  
        if train_superpos==False:
            raise ValueError("This loss function requires training in superposition.")
        
        fx_arr_rounded = [bin_to_dec(dec_to_bin(i,m,'unsigned mag', nint=mint),'unsigned mag',nint=mint) for i in fx_arr]
        distance_arr = np.empty(2**(n+m))
        
        for i in np.arange(2**n):
            bin_i=dec_to_bin(i,n,'unsigned mag') 
            for j in np.arange(2**m):
                bin_j=dec_to_bin(j,m,'unsigned mag')
                ind=int(bin_j + bin_i,2) 
                distance_arr[ind] = 1 + np.abs(bin_to_dec(dec_to_bin(j,m,'unsigned mag'),'unsigned mag',nint=mint) - fx_arr_rounded[i])  
        distance=Tensor(distance_arr)   
        p=WILL_p 
        q=WILL_q 
        reduce='mean' 
        
        def criterion(output, target):
            loss =torch.pow(torch.abs(output-target),p) + torch.mul(torch.abs(output),torch.pow(distance,q)) 
            if reduce=='sum':
                return torch.sum(loss)**(1/p) 
            elif reduce=='mean':
                return torch.sum(loss)**(1/p) / torch.numel(loss)
    elif loss_str=="QRQ":  
        #raise DeprecationWarning("NEEDS FIXING")
        input_register = QuantumRegister(n, "input")
        target_register = QuantumRegister(m, "target")
        circ = QuantumCircuit(input_register, target_register) 
            
        # encode amplitudes 
        circ.compose(A_generate_network(n, 3), input_register, inplace=True)
        circ = circ.assign_parameters(weights_A)

        # get target 
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circuit, backend)
        result = job.result()
        state_vector = result.get_statevector()
        
        target_arr=np.asarray(state_vector)

        # add phase 
        fx_arr_rounded = [bin_to_dec(dec_to_bin(i,m,'unsigned mag', nint=mint),'unsigned mag',nint=mint) for i in fx_arr]
        phase_arr = np.empty(2**(n+m))
        
        for i in np.arange(2**n):
            bin_i=dec_to_bin(i,n,'unsigned mag') 
            for j in np.arange(2**m):
                bin_j=dec_to_bin(j,m,'unsigned mag')
                ind=int(bin_j + bin_i,2) 
                phase_arr[ind] = fx_arr_rounded[i]
        target_arr=target_arr * np.exp(2*1.j*np.pi* phase_arr)      

        target=torch.polar(Tensor(np.abs(target_arr)), Tensor(np.angle(target_arr)))

        phase_tensor_target= Tensor(2*np.pi*phase_arr)

        def criterion(output):

            # get phase of output
            phase_tensor=torch.angle(output) 

            # compare to target 
            chi=torch.abs(phase_tensor - phase_tensor_target)

            return torch.mean(chi)

    # start training 
    print(f"\n\nTraining started. Epochs: {epochs}. Input qubits: {n}. Target qubits: {m}. QCNN layers: {L}. \n")
    start = time.time() 

    warnings.filterwarnings("ignore", category=UserWarning)

    for i in np.arange(epochs)[recovered_k:]:

        if train_superpos == False:
            # get input data
            input = Tensor(binary_to_encode_param(dec_to_bin(x_arr[i],n,'unsigned mag',nint=nint))) 

            # get target data 
            target_arr = np.zeros(2**(n+m))
            index = int(dec_to_bin(fx_arr[i],m,'unsigned mag',nint=mint)+dec_to_bin(x_arr[i],n,'unsigned mag',nint=nint),2)
            target_arr[index]=1 
            target=Tensor(target_arr)

        # train model  
        optimizer.zero_grad()

        if i==recovered_k:
            angle_tensor = Tensor(np.zeros(2**(n+m)))
            sign_tensor=Tensor(np.ones(2**(n+m)))
        else: 
            angle_tensor = Tensor(angle_arr)   
            sign_tensor=Tensor(sign_arr) 
            
        if real:
            if loss_str=="WIM":
                WIM_weights_tensor=Tensor(WIM_weights_arr)
                loss =criterion(torch.mul(torch.sqrt(torch.abs(model(input))+1e-10), sign_tensor), torch.sqrt(target), WIM_weights_tensor)    # add small number in sqrt !
            elif loss_str=="QRQ":
                loss = criterion(torch.polar(torch.sqrt(model(input)+1e-10),angle_tensor))
            else:
                loss =criterion(torch.mul(torch.sqrt(torch.abs(model(input))+1e-10), sign_tensor), torch.sqrt(target))    # add small number in sqrt !
        else: 
            if loss_str=="MM":    
                loss = criterion(torch.polar(torch.sqrt(model(input)+1e-10),angle_tensor), torch.sqrt(target))     # add small number in sqrt !
            else:
                loss = criterion(model(input), target)


        loss.backward()
        optimizer.step()

        # save loss and grad for plotting 
        loss_vals[i]=loss.item()
        grad_vals[i]=np.sum(model.weight.grad.numpy()**2)
        var_grad_vals[i]=np.std(model.weight.grad.numpy())**2
        
        # set up circuit with calculated weights
        if train_superpos:
            if loss_str=="QRQ":
                # set up registers 
                input_register = QuantumRegister(n, "input")
                target_register = QuantumRegister(m, "target")
                circuit = QuantumCircuit(input_register, target_register) 

                # encode amplitudes 
                circuit.compose(A_generate_network(n, 3), input_register, inplace=True)
                circuit = circuit.assign_parameters(weights_A)
                
                # evaluate function
                qc = generate_network(n,m, L, real=real,repeat_params=repeat_params)
                with no_grad():
                    generated_weights = model.weight.detach().numpy()
                qc = qc.assign_parameters(generated_weights)
                inv_qc = qc.inverse()
                circuit.compose(qc, [*input_register,*target_register], inplace=True) 
                
                # extract phases 
                circuit.compose(extract_phase(m),target_register, inplace=True) 

                # clear ancilla register 
                circuit.compose(inv_qc, [*input_register,*target_register], inplace=True)

                circ=circuit

            else:    
                circ = generate_network(n,m,L, encode=False, toggle_IL=True, initial_IL=True, input_H=True, real=real,repeat_params=repeat_params)
                with no_grad():
                    generated_weights = model.weight.detach().numpy()      
                circ = circ.assign_parameters(generated_weights)
        else:
            circ = generate_network(n,m,L, encode=True, toggle_IL=True, initial_IL=True, real=real,repeat_params=repeat_params)

            with no_grad():
                generated_weights = model.weight.detach().numpy()

            input_params = binary_to_encode_param(dec_to_bin(x_arr[i],n,'unsigned mag',nint=nint))
            params = np.concatenate((input_params, generated_weights))           
            circ = circ.assign_parameters(params)    

        # get statevector 
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend)
        result = job.result()
        state_vector = np.asarray(result.get_statevector()) 

        # extract phases of conjugate state vector 
        angle_arr = np.angle(np.conjugate(state_vector))

        # extract signs of state vector 
        sign_arr= np.sign(np.real(state_vector)) 

        # calculate fidelity and mismatch 
        if loss_str =="QRQ":
            fidelity = np.abs(np.dot(target_arr,np.conjugate(state_vector)))**2
        else:        
            fidelity = np.abs(np.dot(np.sqrt(target_arr),np.conjugate(state_vector)))**2
        
        mismatch = 1. - np.sqrt(fidelity)

        # save mismatch for plotting 
        mismatch_vals[i]=mismatch

        # set loss func weights
        if loss_str=="WIM" and (i % tau_2 ==0) and (i >=tau_3):
            
            # initialise arrays to store results 
            temp_mismatch = np.empty(2**n)
            WIM_weights_arr= np.empty(2**(n+m))
            
            # iterate over input states 
            x_arr_temp = np.arange(2**n)
            fx_arr_temp = [func(k) for k in x_arr]

            if phase_reduce: 
                fx_arr_temp = [np.modf(k/ (2* np.pi))[0] for k in fx_arr_temp]

            for q in x_arr:
                
                # prepare circuit 
                enc=binary_to_encode_param(np.binary_repr(q,n))
                params=np.concatenate((enc, generated_weights))  

                qc = generate_network(n,m,L, encode=True,toggle_IL=True, real=real,repeat_params=repeat_params)
                qc = qc.assign_parameters(params) 

                # get target array 
                target_arr_temp = np.zeros(2**(n+m))

                index = int(dec_to_bin(fx_arr_temp[q],m,'unsigned mag',nint=mint)+dec_to_bin(x_arr_temp[q],n,'unsigned mag',nint=nint),2)
                target_arr_temp[index]=1 

                # get statevector 
                backend = Aer.get_backend('statevector_simulator')
                job = execute(qc, backend)
                result = job.result()
                state_vector_temp = np.asarray(result.get_statevector()) 

                # for each output state, calculate the "binary difference" to the target as well as the "coefficient difference"
                sum = 0 
                for j in np.arange(2**m):
                    ind = int(dec_to_bin(j,m,'unsigned mag',nint=m)+dec_to_bin(x_arr_temp[q],n,'unsigned mag',nint=nint),2)
                    
                    num_dif =np.abs(fx_arr_temp[q] -bin_to_dec(dec_to_bin(j,m,'unsigned mag', nint=mint),'unsigned mag',nint=mint))
                    coeff_dif = np.abs(state_vector_temp[ind]- target_arr_temp[ind])
                    sum += num_dif * coeff_dif

                # add to weights_arr 
                for j in np.arange(2**m):
                    ind = int(dec_to_bin(j,m,'unsigned mag',nint=m)+dec_to_bin(x_arr_temp[q],n,'unsigned mag',nint=nint),2) 
                    WIM_weights_arr[ind]= sum    
  
            # focus on outliers: double-weight on states 0.5 sigma or more above the mean
            WIM_weights_arr += (WIM_weights_arr >= np.mean(WIM_weights_arr)+0.5 *np.std(WIM_weights_arr) ).astype(int) 
            
            # smoothen WIM weights 
            WIM_weights_arr=np.exp(tau_1 * WIM_weights_arr)
            
        # temporarily save outputs every hundred iterations
        temp_ind = epochs - 100 
        
        if recover_temp:
            temp_ind = recovered_k -1

        if (i % 100 ==0) and (i != 0) and (i != epochs-1): 

            temp_labels=["weights", "mismatch", "loss", "grad", "vargrad"] 
            temp_arrs=[generated_weights, mismatch_vals, loss_vals, grad_vals, var_grad_vals] 

            for e in np.arange(len(temp_labels)):
                # save temp file 
                file=os.path.join("outputs",f"__TEMP{i}_{temp_labels[e]}{vars_to_name_str(args)}")
                np.save(file,temp_arrs[e])

                # delete previous temp file 
                old_file=os.path.join("outputs",f"__TEMP{i-100}_{temp_labels[e]}{vars_to_name_str(args)}")
                os.remove(old_file) if os.path.isfile(old_file) else None

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
        
        
    print(" ", flush=True, file=sys.stdout)
    
    warnings.filterwarnings("default", category=UserWarning)

    elapsed = time.time()-start
    mins, sec = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    time_str = f"{int(hours):02}:{int(mins):02}:{sec:05.2f}" 

    # decompose circuit for gate count 
    num_CX = dict(circ.decompose(reps=4).count_ops())["cx"]
    num_gates = num_CX + dict(circ.decompose(reps=4).count_ops())["u"]
    print(f"\nTraining completed in {time_str}. Number of weights: {len(generated_weights)}. Number of gates: {num_gates} (of which CX gates: {num_CX}). \n\n")

    # delete temp files
    temp_labels=["weights", "mismatch", "loss", "grad", "vargrad"]  
    for i in np.arange(len(temp_labels)):
        file=os.path.join("outputs",f"__TEMP{temp_ind}_{temp_labels[i]}{vars_to_name_str(args)}.npy")
        os.remove(file) if os.path.isfile(file) else None
                            
    # save outputs 
    with no_grad():
            generated_weights = model.weight.detach().numpy()
    outputs= [generated_weights, mismatch_vals, loss_vals, grad_vals, var_grad_vals]
    output_labels=["weights", "mismatch", "loss", "grad", "vargrad"]  
    for i in np.arange(len(outputs)):
        np.save(os.path.join("outputs", f"{output_labels[i]}{vars_to_name_str(args)}"), outputs[i])      

    return 0 

def test_QNN(n,m,L,seed,epochs, func, func_str,loss_str,meta,nint,mint,phase_reduce,train_superpos,real,repeat_params,WILL_p, WILL_q,verbose=True):   
    """
    Test performance of trained QNN for the various input states
    """
    # compress arguments into dictionary 
    args =compress_args(n,m,L, seed, epochs,func_str,loss_str,meta,nint, mint, phase_reduce, train_superpos, real, repeat_params, WILL_p, WILL_q)
    name_str=vars_to_name_str(args)                    

    # load weights 
    weights = np.load(os.path.join("outputs",f"weights{name_str}.npy"))

    # initialise array to store results 
    mismatch = np.empty(2**n)
    signs = np.empty(2**n)
    
    # iterate over input states 
    x_arr = np.arange(2**n)
    fx_arr = [func(i) for i in x_arr]

    if phase_reduce: 
        fx_arr = [np.modf(i/ (2* np.pi))[0] for i in fx_arr]

    for i in x_arr:
        
        # prepare circuit 
        enc=binary_to_encode_param(np.binary_repr(i,n))
        params=np.concatenate((enc, weights))  

        circ = generate_network(n,m,L, encode=True,toggle_IL=True, real=real,repeat_params=repeat_params)
        circ = circ.assign_parameters(params) 

        # get target array 
        target_arr = np.zeros(2**(n+m))

        index = int(dec_to_bin(fx_arr[i],m,'unsigned mag',nint=mint)+dec_to_bin(x_arr[i],n,'unsigned mag',nint=nint),2)
        target_arr[index]=1 

        # get statevector 
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend)
        result = job.result()
        state_vector = np.asarray(result.get_statevector()) 

        signs[i]=np.sign(np.sum(np.real(state_vector)*np.sqrt(target_arr)))

        # calculate fidelity and mismatch 
        fidelity = np.abs(np.dot(np.sqrt(target_arr),np.conjugate(state_vector)))**2
        mismatch[i] = 1. - np.sqrt(fidelity) 
        
        # save as dictionary 
        dic = dict(zip(x_arr, mismatch)) 
        np.save(os.path.join("outputs",f"bar{name_str}.npy"), dic)
    
    if verbose:
        print("Mismatch by input state:")
        for i in x_arr:
            print(f"\t{np.binary_repr(i,n)}:  {mismatch[i]:.2e} ({signs[i]})")
        print(f"Mean: {np.mean(mismatch):.2e}; STDEV: {np.std(mismatch):.2e}")    
        print("")
        print("")    

    return 0 

def extract_phase(n):
    """
    For an `n`-qubit register storing computational basis state |k> representing a float between 0 and 1, transform to
        |k> -> e^(2 pi k) |k> 
    via single-qubit Ry rotations (based on scheme presented in Hayes 2023).

    This assumes an unsigned magnitude encoding with n precision bits. 
    """
    qc = QuantumCircuit(n, name="Extract Phase")
    qubits = list(range(n))
    
    for k in np.arange(0,n):
        lam = 2.*np.pi*(2.**(k-n))
        qubit = k
        qc.p(lam,qubits[qubit]) 
      
    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(n)
    circuit.append(qc_inst, qubits)    

    return circuit 

def full_encode(n,m, weights_A_str, weights_p_str,L_A,L_p, real_p, repeat_params=None, state_vec_file=None, save=False):
    """
    Amplitude encode (phase and amplitude) a quantum register using pre-trained weights.
    """

    # set up registers 
    input_register = QuantumRegister(n, "input")
    target_register = QuantumRegister(m, "target")
    circuit = QuantumCircuit(input_register, target_register) 

    # load weights 
    weights_A = np.load(weights_A_str)
    if type(weights_p_str) !=str:
        weights_p=weights_p_str
    else:    
        weights_p = np.load(weights_p_str)
    
    # encode amplitudes 
    circuit.compose(A_generate_network(n, L_A), input_register, inplace=True)
    circuit = circuit.assign_parameters(weights_A)

    # evaluate function
    qc = generate_network(n,m, L_p, real=real_p,repeat_params=repeat_params)
    qc = qc.assign_parameters(weights_p)
    inv_qc = qc.inverse()
    circuit.compose(qc, [*input_register,*target_register], inplace=True) 
    
    # extract phases 
    circuit.compose(extract_phase(m),target_register, inplace=True) 

    # clear ancilla register 
    circuit.compose(inv_qc, [*input_register,*target_register], inplace=True) 
 
    # get resulting statevector 
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector()

    state_vector = np.asarray(state_vector).reshape((2**m,2**n))

    state_v = state_vector[0,:].flatten()

    if save:
        # save to file 
        np.save(state_vec_file, state_v)
        return 0
    else:
        return state_v

def ampl_train_QNN(n,L,x_min,x_max,seed, epochs, func,func_str,loss_str,meta, recover_temp, nint, repeat_params):
    """
    Train circuit for amplitude encoding. 
    """

    # compress arguments into dictionary 
    args=compress_args_ampl(n,L,x_min,x_max,seed, epochs,func_str,loss_str,meta, nint, repeat_params)

    # set seed for PRNG 
    algorithm_globals.random_seed= seed
    
    # generate circuit and set up as QNN 
    qc = A_generate_network(n,L,repeat_params)
    qnn = SamplerQNN(
            circuit=qc.decompose(),            # decompose to avoid data copying (?)
            sampler=Sampler(options={"shots": 10000, "seed": algorithm_globals.random_seed}),
            weight_params=qc.parameters, 
            input_params=[],  
            input_gradients=False 
        )
    
    # set precision 
    if nint==None: nint=n   
    
    # choose initial weights
    recovered_k =0

    if recover_temp:    
        recover_labels=["weights", "mismatch", "loss"]
        recover_paths={}

        for k in np.arange(100,epochs, step=100):
            for e in np.arange(len(recover_labels)):
                file=os.path.join("ampl_outputs", f"__TEMP{k}_{recover_labels[e]}{vars_to_name_str_ampl(args)}.npy")
                recover_paths[recover_labels[e]]= (file if os.path.isfile(file) else None)

                if recover_labels[e]=="weights" and os.path.isfile(file):
                    recovered_k=k+1      
        
        if not None in list(recover_paths.values):
            initial_weights=np.load(recover_paths["weights"])
        else:
            initial_weights =np.zeros(len(qc.parameters)) #algorithm_globals.random.random(len(qc.parameters))    
    
    else:
        initial_weights = np.zeros(len(qc.parameters)) #algorithm_globals.random.random(len(qc.parameters))
    
    # initialise TorchConnector
    model = TorchConnector(qnn, initial_weights)

    # choose optimiser and loss function 
    optimizer = Adam(model.parameters(), lr=0.01, betas=(0.7, 0.999), weight_decay=0.005) # Adam optimizer 

    if loss_str=="MSE":
        criterion=MSELoss() 
    elif loss_str=="L1":
        criterion=L1Loss() 
    elif loss_str=="KLD":
        criterion=KLDivLoss()
    elif loss_str=="CE":
        criterion=CrossEntropyLoss()
    elif loss_str=="MM":
        def criterion(output, target):
            return  torch.abs(1. -torch.sum(torch.mul(output, target)))  # redefine to punish sign errors (moved abs outwards)  
                    
    # set up arrays to store training outputs 
    if recover_temp and not None in list(recover_paths.values):
        mismatch_vals=np.load(recover_paths["mismatch"])
        loss_vals=np.load(recover_paths["loss"])
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
        circ = A_generate_network(n,L, repeat_params)

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

            temp_labels=["weights", "mismatch", "loss"] 
            temp_arrs=[generated_weights, mismatch_vals, loss_vals] 

            for e in np.arange(len(temp_labels)):
                # save temp file 
                file=os.path.join("ampl_outputs",f"__TEMP{i}_{temp_labels[e]}{vars_to_name_str_ampl(args)}")
                np.save(file,temp_arrs[e])

                # delete previous temp file 
                old_file=os.path.join("ampl_outputs",f"__TEMP{i-100}_{temp_labels[e]}{vars_to_name_str_ampl(args)}")
                os.remove(old_file) if os.path.isfile(old_file) else None

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
    temp_labels=["weights", "mismatch", "loss"]  
    for i in np.arange(len(temp_labels)):
        file=os.path.join("outputs",f"__TEMP{temp_ind}_{temp_labels[i]}{vars_to_name_str_ampl(args)}.npy")
        os.remove(file) if os.path.isfile(file) else None             

    # save outputs 
    with no_grad():
            generated_weights = model.weight.detach().numpy()
    outputs= [generated_weights, mismatch_vals, loss_vals, np.real(state_vector)]
    output_labels=["weights", "mismatch", "loss", "statevec"]  
    for i in np.arange(len(outputs)):
        np.save(os.path.join("outputs", f"{output_labels[i]}{vars_to_name_str_ampl(args)}"), outputs[i])         

    return 0
