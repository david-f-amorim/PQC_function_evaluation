"""
Collection of useful functions for network training purposes.
"""

import numpy as np 
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_algorithms.utils import algorithm_globals 
from qiskit.primitives import Sampler
from torch.optim import Adam
from torch import Tensor, no_grad
import sys, time, os, warnings, torch  

from .binary_tools import bin_to_dec, dec_to_bin  
from .phase_tools import full_encode, phase_from_state
from .pqc_tools import generate_network, binary_to_encode_param, A_generate_network, get_state_vec  
from .file_tools import compress_args,compress_args_ampl, vars_to_name_str, vars_to_name_str_ampl 
from .psi_tools import psi, A 
from .__input__ import DIR 

#---------------------------------------------------------------------------------------------------

def set_loss_func(loss_str, arg_dict, ampl=False):
    r"""
    Set the loss function to be used in training a network. 

    Arguments:
    ----

    - **loss_str** : *str*

        String specifying the loss function to use. Options are 

        - `'MSE'` : mean squared error loss. Using [pytorch's implementation](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) with default settings. 
          `criterion` takes two pytorch `Tensors` as inputs, corresponding to the network output and the desired output. 

        - `'L1'` : mean absolute error loss. Using [pytorch's implementation](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) with default settings. 
          `criterion` takes two pytorch `Tensors` as inputs, corresponding to the network output and the desired output.    

        - `'KLD'` : Kullback-Leibler divergence loss. Using [pytorch's implementation](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html) with default settings. 
          `criterion` takes two pytorch `Tensors` as inputs, corresponding to the network output and the desired output.   

        - `'CE'` : cross entropy loss. Using [pytorch's implementation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) with default settings. 
          `criterion` takes two pytorch `Tensors` as inputs, corresponding to the network output and the desired output.  

        - `'SAM'` : sign-adjusted mismatch. Defined as $$\text{SAM}(\ket{x}, \ket{y}) =  1 - \sum_k |x_k| |y_k|,$$ where $\ket{x}$, $\ket{y}$ are the 
            network output and desired output, respectively, and $x_k$, $y_k$ are the coefficients w.r.t the two-register computational basis states.  `criterion` takes 
            two pytorch `Tensors` as inputs, corresponding to the network output and the desired output.       

        - `'WIM'` : weighted mismatch. Defined analogously to SAM but with additional weights: $$\text{WIM}(\ket{x}, \ket{y}) =1 - \sum_k w_k |x_k||y_k|.$$
           `criterion` takes three pytorch `Tensors` as inputs, corresponding to the network output, the desired output and the weights. See `set_WIM_weights()`
           for information on how the weights are calculated. This loss function is not an option if `ampl` is True. 

        - `'WILL'` : weighted Lp loss. Defined as $$\text{WILL}(\ket{x}, \ket{y}; p,q) = \sum_k |x_k -y_k|^p + |x_k| |[k]_m - \Psi([k]_n)|^q,$$ where $p$ and $q$ are coefficients stored in 
           `arg_dict['WILL_p']`, `arg_dict['WILL_q']`, $[k]_n$ is the target register bit-string associated with the basis state $\ket{k}$, $[k]_n$ is the input register bit-string 
           associated with the basis state $\ket{k}$, $\Psi$ is the function to be evaluated for the network, and $x_k$, $y_k$ have the same meaning as above.  `criterion` takes 
            two pytorch `Tensors` as inputs, corresponding to the network output and the desired output. This loss function is not an option if `ampl` is True.               

    - **arg_dict** : *dict* 

        A dictionary containing information on training variables, created with `pqcprep.file_tools.compress_args()` (or created 
        with `pqcprep.file_tools.compress_args_ampl()` in the case of `ampl` being True). 

    - **ampl** : *boolean* 

        If True, the loss function is defined for an amplitude-encoding network, as opposed to a function evaluation network. Default is 
        False.     
        
    Returns: 
    ----

    - **criterion** : *callable* 

        The loss function as a callable object. Number and type of arguments depend on the chosen `loss_str` option (see above).    

    """
    
    if loss_str=="MSE":
        from torch.nn import MSELoss
        criterion=MSELoss() 
    elif loss_str=="L1":
        from torch.nn import L1Loss
        criterion=L1Loss() 
    elif loss_str=="KLD":
        from torch.nn import KLDivLoss
        criterion=KLDivLoss()
    elif loss_str=="CE":
        from torch.nn import CrossEntropyLoss
        criterion=CrossEntropyLoss()
    elif loss_str=="SAM":
        def criterion(output, target):
            return  torch.abs(1. -torch.sum(torch.mul(output, target))) 
    elif loss_str=="WIM":  
        if arg_dict["train_superpos"]==False:
            raise ValueError(f"The loss function {loss_str} requires training in superposition, i.e. 'train_superpos==True'.") 
        if ampl:
            raise ValueError(f"The loss function {loss_str} is not available for amplitude training. Allowed options are 'CE', 'MSE', 'L1', 'KLD', 'SAM'.")

        def criterion(output, target, weights):
            output = torch.mul(output, weights)  
            output = output / torch.sum(torch.mul(output, output)) 
            return  torch.abs(1. -torch.sum(torch.mul(output, target)))     
    elif loss_str=="WILL":  
        if arg_dict["train_superpos"]==False:
             raise ValueError(f"The loss function {loss_str} requires training in superposition, i.e. 'train_superpos==True'.")
        if ampl:
            raise ValueError(f"The loss function {loss_str} is not available for amplitude training. Allowed options are 'CE', 'MSE', 'L1', 'KLD', 'SAM'.")
        fx_arr = [psi(i, mode=arg_dict["func_str"]) for i in np.arange(0, 2**arg_dict["n"])]
        if arg_dict["phase_reduce"]:
            fx_arr = [np.modf(i/ (2* np.pi))[0] for i in fx_arr]
        
        fx_arr_rounded = [bin_to_dec(dec_to_bin(i,arg_dict["m"],'unsigned mag', nint=arg_dict["mint"]),'unsigned mag',nint=arg_dict["mint"]) for i in fx_arr]
        distance_arr = np.empty(2**(arg_dict["n"]+arg_dict["m"]))
        
        for i in np.arange(2**arg_dict["n"]):
            bin_i=dec_to_bin(i,arg_dict["n"],'unsigned mag') 
            for j in np.arange(2**arg_dict["m"]):
                bin_j=dec_to_bin(j,arg_dict["m"],'unsigned mag')
                ind=int(bin_j + bin_i,2) 
                distance_arr[ind] = np.abs(bin_to_dec(dec_to_bin(j,arg_dict["m"],'unsigned mag'),'unsigned mag',nint=arg_dict["mint"]) - fx_arr_rounded[i])  
        distance=Tensor(distance_arr)   
        
        def criterion(output, target):
            loss =torch.pow(torch.abs(output-target),arg_dict["WILL_p"]) + torch.mul(torch.abs(output),torch.pow(distance,arg_dict["WILL_q"])) 
            return torch.sum(loss)**(1/arg_dict["WILL_p"]) / torch.numel(loss)
    else:
        raise ValueError("Unrecognised loss function. Options are: 'CE', 'MSE', 'L1', 'KLD', 'SAM', 'WIM', 'WILL'.")

    return criterion 

def set_WIM_weights(generated_weights, arg_dict):
    """ 
    
    Determine the weight coefficients for the WIM loss function. 

    Arguments:
    ----
    - **generated_weights** : *array_like*

        The network weights generated in a given epoch. 

    - **arg_dict** : *dict* 

        A dictionary containing information on training variables, created with `pqcprep.file_tools.compress_args()`. 

    Returns:
    ----
    - **WIM_weights_arr** : *array_like*

        The calculated coefficients for the WIM loss function. 

    """
    
    # initialise arrays to store results 
    WIM_weights_arr= np.empty(2**(arg_dict["n"]+arg_dict["m"]))
    
    # iterate over input states 
    x_arr_temp=np.arange(2**arg_dict["n"])
    fx_arr_temp = [psi(k, mode=arg_dict["func_str"]) for k in x_arr_temp]

    if arg_dict["phase_reduce"]: fx_arr_temp = [np.modf(k/ (2* np.pi))[0] for k in fx_arr_temp]

    for q in x_arr_temp:
        
        # prepare circuit 
        enc=binary_to_encode_param(np.binary_repr(q,arg_dict["n"]))
        params=np.concatenate((enc, generated_weights))  

        qc = generate_network(arg_dict["n"],arg_dict["m"],arg_dict["L"], encode=True,toggle_IL=True, real=arg_dict["real"],repeat_params=arg_dict["repeat_params"])
        qc = qc.assign_parameters(params) 

        # get target array 
        target_arr_temp = np.zeros(2**(arg_dict["n"]+arg_dict["m"]))

        index = int(dec_to_bin(fx_arr_temp[q],arg_dict["m"],'unsigned mag',nint=arg_dict["mint"])+dec_to_bin(x_arr_temp[q],arg_dict["n"],'unsigned mag',nint=arg_dict["nint"]),2)
        target_arr_temp[index]=1 

        # get statevector 
        state_vector_temp = get_state_vec(qc)

        # for each output state, calculate the "binary difference" to the target as well as the "coefficient difference"
        sum = 0 
        for j in np.arange(2**arg_dict["m"]):
            ind = int(dec_to_bin(j,arg_dict["m"],'unsigned mag',nint=arg_dict["m"])+dec_to_bin(x_arr_temp[q],arg_dict["n"],'unsigned mag',nint=arg_dict["nint"]),2)
            
            num_dif =np.abs(fx_arr_temp[q] -bin_to_dec(dec_to_bin(j,arg_dict["m"],'unsigned mag', nint=arg_dict["mint"]),'unsigned mag',nint=arg_dict["mint"]))
            coeff_dif = np.abs(state_vector_temp[ind]- target_arr_temp[ind])
            sum += num_dif * coeff_dif

        # add to weights_arr 
        for j in np.arange(2**arg_dict["m"]):
            ind = int(dec_to_bin(j,arg_dict["m"],'unsigned mag',nint=arg_dict["m"])+dec_to_bin(x_arr_temp[q],arg_dict["n"],'unsigned mag',nint=arg_dict["nint"]),2) 
            WIM_weights_arr[ind]= sum    

    # focus on outliers: double-weight on states 0.5 sigma or more above the mean
    WIM_weights_arr += (WIM_weights_arr >= np.mean(WIM_weights_arr)+0.5 *np.std(WIM_weights_arr) ).astype(int) 
    
    # smoothen WIM weights 
    WIM_weights_arr=np.exp(0.8 * WIM_weights_arr)

    return WIM_weights_arr

def train_QNN(n,m,L, seed, epochs,func_str,loss_str,meta, recover_temp, nint, mint, phase_reduce, train_superpos, real, repeat_params, WILL_p, WILL_q, delta):
    """
    Train a QCNN to perform function evaluation $\ket{j}\ket{0} \mapsto \ket{j}\ket{\Psi(j)}$.

    The QCNN is generated using `pqcprep.pqc_tools.generate_network()`. 

    Arguments:
    ---
    - **n** : *int*

        Number of qubits in the input register. 

    - **m** : *int*

        Number of qubits in the target register. 

    - **L** : *int*

        Number of layers in the network. 

    - **seed** : *int* 

        Seed for random number generation. 

    - **epochs** : *int* 

        Number of training runs. 

    - **func_str** : *str*

        String specifying the function $\Psi$ to be evaluated. Must be a valid option for the argument `mode` of `pqcprep.psi_tools.psi()`. 

    - **loss_str** : *str* 

        String specifying the loss function minimised by the optimiser. Must be a valid option for the argument `loss_str` of `set_loss_func()`. 

    - **meta** : *str*

        String containing meta information to be included in output file names. 

    - **recover_temp** : *boolean* 

        If True, continue training from TEMP files (should they exist). If False and TEMP files exist they will be overwritten. 

    - **nint** : *int*

        Number of integer qubits in input register. 

    - **mint** : *int*

        Number of integer qubits in target register.  

    - **phase_reduce** : *boolean* 

        If True, reduce $\Psi(j)$ to the interval $(0, 2 \pi]$ i.e. perform the mapping $\Psi \to \Psi \text{mod} 2 \pi$. 

    - **train_superpos** : *boolean*

        If True, train on a superposition of input states. If False, train on randomly sampled individual input states. 

    - **real** : *boolean*

        If True, generate a network only involving CX and Ry rotations, resulting in real amplitudes. 

    - **repeat_params** : *str*, *optional* 

        Keep parameters fixed for different layer types, i.e. use the same parameter values for each instance of a layer type. 
        Options are `None` (do not keep parameters fixed), `'CL'` (keep parameters fixed for convolutional layers), 
        `'IL'` (keep parameters fixed for input layers), `'both'` (keep parameters fixed for both convolutional and input layers).    

    - **WILL_p** : *float* 

        The $p$ parameter of the WILL loss function, as described in `set_loss_func()`.

    - **WILL_q** : *float* 

        The $q$ parameter of the WILL loss function, as described in `set_loss_func()`.    

    - **delta** : *float* 

        Hyper-parameter controlling the sampling of input state coefficients when training in superposition (`train_superpos = True`). Must be 
        between 0 and 1. `delta = 0` corresponds to coefficients fixed at $\\frac{1}{\\sqrt{2^n}}$ while `delta = 1` corresponds to coefficients randomly assuming values on the range $(0,1)$. 
        Intermediate values of `delta` result in coefficinets being randomly sampled on an interval around $\\frac{1}{\\sqrt{2^n}}$, with the range of the interval 
        determined by the value of `delta`.     

    Returns:
    ----

    The output produced by the training is saved in binary `.npy` files in the directory `DIR/ampl_outputs` using naming convention `<TYPE>_<NAME_STR>.npy`
    where `<NAME_STR>` is the name string produced by `pqcprep.file_tools.vars_to_name_str()` and `<TYPE>` is one of: 

    - `weights` : file containing the QCNN weights determined by the optimiser;

    - `loss` : file containing the loss value after each epoch; 

    - `mismatch` : file containg the mismatch value after each epoch; 

    - `grad` : file containing the  squared weight gradient norm after each epoch;  

    - `vargrad` : file containing the variance of the weight gradients after each epoch.   

    TAKE DECISION ON DIR HANDLING LATER !!!    

    """
    
    # compress arguments into dictionary 
    args =compress_args(n,m,L, seed, epochs,func_str,loss_str,meta,nint, mint, phase_reduce, train_superpos, real, repeat_params, WILL_p, WILL_q, delta)

    # set precision strings 
    if nint==None: nint=n
    if mint==None: mint=m  
    if phase_reduce: mint=0

    # set seed for PRNG 
    algorithm_globals.random_seed= seed
    rng = np.random.default_rng(seed=seed)

    # generate circuit and set up as QNN 
    qc = generate_network(n,m,L, encode=not train_superpos, toggle_IL=True, initial_IL=True,input_Ry=train_superpos, real=real,repeat_params=repeat_params, wrap=False)
                                                                                        
    qnn = SamplerQNN(
                    circuit=qc.decompose(),           
                    sampler=Sampler(options={"shots": 10000, "seed": algorithm_globals.random_seed}),
                    input_params=qc.parameters[:n], 
                    weight_params=qc.parameters[n:],
                    input_gradients=True
                )
              
    # choose initial weights
    recovered_k =0
    if recover_temp:    
        recover_labels=["weights", "mismatch", "loss", "grad", "vargrad"]
        recover_paths={}
        for k in np.arange(100,epochs, step=100):
            for e in np.arange(len(recover_labels)):
                file=os.path.join(DIR,"outputs", f"__TEMP{k}_{recover_labels[e]}{vars_to_name_str(args)}.npy")
                recover_paths[recover_labels[e]]= (file if os.path.isfile(file) else None)

                if recover_labels[e]=="weights" and os.path.isfile(file):
                    recovered_k=k+1
                     
        if not None in list(recover_paths.values):
            initial_weights=np.load(recover_paths["weights"])
        else:
            initial_weights =np.zeros(len(qc.parameters[n:])) # initialise parameters to zero   
    else:
        initial_weights =np.zeros(len(qc.parameters[n:])) # initialise parameters to zero   
           
    # initialise TorchConnector
    model = TorchConnector(qnn, initial_weights)

    # choose optimiser 
    optimizer = Adam(model.parameters(), lr=0.01, betas=(0.7, 0.999), weight_decay=0.005) 
                
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
        fx_arr = [psi(i, mode=func_str) for i in x_arr]

        if phase_reduce:
            fx_arr = [np.modf(i/ (2* np.pi))[0] for i in fx_arr]

        # convert fx_arr to binary at available target register precision 
        fx_arr_bin = [dec_to_bin(i,m,nint=mint,encoding="unsigned mag") for i in fx_arr]

        if np.max(fx_arr)> 2.**mint - 2.**(-pm) and mint != 0:
            raise ValueError(f"Insufficient number of target (integer) qubits.")
        
        # get bit strings corresponding to target arrays and convert to indices
        target_bin = [fx_arr_bin[i]+x_arr_bin[i] for i in x_arr]
        target_ind = [bin_to_dec(i, encoding='unsigned mag') for i in target_bin]

        # prepare target array 
        target_arr = np.zeros(2**(n+m))
        for k in target_ind:
            target_arr[int(k)]=1

    else:        
        x_min = 0
        x_max = 2.**nint - 2.**(-pn) 
        x_arr = np.array(x_min + (x_max - x_min) *rng.random(size=epochs))
        fx_arr = [psi(i, mode=func_str) for i in x_arr]

        # reduce to phase value between 0 and 1:
        if phase_reduce: 
            fx_arr = [np.modf(i/ (2* np.pi))[0] for i in fx_arr]
        
        if np.max(fx_arr)> 2.**mint - 2.**(-pm) and mint != 0:
            raise ValueError(f"Insufficient number of target (integer) qubits.")

    # choose loss function 
    criterion=set_loss_func(loss_str, args)
    
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
        else:

            # generate random coefficients 
            coeffs = np.array(np.pi / 2 * (1+delta *(2 *rng.random(size=n)-1)))
            
            # get input data 
            input=Tensor(coeffs)

            # get target data 
            
            target_ampl = np.empty(2**(n+m))

            for j in np.arange(2**n):
                for k in np.arange(2**m):
                    ind =int(dec_to_bin(k,m)+dec_to_bin(j,n),2) 
                    bin_n = dec_to_bin(j,n)[:: -1] # reverse bit order !
                    val=1 

                    for l in np.arange(n):
                        val *= np.cos(coeffs[l]/2) if bin_n[l]=='0' else np.sin(coeffs[l]/2)

                    target_ampl[ind] = val * target_arr[ind] 

            target=Tensor(target_ampl**2)

        # train model  
        optimizer.zero_grad()

        # apply loss function          
        if loss_str=="WIM":
            WIM_weights_arr=np.ones(2**(n+m)) if i==recovered_k else WIM_weights_arr
            WIM_weights_tensor=Tensor(WIM_weights_arr)
            loss =criterion(torch.sqrt(torch.abs(model(input))+1e-10), torch.sqrt(target), WIM_weights_tensor)    # add small number in sqrt to avoid zero grad !
        else: 
            loss = criterion(torch.sqrt(torch.abs(model(input))+1e-10), torch.sqrt(target))                       # add small number in sqrt to avoid zero grad !

        # propagate gradients and recompute weights
        loss.backward()
        optimizer.step()

        # save loss and grad for plotting 
        loss_vals[i]=loss.item()
        grad_vals[i]=np.sum(model.weight.grad.numpy()**2)
        var_grad_vals[i]=np.std(model.weight.grad.numpy())**2
        
        # set up circuit with calculated weights
        circ = generate_network(n,m,L, encode=not train_superpos, toggle_IL=True, initial_IL=True,input_Ry=train_superpos, real=real,repeat_params=repeat_params, wrap=False)
        
        with no_grad():
            generated_weights = model.weight.detach().numpy()   
        if train_superpos:
            input_params = coeffs
            params=np.concatenate((input_params, generated_weights))  
        else:
            input_params = binary_to_encode_param(dec_to_bin(x_arr[i],n,'unsigned mag',nint=nint))
            params = np.concatenate((input_params, generated_weights))   
        circ = circ.assign_parameters(params)

        # get statevector 
        state_vector = get_state_vec(circ)

        # calculate fidelity and mismatch
        target_state = target_ampl**2 if train_superpos else target_arr 
        fidelity = np.abs(np.dot(np.sqrt(target_state),np.conjugate(state_vector)))**2
        mismatch = 1. - np.sqrt(fidelity)
        mismatch_vals[i]=mismatch

        # set loss func weights for WIM
        if loss_str=="WIM" and (i % 10 ==0) and (i >=1): WIM_weights_arr= set_WIM_weights(generated_weights, args)
            
        # temporarily save outputs every hundred iterations
        temp_ind = epochs - 100 
        
        if recover_temp:
            temp_ind = recovered_k -1

        if (i % 100 ==0) and (i != 0) and (i != epochs-1): 

            temp_labels=["weights", "mismatch", "loss", "grad", "vargrad"] 
            temp_arrs=[generated_weights, mismatch_vals, loss_vals, grad_vals, var_grad_vals] 

            for e in np.arange(len(temp_labels)):
                # save temp file 
                file=os.path.join(DIR,"outputs",f"__TEMP{i}_{temp_labels[e]}{vars_to_name_str(args)}")
                np.save(file,temp_arrs[e])

                # delete previous temp file 
                old_file=os.path.join(DIR,"outputs",f"__TEMP{i-100}_{temp_labels[e]}{vars_to_name_str(args)}.npy")
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
        file=os.path.join(DIR,"outputs",f"__TEMP{temp_ind}_{temp_labels[i]}{vars_to_name_str(args)}.npy")
        os.remove(file) if os.path.isfile(file) else None
                            
    # save outputs 
    with no_grad():
            generated_weights = model.weight.detach().numpy()
    outputs= [generated_weights, mismatch_vals, loss_vals, grad_vals, var_grad_vals]
    output_labels=["weights", "mismatch", "loss", "grad", "vargrad"]  
    for i in np.arange(len(outputs)):
        np.save(os.path.join(DIR,"outputs", f"{output_labels[i]}{vars_to_name_str(args)}"), outputs[i])      

    return 0 

def test_QNN(n,m,L,seed,epochs, func_str,loss_str,meta,nint,mint,phase_reduce,train_superpos,real,repeat_params,WILL_p, WILL_q,delta,verbose=True):   
    """
    Test performance of a QCNN trained for function evaluation with respect to different metrics. 

    This requires the existence of an appropriate `weights_<NAME_STR>.npy` file (as produced by `train_QNN()`) in the directory `DIR/outputs`. 

    Arguments: 
    ---

    Same arguments as `train_QNN()`. See there for a description. 

    Returns:
    ---

    The testing output produced is saved in binary `.npy` files in the directory `DIR/ampl_outputs` using naming convention `<TYPE>_<NAME_STR>.npy`
    where `<NAME_STR>` is the name string produced by `pqcprep.file_tools.vars_to_name_str()` and `<TYPE>` is one of: 
 
    - `mismatch_by_state` : file containing the mismatch between the state produced by the network and the desired output state for each of the possible input 
        register states. Contains a dictionary with the input bit strings as keys and the associated mismatch as values. 

    - `phase` : file containing the phase function encoded by the network when the input register is in an equal superposition of input states. 

    - `metrics` : file containing a dictionary with several metrics evaluating the performance of the network. These metrics are 
        * `mu` : mean mismatch (mean of the data contained in `mismatch_by_state_<NAME_STR>.npy`); should be zero for ideal performance
        * `sigma` : mismatch standard deviation (standard deviation of the data contained in `mismatch_by_state_<NAME_STR>.npy`);  should be zero for ideal performance
        * `eps` : normalisation error on the state vector associated with the data contained in `phase_<NAME_STR>.npy`;  should be zero for ideal performance
        * `chi` : mean absolute difference between the phase function contained in `phase_<NAME_STR>.npy` and the rounded desired phase function;  should be zero for ideal performance
        * `omega` : a super-metric defined as `1/(mu + sigma + eps + chi)` ; should be maximal for ideal performance

    TAKE DECISION ON DIR HANDLING LATER !!! 

    """
    # compress arguments into dictionary 
    args =compress_args(n,m,L, seed, epochs,func_str,loss_str,meta,nint, mint, phase_reduce, train_superpos, real, repeat_params, WILL_p, WILL_q, delta)
    name_str=vars_to_name_str(args)  

    # set precision strings 
    if nint==None: nint=n
    if mint==None: mint=m  
    if phase_reduce: mint=0                  

    # load weights 
    if os.path.isfile(os.path.join(DIR,"outputs",f"weights{name_str}.npy")):
        weights = np.load(os.path.join(DIR,"outputs",f"weights{name_str}.npy"))
    else:
        raise ValueError("No appropriate QCNN weights could be found. Check the network configuration as well as the relevant directory.")    

    # initialise array to store results 
    mismatch = np.empty(2**n)
    signs = np.empty(2**n)
    
    # iterate over input states 
    x_arr = np.arange(2**n)
    fx_arr = [psi(i, mode=func_str) for i in x_arr]

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
        state_vector = get_state_vec(circ)

        signs[i]=np.sign(np.sum(np.real(state_vector)*np.sqrt(target_arr)))

        # calculate fidelity and mismatch 
        fidelity = np.abs(np.dot(np.sqrt(target_arr),np.conjugate(state_vector)))**2
        mismatch[i] = 1. - np.sqrt(fidelity) 
            
    # get phase target 
    fx_arr_bin = [dec_to_bin(i,m, "unsigned mag", 0) for i in fx_arr]
    phase_target =  np.array([bin_to_dec(i,"unsigned mag", mint) for i in fx_arr_bin])
    if phase_reduce:
        phase_target *= 2 * np.pi 

    # get state vector for full phase extraction 
    state_vec= full_encode(n,m, weights_A_str=None, weights_p_str=weights, L_A=None, L_p=L,real_p=real,repeat_params=repeat_params,full_state_vec=False, no_UA=True, operators="QRQ")
    phase = phase_from_state(state_vec)

    # calculate metrics
    mu = np.mean(mismatch)
    sigma = np.std(mismatch) 
    eps = 1 - np.sum(np.abs(state_vec)**2)
    chi = np.mean(np.abs(phase - phase_target))
    omega= 1/(mu+sigma+eps+chi) 

    # save outputs
    full_dic = dict(zip(x_arr, mismatch)) 
    np.save(os.path.join(DIR,"outputs",f"mismatch_by_state{name_str}.npy"), full_dic) 

    metric_dic = {} 
    metric_dic["mu"]=mu 
    metric_dic["sigma"]=sigma   
    metric_dic["eps"]=eps 
    metric_dic["chi"]=chi
    metric_dic["omega"]=omega  
    np.save(os.path.join(DIR,"outputs",f"metrics{name_str}.npy"), metric_dic) 

    np.save(os.path.join(DIR,"outputs",f"phase{name_str}.npy"), phase) 

    if verbose:
        print("Mismatch by input state:")
        for i in x_arr:
            print(f"\t{np.binary_repr(i,n)}:  {mismatch[i]:.2e} ({signs[i]})")
        print("-----------------------------------")
        print(f"Mu: \t{mu:.3e}") 
        print(f"Sigma: \t{sigma:.3e}") 
        print(f"Eps: \t{eps:.3e}")
        print(f"Chi: \t{chi:.3e}") 
        print(f"Omega: \t{omega:.3f}")
        print("-----------------------------------")
        print("")
        print("")    

    return 0 

def ampl_train_QNN(n,L,x_min,x_max,seed, epochs,func_str,loss_str,meta, recover_temp, nint, repeat_params):
    """

    Train a QCNN to prepare an amplitude distribution: $\ket{0} \mapsto \sum_j A(j) \ket{j}$.

    The QCNN is generated using `pqcprep.pqc_tools.A_generate_network()`. 

    Arguments:
    ---

    - **n** : *int* 

        Number of qubits in the register. 

    - **L** : *int* 

        Number of layers in the network. 

    - **x_min** : *float* 

        Minimum of function domain.         

    - **x_max** : *float* 

        Maximum of function domain.   

    - **seed** : *int* 

        Seed for random number generation. 

    - **epochs** : *int* 

        Number of training runs. 

    - **func_str** : *str*

        String specifying the function $A$ to be prepared. Must be a valid option for the argument `mode` of `pqcprep.psi_tools.A()`. 

    - **loss_str** : *str* 

        String specifying the loss function minimised by the optimiser. Must be a valid option for the argument `loss_str` of `set_loss_func()`. 

    - **meta** : *str*

        String containing meta information to be included in output file names. 

    - **recover_temp** : *boolean* 

        If True, continue training from TEMP files (should they exist). If False and TEMP files exist they will be overwritten. 

    - **nint** : *int*

        Number of integer qubits in the register.  

    - **repeat_params** : *boolean* 

        If True, keep parameters fixed for different layer types, i.e. use the same parameter values for each instance of a layer type.
          

    Returns:
    ---

    The output produced by the training is saved in binary `.npy` files in the directory `DIR/ampl_outputs` using naming convention `<TYPE>_<NAME_STR>.npy`
    where `<NAME_STR>` is the name string produced by `pqcprep.file_tools.vars_to_name_str_ampl()` and `<TYPE>` is one of: 

    - `weights` : file containing the QCNN weights determined by the optimiser;

    - `state_vec` : file containing the statevector corresponding to the register after applying the QCNN; 

    - `loss` : file containing the loss value after each training run; 

    - `mismatch` : file containg the mismatch value after each training run.   

    TAKE DECISION ON DIR HANDLING LATER !!!    
        
    """

    # compress arguments into dictionary 
    args=compress_args_ampl(n,L,x_min,x_max,seed, epochs,func_str,loss_str,meta, nint, repeat_params)

    # set seed for PRNG 
    algorithm_globals.random_seed= seed
    rng = np.random.default_rng(seed=seed)
    
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
                file=os.path.join(DIR,"ampl_outputs", f"__TEMP{k}_{recover_labels[e]}{vars_to_name_str_ampl(args)}.npy")
                recover_paths[recover_labels[e]]= (file if os.path.isfile(file) else None)

                if recover_labels[e]=="weights" and os.path.isfile(file):
                    recovered_k=k+1      
        
        if not None in list(recover_paths.values):
            initial_weights=np.load(recover_paths["weights"])
        else:
            initial_weights =np.zeros(len(qc.parameters)) # initialise parameters to zero   
    
    else:
        initial_weights = np.zeros(len(qc.parameters)) # initialise parameters to zero   
    
    # initialise TorchConnector
    model = TorchConnector(qnn, initial_weights)

    # choose optimiser and loss function 
    optimizer = Adam(model.parameters(), lr=0.01, betas=(0.7, 0.999), weight_decay=0.005) # Adam optimizer 
    criterion=set_loss_func(loss_str, args, ampl=True)
                    
    # set up arrays to store training outputs 
    if recover_temp and not None in list(recover_paths.values):
        mismatch_vals=np.load(recover_paths["mismatch"])
        loss_vals=np.load(recover_paths["loss"])
    else:    
        mismatch_vals = np.empty(epochs)
        loss_vals = np.empty(epochs)

    # calculate target and normalise 
    dx = (x_max-x_min)/(2**n)
    target_arr = np.array([A(i, mode=func_str) for i in np.arange(x_min,x_max, dx)])**2
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

        # train model  
        optimizer.zero_grad()
        loss = criterion(torch.sqrt(torch.abs(model(input))+1e-10), torch.sqrt(target)) # adding 1e-10 to prevent taking sqrt(0) ??!!
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
        state_vector = get_state_vec(circ)
        
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
                file=os.path.join(DIR,"ampl_outputs",f"__TEMP{i}_{temp_labels[e]}{vars_to_name_str_ampl(args)}")
                np.save(file,temp_arrs[e])

                # delete previous temp file 
                old_file=os.path.join(DIR,"ampl_outputs",f"__TEMP{i-100}_{temp_labels[e]}{vars_to_name_str_ampl(args)}.npy")
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
        file=os.path.join(DIR,"ampl_outputs",f"__TEMP{temp_ind}_{temp_labels[i]}{vars_to_name_str_ampl(args)}.npy")
        os.remove(file) if os.path.isfile(file) else None             

    # save outputs 
    with no_grad():
            generated_weights = model.weight.detach().numpy()
    outputs= [generated_weights, mismatch_vals, loss_vals, state_vector]
    output_labels=["weights", "mismatch", "loss", "statevec"]  
    for i in np.arange(len(outputs)):
        np.save(os.path.join(DIR,"ampl_outputs", f"{output_labels[i]}{vars_to_name_str_ampl(args)}"), outputs[i])         

    return 0
