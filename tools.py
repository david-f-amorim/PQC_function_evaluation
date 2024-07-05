import numpy as np 
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute 
from qiskit.circuit import ParameterVector
from itertools import combinations
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit.library import CU3Gate, U3Gate
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss, KLDivLoss
from torch import Tensor, no_grad 
import sys, time, os 

"""
Construct an input layer consisting of controlled single-qubit rotations
with the n input qubits acting as controls and the m target qubits
acting as targets. 
"""

def input_layer(n, m, par_label, ctrl_state=0):

    # set up circuit 
    qc = QuantumCircuit(n+m, name="Input Layer")
    qubits = list(range(n+m))

    # number of parameters used by each gate 
    num_par = 3 

    # number of gates applied per layer 
    num_gates = n

    # set up parameter vector 
    params = ParameterVector(par_label, length= num_par * num_gates)
    param_index = 0

    # apply gates to qubits 
    for i in qubits[:n]:

        j = i 
        if j >=m:
            j -= m

        par = params[int(param_index) : int(param_index + num_par)]    

        cu3 = U3Gate(par[0],par[1],par[2]).control(1, ctrl_state=ctrl_state)
        qc.append(cu3, [qubits[i], qubits[j+n]])

        param_index += num_par
        

    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(n+m)
    circuit.append(qc_inst, qubits)
    
    return circuit 

"""
Construct the two-qubit N gate (as defined in Vatan 2004)
in terms of three parameters, stored in list or tuple 'params'
"""

def N_gate(params):

    circuit = QuantumCircuit(2, name="N Gate")
    circuit.rz(-np.pi / 2, 1)
    circuit.cx(1, 0)
    circuit.rz(params[0], 0)
    circuit.ry(params[1], 1)
    circuit.cx(0, 1)
    circuit.ry(params[2], 1)
    circuit.cx(1, 0)
    circuit.rz(np.pi / 2, 0)

    return circuit 

"""
Construct a linear (neighbour-to-neighbour) convolutional layer
via the cascaded application of the N gate to the m-qubit target 
register.   
"""

def conv_layer_NN(m, par_label):

    # set up circuit 
    qc = QuantumCircuit(m, name="Convolutional Layer (NN)")
    qubits = list(range(m))

    # number of parameters used by each N gate 
    num_par = 3 

    # number of gates applied per layer 
    num_gates = m

    # set up parameter vector 
    param_index = 0
    params = ParameterVector(par_label, length= int(num_par * num_gates))

    # apply N gate linearly between neighbouring qubits
    # (including circular connection between last and first) 
    pairs = [tuple([i, i+1]) for i in qubits[:-1]]
    pairs.append((qubits[-1], 0))

    for j in np.arange(num_gates):
        qc.compose(N_gate(params[int(param_index) : int(param_index + num_par)]),pairs[int(j)],inplace=True)
        if j != num_gates -1:
            qc.barrier()
        param_index += num_par 
    
    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(m)
    circuit.append(qc_inst, qubits)
    
    return circuit 

"""
Construct a quadratic (all-to-all) convolutional layer
via the cascaded application of the N gate to the m-qubit target 
register.   
"""

def conv_layer_AA(m, par_label):

    # set up circuit 
    qc = QuantumCircuit(m, name="Convolutional Layer (AA)")
    qubits = list(range(m))

    # number of parameters used by each N gate 
    num_par = 3 

    # number of gates applied per layer 
    num_gates = 0.5 * m * (m-1)

    # set up parameter vector 
    param_index = 0
    params = ParameterVector(par_label, length= int(num_par * num_gates))

    # apply N gate linearly between neighbouring qubits
    # (including circular connection between last and first) 
    pairs = list(combinations(qubits,2))

    for j in np.arange(num_gates):
        qc.compose(N_gate(params[int(param_index) : int(param_index + num_par)]),pairs[int(j)],inplace=True)
        if j != num_gates -1:
            qc.barrier()
        param_index += num_par 
    
    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(m)
    circuit.append(qc_inst, qubits)
    
    return circuit 

"""
Digitally encode an n-bit binary number onto an n-qubit register. 
The encoding is set by assigning the value 0 to the i-th component of the
parameter vector to represent a bit value of "0" for the i-th bit and assigning
pi for the case of "1". 

"""

def digital_encoding(n):

    qc = QuantumCircuit(n,name="Digital Encoding")
    qubits = list(range(n))
    params = ParameterVector("enc", length=n)

    for i in np.arange(n):
        qc.rx(params[i], qubits[i]) 
        qc.p(params[i]/2, qubits[i])

    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(n)
    circuit.append(qc_inst, qubits)    

    return circuit 

"""
Convert an n-bit binary string to the associated parameter array to feed 
into the digital encoding circuit. 
"""

def binary_to_encode_param(binary):

    params = np.empty(len(binary))

    binary = binary[::-1]  # reverse binary string (small endian for binary, big endian for arrays)

    for i in np.arange(len(binary)):
        if binary[i]=="0":
            params[i]=0 
        elif binary[i]=="1":
            params[i]=np.pi 
        else: 
            raise ValueError("Binary string should only include characters '0' and '1'.")        

    return params 

"""
Set up a network consisting of input and convolutional layers acting on n input 
qubits and m target qubits. For now, use a single input layer and alternating quadratic 
and linear convolutional layers, with L convolutional layers in total. 
Both the input state and the circuit weights can be set by accessing circuit parameters
after initialisation.  
"""

def generate_network(n,m,L, encode=False, toggle_IL=False):

    # initialise empty input and target registers 
    input_register = QuantumRegister(n, "input")
    target_register = QuantumRegister(m, "target")
    circuit = QuantumCircuit(input_register, target_register) 

    # prepare registers 
    circuit.h(target_register)
    if encode:
        circuit.compose(digital_encoding(n), input_register, inplace=True)

    # apply input layer 
    circuit.compose(input_layer(n,m, u"\u03B8_IN"), circuit.qubits, inplace=True)
    circuit.barrier()

    # apply convolutional layers (alternating between AA and NN)
    # if toggle_IL is True, additional input layers are added after 
    # each NN
    for i in np.arange(L):

        if toggle_IL==False:

            if i % 2 ==0:
                circuit.compose(conv_layer_AA(m, u"\u03B8_AA_{0}".format(i // 2)), target_register, inplace=True)
            elif i % 2 ==1:
                circuit.compose(conv_layer_NN(m, u"\u03B8_NN_{0}".format(i // 2)), target_register, inplace=True)
        
        if toggle_IL==True:

            if i % 3 ==0:
                circuit.compose(conv_layer_AA(m, u"\u03B8_AA_{0}".format(i // 3)), target_register, inplace=True)
            elif i % 3 ==1:
                circuit.compose(conv_layer_NN(m, u"\u03B8_NN_{0}".format(i // 3)), target_register, inplace=True)
            elif i % 3 ==2:
                # alternate between layers with control states 0 and 1 
                if i % 2 == 1:
                    circuit.compose(input_layer(n,m, u"\u03B8_IN_{0}".format(i // 3), ctrl_state=1), circuit.qubits, inplace=True) 
                elif i % 2 == 0:
                    circuit.compose(input_layer(n,m, u"\u03B8_IN_{0}".format(i // 3), ctrl_state=0), circuit.qubits, inplace=True)     

        if i != L-1:
            circuit.barrier()

    return circuit 

"""
Initialise circuit as QNN for training purposes.
"""

def train_QNN(n,m,L, seed, shots, lr, b1, b2, epochs, func,func_str,loss_str,meta, recover_temp):

    # set seed for PRNG 
    algorithm_globals.random_seed= seed
    rng = np.random.default_rng(seed=seed)

    # generate circuit and set up as QNN 
    qc = generate_network(n,m,L, encode=True, toggle_IL=True)
    qnn = SamplerQNN(
            circuit=qc.decompose(),            # decompose to avoid data copying (?)
            sampler=Sampler(options={"shots": shots, "seed": algorithm_globals.random_seed}),
            input_params=qc.parameters[:n],    # encoding params treated as input params
            weight_params=qc.parameters[n:],   # encoding params not selected as weights
            input_gradients=True # ?? 
        )

    # choose initial weights
    recovered_k =0

    if recover_temp:    
        recovered_weights=None
        recovered_mismatch=None 
        recovered_loss=None 
        for k in np.arange(100,epochs, step=100):
            if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_weights_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
                recovered_weights=os.path.join("outputs", f"__TEMP{k}_weights_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")
                recovered_k=k+1
            if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
                recovered_mismatch=os.path.join("outputs", f"__TEMP{k}_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")
            if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
                recovered_loss=os.path.join("outputs", f"__TEMP{k}_loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")        
        
        if recovered_weights != None and recovered_mismatch != None and recovered_loss != None:
            initial_weights=np.load(recovered_weights)
        else:
            initial_weights = algorithm_globals.random.random(len(qc.parameters[n:]))    
    
    else:
        initial_weights = algorithm_globals.random.random(len(qc.parameters[n:]))
    
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

    # generate x and f(x) values (IMPROVE LATER!!)
    x_min = 0
    x_max = 2**n 
    x_arr = rng.integers(x_min, x_max, size=epochs)
    fx_arr = [func(i) for i in x_arr]

    if np.log2(np.max(fx_arr))> m:
        raise ValueError(f"Insufficient number of target qubits: at least {int(np.ceil(np.log2(np.max(fx_arr))))} required")

    # start training 
    print(f"\n\nTraining started. Epochs: {epochs}. Input qubits: {n}. Target qubits: {m}. QCNN layers: {L}. \n")
    start = time.time() 

    for i in np.arange(epochs)[recovered_k:]:

        # get input data
        input = Tensor(binary_to_encode_param(np.binary_repr(x_arr[i],n))) 

        # get target data 
        target_arr = np.zeros(2**(n+m))
        index = int(np.binary_repr(fx_arr[i],m)+np.binary_repr(x_arr[i],n),2)
        target_arr[index]=1 
        target=Tensor(target_arr)

        # train model  
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        loss.backward()
        optimizer.step()

        # save loss for plotting 
        loss_vals[i]=loss.item()

        # set up circuit with calculated weights
        circ = generate_network(n,m,L, encode=True,toggle_IL=True)

        with no_grad():
            generated_weights = model.weight.detach().numpy()

        input_params = binary_to_encode_param(np.binary_repr(x_arr[i],n))
        params = np.concatenate((input_params, generated_weights))           
        circ = circ.assign_parameters(params)    

        # get statevector 
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend)
        result = job.result()
        state_vector = np.asarray(result.get_statevector()) 

        # calculate fidelity and mismatch 
        fidelity = np.abs(np.dot(target_arr,np.conjugate(state_vector)))**2
        mismatch = 1. - np.sqrt(fidelity)

        # save mismatch for plotting 
        mismatch_vals[i]=mismatch

        # temporarily save outputs every hundred iterations
        if recover_temp:
            temp_ind = recovered_k -1

        if (i % 100 ==0) and (i != 0) and (i != epochs-1): 
            np.save(os.path.join("outputs", f"__TEMP{i}_weights_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),generated_weights)
            np.save(os.path.join("outputs", f"__TEMP{i}_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),mismatch_vals)
            np.save(os.path.join("outputs", f"__TEMP{i}_loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),loss_vals)
            
            # delete previous temp files
            prev_weights=f"__TEMP{i-100}_weights_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"
            prev_mismatch=f"__TEMP{i-100}_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"
            prev_loss=f"__TEMP{i-100}_loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"

            if os.path.isfile(os.path.join("outputs", prev_weights)):
                os.remove(os.path.join("outputs", prev_weights))

            if os.path.isfile(os.path.join("outputs", prev_mismatch)):
                os.remove(os.path.join("outputs", prev_mismatch))

            if os.path.isfile(os.path.join("outputs", prev_loss)):
                os.remove(os.path.join("outputs", prev_loss))        

            # make not of last created temp files
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
    
    elapsed = time.time()-start
    mins, sec = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    time_str = f"{int(hours):02}:{int(mins):02}:{sec:05.2f}" 

    # decompose circuit for gate count 
    num_CX = dict(circ.decompose(reps=4).count_ops())["cx"]
    num_gates = num_CX + dict(circ.decompose(reps=4).count_ops())["u"]

    print(f"\nTraining completed in {time_str}. Number of weights: {len(generated_weights)}. Number of gates: {num_gates} (of which CX gates: {num_CX}). \n\n")

    # delete temp files 
    temp_weights=f"__TEMP{temp_ind}_weights_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"
    temp_mismatch=f"__TEMP{temp_ind}_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"
    temp_loss=f"__TEMP{temp_ind}_loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"

    if os.path.isfile(os.path.join("outputs", temp_weights)):
            os.remove(os.path.join("outputs", temp_weights))
    if os.path.isfile(os.path.join("outputs", temp_mismatch)):
            os.remove(os.path.join("outputs", temp_mismatch))
    if os.path.isfile(os.path.join("outputs", temp_loss)):
            os.remove(os.path.join("outputs", temp_loss))               

    # save outputs 
    with no_grad():
            generated_weights = model.weight.detach().numpy()

    np.save(os.path.join("outputs", f"weights_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),generated_weights)
    np.save(os.path.join("outputs", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),mismatch_vals)
    np.save(os.path.join("outputs", f"loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),loss_vals)
    

    return 0 

"""
Test performance of trained QNN for the various input states
"""

def test_QNN(n,m,L,epochs, func, func_str,loss_str,meta,verbose=True): 

    # load weights 
    weights = np.load(os.path.join("outputs",f"weights_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))

    # initialise array to store results 
    mismatch = np.empty(2**n)

    # iterate over input states 
    x_arr = np.arange(2**n)

    for i in x_arr:
        
        # prepare circuit 
        enc=binary_to_encode_param(np.binary_repr(i,n))
        params=np.concatenate((enc, weights))  

        circ = generate_network(n,m,L, encode=True,toggle_IL=True)
        circ = circ.assign_parameters(params) 

        # get target array 
        target_arr = np.zeros(2**(n+m))
        index = int(np.binary_repr(func(i),m)+np.binary_repr(i,n),2)
        target_arr[index]=1 

        # get statevector 
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend)
        result = job.result()
        state_vector = np.asarray(result.get_statevector()) 

        # calculate fidelity and mismatch 
        fidelity = np.abs(np.dot(target_arr,np.conjugate(state_vector)))**2
        mismatch[i] = 1. - np.sqrt(fidelity) 

        # save as dictionary 
        dic = dict(zip(x_arr, mismatch)) 
        np.save(os.path.join("outputs",f"bar_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"), dic)
    
    if verbose:
        print("Mismatch by input state:")
        for i in x_arr:
            print(f"\t{np.binary_repr(i,n)}:  {mismatch[i]:.2e}")
        print("")
        print("")    

    return 0 

"""
For a given set of input parameters, check if training and testing results already exist. 
"""

def check_duplicates(n,m,L,epochs,func_str,loss_str,meta):

    check_mismatch = os.path.isfile(os.path.join("outputs", f"mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))
    check_weights = os.path.isfile(os.path.join("outputs", f"loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))
    check_loss = os.path.isfile(os.path.join("outputs", f"weights_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))
    
    return check_mismatch & check_weights & check_loss

"""
For a given set of input parameters, check if temp files already exist. 
"""

def check_temp(n,m,L,epochs,func_str,loss_str,meta):

    check_mismatch=False 
    check_weights=False 
    check_loss=False

    for k in np.arange(100,epochs, step=100):
        if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_weights_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
            check_weights=True 
        if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
            check_mismatch=True 
        if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
            check_loss=True         
    
    return check_mismatch & check_weights & check_loss

"""
For a given set of input parameters, check if plots already exist (excluding compare plots). 
"""

def check_plots(n,m,L,epochs,func_str, loss_str, meta, log):

    log_str= ("" if log==False else "log_")

    check_mismatch =os.path.isfile(os.path.join("plots", f"{log_str}mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.png"))
    check_loss=os.path.isfile(os.path.join("plots", f"{log_str}loss_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.png"))
    check_bars=os.path.isfile(os.path.join("plots", f"{log_str}bar_mismatch_{n}_{m}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.png"))

    return check_mismatch & check_loss & check_bars 

"""
Generate random seed from timestamp
"""

def generate_seed():
    
    return int(time.time())
