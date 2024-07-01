import numpy as np 
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute 
from qiskit.circuit import ParameterVector
from itertools import combinations
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from torch.optim import Adam
from torch.nn import MSELoss
from torch import Tensor, no_grad 
import sys, time, os 

"""
Construct an input layer consisting of controlled Ry rotations
with the n input qubits acting as controls and the m target qubits
acting as targets. 
"""

def input_layer(n, m, par_label):

    # set up circuit 
    qc = QuantumCircuit(n+m, name="Input Layer")
    qubits = list(range(n+m))

    # number of parameters used by each gate 
    num_par = 1 

    # number of gates applied per layer 
    num_gates = n

    # set up parameter vector 
    params = ParameterVector(par_label, length= num_par * num_gates)

    # apply gates to qubits 
    for i in qubits[:n]:

        j = i 
        if j >=m:
            j -= m

        qc.cry(params[i], qubits[i], qubits[j+n])
        

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

def generate_network(n,m,L, encode=False):

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
    for i in np.arange(L):

        if i % 2 ==0:
            circuit.compose(conv_layer_AA(m, u"\u03B8_AA_{0}".format(i // 2)), target_register, inplace=True)
        elif i % 2 ==1:
            circuit.compose(conv_layer_NN(m, u"\u03B8_NN_{0}".format(i // 2)), target_register, inplace=True)
        
        if i != L-1:
            circuit.barrier()

    return circuit 

"""
Initialise circuit as QNN for training purposes.
"""

def train_QNN(n,m,L, seed, shots, lr, b1, b2, epochs, func):

    # set seed for PRNG 
    algorithm_globals.random_seed= seed
    rng = np.random.default_rng(seed=seed)

    # generate circuit and set up as QNN 
    qc = generate_network(n,m,L, encode=True)
    qnn = SamplerQNN(
            circuit=qc.decompose(),            # decompose to avoid data copying (?)
            sampler=Sampler(options={"shots": shots, "seed": algorithm_globals.random_seed}),
            input_params=qc.parameters[:n],    # encoding params treated as input params
            weight_params=qc.parameters[n:],   # encoding params not selected as weights
            input_gradients=True # ?? 
        )

    # choose random initial weights and initalise TorchConnector 
    initial_weights = algorithm_globals.random.random(len(qc.parameters[n:]))
    model = TorchConnector(qnn, initial_weights)

    # choose optimiser and loss function 
    optimizer = Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005) # Adam optimizer 
    criterion = MSELoss(reduction="sum") # MSE loss 

    # set up arrays to store training outputs 
    mismatch_vals = np.empty(epochs)
    loss_vals = np.empty(epochs)

    # generate x and f(x) values (IMPROVE LATER!!)
    x_min = 0
    x_max = 2**n 
    x_arr = rng.integers(x_min, x_max, size=epochs)
    x_arr =0 * np.ones(epochs, dtype=int)
    fx_arr = [func(i) for i in x_arr]

    # start training 
    print("\n\nTraining started \n")
    start = time.time() 

    for i in np.arange(epochs):

        # get input data
        input = Tensor(binary_to_encode_param(np.binary_repr(x_arr[i],n))) 

        # get target data (RETHINK THIS!!!)
        target_arr = np.zeros(2**(n+m))
        index = int(np.binary_repr(x_arr[i],n)+np.binary_repr(fx_arr[i],m),2)
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
        circ = generate_network(n,m,L, encode=True)

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
        fidelity = np.abs(np.dot(np.sqrt(target_arr),np.conjugate(state_vector)))**2
        mismatch = 1. - np.sqrt(fidelity)

        # save mismatch for plotting 
        mismatch_vals[i]=mismatch

        # print status
        a = int(20*(i+1)/epochs)

        if i==0:
            time_str="--:--:--.--"
        elif i==epochs-1:
            time_str="00:00:00.00"    
        else:
            remaining = ((time.time() - start) / i) * (epochs - i)
            mins, sec = divmod(remaining, 60)
            hours, mins = divmod(mins, 60)
            time_str = f"{int(hours):02}:{int(mins):02}:{sec:05.2f}"

        prefix="\t" 
        print(f"{prefix}[{u'█'*a}{('.'*(20-a))}] {100.*((i+1)/epochs):.2f}% ; Loss {loss_vals[i]:.2f} ; Mismatch {mismatch:.2e} ; ETA {time_str}", end='\r', file=sys.stdout, flush=True)
        
        
    print(" ", flush=True, file=sys.stdout) 

    elapsed = time.time()-start
    mins, sec = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    time_str = f"{int(hours):02}:{int(mins):02}:{sec:05.2f}" 

    print(f"\nTraining completed in {time_str}\n\n")
    ## GIVE MORE INFO... NUMBER OF CX GATES ETC

    # save outputs (FIND BETTER NAMING CONVENTIONS!)
    with no_grad():
            generated_weights = model.weight.detach().numpy()

    np.save(os.path.join("outputs", "weights_"+str(int(n))+"_"+str(int(m))+"_"+str(int(L))+"_"+str(int(epochs))),generated_weights)
    np.save(os.path.join("outputs", "mismatch_"+str(int(n))+"_"+str(int(m))+"_"+str(int(L))+"_"+str(int(epochs))),mismatch_vals)
    np.save(os.path.join("outputs", "loss_"+str(int(n))+"_"+str(int(m))+"_"+str(int(L))+"_"+str(int(epochs))),loss_vals)
    
    return 0 

"""
Analytical test function
"""

def f(x):

    return x

####

train_QNN(n=2,m=2,L=8, seed=1680458526, shots=500, lr=0.01, b1=0.7, b2=0.99, epochs=200, func=f)


"""
NOTES:
    - when given fixed input of 3 (11): learns well; when given 2 (10) or 1 (01): does not learn at all [loss always 1.25, mismatch always 1]; given 1 (01):
       does not learn at all [loss always 0.73, mismatch always 0.5]
        --> related to random seed/initial weights? something about circuit layout??
        ----> try add more input layers? 


"""





