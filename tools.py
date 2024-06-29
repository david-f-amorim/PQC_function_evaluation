import numpy as np 
from qiskit import QuantumCircuit, QuantumRegister 
from qiskit.circuit import ParameterVector
from itertools import combinations

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
Set up a network consisting of input and convolutional layers acting on n input 
qubits and m target qubits. For now, use a single input layer and alternating quadratic 
and linear convolutional layers, with L convolutional layers in total. 
Both the input state and the circuit weights can be set by accessing circuit parameters
after initialisation.  
"""

def generate_network(n,m,L):

    # initialise empty input and target registers 
    input_register = QuantumRegister(n, "input")
    target_register = QuantumRegister(m, "target")
    circuit = QuantumCircuit(input_register, target_register) 

    # prepare registers 
    circuit.h(target_register)
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

####

circ = digital_encoding(2)
print(circ)
circ = circ.assign_parameters([0, np.pi])
#circ = circ.bind_parameters([0, np.pi])
print(circ)

