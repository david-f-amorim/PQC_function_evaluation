import numpy as np 
from qiskit import QuantumCircuit 
from qiskit.circuit import ParameterVector
from itertools import combinations

def input_layer(circuit, input_register, target_register):

    return 0 

"""
Construct the two-qubit N gate (as defined in Vatan 2004)
in terms of three parameters, stored in list or tuple 'params'
"""

def N_gate(params):

    circuit = QuantumCircuit(2)
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
    qubits = np.arange(m)

    # number of parameters used by each N gate 
    num_par = 3 

    # number of gates applied per layer 
    num_gates = m

    # set up parameter vector 
    param_index = 0
    params = ParameterVector(par_label, length= num_par * num_gates)

    # apply N gate linearly between neighbouring qubits
    # (including circular connection between last and first) 
    pairs = [tuple([i, i+1]) for i in qubits[:-1]]
    pairs.append((qubits[-1], 0))

    for j in np.arange(num_gates):
        qc.compose(N_gate(params[param_index : (param_index + num_par)]),pairs[j],inplace=True)
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
    qubits = np.arange(m)

    # number of parameters used by each N gate 
    num_par = 3 

    # number of gates applied per layer 
    num_gates = 0.5 * m * (m-1)

    # set up parameter vector 
    param_index = 0
    params = ParameterVector(par_label, length= num_par * num_gates)

    # apply N gate linearly between neighbouring qubits
    # (including circular connection between last and first) 
    pairs = list(combinations(qubits,2))

    for j in np.arange(num_gates):
        qc.compose(N_gate(params[param_index : (param_index + num_par)]),pairs[j],inplace=True)
        param_index += num_par 
    
    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(m)
    circuit.append(qc_inst, qubits)
    
    return circuit 