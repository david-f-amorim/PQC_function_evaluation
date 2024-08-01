"""
Collection of functions regarding phase extraction. 
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from .pqc_tools import generate_network, A_generate_network 
from .tools import get_state_vec


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
    state_vector = get_state_vec(circuit).reshape((2**m,2**n))

    state_v = state_vector[0,:].flatten()

    if save:
        # save to file 
        np.save(state_vec_file, state_v)
        return 0
    else:
        return state_v