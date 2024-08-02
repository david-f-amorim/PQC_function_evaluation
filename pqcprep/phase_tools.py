"""
Collection of functions regarding phase extraction. 
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from .pqc_tools import generate_network, A_generate_network 
from .tools import get_state_vec


def extract_phase(n):
    r"""

    Constructs an operator that extracts the phase associated with a given computational basis state:
    $ \ket{k} \mapsto e^{2 \pi k} \ket{k}$. 

    This is based on a sequence of single-qubit Rz rotations, as shown in Eqs. (13)-(15) in [Hayes 2023](https://arxiv.org/pdf/2306.11073). 
    Unsigned magnitude encoding is assumed. 

    Arguments: 
    ---
    - **n** : *int* 

        Number of qubits in the system. 

    Returns:
    ---
    - **circuit** : *QuantumCircuit* 

        Implementation of the gate as a qiskit `QuantumCircuit`.       

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

def full_encode(n,m, weights_A_str, weights_p_str,L_A,L_p, real_p, repeat_params=None, state_vec_file=None, save=False, full_state_vec=False, Q_only=False):
    """
    
    Execute the quantum state preparation protocol using pre-trained QCNN weights. 

    Arguments:
    ---
    - **n** : *int* 

        Number of qubits in the input register. 

    - **m** : *int* 

        Number of qubits in the target register. 

    - **weights_A_str** : *str*

        File path to storage location of amplitude encoding QCNN weights. 

    - **weights_p_str** : *str*

        File path to storage location of phase encoding QCNN weights.    

    - **L_A** : *int* 

        Number of layers in the amplitude encoding QCNN.     

    - **L_p** : *int* 

        Number of layers in the phase encoding QCNN. 

    - **real_p** : *boolean* 

        Value of the `real` argument of the phase encoding QCNN. See `pqcprep.pqc_tools.generate_network()` for details. 

    - **repeat_params** : *boolean*

        Value of the `repeat_params` argument of the phase encoding QCNN. See `pqcprep.pqc_tools.generate_network()` for details. Default is False. 

    - **state_vec_file** : *str*, *optional* 

        File path to store output state vector. 

    - **save** : *boolean* 

        If True, output statevector is saved at `state_vec_file`. Default is False.      

    - **full_state_vec** : *boolean* 

        If True, the full statevector is additionally returned, including non-cleared ancilla states. Default is False. 

    - **Q_only** : *boolean* 

        If True, only apply the function evaluation operator and leave out the phase extraction. Default is False    


    Returns:
    ---
    - **state_v** : *array_like* 

        Array representing the statevector of the input register, assuming a cleared target register. 

    - **state_v_full** : *array_like*, *optional* 

        Array representing the statevector of the input register for the various target register configurations.    
        Only returned if `full_state_vec` is True. 

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
    
    if not Q_only:
        # extract phases 
        circuit.compose(extract_phase(m),target_register, inplace=True) 

        # clear ancilla register 
        circuit.compose(inv_qc, [*input_register,*target_register], inplace=True) 
 
    # get resulting statevector 
    state_vector = get_state_vec(circuit).reshape((2**m,2**n))

    state_v = state_vector[0,:].flatten()

    if save:
        if state_vec_file==None: raise ValueError("Must provide file path to save output.")
        
        # save to file 
        np.save(state_vec_file, state_v)
        if full_state_vec:
            np.save(state_vec_file+"full", state_vector)

        return 0
    else:
        if full_state_vec:
            return state_v, state_vector 
        else:
            return state_v
