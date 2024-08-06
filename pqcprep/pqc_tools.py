"""
Collection of functions relating to setting up a QCNN.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import StatevectorSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import U3Gate
from itertools import combinations

def N_gate(params, real=False):
    r"""
    Constructs the two-qubit three-parameter $\mathcal{N}$ gate, defined as 
    $$\mathcal{N}(\alpha, \beta, \gamma) = \exp \left( i \left[ \alpha X \otimes X + \beta Y \otimes Y + \gamma Z \otimes Z \right] \right),$$
    where $X$, $Y$, $Z$ are the Pauli operators. 

    This is implemented via three CX gates, three Rz gates and two Ry gates, as shown in [Vatan 2004](https://arxiv.org/pdf/quant-ph/0308006).  

    Arguments:
    ----
    - **params** : *array_like*

        Array-like object containing the gate parameters, corresponding to the qubit rotation angles. 

    - **real** : *boolean*
       
        If True, a two-parameter version of the $\mathcal{N}$ gate is implemented instead, with one of CX gatess and all of the 
        Rz gates removed. This ensures real amplitudes. Default is False.   
             
    Returns:
    ----
    - **circuit** : *QuantumCircuit*

        Implementation of the $\mathcal{N}$ gate as a qiskit `QuantumCircuit`. 

    """

    if real:
        circuit = QuantumCircuit(2, name="RN Gate")
        circuit.cx(1, 0)
        circuit.ry(params[0], 0)
        circuit.ry(params[1], 1)
        circuit.cx(0, 1)
    else:    
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

def input_layer(n, m, par_label, ctrl_state=0, real=False, params=None, AA=False, shift=0, wrap=False): 
    r"""

    Implement a QCNN input layer as a parametrised quantum circuit. 

    The layer consists of a sequence of controlled arbitrary single-qubit operations (CU3 operations)
    applied to the target register with the input qubits acting a controls. This feeds information 
    about the input register state into the target register. 
    
    Arguments:
    ----
    - **n** : *int*

        Number of qubits in the input register. 

    - **m** : *int* 

        Number of qubits in the target register. 

    - **par_label** : *str*

        Label to assign to the parameter vector. 

    - **ctrl_state** : *int* 

        Control state of the controlled gates. If equal to 0, the controlled operation is applied 
        when the control qubit is in state $\ket{0}$. If equal to 1, the controlled operation is 
        applied when the control qubit is in state $\ket{1}$. Default is 0. 

    - **real** : *boolean*

        If True, controlled Ry rotations are applied instead of CU3 operations. This ensures real 
        amplitudes. Default is False.     

    - **params** : *array_like*, *optional*

        Directly assign values, stored in `params`, to the circuit parameters instead of creating a `ParameterVector`.      

    - **AA** : *boolean* 

        If True, each input qubit controls an operation on each target qubit, corresponding to an "all-to-all" layer topology. 
        Default is False. 

    - **shift** : *int* 

        If `AA` is False, the $j$th input qubit controls an operation applied on the $(j+s)$th target qubit with wrap-around 
        for `n > m`. Default is 0.    

    - **wrap**: *boolean* 

        If True, map rotation angles to an interval specified by `map_angles()`. Default is False.     

    Returns:
    ---
    - **circuit** : *QuantumCircuit* 

        The qiskit `QuantumCircuit` implementation of the input layer. 
    
    """

    # set up circuit 
    qc = QuantumCircuit(n+m, name="Input Layer")
    qubits = list(range(n+m))

    # number of parameters used by each gate 
    num_par = 3 if real==False else 1

    # number of gates applied per layer 
    num_gates = n if AA==False else n*m

    # set up parameter vector 
    if params == None:
        params = ParameterVector(par_label, length= num_par * num_gates)
    param_index = 0

    # define weight re-mapping function:
    def wrap_angle(theta):
        if wrap:
            return map_angle(theta)  
        else:
            return theta 

    # apply gates to qubits 
    if AA: 
        for i in qubits[:n]:
            for j in qubits[n:]:
                if real:
                    qc.cry(wrap_angle(params[int(param_index)]), qubits[i], qubits[j])
                    param_index += 1
                else:  
                    par = params[int(param_index) : int(param_index + num_par)] 
                    cu3 = U3Gate(wrap_angle(par[0]),wrap_angle(par[1]),wrap_angle(par[2])).control(1, ctrl_state=ctrl_state)
                    qc.append(cu3, [qubits[i], qubits[j]])
                    param_index += num_par   
    else:    
        for i in qubits[:n]:

            j = i + shift  
            if np.modf(j/m)[1] >= 1:
                j -=int(np.modf(j/m)[1] * m)

            if real:
                qc.cry(wrap_angle(params[i]), qubits[i], qubits[j+n])
            else:
                par = params[int(param_index) : int(param_index + num_par)] 
                cu3 = U3Gate(wrap_angle(par[0]),wrap_angle(par[1]),wrap_angle(par[2])).control(1, ctrl_state=ctrl_state)
                qc.append(cu3, [qubits[i], qubits[j+n]])
                param_index += num_par
        
    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(n+m)
    circuit.append(qc_inst, qubits)
    
    return circuit 

def conv_layer_NN(m, par_label, real=False, params=None, wrap=False):
    """

    Implement a QCNN neighbour-to-neighbour convolutional layer as a parametrised quantum circuit. 

    The layer consists of the cascaded application of the two-qubit $\mathcal{N}$ gate on
    the target register. $\mathcal{N}$ is applied to all neighbouring target qubits, including a connection between
    the first and last qubit ("neighbour-to-neighbour" topology), resulting in a gate cost linear in the size of the target register. 
    
    Arguments:
    ----
    - **m** : *int* 

        Number of qubits in the target register. 

    - **par_label** : *str*

        Label to assign to the parameter vector. 

    - **real** : *boolean*

        If True, the real version of the $\mathcal{N}$ gate is used (which only involves CX and Ry operations). This ensures real 
        amplitudes. Default is False.     

    - **params** : *array_like*, *optional*

        Directly assign values, stored in `params`, to the circuit parameters instead of creating a `ParameterVector`. 

    - **wrap**: *boolean* 

        If True, map rotation angles to an interval specified by `map_angles()`. Default is False.         
  
    Returns:
    ---
    - **circuit** : *QuantumCircuit* 

        The qiskit `QuantumCircuit` implementation of the input layer.
    """

    # set up circuit 
    qc = QuantumCircuit(m, name="Convolutional Layer (NN)")
    qubits = list(range(m))

    # number of parameters used by each N gate 
    num_par = 3 if real==False else 2 

    # number of gates applied per layer 
    num_gates = m

    # define weight re-mapping function:
    def wrap_angle(theta):
        if wrap:
            return map_angle(theta)  
        else:
            return theta

    # set up parameter vector 
    param_index = 0
    if params == None:
        params = ParameterVector(par_label, length= int(num_par * num_gates))

    # apply N gate linearly between neighbouring qubits
    # (including circular connection between last and first) 
    pairs = [tuple([i, i+1]) for i in qubits[:-1]]
    pairs.append((qubits[-1], 0))

    for j in np.arange(num_gates):
        pars = params[int(param_index) : int(param_index + num_par)] 
        qc.compose(N_gate([wrap_angle(i) for i in pars], real=real),pairs[int(j)],inplace=True)
        param_index += num_par 
    
    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(m)
    circuit.append(qc_inst, qubits)
    
    return circuit 

def conv_layer_AA(m, par_label, real=False, params=None, wrap=False): 
    """

    Implement a QCNN all-to-all convolutional layer as a parametrised quantum circuit. 

    The layer consists of the cascaded application of the two-qubit $\mathcal{N}$ gate on
    the target register. $\mathcal{N}$ is applied to all combinations of target qubits ("all-to-all" topology),
    resulting in a gate cost quadratic in the size of the target register. 
    
    Arguments:
    ----
    - **m** : *int* 

        Number of qubits in the target register. 

    - **par_label** : *str*

        Label to assign to the parameter vector. 

    - **real** : *boolean*

        If True, the real version of the $\mathcal{N}$ gate is used (which only involves CX and Ry operations). This ensures real 
        amplitudes. Default is False.     

    - **params** : *array_like*, *optional*

        Directly assign values, stored in `params`, to the circuit parameters instead of creating a `ParameterVector`.   

    - **wrap**: *boolean* 

        If True, map rotation angles to an interval specified by `map_angles()`. Default is False.        
  
    Returns:
    ---
    - **circuit** : *QuantumCircuit* 

        The qiskit `QuantumCircuit` implementation of the input layer.
    """

    # set up circuit 
    qc = QuantumCircuit(m, name="Convolutional Layer (AA)")
    qubits = list(range(m))

    # number of parameters used by each N gate 
    num_par = 3 if real==False else 2 

    # number of gates applied per layer 
    num_gates = 0.5 * m * (m-1)

    # define weight re-mapping function:
    def wrap_angle(theta):
        if wrap:
            return map_angle(theta)  
        else:
            return theta

    # set up parameter vector 
    param_index = 0
    if params == None:
        params = ParameterVector(par_label, length= int(num_par * num_gates))

    # apply N gate linearly between neighbouring qubits
    # (including circular connection between last and first) 
    pairs = list(combinations(qubits,2))

    for j in np.arange(num_gates):
        pars = params[int(param_index) : int(param_index + num_par)]
        qc.compose(N_gate([wrap_angle(i) for i in pars], real=real),pairs[int(j)],inplace=True)
        if j != num_gates -1:
            #qc.barrier()
            s=" "
        param_index += num_par 
    
    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(m)
    circuit.append(qc_inst, qubits)
    
    return circuit 

def digital_encoding(n):   
    r"""
    Set up a parametrised quantum circuit for the digital encoding of a binary number onto 
    a quantum register. 

    The encoding is set by assigning the value `0` to the $i$th parameter of the circuit to 
    represent a state $\ket{0}$ for the $i$th bit and assigning $\pi$ for the case $\ket{1}$. 
    Values can be assigned to circuit parameters using qiskit's `QuantumCircuit.assign_parameters()`.  

    This is to be used in conjunction with the parameter array generated by `binary_to_encode_param()`. 

    Example usage: 

            >>> binary='011001'                         # bit string to encode 
            >>> params=binary_to_encode_param(binary)   # generate parameter array 
            >>> qc=digital_encoding(len(binary))        # generate circuit 
            >>> qc.assign_parameters(params)            # assign parameters to circuit 

    Arguments:
    ----
    - **n** : *int* 

        Number of qubits in the register. Should be equal to the number of bits 
        in the bit strings that are to be encoded. 

    Returns:
    ----
    - **circuit** : *QuantumCircuit* 

        The qiskit `QuantumCircuit` implementation of the encoding circuit, with `n` unassigned parameters. 

    """

    # set up circuit and parameter vector
    qc = QuantumCircuit(n,name="Digital Encoding")
    qubits = list(range(n))
    params = ParameterVector("enc", length=n)

    # flip ith qubit if params[i]==np.pi
    for i in np.arange(n):
        qc.rx(params[i], qubits[i]) 
        qc.p(params[i]/2, qubits[i])

    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(n)
    circuit.append(qc_inst, qubits)    

    return circuit 

def binary_to_encode_param(binary):        
    """

    Generate the parameter array for digital encoding corresponding to a binary string. 

    This is to be used in conjunction with the encoding circuit generated by `digital_encoding()`. 

    Example usage: 

            >>> binary='011001'                         # bit string to encode 
            >>> params=binary_to_encode_param(binary)   # generate parameter array 
            >>> qc=digital_encoding(len(binary))        # generate circuit 
            >>> qc.assign_parameters(params)            # assign parameters to circuit 

    Arguments:
    ----
    - **binary** : *str* 

        The binary string. Little endian convention is assumed. 

    Returns:
    ----
    - **params** : *array_like* 

        The parameter array of length `len(binary)` corresponding to `binary`. Big endian convention is used. 


    Convert an n-bit binary string to the associated parameter array to feed 
    into the digital encoding circuit. 
    """

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

def generate_network(n,m,L, encode=False, toggle_IL=True, initial_IL=True, input_Ry=False, real=False, inverse=False, repeat_params=None, wrap=False):
    r"""
    Set up a QCNN consisting of input and convolutional layers acting on two distinct registers, the 'input register' and the 'target register'. 

    Input layers consist of single-qubit operations applied to the target register, controlled by the input register qubits. Convolutional layers
    consist of two-qubit operations applied to the target register. See `input_layer()` for more information on input layers and `conv_layer_AA()`, 
    `conv_layer_NN()` for more information on convolutional layers. 

    The network was designed for the task of performing function evaluation, i.e. to implement an operator $\hat{Q}_\Psi$ such that 
    $$ \hat{Q}_\Psi \ket{j}_i \ket{0}_t  = \ket{j}_i \ket{\Psi(j)}_t,$$
    for some function $\Psi$ with the subscripts $i$ and $t$ denoting the input and target registers, respectively. 
    With some adaptation, the network structure could also be used for other applications. 

    Arguments:
    ---
    - **n** : *int* 

        Number of qubits in the input register. 

    - **m** : *int* 

        Number of qubits in the target register. 

    - **L** : *int* 

        Number of layers in the network. Note that `L` does not take into account the optional initial input layer        
        added by `initial_IL`. Further, `L` does not take into account the padding of the network with additional input 
        layers to ensure the number of input layers is at least equal to `m`. 

    - **encode** : *boolean*

        If True, apply an initial encoding circuit to the input register which can be used control the input states of the network. 
        Default is False. 

    - **toggle_IL** : *boolean*  

        If True, every third layer added is an input layer and additional input layers are added to ensure the number of input layers is at least equal to
        `m`, with input layers alternating between control states 0 and 1. The input layer `shift` parameter is successively increased for each new input layer, 
        resulting in each input qubit controlling an operation on each target qubit at some point in the network. 
        
        If False, only convolutional layers are added. In either case, convolutional layers alternate
        between all-to-all and neighbour-to-neighbour layers. Default is True.    
    
    - **initial_IL** : *boolean*

        If True, add an input layer at the beginning of the circuit. Default is True. 

    - **input_Ry** : *boolean*

        If True, initially apply a sequence of parametrised Ry rotations to the input register. Default is False. 

    - **real** : *boolean*

        If True, use the real versions of input and convolutional layers, which only contain CX and Ry gates and hence ensure 
        real amplitudes. Default is False. 

    - **inverse** : *boolean* 

        If True, invert the circuit and return the inverse of the network. 

    - **repeat_params** : *str*, *optional*

        Keep parameters fixed for different layer types, i.e. use the same parameter values for each instance of a layer type. 
        Options are `CL` (keep parameters fixed for convolutional layers), `IL` (keep parameters fixed for input layers), `both` 
        (keep parameters fixed for both convolutional and input layers). 

    - **wrap**: *boolean* 

        If True, map rotation angles to an interval specified by `map_angles()`. Default is False.     

        
    Returns:
    ----
    - **circuit** : *QuantumCircuit* 

        The qiskit `QuantumCircuit` implementation of the network, with unassigned parameters.  
 
    """
    # initialise parameter vector 
    if repeat_params==None:
        AA_CL_params=None 
        NN_CL_params=None 
        IL_params=None 
    elif repeat_params=="CL":
        IL_params=None  
        AA_CL_params=ParameterVector(u"\u03B8_CL_AA", length= int((3 if real==False else 2 ) * (0.5 * m * (m-1))))
        NN_CL_params=ParameterVector(u"\u03B8_CL_NN", length= int((3 if real==False else 2) * m))     
    elif repeat_params=="IL":
        IL_params=ParameterVector(u"\u03B8_IL", length= int((3 if real==False else 1) * n))
        AA_CL_params=None 
        NN_CL_params=None 
    elif repeat_params=="both":
        AA_CL_params=ParameterVector(u"\u03B8_CL_AA", length= int((3 if real==False else 2 ) * (0.5 * m * (m-1))))
        NN_CL_params=ParameterVector(u"\u03B8_CL_NN", length= int((3 if real==False else 2) * m))
        IL_params=ParameterVector(u"\u03B8_IL", length= int((3 if real==False else 1) * n))    
        
    # initialise empty input and target registers 
    input_register = QuantumRegister(n, "input")
    target_register = QuantumRegister(m, "target")
    circuit = QuantumCircuit(input_register, target_register) 

    # prepare registers 
    circuit.h(target_register)
    if encode:
        circuit.compose(digital_encoding(n), input_register, inplace=True)

    if input_Ry:
        input_Ry_params = ParameterVector("\u03B8_input_Ry",n)
        for i in np.arange(n):
            circuit.ry(input_Ry_params[i], input_register[i])
         
    if initial_IL: 
        # apply input layer 
        circuit.compose(input_layer(n,m, u"\u03B8_IL_0", real=real, params=IL_params, wrap=wrap), circuit.qubits, inplace=True)
        
    # apply convolutional layers (alternating between AA and NN)
    # if toggle_IL is True, additional input layers are added after each NN
    for i in np.arange(L):

        if toggle_IL==False:

            if i % 2 ==0:
                circuit.compose(conv_layer_AA(m, u"\u03B8_CL_AA_{0}".format(i // 2), real=real, params=AA_CL_params,wrap=wrap), target_register, inplace=True)
            elif i % 2 ==1:
                circuit.compose(conv_layer_NN(m, u"\u03B8_CL_NN_{0}".format(i // 2),real=real, params=NN_CL_params, wrap=wrap), target_register, inplace=True)
        
        if toggle_IL==True:

            # difference between number of input layers and number of target qubits:
            del_IL = m - (int(initial_IL)+ L //3) 

            if del_IL <= 0:
                if i % 3 ==0:
                    circuit.compose(conv_layer_AA(m, u"\u03B8_CL_AA_{0}".format(i // 3),real=real,params=AA_CL_params, wrap=wrap), target_register, inplace=True)
                elif i % 3 ==1:
                    circuit.compose(conv_layer_NN(m, u"\u03B8_CL_NN_{0}".format(i // 3),real=real,params=NN_CL_params, wrap=wrap), target_register, inplace=True)
                elif i % 3 ==2:
                    circuit.compose(input_layer(n,m, u"\u03B8_IL_{0}".format(i // 3 +1),shift=(i//3)+1, ctrl_state=(i % 2),real=real,params=IL_params, wrap=wrap), circuit.qubits, inplace=True) 
            else: 
                if i % 3 ==0:
                    circuit.compose(conv_layer_AA(m, u"\u03B8_CL_AA_{0}".format(i // 3),real=real,params=AA_CL_params, wrap=wrap), target_register, inplace=True)
                elif i % 3 ==1:
                    circuit.compose(conv_layer_NN(m, u"\u03B8_CL_NN_{0}".format(i // 3),real=real,params=NN_CL_params, wrap=wrap), target_register, inplace=True)
                elif i % 3 ==2:
                    # padd with additional input layers
                    for j in np.arange(del_IL // (L //3) +2):
                        shift = (i // 3) +int(initial_IL) + j * (L //3)
                        if shift <= m-1:
                            circuit.compose(input_layer(n,m, u"\u03B8_IL_{0}".format(shift),shift=shift, ctrl_state=((shift-int(initial_IL)) % 2),real=real,params=IL_params, wrap=wrap), circuit.qubits, inplace=True) 

        if inverse:
            circuit=circuit.inverse()   

    return circuit

def A_generate_network(n,L, repeat_params=False, wrap=False):
    r"""
    Set up a network consisting of real convolutional layers acting on a single qubit register. 

    Layers alternate between all-to-all and neighbour-to-neighbour convolutional layers. For more 
    information see `conv_layer_AA()`, `conv_layer_NN()`. The convolutional layers are *real* in 
    the sense of involving only CX and Ry operations, ensuring real amplitudes. 

    The network was designed for the task of encoding a suitably normalised real function, $A(j)$, into the amplitudes of the 
    register, i.e. to implement an operator $\hat{U}_A$ such that  
    $$ \hat{U}_A \ket{j}= A(j) \ket{j}.$$
    With some adaptation, the network structure could also be used for other applications. 

    Arguments:
    ---
    - **n** : *int* 

        Number of qubits in the register. 

    - **L** : *int* 

        Number of layers in the network. 

    - **repeat_params** : *boolean* 

        If True, use the same parameter values for each layer type. Default is False. 

    - **wrap**: *boolean* 

        If True, map rotation angles to an interval specified by `map_angles()`. Default is False.      

    Returns:
    ---
    - **circuit** : *QuantumCircuit* 

        The qiskit `QuantumCircuit` implementation of the network, with unassigned parameters.  
                    
    """

    # initialise parameter vector 
    if repeat_params:
        AA_CL_params=ParameterVector(u"\u03B8_AA", length= int(2 * (0.5 * n * (n-1))))
        NN_CL_params=ParameterVector(u"\u03B8_NN", length= int(2 * n))
    else:
        AA_CL_params = None 
        NN_CL_params = None      


    # initialise empty input register 
    register = QuantumRegister(n, "reg")
    circuit = QuantumCircuit(register) 

    # prepare register
    circuit.h(register)
    
    # apply L convolutional layers (alternating between AA and NN)
    for i in np.arange(L):

        if i % 2 ==0:
            circuit.compose(conv_layer_AA(n, u"\u03B8_R_AA_{0}".format(i // 2), real=True, params=AA_CL_params,wrap=wrap), register, inplace=True)
        elif i % 2 ==1:
            circuit.compose(conv_layer_NN(n, u"\u03B8_R_NN_{0}".format(i // 2), real=True, params=NN_CL_params,wrap=wrap), register, inplace=True)
         
    return circuit 

def get_state_vec(circuit):
    """
    Get statevector produced by a quantum circuit. 

    Uses the `qiskit.Aer` backend. 

    Arguments:
    ----
    - **circuit** : *QuantumCircuit* 

        The circuit to be evaluated. 

    Returns:
    ----
    - **state_vector** : *array_like* 

        Array storing the complex amplitudes of the system to be in each of the 
        `2**circuit.num_qubits` basis states.     

    """
    
    # Transpile for simulator
    simulator = StatevectorSimulator()
    circuit = transpile(circuit, simulator)

    # Run and get counts
    result = simulator.run(circuit).result()
    state_vector= result.get_statevector(circuit)

    return np.asarray(state_vector)

def map_angle(theta):
    r"""
    Map a rotation angle to the interval $(0, \frac{\pi}{2})$.

    An inverse tan function is used for the mapping. 

    Arguments:
    ----
    - **theta** : *float* 

        Rotation angle. 

    Returns:
    ---
    - **theta_mapped** : *float* 

        Rotation angle mapped to the given interval.     

    """

    return np.arctan(theta) + np.pi /2  