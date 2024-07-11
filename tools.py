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
import torch 

def dec_to_bin(digits,n,encoding,nint=None, overflow_error=True):
    """
    Encode a float `digits` in base-10 to a binary string `bits`. 
     
    Binary encoding must be specified as  `'unsigned mag'`, `'signed mag'`, 
    or `'twos comp'` for unsigned magnitude, signed magnitude, and two's
    complement representations, respectively. The fractional part is rounded to 
    the available precision: unless otherwise specified via `nint`, all bits (apart 
    from the sign bit) are assumed to be integer bits. Little endian convention is used.
    
    """

    # set number of integer bits
    if nint==None:
        nint=n 

    # at least one bit is reserved to store the sign 
    if nint==n and (encoding=='signed mag' or encoding=='twos comp'):
        nint-= 1 

    # determine number of precision bits 
    if encoding=='signed mag' or encoding=='twos comp': 
        p = n - nint - 1
    elif encoding=='unsigned mag':
        p = n - nint 
    else: 
        raise ValueError("Unrecognised type of binary encoding. Should be 'unsigned mag', 'signed mag', or 'twos comp'.")

    # raise overflow error if float is out of range for given encoding
    if overflow_error:

        if encoding=='unsigned mag':
            min_val = 0
            max_val = (2.**nint) - (2.**(-p)) 
        elif encoding=='signed mag':
            min_val = - (2.**nint - 2.**(-p))
            max_val = + (2.**nint - 2.**(-p))     
        elif encoding=='twos comp':
            min_val = - (2.**nint)
            max_val = + (2.**nint - 2.**(-p)) 

        if (digits>max_val and nint != 0) or digits<min_val:
            raise ValueError(f"Float {digits} out of available range [{min_val},{max_val}].")   

    
    # take absolute value and separate integer and fractional part:
    digits_int = np.modf(np.abs(digits))[1]
    digits_frac = np.modf(np.abs(digits))[0] 

    # add fractional parts
    bits_frac=''
    for i in range(p):
        bits_frac +=str(int(np.modf(digits_frac * 2)[1])) 
        digits_frac =np.modf(digits_frac * 2)[0]

    # add integer parts
    bits_int=''
    for i in range(nint):
        bits_int +=str(int(digits_int % 2))
        digits_int = digits_int // 2 

    bits_int= bits_int[::-1]    

    if encoding=="unsigned mag":
        bits = bits_int + bits_frac 
    if encoding=="signed mag":
        if digits >= 0:
            bits = '0' +  bits_int + bits_frac
        else:
            bits = '1' +  bits_int + bits_frac  
    if encoding=="twos comp":
        if digits >=0:
            bits = '0' +  bits_int + bits_frac
        elif digits== min_val:
            bits = '1' +  bits_int + bits_frac   
        else:
            bits = twos_complement('0' +  bits_int + bits_frac)                         

    return bits

def bin_to_dec(bits,encoding,nint=None):
    """
    Decode a binary string `bits` to a float `digits` in base-10. 
    
    Binary encoding must be specified as 
    `'unsigned mag'`, `'signed mag'`, or `'twos comp'` for unsigned magnitude, signed magnitude, and two's
    complement representations, respectively. Unless the number of integer bits is specified using `nint`,
    all bits (apart from the sign bit) are assumed to be integer bits. Little endian convention is used.
    """

    n = len(bits)
    if nint==None:
        nint=n

    bits=bits[::-1]
          
    bit_arr = np.array(list(bits)).astype('int')

    if encoding=="unsigned mag":
        p = n - nint 
        digits = np.sum([bit_arr[i] * (2.**(i-p)) for i in range(n)])
    elif encoding=="signed mag":
        if nint==n: 
            nint -= 1
        p = n - nint -1
        digits = ((-1.)**bit_arr[-1] ) * np.sum([bit_arr[i] * (2.**(i-p)) for i in range(n-1)])
    elif encoding=="twos comp":
        if nint==n: 
            nint -= 1
        p = n - nint -1
        digits = (-1.)*bit_arr[-1]*2**nint + np.sum([bit_arr[i] * (2.**(i-p)) for i in range(n-1)])
    else: 
        raise ValueError("Unrecognised type of binary encoding. Should be 'unsigned mag', 'signed mag', or 'twos comp'.") 

    return digits 

def twos_complement(binary):
    """
    For a bit string `binary` calculate the two's complement binary string `compl`. 
    
    Little endian convention is used. An all-zero bit string is its own complement.
    """   
    binary_to_array = np.array(list(binary)).astype(int)
   
    if np.sum(binary_to_array)==0:
        return binary 
   
    inverted_bits = ''.join((np.logical_not(binary_to_array).astype(int)).astype(str))

    compl = dec_to_bin(bin_to_dec(inverted_bits, encoding='unsigned mag')+ 1,len(binary),encoding='unsigned mag', round=False) 
    
    return compl

def input_layer(n, m, par_label, ctrl_state=0): 
    """
    Construct an input layer consisting of controlled single-qubit rotations
    with the n input qubits acting as controls and the m target qubits
    acting as targets. 
    """

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
        if np.modf(j/m)[1] >= 1:
            j -=int(np.modf(j/m)[1] * m)

        par = params[int(param_index) : int(param_index + num_par)]    

        cu3 = U3Gate(par[0],par[1],par[2]).control(1, ctrl_state=ctrl_state)
        qc.append(cu3, [qubits[i], qubits[j+n]])

        param_index += num_par
        

    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(n+m)
    circuit.append(qc_inst, qubits)
    
    return circuit 

def N_gate(params):
    """
    Construct the two-qubit N gate (as defined in Vatan 2004)
    in terms of three parameters, stored in list or tuple 'params'
    """

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

def conv_layer_NN(m, par_label):
    """
    Construct a linear (neighbour-to-neighbour) convolutional layer
    via the cascaded application of the N gate to the m-qubit target 
    register.   
    """

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

def conv_layer_AA(m, par_label): 
    """
    Construct a quadratic (all-to-all) convolutional layer
    via the cascaded application of the N gate to the m-qubit target 
    register.   
    """
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

def digital_encoding(n):   
    """
    Digitally encode an n-bit binary number onto an n-qubit register. 
    The encoding is set by assigning the value 0 to the i-th component of the
    parameter vector to represent a bit value of "0" for the i-th bit and assigning
    pi for the case of "1". 

    """

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

def binary_to_encode_param(binary):        
    """
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

def generate_network(n,m,L, encode=False, toggle_IL=True, initial_IL=True, input_H=False):
    """
    Set up a network consisting of input and convolutional layers acting on n input 
    qubits and m target qubits.  
    Both the input state and the circuit weights can be set by accessing circuit parameters
    after initialisation.  
    """

    # initialise empty input and target registers 
    input_register = QuantumRegister(n, "input")
    target_register = QuantumRegister(m, "target")
    circuit = QuantumCircuit(input_register, target_register) 

    # prepare registers 
    circuit.h(target_register)
    if encode:
        circuit.compose(digital_encoding(n), input_register, inplace=True)

    if initial_IL: 
        # apply input layer 
        circuit.compose(input_layer(n,m, u"\u03B8_IN"), circuit.qubits, inplace=True)
        circuit.barrier()

    if input_H:
        circuit.h(input_register)    

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

def train_QNN(n,m,L, seed, shots, lr, b1, b2, epochs, func,func_str,loss_str,meta, recover_temp, nint, mint, phase_reduce, train_superpos):
    """
    Initialise circuit as QNN for training purposes.
    """
    
    # set precision strings 
    if nint==None:
        nint=n 
    if mint==None:
        mint=m  
    if nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if train_superpos:
        phase_reduce=True
        meta+='(S)'    
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)'

    # set seed for PRNG 
    algorithm_globals.random_seed= seed
    rng = np.random.default_rng(seed=seed)

    # generate circuit and set up as QNN 
    if train_superpos: 
        qc = generate_network(n,m,L, encode=False, toggle_IL=True, initial_IL=True, input_H=True)
        qnn = SamplerQNN(
                circuit=qc.decompose(),            # decompose to avoid data copying (?)
                sampler=Sampler(options={"shots": shots, "seed": algorithm_globals.random_seed}),
                input_params=[],    # no input params
                weight_params=qc.parameters,   # weights
                input_gradients=False # ?? 
            ) 
    else:     
        qc = generate_network(n,m,L, encode=True, toggle_IL=True, initial_IL=True)
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
            if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_weights_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
                recovered_weights=os.path.join("outputs", f"__TEMP{k}_weights_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")
                recovered_k=k+1
            if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
                recovered_mismatch=os.path.join("outputs", f"__TEMP{k}_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")
            if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
                recovered_loss=os.path.join("outputs", f"__TEMP{k}_loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")        
        
        if recovered_weights != None and recovered_mismatch != None and recovered_loss != None:
            initial_weights=np.load(recovered_weights)
        else:
            if train_superpos: 
                 initial_weights = algorithm_globals.random.random(len(qc.parameters))
            else:    
                initial_weights = algorithm_globals.random.random(len(qc.parameters[n:]))    
    
    else:
        if train_superpos: 
            initial_weights = algorithm_globals.random.random(len(qc.parameters))
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
    elif loss_str=="MM":

        def criterion(output, target):

            val = 1. - torch.abs(torch.sum(torch.mul(output, target)))

            return val 
      
                    
    # set up arrays to store training outputs 
    if recover_temp and recovered_weights != None and recovered_mismatch != None and recovered_loss != None:
        mismatch_vals=np.load(recovered_mismatch)
        loss_vals=np.load(recovered_loss)
    else:    
        mismatch_vals = np.empty(epochs)
        loss_vals = np.empty(epochs)

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
        fx_arr = [np.modf(i/ (2* np.pi))[0] for i in fx_arr]

        # convert fx_arr to binary at available target register precision 
        fx_arr_bin = [dec_to_bin(i,m,nint=0,encoding="unsigned mag") for i in fx_arr]
        
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
       
    # start training 
    print(f"\n\nTraining started. Epochs: {epochs}. Input qubits: {n}. Target qubits: {m}. QCNN layers: {L}. \n")
    start = time.time() 

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

        if loss_str=="MM":
            if i==recovered_k:
                angle_tensor = Tensor(np.zeros(2**(n+m)))
            else: 
                angle_tensor = Tensor(angle_arr)   

            loss = criterion(torch.polar(torch.sqrt(model(input)+1e-10),angle_tensor), torch.sqrt(target))     # add small number 

        else:
            loss = criterion(model(input), target)

        loss.backward()
        optimizer.step()

        # save loss for plotting 
        loss_vals[i]=loss.item()

        # set up circuit with calculated weights
        if train_superpos:
            circ = generate_network(n,m,L, encode=False, toggle_IL=True, initial_IL=True, input_H=True)
            with no_grad():
                generated_weights = model.weight.detach().numpy()      
            circ = circ.assign_parameters(generated_weights)
        else:
            circ = generate_network(n,m,L, encode=True, toggle_IL=True, initial_IL=True)

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
            np.save(os.path.join("outputs", f"__TEMP{i}_weights_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),generated_weights)
            np.save(os.path.join("outputs", f"__TEMP{i}_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),mismatch_vals)
            np.save(os.path.join("outputs", f"__TEMP{i}_loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),loss_vals)
            
            # delete previous temp files
            prev_weights=f"__TEMP{i-100}_weights_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"
            prev_mismatch=f"__TEMP{i-100}_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"
            prev_loss=f"__TEMP{i-100}_loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"

            if os.path.isfile(os.path.join("outputs", prev_weights)):
                os.remove(os.path.join("outputs", prev_weights))

            if os.path.isfile(os.path.join("outputs", prev_mismatch)):
                os.remove(os.path.join("outputs", prev_mismatch))

            if os.path.isfile(os.path.join("outputs", prev_loss)):
                os.remove(os.path.join("outputs", prev_loss))        

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
    
    elapsed = time.time()-start
    mins, sec = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    time_str = f"{int(hours):02}:{int(mins):02}:{sec:05.2f}" 

    # decompose circuit for gate count 
    num_CX = dict(circ.decompose(reps=4).count_ops())["cx"]
    num_gates = num_CX + dict(circ.decompose(reps=4).count_ops())["u"]

    print(f"\nTraining completed in {time_str}. Number of weights: {len(generated_weights)}. Number of gates: {num_gates} (of which CX gates: {num_CX}). \n\n")

    # delete temp files 
    temp_weights=f"__TEMP{temp_ind}_weights_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"
    temp_mismatch=f"__TEMP{temp_ind}_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"
    temp_loss=f"__TEMP{temp_ind}_loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"

    if os.path.isfile(os.path.join("outputs", temp_weights)):
            os.remove(os.path.join("outputs", temp_weights))
    if os.path.isfile(os.path.join("outputs", temp_mismatch)):
            os.remove(os.path.join("outputs", temp_mismatch))
    if os.path.isfile(os.path.join("outputs", temp_loss)):
            os.remove(os.path.join("outputs", temp_loss))               

    # save outputs 
    with no_grad():
            generated_weights = model.weight.detach().numpy()

    np.save(os.path.join("outputs", f"weights_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),generated_weights)
    np.save(os.path.join("outputs", f"mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),mismatch_vals)
    np.save(os.path.join("outputs", f"loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}"),loss_vals)
    

    return 0 

def test_QNN(n,m,L,epochs, func, func_str,loss_str,meta,nint,mint,phase_reduce,train_superpos, verbose=True):   
    """
    Test performance of trained QNN for the various input states
    """
    # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"    
    if train_superpos:
        phase_reduce=True
        meta+='(S)'
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)'          

    # load weights 
    weights = np.load(os.path.join("outputs",f"weights_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))

    # initialise array to store results 
    mismatch = np.empty(2**n)

    # iterate over input states 
    x_arr = np.arange(2**n)
    fx_arr = [func(i) for i in x_arr]

    if phase_reduce: 
        fx_arr = [np.modf(i/ (2* np.pi))[0] for i in fx_arr]

    for i in x_arr:
        
        # prepare circuit 
        enc=binary_to_encode_param(np.binary_repr(i,n))
        params=np.concatenate((enc, weights))  

        circ = generate_network(n,m,L, encode=True,toggle_IL=True)
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

        # calculate fidelity and mismatch 
        fidelity = np.abs(np.dot(np.sqrt(target_arr),np.conjugate(state_vector)))**2
        mismatch[i] = 1. - np.sqrt(fidelity) 

        # save as dictionary 
        dic = dict(zip(x_arr, mismatch)) 
        np.save(os.path.join("outputs",f"bar_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"), dic)
    
    if verbose:
        print("Mismatch by input state:")
        for i in x_arr:
            print(f"\t{np.binary_repr(i,n)}:  {mismatch[i]:.2e}")
        print("")
        print("")    

    return 0 

def check_duplicates(n,m,L,epochs,func_str,loss_str,meta,nint,mint, phase_reduce,train_superpos):
    """
    For a given set of input parameters, check if training and testing results already exist. 
    """
    # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if train_superpos:
        phase_reduce=True
        meta+='(S)'
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)'    

    check_mismatch = os.path.isfile(os.path.join("outputs", f"mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))
    check_weights = os.path.isfile(os.path.join("outputs", f"loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))
    check_loss = os.path.isfile(os.path.join("outputs", f"weights_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy"))
    
    return check_mismatch & check_weights & check_loss

def check_temp(n,m,L,epochs,func_str,loss_str,meta,nint,mint, phase_reduce,train_superpos):   
    """
    For a given set of input parameters, check if temp files already exist. 
    """

    # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if train_superpos:
        phase_reduce=True
        meta+='(S)'    
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)'

    check_mismatch=False 
    check_weights=False 
    check_loss=False

    for k in np.arange(100,epochs, step=100):
        if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_weights_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
            check_weights=True 
        if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
            check_mismatch=True 
        if os.path.isfile(os.path.join("outputs", f"__TEMP{k}_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.npy")):
            check_loss=True         
    
    return check_mismatch & check_weights & check_loss

def check_plots(n,m,L,epochs,func_str, loss_str, meta, log, mint, nint, phase_reduce,train_superpos):    
    """
    For a given set of input parameters, check if plots already exist (excluding compare plots). 
    """
    # set precision strings 
    if nint==None or nint==n:
        nis=""
    else:
        nis=f"({nint})"
    if mint==None or mint==m:
        mis=""
    else:
        mis=f"({mint})"
    if train_superpos:
        phase_reduce=True
        meta+='(S)'    
    if phase_reduce:
        mint = 0
        mis=f"({mint})"
        meta+='(PR)'    

    log_str= ("" if log==False else "log_")

    check_mismatch =os.path.isfile(os.path.join("plots", f"{log_str}mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.png"))
    check_loss=os.path.isfile(os.path.join("plots", f"{log_str}loss_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.png"))
    check_bars=os.path.isfile(os.path.join("plots", f"{log_str}bar_mismatch_{n}{nis}_{m}{mis}_{L}_{epochs}_{func_str}_{loss_str}_{meta}.png"))

    return check_mismatch & check_loss & check_bars 

def generate_seed():   
    """
    Generate random seed from timestamp
    """

    return int(time.time())

def extract_phase(n):
    """
    For an `n`-qubit register storing computational basis state |k> representing a float between 0 and 1, transform to
        |k> -> e^(2 pi k) |k> 
    via single-qubit Ry rotations (based on scheme presented in Hayes 2023).

    This assumes an unsigned magnitude encoding with n precision bits. 
    """

    qc = QuantumCircuit(n, "Extract Phase")
    qubits = list(range(n))

    for k in np.arange(1,n+1):
        lam = 2.*np.pi*(2.**(-k))
        qubit = n-k
        qc.p(lam,qubits([qubit]))

    # package as instruction
    qc_inst = qc.to_instruction()
    circuit = QuantumCircuit(n)
    circuit.append(qc_inst, qubits)    

    return 0 
