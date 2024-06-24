import numpy as np 
from qiskit import QuantumRegister, QuantumCircuit, execute, Aer 

""" 
taken from qiskit_tools.py (Fergus Hayes)
"""
def input_bits_to_qubits(binary, circ, reg, wrap=False, inverse=False, phase=False, qphase=None, label='Input'):
    # Flips the qubits to match a classical bit string
    
    n = len(binary)
    
    if inverse:
        wrap = True

    if wrap:
        regs = []
        reg = QuantumRegister(n, 'reg')
        regs.append(reg)
        if qphase!=None:
            qphase = QuantumRegister(1, 'phase')
            regs.append(qphase)
        circ = QuantumCircuit(*regs)

    if phase and qphase==None:
        qphase = QuantumRegister(1, 'phase')
        circ.add(qphase)

    for bit in np.arange(n):
        if int(binary[::-1][bit])==1:
            circ.x(reg[bit])

    if phase<0.:
        #circ.p(np.angle(phase), phase_reg[0])
        circ.x(qphase[0])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'†'
    
    return circ

""" 
taken from qiskit_tools.py (Fergus Hayes)
"""
def my_binary_repr(digit, n, nint=None, phase=False, nround=True, overflow_error=True):
    """
    Convert a floating point digit to binary string
    digit - input number (float)
    n - total number of bits (int)
    nint - number of integer bits. Default to lowest required (int)
    """

    if nint is None:# or nint==n:
        if phase:
            nint = n - 1
        else:
            nint = n

    if phase:
        p = n - nint - 1
        dmax = 2.**(nint) - 2.**(-p)
        dmin = -2.**(nint)
    else:
        p = n - nint
        dmax = 2.**(nint) - 2.**(-p)
        dmin = 0.

    if overflow_error:
        if digit>dmax or digit<dmin:
            raise ValueError('Digit '+str(digit)+' does not lie in the range:',dmin,'-',dmax,n,nint,p)

    if nround:
        n += 1
        p += 1

    value = digit
    bin_out = ''
    if phase:
        if value<0.:
            value+=2.**nint
            bin_out+='1'
        else:
            bin_out+='0'
    
    for i,bit in enumerate(np.arange(-p,nint)[::-1]):
        bin_out+=str(int(np.floor(value/2.**bit)))
        if value>=2.**bit:
            value-=2.**bit

    if nround:
        carry = True
        bin_out = np.array(list(bin_out))
        for i in np.arange(n)[::-1]:
            if not carry:
                break
            if bin_out[i]=='1':
                bin_out[i]='0'
            elif bin_out[i]=='0':
                bin_out[i]='1'
                carry = False
        bin_out = ("").join(list(bin_out[:-1]))

    return bin_out

"""
take a float and encode it onto an n-qubit register;
return circuit 
"""

def float_to_input_qubits(digit, n, nint = None): 
    # digit: float to be digitally encoded onto qubit register 
    # n: number of qubits in register 

    #get binary string:
    binary = my_binary_repr(digit, n, nint=nint) 

    # get circuit 
    reg = QuantumRegister(n)
    circ = QuantumCircuit(reg)
    qc = input_bits_to_qubits(binary, circ=circ, reg=reg)

    # return circuit 
    return qc 

"""
take a float and get corresponding n-qubit state vector 
"""

def float_to_state_vector(digit,n, nint=None):

    # get binary string:
    binary = my_binary_repr(digit, n, nint=nint)

    # convert to integer: (from qiskit_tools.py, Fergus Hayes)
    digit = 0.
    for i in np.arange(n):
        digit += 2.**(n-1-i) * float(binary[i])
        
    # produce state vector array 
    state_vector = np.zeros(2**n)
    state_vector[int(digit)] = 1

    return state_vector  

""" 
take a float, x, and define an analytical function, f(x), to produce 
the state vector corresponding to |f(x)>
"""  

def x_to_fx_statevec(x,n, nint=None):

    def f(x):

        a = 0.33 
        b = 0.17

        return a*x +b
    
    return float_to_state_vector(f(x),n,nint)



################################

digit = 15.1267 #np.random.rand()*15.75
n = 6 

qc = float_to_input_qubits(digit, n, nint = n-2)
print(qc)
print(digit)
print(my_binary_repr(digit,n, nint = n-2))

backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
result = job.result()
state_vector = result.get_statevector()
state_vector = np.asarray(state_vector).reshape(2**n)

print(state_vector)
print(float_to_state_vector(digit,n,nint=n-2))