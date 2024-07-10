import numpy as np 
from qiskit.circuit.library import IntegerComparator, DraperQFTAdder
from qiskit import QuantumCircuit, QuantumRegister

"""
FUNCTIONS ASSOCIATED WITH CLASSICAL BINARY OPERATIONS 
"""

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

        if digits>max_val or digits<min_val:
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

def bin_to_qubits(binary,inverse=False):
    """
    Prepare a quantum register corresponding to a classical bit string `binary`.  

    If `inverse` is `True`, the inverse operation is applied.
    """
    
    n = len(binary)
    qreg = QuantumRegister(n)
    circ = QuantumCircuit(qreg)
    
    for i in np.arange(n):
        if int(binary[::-1][i])==1:
            circ.x(qreg[i])

    circ = circ.to_gate()
    circ.label ="bin_to_qubits"

    if inverse:
        circ = circ.inverse()

    return circ

""""
FUNCTIONS ASSOCIATED WITH LPF EVALUATION
"""

def label_gate(circ, q_x,q_label,q_anc,bounds,inverse=False):
    """
    Implementation of the LABEL gate, as described in Haener 2018. 

    Given a register `q_x` encoding a value in a domain partitioned
    into subdomains with boundaries specified in `bounds`, the LABEL
    gate determines into which subdomain `q_x` falls. The result is 
    stored in the label register `q_label`. An ancillary register 
    `q_anc` is required as part of the routine. The register are 
    taken to be part of the QuantumCircuit `circ`.  

    PRESENTLY REQUIRES POSITIVE INTEGER VALUES. GENERALISE THIS LATER!
    
    If `inverse` is `True`, the inverse operation is applied.
    """

    n=len(q_x) 
    nlab=len(q_label)

    if len(q_anc) < n+ nlab:
        raise ValueError("Ancilla register must be as large as the input register and the label register combined.")
    if len(q_label) != int(np.ceil(np.log2(len(bounds)))):
        raise ValueError("There must be one label for each subdomain.")
    if np.max(bounds)>=2**n:
        raise ValueError(f"Domain out of bounds for {n} qubits.")
    
    # encode the number 1 into the ancilla register 
    circ.x(q_anc[n])

    for i, bound in enumerate(bounds):

        # set up circuit to compare q_x to the bound value 
        comp_gate = IntegerComparator(n, bound, geq=True, name=f'P{i}').to_gate()
        circ.append(comp_gate, [*q_x, *q_anc[:n]])

        # increment the label gate if ancilla qubit has been flipped 
        incr_gate = DraperQFTAdder(nlab, name=f'SET{i}').to_gate().control(1)
        circ.append(incr_gate, [q_anc[0],*q_anc[n:], *q_label])

        # undo comparison 
        comp_gate_inv = IntegerComparator(n, bound, geq=True, name=f'P{i}').inverse().to_gate()
        comp_gate_inv.name = f'P{i}†'
        circ.append(comp_gate_inv, [*q_x, *q_anc[:n]])

    circ.name= "LABEL"
    circ = circ.to_gate()
    
    if inverse:
        circ = circ.inverse()

    return circ 

#####
# 
#  NEXT UP: 
#   - integer_compare ; QFTBinaryAdd ; TwosCompliment ; QFTMultiply ; QFTAddition_ ;  QFTPosMultiplicand ; QFTAddition ;

#   USE DraperQFTAdder, IntegerComparator, and RGQFTMultiplier ???!!!
#       
#       which are required for label_gate ; first_gate ; increment_gate
#       which are required for piecewise_function_posmulti 
 





