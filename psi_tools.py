import numpy as np 

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

#####
# 
#  NEXT UP: 
#   - input_bits_to_qubits ; integer_compare ; QFTBinaryAdd ; TwosCompliment ; QFTMultiply ; QFTAddition_
#       which are required for label_gate ; first_gate ; QFTPosMultiplicand ; QFTAddition 
#       which are required for piecewise_function_posmulti 
 

b =dec_to_bin(-1.499999,4,nint=2, encoding="twos comp")

print(b)

