"""
Collection of functions relating to binary encoding and decoding.
"""

import numpy as np 

def dec_to_bin(digits,n,encoding='unsigned mag',nint=None, overflow_error: bool=True):
    """
    Encode a base-10 float to a binary string. 
    
    The fractional part is rounded to the available precision. Little endian convention 
    is used. 

    Arguments:
    ----------
    - **digits** : *float*

        The float to encode.

    - **n** : *int*  

        The number of bits in the binary string 

    - **encoding** : *str*

        The type of binary encoding used. Possible options are `'unsigned mag'` for unsigned magnitude 
        encoding, `'signed mag'` for signed magnitude encoding, and `'twos comp'` for two's complement 
        representation. Default is `'unsigned mag'`.  

    - **nint** : *int*, *optional* 

        Number of integer bits used in the encoding. If `nint` is not given all bits are taken to be
        integer bits. 

    - **overflow_error** : *bool*

        Raises a `ValueError` if `digits` lies outside the available range for the given encoding. Default 
        is True. 

    Returns:
    --------  
    - **bits** : *str* 

        The binary string representing `digits`.  
    
    """

    # set number of integer bits
    if nint==None:
        nint=n 

    # one bit is reserved to store the sign for signed encodings 
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

    # reverse array (little endian convention)
    bits_int= bits_int[::-1]    

    # assemble bit string
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

def bin_to_dec(bits,encoding='unsigned int',nint=None):
    """
    Decode a binary string to a float in base-10.
    
    Little endian convention is used. 

    Arguments:
    ----------
    - **bits** : *str*

        The binary string to decode.

    - **encoding** : *str*

        The type of binary encoding used. Possible options are `'unsigned mag'` for unsigned magnitude 
        encoding, `'signed mag'` for signed magnitude encoding, and `'twos comp'` for two's complement 
        representation. Default is `'unsigned mag'`.  

    - **nint** : *int*, *optional* 

        Number of integer bits used in the encoding. If `nint` is not given all bits are taken to be
        integer bits. 

    Returns:
    --------  
    - **digits** : *float* 

        The base-10 float represented by `bits`.  
    
    """
   
    # get integer bits 
    n = len(bits)
    if nint==None:
        nint=n

    # reverse bit string (little endian convention)
    bits=bits[::-1]

    # cast as array of integers       
    bit_arr = np.array(list(bits)).astype('int')

    # decode binary string 
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
    Calculate the [two's complement](https://en.wikipedia.org/wiki/Two%27s_complement) of a bit string. 
    
    Little endian convention is used. An all-zero bit string is its own complement.

    Arguments:
    ----------

    - **binary** : *str* 

        The binary string. 

    Returns:
    --------

    - **compl** : *str* 

        The two's complement of `binary`.    

    """   
    # cast as list of integers 
    binary_to_array = np.array(list(binary)).astype(int)
   
    # check if bit string is all zeros and return itself if True 
    if np.sum(binary_to_array)==0:
        return binary 
   
    # calculate two's complement
    inverted_bits = ''.join((np.logical_not(binary_to_array).astype(int)).astype(str))
    compl = dec_to_bin(bin_to_dec(inverted_bits, encoding='unsigned mag')+ 1,len(binary),encoding='unsigned mag') 
    
    return compl