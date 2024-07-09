import numpy as np 

"""
FUNCTIONS ASSOCIATED WITH CLASSICAL BINARY OPERATIONS 
"""

def get_nint(digits):
    if np.array(digits).ndim==0:
        digits=np.array([digits])
    digits = np.where(np.abs(digits)>1.,np.modf(digits)[1],digits)
    digits = digits[digits!=0.]
    if len(digits)==0:
        return 0
    nint = int(np.ceil(np.log2(np.max(np.abs(digits))))) + 1
    #if np.max(np.abs(digits))<np.sum(np.power(2., np.arange(nint-1))):
    #    nint-=1
    return nint

def get_npres(digits):
    if np.array(digits).ndim==0:
        digits=np.array([digits])
    digdecs = np.modf(digits)[0]
    digdecs = digdecs[digdecs!=0]
    if len(digdecs)==0:
        return 0
    mindec = np.min(np.abs(digdecs))
    switch = True
    p = 0
    while switch:
        if mindec%(2.**-p)==0.:
            switch=False
        p+=1
    return p-1#-int(np.floor(np.log2(np.min(np.abs(digdecs)))))


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


def get_n(digits):


    return nint +npres +1 

def dec_to_bin(digits,n,encoding,nint=None, round=True, overflow_error=True):

    """
    Encode a float `digits` in base-10 to a binary string `bits`. 

    Little endian convention is used.
    
    ....
    """

    bits=''

    if nint==None:
        nint=n 

    if nint==n and (encoding=='signed mag' or encoding=='twos comp'):
        nint-= 1 

    if encoding=='signed mag' or encoding=='twos comp':
        bits += ('1' if digits<0 else '0')  
        p = n - nint - 1
    elif encoding=='unsigned mag':
        p = n - nint 
    else: 
        raise ValueError("Unrecognised type of binary encoding. Should be 'unsigned mag', 'signed mag', or 'twos comp'.")

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

    if      

    return bits

def bin_to_dec(bits,encoding,nint=None):
    """
    Decode a binary string `bits` to a float `digits` in base-10. 
    
    Binary encoding must be specified as 
    `'unsigned mag'`, `'signed mag'`, or `'twos comp'` for unsigned magnitude, signed magnitude, and twos
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

def twos_complement(binary, nint=None):
    """
    For a bit string `binary` calculate the two's complement binary string `compl`. 
    
    Little endian convention is used. 

    An all-zero bit string is its own complement.
    """
   
    binary_to_array = list(int(binary)) 
   
    if np.sum(binary_to_array)==0:
        return binary 
   
    inverted_bits = ''.join((np.logical_not(binary_to_array).astype(int)).astype(str))

    compl = dec_to_bin(bin_to_dec(inverted_bits, encoding='unsigned mag')+ 1,len(binary),encoding='unsigned mag', round=False) 
    
    return compl 

### 

binary = '10000'
nint=None 

new =dec_to_bin(1002,5,'twos comp', nint=2)

print(new)