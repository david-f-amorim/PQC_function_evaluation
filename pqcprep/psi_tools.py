r"""
Collection of functions relating to the function $\Psi$ to be evaluated by the QCNN.
"""
import numpy as np 
from .binary_tools import bin_to_dec, dec_to_bin

def x_trans(x):
    """
    Transform a scalar to match the frequency scaling used for the gravitational wave binary system inspiral
    in [Hayes 2023](https://arxiv.org/pdf/2306.11073). 

    Arguments:
    ----
    - **x** : *float*

        Scalar to be transformed. 

    Returns:
    ----
    - **x_t** : *float*

        Transformed scalar.     

    """
    n = 6 
    nint = n 
    fmin=40.
    fmax=168. 
    df = (fmax-fmin)/(2**n)

    xmax = np.power(2,nint) - np.power(2,nint-n)
    x = x/xmax
    x = x*(fmax-fmin-df)
    x = x + fmin
    x_t =x
    return x_t 

def psi_H(x):
    r"""
    The phase function $\Psi$ for the gravitational wave binary system inspiral
    in [Hayes 2023](https://arxiv.org/pdf/2306.11073). 

    Arguments:
    ----
    - **x** : *float*

        $x$. 

    Returns:
    ----
    - **out** : *float*

        $\Psi(x)$.     

    """

    # gravitational wave features 
    n = 6 
    fmin=40.
    fmax=168. 
    m1=(4.926e-6)*35
    m2=(4.926e-6)*30. 
    beta=0.
    sig=0.
    Tfrac = 100.
    df = (fmax-fmin)/(2**n)
    T = 1./df
    tc = T + (T/Tfrac)
    DT = tc%T
    Mt = m1 + m2
    nu = (m1*m2)/Mt
    eta = nu/Mt
    Mc = Mt*eta**(3./5)

    # transform x to frequency scale 
    x = x_trans(x)

    # apply function
    out = (((3./128))*((np.pi*Mc*x)**(-5./3))*( 1.+ (20./9)*((743./336)+(11./4)*eta)*(np.pi*Mt*x)**(2./3) -4.*(4.*np.pi - beta)*(np.pi*Mt*x) + 10.*((3058673./1016064) + (eta*5429./1008) + (617*(eta**2)/144) - sig)*(np.pi*Mt*x)**(4./3)) + 2.*np.pi*x*DT)/(2.*np.pi) 

    return out 

def psi(x, mode="psi"):
    r"""
    A wrapper for different phase functions. 

    Inputs are scaled via `x_trans()` in all cases. 

    Arguments:
    ---
    - **x** : *float* 

        $x$.  

    - **mode** : *str*

        Options are `'psi'` (corresponding to `psi_H()`), `'linear'` (corresponding to `psi_linear()`), `'quadratic'` (corresponding to `psi_quadratic()`),
        `'sin'` (corresponding to `psi_sine()`). Default is `'psi'`. 

    Returns:
    ----
    - **out** : *float* 

        $\Psi(x)$ for the $\Psi$ selected via `mode`. 
        

    """
    if mode=="sin":
        out = psi_sine(x)
    elif mode=="quadratic":    
        out = psi_quadratic(x)
    elif mode=="linear":    
        out = psi_linear(x)
    elif mode=="psi":     
        out = psi_H(x)
    
    return out

def psi_linear(x):
    r"""
    A simple linear phase function $\Psi$. 

    Arguments:
    ----
    - **x** : *float*

        $x$. 

    Returns:
    ----
    - **out** : *float*

        $\Psi(x)$.     

    """    
    x = x_trans(x)
    out= 0.5 + 0.01*x

    return out 

def psi_quadratic(x):
    r"""
    A simple quadratic phase function $\Psi$. 

    Arguments:
    ----
    - **x** : *float*

        $x$. 

    Returns:
    ----
    - **out** : *float*

        $\Psi(x)$.     

    """    
    x = x_trans(x)
    out= 0.0001* x**2

    return out 

def psi_sine(x):
    r"""
    A simple sinusoidal phase function $\Psi$. 

    Arguments:
    ----
    - **x** : *float*

        $x$. 

    Returns:
    ----
    - **out** : *float*

        $\Psi(x)$.     

    """     
    x = x_trans(x)
    out =  np.pi /2 *(1+ np.sin(x /4 )) 

    return out 

def get_phase_target(m, func):
    """
    
    Generate an array of phase function values taking into account rounding due to the limited 
    size of the target register. 

    The number of values in the array is determined for the same frequency scaling as in `x_trans()`. 

    Arguments:
    ---
    - **m** : *int* 

        Number of qubits in the target register. 

    - **func** : *callable* 

        Phase function.     

    Returns:
    ---
    - **phase_rounded** : *array_like*

        Array containing phase function values rounded to the precision 
        afforded by the target register size. 


    """

    # define x array 
    n = 6
    x_min = 40
    x_max = 168 
    dx = (x_max-x_min)/(2**n) 
    x_arr = np.arange(x_min, x_max, dx) 

    # calculate target output for phase 
    phase_target = func(np.linspace(0, 2**n, len(x_arr)))

    # calculate target for phase taking into account rounding 
    phase_reduced = np.modf(phase_target / (2* np.pi))[0] 
    phase_reduced_bin = [dec_to_bin(i,m, "unsigned mag", 0) for i in phase_reduced]
    phase_reduced_dec =  np.array([bin_to_dec(i,"unsigned mag", 0) for i in phase_reduced_bin])
    phase_rounded = 2 * np.pi * phase_reduced_dec

    return phase_rounded
