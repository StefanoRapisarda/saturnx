import numpy as np
import pandas as pd

def sbend_pl(f,a1,a2,b=1,f0=1,norm=None,dc=10,frac_rms=0.3,df=1):
    '''
    It returns a (smoothly) bending power law power spectral shape
    given a certain total fractional RMS amplitude and a normalizaion
    option.

    PARAMETERS
    ----------
    f: sequence or float
        Frequency array 
    a1: float
        Power law index at low frequencies
    a2: float
        Power law index at high frequencies
    b: float
        Bending parameter (the highest the sharper the break)
    f0: float
        Bending/breaking frequency
    norm: str or None
        Normalization option. According to the normalization option,
        the total fractional RMS is computed with a different expression.
        Each power is then divided by this value and multiplied by
        the fractional RMS value specified by the user.
        If None (defualt), no normalization is applied.
        Other options are:
        - POWER
        - LEAHY
        - RMS
        
    frac_rms: float
        Total fractional RMS
    df: float
        Frequency resolution. This is used only in the case the 
        frequency array is longer than 2. It is supposed to be 
        specified when you want to evaluate the power at a single
        frequency

    RETURNS
    -------
    result: numpy.ndarray
        Power spectral shape

    HISTORY
    -------
    2021 06 01, Stefano Rapisarda (Uppsala), Creation date
    '''

    if not type(f) in [list,np.ndarray,pd.Series]:
        f = np.array([f])
    elif type(f) == list:
        f = np.array(f)

    x = f/f0
    result = x**a1/( (1.0+x**b)**((-a2+a1)/b) )
    
    if len(f) > 2:
        df = f[2]-f[1]

    # Get rid of the zero frequency component
    start=0
    if f[0] == 0.: start=1
        
    # Applying normalization according to specified option
    if not norm is None:
        if norm.upper() == 'POWER':
            # There is not 2 because the None normalization takes
            # into account both positive and negative power
            current_frac_rms2 = np.sum(result[start:])/dc**2
        elif norm.upper() == 'LEAHY':
            current_frac_rms2 = np.sum(result[start:])/dc
        elif norm.upper() == 'RMS':
            current_frac_rms2 = np.sum(result[start:])*df

        result = result/current_frac_rms2*frac_rms**2
    
    return result