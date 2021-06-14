import numpy as np
import pandas as pd
from scipy.fft import rfftfreq

def sbend_pl(dt,nt,freq_array=None,a1=0,a2=-1,b=10,f0=1,norm=None,dc=10,frac_rms=0.3):
    '''
    It returns a (smoothly) bending power law power spectral shape
    given a certain total fractional RMS amplitude and a normalizaion
    option.

    PARAMETERS
    ----------
    dt: float
        Time frequency
    nt: int
        Number of time bins
    freq_array: sequence, float, or None
        Frequency values where to evaluate the (normalized) power.
        If None (default), an array of positive frequencies will
        be evaluated according to the provided time resolution and
        number of time bins
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


    RETURNS
    -------
    result: numpy.ndarray
        Power spectral shape

    HISTORY
    -------
    2021 06 01, Stefano Rapisarda (Uppsala), Creation date
    2021 06 14, Stefano Rapisarda (Uppsala)
        Mistake about RMS comutation corrected. I did not take into 
        account that the initially provided frequency array could be
        not linearly spaced. I added two parameters (dt,nt), so that
        the script provide a valid frequency array for computing RMS
        normalization
    '''

    f = rfftfreq(n=nt,d=dt)
    df = f[2]-f[1]
    
    if not norm is None:
        
        f = rfftfreq(n=nt,d=dt)
        xtmp = f/f0
        power = xtmp**a1/( (1.0+xtmp**b)**((-a2+a1)/b))

        if norm.upper() == 'POWER':
            # There is not 2 because the None normalization takes
            # into account both positive and negative power
            current_frac_rms2 = np.sum(power)/dc**2
        elif norm.upper() == 'LEAHY':
            current_frac_rms2 = np.sum(power)/dc
        elif norm.upper() == 'RMS':
            current_frac_rms2 = np.sum(power)*df

        norm_factor = frac_rms**2/current_frac_rms2
                       
    if freq_array is None:
        x = f/f0
    else:
        if type(freq_array) == list: freq_array = np.array(freq_array)  
        x = freq_array/f0
                       
    result = norm_factor * x**a1/( (1.0+x**b)**((-a2+a1)/b))
    
    return result