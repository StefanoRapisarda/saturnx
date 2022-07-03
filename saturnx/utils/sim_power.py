import numpy as np
import pandas as pd
from scipy.fft import rfftfreq

def sbend_pl(dt=1,nt=1000,freq_array=None,a1=0,a2=1,b=100,f0=1,norm=1,
    dc=100,frac_rms=0.3):
    '''
    It returns a (smoothly) bending power law power spectral shape
    given a certain total fractional RMS amplitude and a normalizaion
    option.

    PARAMETERS
    ----------
    dt: float (optional)
        Time frequency (default is 1)
    nt: int (optional)
        Number of time bins (default is 1000)
    freq_array: sequence, float, or None (optional)
        Frequency values to evaluate the (normalized) power.
        If None (default), an array of positive frequencies will
        be initialized according to the provided time resolution and
        number of time bins
    a1: float (optional)
        Power law index at low frequencies (propto e^-a1, default is 0)
    a2: float (optional)
        Power law index at high frequencies (propto e^-a2, default is 1)
    b: float (optional)
        Bending parameter (the highest b the sharper the break, default
        is 100)
    f0: float (optional)
        Bending/breaking frequency (default is 1)
    norm: str or float (optional)
        If float (default is 1), the norm factor is just the bending 
        power law normalization
        If str, it specifies a normalization option. This means that a 
        normalization will be computed according do the specified DC
        and fractional rms.
        First, the current total fractional RMS is computed with an
        expression depending on the specified normalization type,
        each power is then divided by this value and multiplied by the 
        fractional fractional RMS value specified by the user.
        The normalization options are:
        - POWER 
        - LEAHY
        - RMS    
        !!! If the eventually provided frequency array is not evenly
        spaced, the fractional RMS will be computed from dt and nt 
    dc: float
        Zero frequency component. It is equal to the sum of all photons   
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
    2021 07 08, Stefano Rapisarda (Uppsala)
        The story about analytical and user frequency array is 
        dangerously confusing. The frequency resolution is needed when
        computing the fractional RMS. However, if the specified 
        frequency array is not evenly spaced (due to rebinning, for 
        example), the frequency resolution is not extractable, so it 
        needs to be computed from dt and nt. dt and nt are also 
        parameters and, to get a correct result, they must correspond 
        to the original (not binned), intended, frequency array.
        Now, when the program finds out that the frequency array is not
        evenly spaced, it prints a warning and use dt and nt to compute
        a new frequency array and get the correct frequency resolution.

    '''

    f = rfftfreq(n=nt,d=dt)
    if type(freq_array) == list: 
        freq_array = np.array(freq_array)  
    elif freq_array is None:
        freq_array = f

    xtemp = f/f0
    x = freq_array/f0

    power = x**a1 / ( (1.0+x**b)**(-(a2-a1)/b) )
    
    if type(norm) == str:

        # Checking if the frequency array is equally spaced
        diff = np.ediff1d(freq_array)
        test = np.sum(diff==diff[1])
        if test > 1:
            df = np.median(diff)
            power_for_norm = power
        else:
            print('!!! Frequencies are not evenly spaced !!!')
            print('I will use dt and nt to compute the normalization')
            df = f[2]-f[1]
            power_for_norm = xtemp**a1/( (1.0+xtemp**b)**(-(a2-a1)/b))
            
        if norm.upper() == 'POWER':
            current_frac_rms2 = np.sum(2*power_for_norm)/dc**2
        elif norm.upper() == 'LEAHY':
            current_frac_rms2 = np.sum(power_for_norm)/dc
        elif norm.upper() == 'RMS':
            current_frac_rms2 = np.sum(power_for_norm)*df

        norm_factor = frac_rms**2/current_frac_rms2
    else:
        norm_factor = norm
                                        
    result = norm_factor * power
    
    return result