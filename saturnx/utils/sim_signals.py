import numpy as np
from scipy.signal import chirp
from saturnx.core.lightcurve import Lightcurve
from saturnx.core.power import PowerSpectrum

def sin_signal_from_bkg_lc(bkg_lc,freq=1,snr=1,mean=0):
    '''
    Returns a sinusoid signal in the form of a saturnx.Lightcurve
    with the specified frequency and signal to noise ratio
    relative to the given background level

    The sinusoid signal is computed on the time array of the background
    lightcurve. This is because the output lightcurve is meant to be
    added to background noise to simulate the presence of a sinusoid
    signal with a certain SNR on top of the noise

    PARAMETERS
    ----------
    bkg_lc: saturnx.Lightcurve 
        Background lightcurve 
    freq: float
        Frequency of the sinusoid
    snr: float
        Signal to noise ratio
    mean: float
        Mean of the sinusoid

    RETURNS
    -------
    signal_lc: saturnx.Lightcurve
        saturnx.Lightcurve object containing the sinusoid signal

    HISTORY
    -------
    2021 05 11, Stefano Rapisarda (Uppsala), creation date
    '''

    # Extracting time array
    t = bkg_lc.time.to_numpy()
    
    # Computing fractional RMS of bkg_lc
    # at the given frequency
    bkg_power = PowerSpectrum.from_lc(bkg_lc)
    df = bkg_power.df
    rms_bkg_power = bkg_power.normalize('rms')
    bkg_frac_rms = rms_bkg_power.comp_frac_rms(low_freq=freq-df/2,high_freq=freq+df/2)[0]
    
    # Initializing sinusoid signal
    # NOTE: The variance of a sinusoid signal is amp**2/2
    amp = np.sqrt(2)*snr*bkg_frac_rms*np.mean(bkg_lc.counts)
    signal = amp*np.sin(2*np.pi*t*freq)+mean
    
    # Making sinusoid signal lightcurve
    signal_lc = Lightcurve(time_array = t,count_array=signal)
    
    return signal_lc

def chirp_signal_from_bkg_lc(bkg_lc,freq1=1,freq2=2,snr=1,mean=0):
    '''
    Returns a chirp signal in the form of a saturnx.Lightcurve
    with the specified frequencies and signal to noise ratio
    relative to the given background level

    The chirp signal is computed on the time array of the background
    lightcurve. This is because the output lightcurve is meant to be
    added to background noise to simulate the presence of a sinusoid
    signal with a certain SNR on top of the noise

    PARAMETERS
    ----------
    bkg_lc: saturnx.Lightcurve 
        Background lightcurve 
    freq1: float
        Start frequency of the chirp signal
    freq2: float
        End frequency f the chirp signal
    snr: float
        Signal to noise ratio
    mean: float
        Mean of the sinusoid

    RETURNS
    -------
    signal_lc: saturnx.Lightcurve
        saturnx.Lightcurve object containing the sinusoid signal

    HISTORY
    -------
    2021 05 11, Stefano Rapisarda (Uppsala), creation date
    '''

    # Extracting time array
    t = bkg_lc.time.to_numpy()
    
    # Computing fractional RMS of bkg_lc
    # at the given frequency
    bkg_power = PowerSpectrum.from_lc(bkg_lc)
    df = bkg_power.df
    rms_bkg_power = bkg_power.normalize('rms')
    bkg_frac_rms = rms_bkg_power.comp_frac_rms(low_freq=freq1-df/2,high_freq=freq2+df/2)[0]
    
    # Initializing chirp signal
    factor = 1.4 # This is determined empirically
    amp = np.sqrt(2)*snr*bkg_frac_rms*np.mean(bkg_lc.counts)*factor
    signal = amp*chirp(t, f0=freq1, f1=freq2, t1=t[len(t)//2], method='linear')+mean
    
    # Making sinusoid signal lightcurve
    signal_lc = Lightcurve(time_array = t,count_array=signal)
    
    return signal_lc

def sin_signal_from_bkg_power(dt,nt,bkg_frac_rms,freq=1,snr=1,mean=1):
    '''
    '''

    # Extracting time array
    t = dt*np.arange(nt)
    
    # Initializing sinusoid signal
    # NOTE: The variance of a sinusoid signal is amp**2/2
    amp = np.sqrt(2)*snr*bkg_frac_rms*mean
    signal = amp*np.sin(2*np.pi*t*freq)+mean
    
    # Making sinusoid signal lightcurve
    signal_lc = Lightcurve(time_array = t,count_array=signal)
    
    return signal_lc

def chirp_signal_from_bkg_power(dt,nt,bkg_frac_rms,freq1=1,freq2=2,snr=1,mean=1):
    '''
    '''

    # Extracting time array
    t = dt*np.arange(nt)
    
    # Initializing chirp signal
    factor = 1.4 # This is determined empirically
    amp = np.sqrt(2)*snr*bkg_frac_rms*mean*factor
    signal = amp*chirp(t, f0=freq1, f1=freq2, t1=t[len(t)//2], method='linear')+mean
    
    # Making sinusoid signal lightcurve
    signal_lc = Lightcurve(time_array = t,count_array=signal)
    
    return signal_lc

