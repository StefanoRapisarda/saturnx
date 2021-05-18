import numpy as np
import math
from scipy.signal import savgol_filter
from scipy.signal import convolve, fftconvolve
from scipy.fft import fft,ifft
from .wavelets import Wavelet

def comp_scales(s_min,s_max,dj=0.25,family='mexhat',method='fft'):

    #assert s_max > s_min, 's_max must be larger then s_min'

    #n = int( np.log(s_max/s_min)/np.log(2)/ds )
    nj = int(( np.log(s_max/s_min)/np.log(2) )/dj)
    j = np.arange(0,nj+1,1)
    print('{} Computing scales {}'.format(10*'*',10*'*'))
    #print(len(j),(np.log(s_max/s_min)/np.log(2)),dj)
    scales = np.asarray(s_min*2**(j*dj))
    freqs = scale2freq(scales,family=family,method=method)

    print('s_max = {}, s_min = {}'.format(scales[-1],scales[0]))
    print('f_min = {}, f_max = {}'.format(freqs[-1],freqs[0]))
    print('n_scales =',len(scales))
    print('{} {} {}'.format(10*'*',16*'*',10*'*'))
    return scales

def scale2freq(scales,family='mexhat',method='fft'):
    '''
    Compute Fourier frequencies corresponding to scales

    The characteristic frequency of a wavelet is computed with 
    scale = 1

    PARAMETERS
    ----------
    method: str, optional
        can be an (analytical) or fft 
    '''

    if isinstance(scales,list): scales = np.asarray(scales)

    #dur = 100*2
    #n = int(2**12)

    #t = np.linspace(0,dur,n)
    mother_wavelet = Wavelet(scale=1,family=family)
    fc = mother_wavelet.fc

    freqs = np.zeros(len(scales))
    if method == 'an':
        for i,scale in enumerate(scales):
            if family == 'mexhat':
                freqs[i]=1./(2*np.pi*scale/np.sqrt(2.5))
            elif family == 'morlet':
                omega = mother_wavelet.f0*2*np.pi
                freqs[i]=1/(4*np.pi*scale/(omega+np.sqrt(2+omega**2)))

    elif method == 'fft':
        for i,scale in enumerate(scales):
            freqs[i]=fc/scale

    return freqs


def cwt(data, dt, scales, family=None, 
        sub_mean=True, pad=False, method='fft',coi_comp='cpeak',
        print_progress=False):
    '''
    Compute continous wavelet transform using a specified wavelet

    PARAMETERS
    ----------
    data: np.ndarray
        Usually contains a time series

    df: float
        Time resolution of your data (as with data you provide only
        amplitudes and not time information)

    scales: np.ndarray or list
        Array of scales to be explored

    family: str
        Six digit identifier of the wavelet 

    sub_mean: boolean (optional)
        If True the time series is subtracted by the man. Default value
        is False

    method: str
        Can be conv or fft. It specifies the way you want to compute the 
        transform

    coi_comp: str
        Sets the way the cone of influence is computed
        if 'cpeak', it is estimated according to a central peak
        if 'speak', it is estimated according to side peaks
        if 'anal', it is computed according to an analytical formula
    '''

    n = len(data)

    if sub_mean:
        data = data-np.mean(data)

    if pad:
        log2n = np.log(n)/np.log(2)
        x = int(log2n)+1
        if (2**x - n) < ((2**x-2**(x-1))/3):
            x+=1
        diff = int(2**x-n)
        to_pad = int(diff/2)
        if diff%2 == 0:
            left_pad,right_pad = to_pad,to_pad
            ndata = np.pad(data,(to_pad,to_pad),'constant',constant_values=(0,0))
        else:
            left_pad,right_pad = to_pad,to_pad+1
            ndata = np.pad(data,(to_pad,to_pad+1),'constant',constant_values=(0,0))
        nn = len(ndata)
        assert nn == 2**x, 'Something is wrong in padding ({},{})'.format(nn,2**x)
        print('Data padded with zero, {} ---> {}'.format(len(data),nn))
    else:
        ndata = data
        nn = n

    # Determining wavelet dtype, wavelet transform has the same dtype
    if family is None: family = 'mexhat'
    mather_wavelet = Wavelet(scale=1,family=family)
    out_dtype = mather_wavelet.y.dtype
    fc = mather_wavelet.fc

    #scales = comp_scales(f_min=f_min,f_max=f_max,ds=ds,family=family)

    output = np.zeros((len(scales), nn), dtype=out_dtype)
    freqs = np.zeros(len(scales))
    coi_times = np.zeros(len(scales))

    if method == 'fft':
        data_fft = fft(ndata)

    for i, scale in enumerate(scales):
        if i%20 == 0 and print_progress:
            print('Computing transform {} %'.format(int(i/(len(scales)-1)*100)))

        # Computing wavelet with the same data time resolution
        nw = 10*scale*2
        warray = np.arange(0,nw,dt)
        wavelet = Wavelet(x=warray,scale=scale,family=family,coi_comp=coi_comp)
        wavelet_y = wavelet.y
        if len(wavelet_y) < len(ndata):
            diff = len(ndata)-len(wavelet_y)
            to_pad2 = int(diff/2)
            if diff%2 == 0:
                wavelet_y = np.pad(wavelet_y,(to_pad2,to_pad2),'constant',constant_values=(0,0))
            else:
                wavelet_y = np.pad(wavelet_y,(to_pad2,to_pad2+1),'constant',constant_values=(0,0))
        elif len(wavelet_y) > len(ndata):
            diff = len(wavelet_y)-len(ndata)
            to_crop = int(diff/2)
            if diff%2 == 0:
                wavelet_y = wavelet_y[to_crop:-to_crop]
            else:
                wavelet_y = wavelet_y[to_crop:-to_crop-1]

        freqs[i] = fc/scale

        # Computing coi
        coi_times[i] = wavelet.coi

        if method == 'conv':
            wavelet_data = np.conj(wavelet_y[::-1])
            output[i] = convolve(ndata, wavelet_data, mode='same')
        
        elif method == 'fft':

            fft_wavelet = fft(wavelet_y)
            output[i,:] = np.fft.fftshift(ifft(data_fft*fft_wavelet))

    if len(coi_times)%2==0:
        window = len(coi_times)-1
    else:
        window = len(coi_times)
    coi_times = savgol_filter(coi_times, window, 5)

    if pad:
        output = output[:,left_pad:-right_pad]

    print('Done!')
    return output, freqs, coi_times


    

