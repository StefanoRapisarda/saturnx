import numpy as np
from scipy.signal import convolve, fftconvolve
from scipy.fft import fft,ifft
from .wavelets import Wavelet

def comp_scales(s_min,tdur,dj=0.25,family='mexhat'):

    #assert s_max > s_min, 's_max must be larger then s_min'

    #n = int( np.log(s_max/s_min)/np.log(2)/ds )
    nj = int(( np.log(tdur/s_min)/np.log(2) )/dj)
    j = np.arange(0,nj+1,1)
    print(len(j),(np.log(tdur/s_min)/np.log(2)),dj)
    scales = np.asarray(s_min*2**(j*dj))
    freqs = scale2freq(scales,family=family,method='fft')

    print('s_max = {}, s_min = {}'.format(scales[-1],scales[0]))
    print('f_min = {}, f_max = {}'.format(freqs[-1],freqs[0]))
    print('n_scales =',len(scales))
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

    dur = 100*2
    n = int(2**12)

    t = np.linspace(0,dur,n)
    wavelet = Wavelet(t,scale=1,family=family)
    fc = wavelet.fc

    if method == 'an':

        freqs = fc/scales

    elif method == 'fft':
        freqs = np.zeros(len(scales))

        for i,scale in enumerate(scales):
            dur = 2*100*scale
            f_attempt = fc/scale
            n = int(f_attempt*dur*100)
            t = np.linspace(0,dur,n)
            wavelet = Wavelet(t,scale=scale,family=family)
            freqs[i]=wavelet.fc

    return freqs


def cwt(data, dt, scales, family=None, 
        sub_mean=False, pad=False, method='conv'):
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
    '''

    n = len(data)

    if sub_mean:
        data -= np.mean(data)

    if pad:
        log2n = np.log(n)/np.log(2)
        x = int(log2n)
        if log2n - x != 0:
            x+=1
        pads = np.zeros(int(2**x-n)).astype(data.dtype)
        ndata = np.concatenate((data,pads))
        nn = len(data)
        assert nn == 2**x, 'Something is wrong in padding'
    else:
        ndata = data
        nn = n

    # Determining wavelet dtype, wavelet transform has the same dtype
    if family is None: family = 'mexhat'
    wavelet = Wavelet(family=family)
    out_dtype = wavelet.y.dtype

    #scales = comp_scales(f_min=f_min,f_max=f_max,ds=ds,family=family)

    output = np.zeros((len(scales), len(data)), dtype=out_dtype)
    freqs = np.zeros(len(scales))
    for i, scale in enumerate(scales):

        # Computing wavelet with the same data time resolution
        nw = 10*scale*2
        warray = np.arange(0,nw,dt)
        wavelet = Wavelet(x=warray,scale=scale,family=family)
        
        freqs[i] = wavelet.fc

        if method == 'conv':
            wavelet_data = np.conj(wavelet.y[::-1])
            output[i] = convolve(data, wavelet_data, mode='same')
        
        elif method == 'fft':

            # not reliable
            tot_size = len(wavelet.y)+len(data)
            log2n = np.log(tot_size)/np.log(2)
            x = int(log2n)
            if log2n - x != 0:
                x+=1
            closest_p2_size = int(2**x)
            print('closest_p2_size',closest_p2_size)

            wavelet_data = np.conj(fft(wavelet.y,n=len(data)))
            fourier_data = fft(data,n=len(data))
            conv = ifft(fourier_data*wavelet_data)
            #print('conv shape',conv.shape)
            #print('output shape',output.shape)
            output[i,:] = conv*np.sqrt(scale)

    return output, freqs


    

