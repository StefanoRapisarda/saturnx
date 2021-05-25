import numpy as np
import pandas as pd
from random import gauss
import math
from scipy.fftpack import ifft
import multiprocessing as mp
from functools import partial

def poi_events(tres=1, nbins=1000, cr=1):
    '''
    Creates a numpy array of Poisson distributed events

    PARAMETERS
    ----------
    tres: float
        Time resolution
    
    nbins: int
        Number of bins with length equal to the time resolution

    cr: float
        Count rate [counts/seconds]

    RETURNS
    -------
    time: np.array
        Array of time tags

    HISTORY
    -------
    2021 02 18, Stefano Rapisarda, Uppsala (creation date)
        I made this for testing purposes

    NOTES
    -----
    The number of bins refers to the number of bins you would obtain
    if grouping the time arrival of photons with np.histogram and the
    provided time resolution
    '''

    # This is the histogram
    poi = np.array([np.random.poisson(tres*cr) for i in range(nbins)])
    time = np.concatenate([np.random.uniform(0+i*tres,(i+1)*tres,poi[i]) for i in range(nbins)],axis=0)
    time = sorted(time)

    return time
    
def my_slice(k,nk,array,step,exp=1,average=True):
    '''
    It average or sum an input array from k*step to (k+1)*step.
    Before average or sum array is elevated to exp

    PARAMETERS
    ----------
    k: int
        First index to average/sum
    nk: int
        Last index to average/sum
    array: numpy.ndarray
        Array to average/sum
    step: int
        Interval to average (in bin units)
    exp: float
        Exponent
    average: boolean, optional
        If True, array elements are averaged between k*step and
        (k+1)*step, if False elements are summed. Default is True

    RETURNS
    -------
    result: float
        Average or sum of the array elements between bins k*step
        and k*step+step

    HISTORY
    -------
    2019 10 09, Stefano Rapisarda, 09/10/2019
    '''

    if type(exp) == int: exp = float(exp)

    if k == 0:
        pass
        #print('---->',type(array))
    if k == nk-1:
        slice = (array[k*step:]**exp)
    else:
        slice = (array[k*step:k*step+step]**exp)
    
    if average:
        result = np.mean(slice)
    else:
        result = np.sum(slice)
    
    return result



def my_rebin(x,y,xe=0,ye=0,rf=-30,remove_neg=False,average=True):
    '''
    This function rebin x,y,xe,ye arrays linearly (rf>0) or
    logarithmically (rf<0).

    PARAMETERS
    ----------
    x,y,xe,ye: numpy arrays
               Arrays containing data, x,y values and corresponding errors
    
    rf: float
        Values of the rebinning factor. A positive value means linear rebin,
        a negative one means logarithmic rebin.
        All the values between x_(i+1) and x_(i) will be rebinned, where 
        x_(i+1) = x_(i) * c and c = 10*(1/abs(rf)).
        In logarithmic scale, all the values within a (logarithmic) distance equal to
        1/abs(rf) will be rebinned.


    HISTORY
    -------
    Stefano Rapisarda, 24/03/2019, SHAO, creation
    Stefano Rapisarda, 24/09/2019, SHAO
            Implemented with linear rebin. Instead of using Michiel's approach,
            so making the rebing working on the array index, I sligthly modified 
            existing code that still use the x array values, but, this time, with
            linear steps
    Stefano Rapisarda, 09/10/2019
            Previous way to perform linear rebin is too slow. I modified the code
            to use numpy histogram. Also, in Michiel's routine eventual points equal to
            zero are skipped. I confused this to DC component treatment (...). Actually
            when I rebin powers I already skip the zero frequency component with
            masking. I suspect that this option of skipping zeros comes in case some
            other routine introduced zeros before powers because of shifting frequencies.
            Until I figure out, I will just remove those lines of code.
            Later on today, I decided for linear bin with multiprocessing instead 
            of histogram

    NOTE
    ----
    2021 02 23, Stefano Rapisarda (uppsala)
        This works only if there is a ordinated time array!
    '''

    
    rx = np.array([])
    ry = np.array([])
    rxe = np.array([])
    rye = np.array([])
    print('Rebinning...')
    if rf < 0:
        # Initializing rebinned arrays

        check = y<0
        if check.any():
            print('WARNING: There are negative powers')
            #print(y[check]) 
        #check2 = ye<0
        #if check2.any():
        #print('WARNING: There are negative sigmas')
        #print(ye[check2])    

        # Checking x value of the first imput point
        # If this is equal to zero (so if the zero frequency
        # component is specidied), the first rebinned point
        # will be equal to the first imput point
        i = 0
            
        # Initializing rebin factor
        factor = 10.**(1./np.absolute(rf))

        start=x[i]
        #if rf < 0:
        stop =start*factor
        #elif rf > 0:
        #    stop = start+factor
        #    print('Frequency resolution will be {} Hz'.format(round((stop-start),3)))

        while stop <= x[-1]:
            #print(start,stop)
            if remove_neg and check.any():
                mask_y = (x >= start)&(x < stop)&(y>0)
                mask_x = (x >= start)&(x < stop)

                if mask_y.sum() == 0:
                    rebinned_y = 0.
                    rebinned_x = 0.
                    if np.any(ye): rebinned_ye = -1
                    if np.any(xe): rebinned_xe = -1
                else:
                    rebinned_x = np.mean(x[mask_x])
                    if average:
                        rebinned_y = np.mean(y[mask_y])
                    else:
                        rebinned_y = np.sum(y[mask_y])
                    print(len(ye[mask_y]),len(x[mask_y]))
                    if np.any(ye): rebinned_ye = np.sqrt( np.sum(ye[mask_y]**2))/len(x[mask_y] )                        
                    if np.any(xe): rebinned_xe = np.sqrt( np.sum(xe[mask_x]**2))/len(x[mask_x] )                  
            else:
                mask = (x >= start)&(x < stop)

                rebinned_x = np.mean(x[mask])
                if average:
                    rebinned_y = np.mean(y[mask])
                else:
                    rebinned_y = np.sum(y[mask])
                if np.any(xe): rebinned_xe = np.sqrt(np.sum(xe[mask]**2))/len(x[mask])   
                if np.any(ye): rebinned_ye = np.sqrt(np.sum(ye[mask]**2))/len(x[mask])

            # Appending rebinned arrays
            rx = np.append(rx,rebinned_x)
            ry = np.append(ry,rebinned_y)
            if np.any(xe): rxe = np.append(rxe,rebinned_xe)
            if np.any(ye): rye = np.append(rye,rebinned_ye)

            # If stop is exactely in the middle of two bins, closest will be
            # the index of the smallest.
            closest = np.absolute(x-stop).argmin()
            if stop < x[closest]:
                i = closest
            elif stop >= x[closest]:
                i = closest + 1

            start=x[i]
            #if rf < 0:
            stop =start*factor
            #elif rf > 0:
            #    stop = start+factor

        print('---->',stop)

    elif rf > 0:
        #In this case the rebin factor is the number of old
        #bins that will be included in the new bins
        #Computing the new number of bins
        # As I am using numpy histogram, it requires the the
        # bin edges

        # Again this is to take care of the DC component
        i = 0
        if len(x[i:])%rf==0:
            nb = int(len(x[i:])/rf)
        else:
            nb = len(x[i:])//rf + 1
        #else:
        #    nb = round(len(x[i:])/rf)+1

        print('Linear rebinning, old res {}, new res {}'.format(x[5]-x[4],(x[5]-x[4])*rf))
        iterable = [j for j in range(i,nb)]
        pool = mp.Pool(mp.cpu_count())
        
        rx1d = partial(my_slice,nk=nb,array=x,step=rf)
        rx = np.asarray( pool.map( rx1d,iterable ) )
        ry1d = partial(my_slice,nk=nb,array=y,step=rf,average=average)
        ry = np.asarray( pool.map( ry1d,iterable ) )
        if np.any(xe):
            rxe1d = partial(my_slice,nk=nb,array=xe,step=rf,exp=2)
            rxe = np.sqrt(np.asarray( pool.map(rxe1d,iterable) ) )
        if np.any(ye):
            rye1d = partial(my_slice,nk=nb,array=ye,step=rf,exp=2)
            rye = np.sqrt(np.asarray( pool.map(rye1d,iterable) ) )
        pool.close()
        
    print('Done!')
    return rx,ry,rxe,rye

def rebin_xy(x,y=None,xe=None,ye=None,rf=-30,start_x=0,stop_x=np.inf,
    mode='average'):
    '''
    PARAMETERS
    ----------
    mode: string (optional)
        can be average or sum, it specified the way the rebinned y 
        values are computed (default is average)

    HISTORY
    -------
    Stefano Rapisarda, 2021 02 04 (Uppsala), creation date
    '''

    if type(x) == pd.Series: x = x.to_numpy()
    if type(x) == list: x = np.array(x)
    rbx = np.array([])
    rby = None
    rbxe = None
    rbye = None

    if not y is None:
        if type(y) == pd.Series: y = y.to_numpy()
        if type(y) == list: y = np.array(y)
        if len(x) != len(y):
            raise ValueError('x and y must have the same dimension')
        rby = np.array([])
    if not xe is None:
        if type(xe) == pd.Series: xe = xe.to_numpy()
        if type(xe) == list: xe = np.array(xe)
        if len(x) != len(xe):
            raise ValueError('x and xe must have the same dimension')        
        rbxe = np.array([])
    if not ye is None:
        if type(ye) == pd.Series: ye = ye.to_numpy()
        if type(ye) == list: ye = np.array(ye)
        if len(ye) != len(y):
            raise ValueError('y and ye must have the same dimension') 
        rbye = np.array([])

    # Checking that x is monotonically increasing
    if not np.all(np.diff(x) > 0):
        raise ValueError('x must be monotonically increasing')

    #print('Rebinning logarithmically (factor={}) ... '.format(rf))
    
    if rf < 0:

        check = y<0
        if check.any():
            print('WARNING: There are negative powers')
 
        # Checking x value of the first imput point
        # If this is equal to zero (so if the zero frequency
        # component is specidied), the first rebinned point
        # will be equal to the first imput point
        i = 0

        # Determining starting index, if start x is specified
        if start_x != 0:
            closest = np.absolute(x-start_x).argmin()
            if x[closest] > start_x:
                i = closest
            elif x[closest] < start_x:
                i = closest - 1
            
        # Initializing rebin factor
        factor = 10.**(1./np.absolute(rf))

        while (x[i] < x[-1]) and (x[i] < stop_x):

            step_rbx = 0
            if not y is None: step_rby = 0
            if not ye is None: step_rbye = 0
            if not xe is None: step_rbxe = 0
            steps = 0

            # Stop of the single logarithmic bin
            stop = x[i]*factor

            #print('index',i)
            #print('-'*10)

            while x[i] <= stop:
                #print(x[i],stop)

                step_rbx += x[i]
                if not y is None: step_rby += y[i]
                if not ye is None: step_rbye += ye[i]**2
                if not xe is None: step_rbxe += xe[i]**2
                
                steps += 1
                i += 1
                
                if i >= len(x): break  
            
            #print('-'*10)

            if steps != 0:
                #print('Averaging {} points'.format(steps))
                # Appending rebinned arrays
                rbx = np.append(rbx,step_rbx/steps)
                if not y is None:
                    if mode == 'average':
                        rby = np.append(rby,step_rby/steps)
                    elif mode == 'sum':
                        rby = np.append(rby,step_rby)
                if not xe is None: rbxe = np.append(rbxe,np.sqrt(step_rbxe)/steps)
                if not ye is None: rbye = np.append(rbye,np.sqrt(step_rbye)/steps)
            else:
                rbx = np.append(rbx,0)
                if not y is None: rby = np.append(rby,0) 
                if not ye is None: rbye = np.append(rbye,0) 
                if not xe is None: rbxe = np.append(rbxe,0) 

            #print()
            if i >= len(x): break   

    elif rf > 0:

        if mode == 'average': average = True
        if mode == 'sum': average = False
        #In this case the rebin factor is the number of old
        #bins that will be included in the new bins
        #Computing the new number of bins
        # As I am using numpy histogram, it requires the the
        # bin edges

        # Again this is to take care of the DC component
        i = 0
        if len(x[i:])%rf==0:
            nb = int(len(x[i:])/rf)
        else:
            nb = len(x[i:])//rf + 1
        #else:
        #    nb = round(len(x[i:])/rf)+1

        print('Linear rebinning, old res {}, new res {}'.format(x[5]-x[4],(x[5]-x[4])*rf))
        iterable = [j for j in range(i,nb)]
        pool = mp.Pool(mp.cpu_count())
        
        rx1d = partial(my_slice,nk=nb,array=x,step=rf)
        rbx = np.asarray( pool.map( rx1d,iterable ) )
        if not y is None:
            ry1d = partial(my_slice,nk=nb,array=y,step=rf,average=average)
            rby = np.asarray( pool.map( ry1d,iterable ) )
        if not xe is None:
            rxe1d = partial(my_slice,nk=nb,array=xe,step=rf,exp=2)
            rbxe = np.sqrt(np.asarray( pool.map(rxe1d,iterable) ) )
        if not ye is None:
            rye1d = partial(my_slice,nk=nb,array=ye,step=rf,exp=2)
            rbye = np.sqrt(np.asarray( pool.map(rye1d,iterable) ) )
        pool.close()
        
    print('Done!')
    return rbx,rby,rbxe,rbye


def rebin_arrays(time_array,tres=None,rf=-30,
    arrays=[],bin_mode=[],exps=[]):

    if not type(time_array) in [np.ndarray,pd.Series,list]:
        raise ValueError('time_array must be an array')
    else:
        if type(time_array) == list:
            time_array = np.array(time_array)
        elif type(time_array) == pd.Series:
            time_array = time_array.to_numpy()

    if type(arrays) != list: arrays = [arrays]

    if tres is None:
        tres = np.median(np.ediff1d(time_array))
        tres = np.round(tres,int(abs(math.log10(tres/1e+6))))

    if len(arrays) != 0:
        if len(bin_mode) == 0:
            bin_mode = ['ave' for i in range(len(arrays))]
        if len(exps) == 0:
            exps = [1 for i in range(len(arrays))]

    if rf > 0:
        # Binning via bins
        nb = int(len(time_array)/rf)

        print('Linear rebinning, old res {}, new res {}'.\
            format(tres,tres*rf))
        iterable = [j for j in range(nb)]
        
        pool = mp.Pool(mp.cpu_count())
        
        rta1d = partial(my_slice,nk=nb,array=time_array,step=rf)
        rta = np.asarray( pool.map( rta1d,iterable ) )
        
        rebinned_arrays = []
        for array,mode,exp in zip(arrays,bin_mode,exps):
            if mode == 'ave':
                average = True
            elif mode == 'sum':
                average = False
            ra1d = partial(my_slice,nk=nb,array=array,step=rf,
                average=average,exp=exp)
            ra = np.asarray( pool.map( ra1d,iterable ) )
            rebinned_arrays += [ra]
        pool.close()

    elif rf < 0:

        print('log_rebin')

        base = 10
        step = base**(1/abs(rf))

        #binning via values (edges of the binned array)
        start = time_array[0] - tres/2
        if start == 0:
            start = tres/2
        stop  = time_array[-1]+ tres/2
        value = start*step

        log_grid = [start]
        # making the grid
        while value <= stop:
            if value-log_grid[-1] > tres:
                log_grid += [value]
            value *= step

        rebinned_arrays = [np.array([]) for i in range(len(arrays))]
        rta = np.array([])
        for i in range(1,len(log_grid)):
            mask = (time_array >= log_grid[i-1]) & (time_array < log_grid[i])
            rta = np.append(rta,np.mean(time_array[mask]))

            for a,(array,mode,exp) in enumerate(zip(arrays,bin_mode,exps)):
                if mode == 'ave':
                    method = np.mean
                elif mode == 'sum':
                    method = np.sum
                rebinned_arrays[a] = np.append(rebinned_arrays[a],
                    method(array[mask]**exp))
  
    print('Done!')

    return rta,rebinned_arrays


def white_noise(nbins=1000,mean=0,std=2):
    noise = [gauss(mean, std) for i in range(nbins)]
    return np.array(noise)

def poi_noise(counts=100,nbins=1000):
    poi = [np.random.poisson(counts) for i in range(nbins)]
    return np.array(poi)

def timmer_koenig2(freq,ps,mean=0):
    '''
    It is like timmer_koenig, but works with a given power spectrum
    '''

    t_dur = 1./abs(freq[2]-freq[1])
    t_res = 1./2/max(freq)
    n_bins = len(freq)

    # Initializing fourier amplitudes
    fourier = np.zeros(n_bins,dtype=complex)
    fourier[0] = n_bins*mean

    # Loop on frequency excluding the zero-frequency component 
    for i in range(1,n_bins):
        amp  = np.sqrt(ps[i])
        if i < int(n_bins/2):
            fourier[i] = np.random.normal()*amp+np.random.normal()*amp*1j
        else:
            fourier[i] = np.conj(fourier[i-int((i-n_bins/2)*2)])

    # Obtaining the time series via inverse Fourier transform
    time = irfft(fourier).real#+t_dur*cr
    
    # Array of time bins boundaries
    t_bins = np.linspace(0,t_dur+t_res,n_bins+1)
    
    # Array with the center of the time bin
    t = np.array([(t_bins[i]+t_bins[i-1])/2. for i in range(1,len(t_bins))])

    #time = time-np.mean(time)+cr*t_dur
    #time = time/np.std(time)*std
    #time = time-np.mean(time)+t_dur*cr/n_bins
    time = time-np.mean(time)+mean

    return t,time

def timmer_koenig_from_ps(dt,nt,ps,dc=0):
    '''
    Returns a realization of the specified power spectrum
    according to the Timmer and Koenig prescription 
    (Timmer and Koenig 1995)

    DESCRIPTION
    -----------
    This function was designed to obtain a realization of a power 
    spectrum of a certain analytical form. The way the user should
    use it is, first decide the time resolution dt and the number
    of bins of the output lightcurve. Then, the user should create
    an array of frequencies with scipy/numpy fftfreq and the previously
    defined time resolution and number of bins. User this array of 
    frequency, the user can estimate according to an analytic 
    expression the power spectrum corresponding to positive frequencies
    and feed it to this function.

    PARAMETERS
    ----------
    dt: float
        Time resolution of the original time series
    nt: int
        Number of time bins in the original time series
    ps: numpy.array
        Power spectrum (positive frequencies)
    dc: float, optional
        dc or zero-frequency component, it should be equal
        to the desired total counts of the time series.
        !!! This correspond to the Fourier amplitude at
        zero frequency, so np.sqrt(ps[0])!!!

    RETURNS
    -------
    t, lc: tuple
        t is the time array and lc the corresponding amplitude
        of the lightcurve


    HISTORY
    -------
    2021 05 18, Stefano Rapisarda (Uppsala), creation date
        Compared to the previous versions, this is more efficient
        as a tried to use "pythonian" language as much as I can.
    '''

    # Initializing two series of random numbers
    n1 = np.random.normal(size=len(ps))
    n2 = np.random.normal(size=len(ps))

    # Initializing Fourier amplitudes
    pos_amp = np.sqrt(0.5*ps)*(n1+n2*1j)
    
    if nt%2 == 0:
        amp = np.hstack([dc,pos_amp,1/2/dt,np.flip(np.conj(pos_amp))])
    else:
        amp = np.hstack([dc,pos_amp,np.flip(np.conj(pos_amp))])

    # Perform inverse Fourier transform
    lc = (ifft(amp)).real

    # Initializing time array of bin center
    t_bins = np.linspace(0,dt*(nt+1),nt+1)
    t = np.array([(t_bins[i]+t_bins[i-1])/2. for i in range(1,len(t_bins))])

    return t,lc