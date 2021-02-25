import numpy as np
import pandas as pd
import math
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

    if k == 0:
        print('---->',type(array))
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
