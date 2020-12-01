import re
import sys
from datetime import datetime

import numpy as np
import multiprocessing as mp
from functools import partial

import scipy.fftpack as fftpack
from scipy import signal

import logging
from ..utils import my_classes as mc

def clean_gti(start,stop):
    '''
    Re-arrange GTIs in order of starting time and merges
    overlapping GTIs
    '''

    start = np.asarray(start)
    stop  = np.asarray(stop)

    sorting_indices = np.argsort(start)
    sorted_start = start[sorting_indices]
    sorted_stop = stop[sorting_indices]
    
    clean_start = [sorted_start[0]]
    clean_stop = [sorted_stop[0]]
    #print('clean_start',clean_start,'clean_stop',clean_stop)
    flag=False
    for i in range(1,len(start)):
        #print('iteration',i)
        # Case A
        #print('sorted_start',sorted_start[i],'clean_stop',clean_stop[-1])
        if sorted_start[i] <= clean_stop[-1]:
            flag = True
            if sorted_stop[i] <= clean_stop[-1]:
                #print('CaseA1')
                continue
            else:
                #print('CaseA2')
                clean_stop[-1] = sorted_stop[i]
        # Case B
        elif sorted_start[i] > clean_stop[-1]:
            clean_start += [sorted_start[i]]
            clean_stop  += [sorted_stop[i]]

    if flag: print('Some of the GTIs were overlapping')
    return np.array(clean_start),np.array(clean_stop) 

def get_nn_var(expr,user_char_set=''):
    '''
    Return a list of not numerical variables in a string expression
    excluding the characters '()+-/*[]= '

    PARAMETERS
    ----------
    expr: string
        Expression to examine
    user_char_set: string (optional)
        Character to exclude 

    RETURNS
    -------
    nn_vars: list
        List of not numerical characters in expr (excluding
        the characters specified in char_set)

    HISTORY
    -------
    2020, Stefano Rapisarda (Uppsala), creation date
    '''

    char_set = '()+-/*[]= '
    char_set+=user_char_set

    if '[' in char_set: char_set=char_set.replace('[',r'\[')
    if ']' in char_set: char_set=char_set.replace(']',r'\]')
    
    vars = re.split('['+char_set+']',expr)
    nn_vars = []
    for var in vars:
        var = var.strip()
        try:
            float(var)
        except ValueError:
            if var != '':
                nn_vars += [var]

    return nn_vars

def print_history(cls):
    for key, value in cls.history.items():
        print('{}: {}'.format(key,value))

def my_cdate():
    now = datetime.utcnow()
    date = f'{now.year}-{now.month}-{now.day},{now.hour}:{now.minute}:{now.second}'
    return date

def my_slice(k,nk,array,step,exp=1):
    if k == nk-1:
        tmp = (array[k*step:]**exp).mean()
    else:
        tmp = (array[k*step:k*step+step]**exp).mean()
    return tmp

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
                    rebinned_y = np.mean(y[mask_y])
                    rebinned_x = np.mean(x[mask_x])
                    print(len(ye[mask_y]),len(x[mask_y]))
                    if np.any(ye): rebinned_ye = np.sqrt( np.sum(ye[mask_y]**2))/len(x[mask_y] )                        
                    if np.any(xe): rebinned_xe = np.sqrt( np.sum(xe[mask_x]**2))/len(x[mask_x] )                  
            else:
                mask = (x >= start)&(x < stop)

                rebinned_x = np.mean(x[mask])
                rebinned_y = np.mean(y[mask])
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
        ry1d = partial(my_slice,nk=nb,array=y,step=rf)
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

def timmer_koenig(t_dur,t_res,fv,cr=0,std=1):
    '''
    Stefano Rapisarda 06/04/2019
    
    Assuming that the underlying process (power spectrum) is described
    by a zero frequency Lorentzian, it produces a single realization (time series)
    of the process following Timmer and Koenig (1995) prescription.
    
    In particular:
    - For each frequency draw two Gaussian numbers and multiply them by the square
      root of the power spectrum and use the result as real and imaginary part of
      the Furier transform of the desired data;
    - To obtain a real valued time series, choose the Fourier components for the 
      negative frequencies according to f(-freq) = *f(freq), where * denotes complex
      conjugation;
    - Obtain the time series by backward Fourier transformation of f(freq) from the
      frequency domain to the time domain.
    
    PARAMETERS
    ----------
    t_dur: float
           Duration of the time series
    t_res: float
           Time resolution
    fv: float
        Local viscous frequency. 
        Plotting the Lorentzian using both negative and positive frequencies, 
        fv is the half width at half maximum of the curve.
        Plotting the Lorentzian over only the positive frequencies, fv is the
        point where the Lorentzian changes concavity.
        Plotting the Lorentian in logarithmic scale and in the P*freq versus freq
        representation, the curve peak is at fv.
        
    RETURNS
    -------
    t: numpy array
       array of time bins
       
    time: numpy array
          time series
    '''
    
    # Setting Fourier transform/time series number of bins
    n_bins = int((t_dur)/t_res)
    
    # Setting frequency array
    f = fftpack.fftfreq(n_bins,t_res)
    
    # Because in python, when n_bins is even, the Nyquist frequency is negative
    # I set it positive
    f[int(n_bins/2)]=np.abs(f[int(n_bins/2)]) 
    
    # Computing the Lorentzian
    lor = 1./(1.+(f/fv)**2)
    
    # Initializing fourier amplitudes
    fourier = np.zeros(n_bins,dtype=complex)
    
    # Setting the zero frequency component
    fourier[0] = t_dur*cr
    
    # Loop on frequency excluding the zero-frequency component 
    for i in range(1,n_bins):
        amp  = np.sqrt(lor[i])
        if i < int(n_bins/2):
            fourier[i] = np.random.normal()*amp+np.random.normal()*amp*1j
        else:
            fourier[i] = np.conj(fourier[i-int((i-n_bins/2)*2)])
    
    # Obtaining the time series via inverse Fourier transform
    time = fftpack.ifft(fourier).real#+t_dur*cr
    
    # Array of time bins boundaries
    t_bins = np.linspace(0,t_dur+t_res,n_bins+1)
    
    # Array with the center of the time bin
    t = np.array([(t_bins[i]+t_bins[i-1])/2. for i in range(1,len(t_bins))])

    #time = time-np.mean(time)+cr*t_dur
    time = time/np.std(time)*std
    time = time-np.mean(time)+t_dur*cr/n_bins
    
    return t,time

def initialize_logger(log_name=False,level=logging.DEBUG,text_widget=None):
    '''
    Initialize logging options to pring log messages on the screen
    and on a file (if log_name is specified)

    HISTORY
    -------
    unknown   , Stefano Rapisarda (SHAO), creation date
    2020 09 23, Stefano Rapisarda (Uppsala), efficiency improved
    2020 11 06, Stefano Rapisarda (Uppsala), 
    '''

    # Creating an instance of the object logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Setting a format for the log
    formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(message)s')

    if log_name:
        logging.basicConfig(level=level,
                            format='%(levelname)s: %(asctime)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=log_name+'.log',
                            filemode='w')

    # Creating the log handler
    #handler = logging.StreamHandler(stream=sys.stderr)
    handler = mc.GuiHandler(stream=sys.stderr,text_widget=text_widget)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter) 

    # Configuring the logger with the handler
    logger.addHandler(handler)   

    #if log_name:
        
    #    handler = logging.FileHandler(log_name,"w", encoding=None, delay="true")
    #    handler.setLevel(logging.DEBUG)
    #    handler.setFormatter(formatter)
    #    logger.addHandler(handler)

    return logger
