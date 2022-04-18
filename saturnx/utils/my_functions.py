import os
import re
import sys
from datetime import datetime

import numpy as np
import multiprocessing as mp
from functools import partial

import math

import scipy.fftpack as fftpack
from scipy import signal

import logging
import pathlib
from ..utils import my_classes as mc

import matplotlib
import matplotlib.pyplot as plt

def round_half_up(n, decimals=0):
    '''
    From https://realpython.com/python-rounding/
    '''
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def list_items(path=pathlib.Path.cwd(),itype = 'dir',ext = '',prefix='',
                include_or=[],include_and=[],exclude_or=[],exclude_and=[],
                choose=False,show=False,sort=True,digits=False):
    '''
    Lists files or directories in a folder 

    DESCRIPTION
    -----------
    Depending on the specified option, it lists either all the files
    or the folders inside a specified path. 
    If "choose" is True, returns an interactively chosen file/directory

    PARAMETERS
    ----------
    path: string or pathlib.Path (optional)
        target directory, default is current working directory
    itype: string, optional 
        'dir' for directory, 'file' for files (default = 'dir')
    ext: string or list of strings, optional
        Only files with extension equal to .ext will be returned
    prefix: string or list of strings, optional
        Only files/dirs beginning with prexix will be returned 
        (default is '')
    include_or: string or list (optional)
        Only items including one of the specified string will be
        returned
    include_all: list (optional)
        Only items including all the strings in the list
        will be included
    exclude_or: string or list (optional)
        Only items excluding at least one of the elements in this 
        list will be returned
    exclude_and: string or list (optional)
        Only items excluding all the elements in this list will be
        returned
    digits: boolean, int, string, or list (optional) 
        If True only items whose names contain only digits will 
        be considered (default=False).
        If string or list, the item name will be split excluding 
        the characters in the list. If all the split elements include
        digits, the item will be returned.
        If int, only items containing only digits of len == in will be
        returned.
    sort: boolean (optional) 
        If True the returned list of items will be sorted 
        (default=True)

    RETURNS
    -------
    list
        List of items

    HISTORY
    -------
    2019 07 11, Stefano Rapisarda (SHAO), creation date
    2019 07 16, Stefano Rapisarda (SHAO), 
        Including ext and include options
    2019 07 19, Stefano Rapisarda (SHAO), 
        Changed approach, from listing with next and os.walk to glob
    2019 07 23, Stefano Rapisarda (SHAO), 
        Going back to os.walk and next to isolate folders and files
    2019 11 08, Stefano Rapisarda (SHAO), 
        Introduced the option to specify a list of extension
    2019 11 21, Stefano Rapisarda (SHAO), 
        Corrected some bugs and added the option include_all
    2019 11 23, Stefano Rapisarda (SHAO), 
        Added the option all digits. Also added sort option.
    2021 05 05, Stefano Rapisarda (Uppsala)
        - The option to_removed was removed (you can sort files 
          using other options);
        - include and exclude parameters have been extended with
          or and and;
        - The digit parameter now allows to exclude characters from
          the item name when checking if all the characters are digits
          (see description of the parameter);
        - Directories now are treated as pathlib.Path(s);
    2021 12 06, Stefano Rapisarda (Uppsala)
        Added Prefix option
    2021 03 03, Stefano Rapisarda (Uppsala)
        Added int option in digits parameter
    '''

    if type(path) != type(pathlib.Path()):
        path = pathlib.Path(path)

    if ext != '': itype='file'

    # Listing items
    items = []
    for item in path.iterdir():
        if item.is_dir() and itype == 'dir':
            items += [item]
        elif item.is_file() and itype == 'file':
            items += [item]

    # Filtering files with a certain extension
    if itype == 'file' and ext != '':
        if type(ext) == str: ext = [ext]

        # Removing first dot from ext
        new_ext = [ex[1:]  if ex[0]=='.' else ex for ex in ext]
        ext = new_ext

        new_items = []
        for item in items:
            # Finding the first occurrence of a dot
            #file_name = str(item.name)
            #target_index = file_name.find('.')
            # .suffix returns the extension with the point
            ext_to_test = item.suffix
            if ext_to_test[1:] in ext:
                new_items += [item]
        items = new_items

    # Filtering according to pre (first string)
    if prefix != '':
        if type(prefix) == str: prefix = [prefix]

        new_items = []
        for item in items:
            flag = False
            for p in prefix:
                if item.name[0:len(p)] == p: flag=True
                    
            if flag: new_items += [item]
        items = new_items


    # Filtering files according to include_or
    if include_or != []:
        if type(include_or) == str: include_or = [include_or]

        new_items = []
        for item in items:
            file_name = str(item.name)   
            flags = [True if inc in file_name else False for inc in include_or]
            if sum(flags) >= 1: new_items += [item]
        items = new_items

    # Filtering files according to include_and
    if include_and != []:
        if type(include_and) == str: include_and = [include_and]

        new_items = []
        for item in items:
            file_name = str(item.name)  
            flags = [True if inc in file_name else False for inc in include_and]
            if sum(flags) == len(include_and): new_items += [item]  
        items = new_items

    # Filtering files according to exclude_or
    if exclude_or != []:
        if type(exclude_or) == str: exclude_or = [exclude_or]

        new_items = []
        for item in items:
            file_name = str(item.name)   
            flags = [True if inc in file_name else False for inc in exclude_or]
            if sum(flags) == 0: new_items += [item]
        items = new_items

    # Filtering files according to exclude_and
    if exclude_and != []:
        if type(exclude_and) == str: exclude_and = [exclude_and]

        new_items = []
        for item in items:
            file_name = str(item.name)  
            flags = [True if inc in file_name else False for inc in exclude_and]
            if sum(flags) != len(exclude_and): new_items += [item]  
        items = new_items

    # Filtering according to digit
    if not digits is False:
        new_items = []
        if digits is True:
            new_items = [item for item in items if str(item.name).isdigit()]
        elif type(digits) == int:
            new_items = [item for item in items if \
                (str(item.name).isdigit() and len(item.name) == digits)]
        else:
            if type(digits) == str: digits = [digits]
            split_string = '['
            for dig in digits: 
                if str(dig) != '-':
                    split_string += str(dig)
                else:
                    split_string += '\-'
            split_string += ']'
            #print(split_string)

            for item in items:
                file_name = str(item.name)
                prediv = re.split(split_string,file_name) 
                div = [p  for p in prediv if p != '']
                flags = [True if d.isdigit() else False for d in div]
                if sum(flags) == len(div): new_items += [item]
        items = new_items

    if choose: show=True

    item_name = 'Directories'
    if itype == 'file': item_name = 'Files'
    if show: print('{} in {}:'.format(item_name,str(path)))
    happy = False
    while not happy:
        if show:
            for i, item in enumerate(items):
                print('{}) {}'.format(i+1,item.name))

        if choose:
            target_name = 'directory'
            if itype == 'file': target_name = 'file'
            index = int(input(f'Choose a {target_name} ====> '))-1
            target = items[index]
            ans=input('You chose "{}", are you happy?'.format(target.name))
            if not ('N' in ans.upper() or 'O' in ans.upper()):
                happy = True
        else:
            target = items
            happy = True

    if type(target) == list:
        if sort: target = sorted(target)
        #if len(target) == 1: 
        #    target = target[0]
        # if len(target) == 0:
        #    target = False

    return target

def read_args(usr_args={}):
    '''
    Read arguments and corresponding values storing them in a 
    dictionary

    DESCRIPTION
    -----------
    Arguments should be specified in the form arg=value or arg:value or
    simply arg.
    If no value is specified, the value of that argument will be True.
    Even if no argument is specified, there will always an argument 
    called "name" containing the name of the script itself

    PARAMETERS
    ----------
    usr_args: dictionary
        Dictionary of default arguments and corresponding values
        specified by the user

    RETURNS
    -------
    arg_dict: dictionary
        Dictionary containing argument:value

    HISTORY
    -------
    2021 10 13, Stefano Rapisarda (Uppsala), creation date
    2021 11 03, Stefano Rapisarda (Uppsala)
        Bug corrected, when you where writing True or False 
        the argument was interpreted as a string
    2021 12 10 Stefano Rapisarda (Uppsala)
        Bug fixed, it was trying to capitalize list
    '''
    args=sys.argv

    arg_dict = usr_args
    for i,arg in enumerate(args):
        if i == 0:
            div = ['name',arg]
        else:
            if ':' in arg:
                div = arg.split(':')
            elif '=' in arg:
                div = arg.split('=')
            else:
                div = [arg.strip(),True]
        if type(div[0]) == str: div[0]=div[0].strip()
        if type(div[1]) == str: 
            div[1]=div[1].strip()

            if ',' in div[1]:
                value = div[1].split(',')
            else:
                value = div[1]
        else:
            value = div[1]

        if type(value) == str:
            if value.upper() == 'TRUE': value = True
            if value.upper() == 'FALSE': value = False
 
        arg_dict[div[0]] = value

    return arg_dict




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

def chunks(in_obj,n_chunks):  
    '''
    This small function was designed for plotting purposes
    It splits a new list or an existing one in n_chunks

    PARAMETERS
    ----------
    in_obj: integer or list
        If in_obj is an interger, a list of integer between 0 and 
        in_obj-1 will be splitted in n_chunks.
        If in_obj is a list, the list will be splitted in n_chunks.
    n_chunks: integer
        Number of chunks

    HISTORY
    -------
    Stefano Rapisarda, 2019 06 21 (Shanghai), creation date
    '''

    if type(in_obj) == type([]) or type(in_obj) == type(np.array([])):
        n = len(in_obj)
        array = in_obj
    elif type(in_obj) == type(1):
        n = in_obj
        array = list(range(in_obj))
    else:
        logging.error('Wrong object type')
        return

    if n%n_chunks == 0:
        sub = [[array[i+n_chunks*j] for i in range(n_chunks)] for j in range(int(n/n_chunks))]
    else:
        sub = [[array[i+n_chunks*j] for i in range(n_chunks)] for j in range(int(n/n_chunks))]+\
              [[array[n+i-n%n_chunks] for i in range(n%n_chunks)]]

    return sub

def ask_opts(opts,title=''):
    '''
    Prints a series of option, allowing the user to select one. 

    PARAMETERS
    ----------
    opts: list
          List of string containing the different options
    title: string optional
          Text to print before printing the different options

    RETURNS
    -------
    index: integer
          positional index of the selected option
          (the first option corresponds to index 0)

    HISTORY
    -------
    2019 06 06, Stefano Rapisarda (SHAO), creation date

    '''
    # Function for selecting an option from the list
    # Listing the options
    if title:
        print(title)
    n = len(opts)
    for i in range(n):
        print('{:3}) {}'.format(i+1, opts[i]))
    sel = input('=============> ')
    sel = sel.strip()
    index = int(sel)-1
    return index

def to_terminal(cmds):
    '''
    Execute on terminal commands in cmds

    HISTORY
    -------
    Stefano Rapisarda 23 09 2020, Uppsala (Creation date)
    '''

    if isinstance(cmds,str): 
        os.system(cmds)
    elif isinstance(cmds,list):
        for cmd in cmds:
            os.system(cmd)

def yesno(prompt):
    '''
    Asks a question that requires a yes or a no

    PARAMETERS
    ----------
    prompt: string
            Question to be asked. (Yes/No) will be added at the end of the text

    RETURNS
    -------
    ans: logical
            True if yes, False if no

    HISTORY
    -------
    2017 09 20, Stefano Rapisarda (Amsterdam), creation date
    2019 04 22, Stefano Rapisarda (SHAO), cleaned up
    '''
    
    ctrflag = True
    while ctrflag:
        ans = input(prompt+'(Yes/No) ')
        if ans[0].lower() == 'y':
            ctrflag = not ctrflag
            ans = True
        elif ans[0].lower() == 'n':
            ctrflag = not ctrflag
            ans = False
        else:
            print('Wrong choice, try again...')
    return ans
