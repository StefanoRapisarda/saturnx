'''This module contains the definition of Lightcurve and LightcurveList 
classes

The Lightcurve object is a container of a binned lightcurve. Computing or
retrieving this object is a fundamental step for computing Fourier products
such us PowerSpectrum and CrossSpectrum
'''

import numpy as np
import pandas as pd 
import pickle
import pathlib
import copy
import logging
import matplotlib.pyplot as plt

from astropy.io.fits import getdata,getval
from astropy.io import fits

from saturnx.core.gti import Gti
from saturnx.core.event import Event, EventList
from saturnx.utils.time_series import rebin_arrays
from saturnx.utils.fits import read_fits_keys, get_basic_info
from saturnx.utils.generic import is_number, my_cdate, round_half_up

class Lightcurve(pd.DataFrame):
    '''
    Lightcurve object. It computes/stores binned events for a certain 
    range of energy

    ATTRIBUTES
    ----------
    low_en: float, string, or None
        Lower energy in keV
    high_en: float, string, or None
        Upper energy in keV
    tot_counts: float or None
        Total number of counts. If the counts column is empty or full
        of None, None is returned.
    count_std: float or None
        Standard deviation of counts. If the counts column is empty 
        or full of None, None is returned.
    cr: float or None
        Count rate computed as total counts divided by total exposure.
        If the counts column is empty or full of None, None is returned.
    cr_std: float or None
        Standard deviation of count rates. If the counts column is empty 
        or full of None, None is returned.
    texp: float or None
        Total exposure time computed as the time resolution times the
        number of time bins (this is equivalent to the time interval
        between the lower edge of the first time bin and the upper edge
        of the last time bin). If the time array is empty, None is 
        returned.
    tres: float or None
        Time resolution computed as the median of the difference of 
        consecutive time bins rounded up to the 12th order (to the 
        picosecond). If the time array is empty, None is returned.
    rms: float or None
        Root-Mean-Squared. If the count array is empty or full of None,
        None is returned.
    frac_rms: float
        Fractional RMS, computed as the squared root of the count variance
        divided by the count mean, squared. If the count array is empty 
        or full of None, None is returned.
    meta_data: dictionary
        Container of useful information, user notes (key NOTES), and 
        data reduction history (key HISTORY)

    METHODS
    -------
    __add__(self, other)
        - If other is a Lightcurve...
        The other Lightcurve must have the same number of bins and time
        resolution of the current one. The methods adds bin by bin the 
        counts of the two Lightcurves. If the tags of the time arrays
        are not the same, a new time array starting from tres/2
        will be defined. 
        !!! meta_data are copied from the current Lightcurve !!!
        - If other is a value...
        Adds the value to counts and returns the modified Lightcurve

    __sub__(self,other)
        Similar to __add__

    __mul__(self, value)
        - If value is a list, a numpy.ndarray, or a pandas.Series...
        Returns a LightcurveList where each Lightcurve bin counts are
        multiplied by the i-th array element
        - If value object is a number...
        Multiplies counts and value and returns the modified Lightcurve

    __truediv__(self, value)
        Divide counts by a value and returns the modified Lightcurve

    split(splitter=16)
        Splits the lightcurve according to a Gti or a time segment 
        returning a LightcurveList

    rebin(factors=1)
        Rebins the lightcurve linearly returning a rebinned lightcurve

    plot(norm=None,ax=None,cr=True,title=False,lfont=16,
         norm_ylabel='',zero_start=True,**kwargs)
        Plots either counts or count rates over time on a new or user
        defined axis

    from_event(events,time_res=1.,user_start_time=0,user_dur=None,
               low_en=0.,high_en=np.inf)
        Computes a Lightcurve according to the given time resolution, 
        start_time, duration, low energy and high energy [keV]. Lightcurve
        computed from a single Event object will return a single Lightcurve,
        computation from an EventList will return a LightcurveList

    read_fits(fits_file, ext='COUNTS', time_col='TIME', 
              count_col='COUNTS',rate_col='RATE', keys_to_read=None)
        Reads specified extension and columns of a FITS file and 
        returns a Lightcurve 

    to_fits(file_name='lightcurve.fits',fold=pathlib.Path.cwd())
        Saves Lightcurve in a FITS file

    save(file_name='lightcurve.pkl',fold=pathlib.Path.cwd())
        Saves Lightcurve in a pickle file

    load(fold=pathlib.Path.cwd())
        Loads Lightcurve from a pickle file

    HISTORY
    -------
    2020 04 ##, Stefano Rapisarda (Uppsala)
    2021 03 01, Stefano Rapisarda (Uppsala)
        Last day of a week of major updates
    2021 02 01, Stefano Rapisarda (Uppsala)
        Corrected bug about splitting with time segment equal or larger
        than the time resolution
    '''

    _metadata = ['_low_en','_high_en','low_en','high_en',
                 'meta_data','_meta_data']

    # >>> METHODS <<<

    def __init__(self, time_array = np.array([]), count_array = None,
                 low_en = None, high_en = None,rate_array = None,
                 meta_data = None):  
        '''
        PARAMETERS
        ----------
        time_array: numpy.ndarray, pandas.Series, or list (optional)
            Contains bin center time (default, empty numpy.ndarray)
        count_array: numpy.ndarray, pandas.Series, list, or None (optional)
            Contains counts per bin (default is None)
        rate_array: numpy.ndarray, pandas.Series, list, or None (optional)
            Contains count rate per bin (default is None)
        low_en: float or string (optional)
            Low energy in keV (default is 0)
        high_en: float or string (optional)
            High energy in keV (default is numpy.inf)
        meta_data: dictionary (optional)
            Contains useful information, including mission (default is {})   
        '''

        # Initialisation  
        column_dict = {'time':time_array,'counts':count_array,'rate':rate_array}
        super().__init__(column_dict)

        # Rate or count rate initialization
        # Assuming only one of the two is not None and preferring counts
        if len(time_array) != 0:
            if self.counts.any():
                self.rate = self.counts/self.tres
            elif np.all((self.counts == 0.)):
                print('WARNING: Lightcurve counts are all zero')
                self.rate = self.counts
            elif self.rate.any():
                self.counts = self.rate*self.tres
            elif np.all((self.rate == 0.)):
                print('WARNING: Lightcurve rates are all zero')
                self.counts = self.rate

        # Energy range
        self.low_en = low_en
        self.high_en = high_en

        self.meta_data = meta_data
        if not 'LC_CRE_DATE' in self.meta_data['HISTORY'].keys():
            self.meta_data['HISTORY']['LC_CRE_DATE'] = my_cdate()    

    def __add__(self, other):
        if isinstance(other,Lightcurve):
            # Checking n. of bins and time resolution
            if len(self) != len(other):
                raise ValueError('You cannot add Lightcurves having different number of bins')
            if self.tres != other.tres:
                raise ValueError('You cannot add Lightcurves with different time resolution')
        
            # Initialize time 
            if np.array_equal(self.time, other.time):
                time = self.time
            else:
                # Defining a new time ax 
                time = self.time - self.time.iloc[0] +self.tres/2

            # Initialize counts
            counts = self.counts + other.counts

            # Initialize energy bands
            # ---------------------------------------------------------
            if self.low_en and self.high_en and other.low_en and other.high_en:
                low_en = min([self.low_en,other.low_en])
                high_en = max([self.high_en,other.high_en])
            else:
                low_en = None
                high_en = None
            # ---------------------------------------------------------
        
        else:  
            if type(other) == str:
                if is_number(other): 
                    other=eval(other)
                else:
                    raise TypeError('Cannot add string to Lightcurve')
            time = self.time
            counts = self.counts + other
            low_en, high_en = self.low_en, self.high_en

        return Lightcurve(time_array = time, count_array = counts,
                          low_en = low_en, high_en = high_en,
                          meta_data = self.meta_data)

    def __sub__(self, other):
        if isinstance(other,Lightcurve):
            if len(self) != len(other):
                raise ValueError('You cannot subtract Lightcurves with different dimensions')
            if self.tres != other.tres:
                raise ValueError('You cannot subtract Lightcurves with different time resolution')
        
            # Initialize time 
            if np.array_equal(self.time, other.time):
                time = self.time
            else:
                # Defining a new time ax 
                time = self.time - self.time.iloc[0] +self.tres/2

            # Initialize counts
            counts = self.counts - other.counts

            # Initialize energy bands # implement with range
            # ---------------------------------------------------------
            if self.low_en and self.high_en and other.low_en and other.high_en:
                low_en = min([self.low_en,other.low_en])
                high_en = max([self.high_en,other.high_en])
            else:
                low_en = None
                high_en = None
            # ---------------------------------------------------------
        
        else:  
            if isinstance(other,str):
                if is_number(other): 
                    other=eval(other)
                else:
                    raise TypeError('Cannot subtract string to Lightcurve')
            time = self.time
            counts = self.counts - other
            low_en, high_en = self.low_en, self.high_en

        return Lightcurve(time_array = time, count_array = counts,
                          low_en = low_en, high_en = high_en,
                          meta_data = self.meta_data)

    def __mul__(self,value):
        
        if isinstance(value,(list,np.ndarray,pd.Series)):
            if len(value) == 0: return self
            
            lcs = []
            for item in value:
                if type(item) == str:
                    if is_number(item): 
                        value=eval(item)
                    else:
                        raise TypeError('Cannot multiply string to Lightcurve')                
                else:
                    try:
                        float(item)
                    except Exception:
                        raise TypeError('Array items must be numbers')

                counts = self.counts*item
                lcs += [
                    Lightcurve(time_array = self.time, count_array = counts,
                               low_en = self.low_en, high_en = self.high_en,
                               meta_data=self.meta_data)
                        ]
            
            return LightcurveList(lcs)

        else:
            if isinstance(value,str):
                if is_number(value): 
                    value=eval(value)
                else:
                    raise TypeError('Cannot multiply string to Lightcurve')                
            else:
                try:
                    float(value)
                except Exception:
                    raise TypeError('Value must be a number')            
            counts = self.counts*value    

            return Lightcurve(time_array = self.time,count_array = counts,
                              low_en = self.low_en,high_en = self.high_en,
                              meta_data=self.meta_data)

    def __rmul__(self,value):
        return self*value

    def __truediv__(self,value):
        if isinstance(value,str):
            if is_number(value): 
                value=eval(value)
            else:
                raise TypeError('Cannot divide Lightcurve by string')                
        else:
            try:
                float(value)
            except Exception:
                raise TypeError('Value must be a number')   

        if value == 0:
            raise ZeroDivisionError('Dude, you cannot divide by zero')
        counts = self.counts/value

        return Lightcurve(time_array = self.time, count_array = counts,
                          low_en = self.low_en, high_en = self.high_en,
                          meta_data=self.meta_data)  
   
    def split(self,splitter=16):
        '''
        Splits the Lightcurve into a LightcurveList according to a
        Gti object or a time segment

        In case the time segment is larger than the Ligtcurve time exposure
        or smaller than the Lightcurve time resolution, a LightcurveList
        containing only the full Lightcurve is returned.
        In case of splitting according to GTI, it is assumed that the
        GTI is included in the Lightcurve time array. If not, no Lightcurve
        is computed and there will be no Lightcurve with that specific
        GTI index in the final LightcurveList. This means that the largest 
        GTI index of a Lightcurve in the LightcurveList is not 
        representative of the number of GTIs in the Gtis object used to 
        split the Lightcurve. Viceversa, the number of GTIs in the Gti
        object used for the split will be not necesseraly equal to the
        number of Lightcurves in the final LightcurveList. Take that in
        mind in your analysis.

        PARAMETERS
        ----------
        time_seg: float, string, or saturnx.core.Gti (optional)
            Time segment (duration) or Gti object (default is 16)

        RETURNS
        -------
        LightcurveList
            Each lightcurve in the Lightcurve list will have duration
            corresponding to the time segment or to the i-th GTI 
            duration in Gti(). If time_seg > Lightcurve duration, a
            LightcurveList containing only the initial Lightcurve is 
            returned
        '''

        tmp_meta_data = copy.deepcopy(self.meta_data)

        if isinstance(splitter,Gti):
            print('===> Splitting Lightcurve according to GTI')

            # In case Lightcurve has already been split according to GTI
            splitting_keys = [key for key in tmp_meta_data['HISTORY'].keys() if 'SPLITTING_GTI' in key]
            if len(splitting_keys) == 0:
                suffix = ''
            else:
                suffix = '_{}'.format(len(splitting_keys))  

            gti = splitter

            tmp_meta_data['HISTORY']['SPLITTING_GTI{}'.format(suffix)] = my_cdate()
            tmp_meta_data['N_GTIS{}'.format(suffix)] = len(gti)

            lcs = []
            for gti_index,(start,stop) in enumerate(zip(gti.start,gti.stop)):
                mask = (self.time>= start) & (self.time<stop)

                if stop > self.time.iloc[-1]:
                    print(f'WARNING: GTI {gti_index}(+1) stop is larger than the last Lightcurve bin')
                if start < self.time.iloc[0]:
                    print(f'WARNING: GTI {gti_index}(+1) start is smaller than the first Lightcurve bin')

                if np.sum(mask) != 0:
                    time=self.time[mask]
                    meta_data_gti = copy.deepcopy(tmp_meta_data)
                    meta_data_gti['GTI_INDEX{}'.format(suffix)] = gti_index
                    counts = self.counts[mask]
                    lc = Lightcurve(time_array = time, count_array = counts,
                                    low_en = self.low_en, high_en = self.high_en,
                                    meta_data = meta_data_gti)
                    lcs += [lc]
                else:
                    print(f'WARNING: GTI {gti_index}(+1) does not contain data')

        else:
            print('===> Splitting Lightcurve according to time segment')

            splitting_keys = [key for key in tmp_meta_data['HISTORY'].keys() if 'SPLITTING_SEG' in key]
            if len(splitting_keys) == 0:
                suffix = ''
            else:
                suffix = '_{}'.format(len(splitting_keys))  

            time_seg = splitter

            if isinstance(time_seg,str):
                if is_number(time_seg): 
                    time_seg=eval(time_seg)
                else:
                    raise TypeError('Time segment must be a number')                
            else:
                try:
                    float(time_seg)
                except Exception:
                    raise TypeError('Time segment must be a number')   

            if time_seg == 0:
                raise ValueError('Time segment cannot be zero')

            if time_seg >= self.texp:
                print('Lightcurve duration is less or equal than the specified segment ({} < {})'.\
                    format(self.texp,time_seg))
                print('Returning original Lightcurve')
                return LightcurveList([self])

            if time_seg <= self.tres:
                print('Lightcurve time resolution is larger or equal than the specified segment ({} > {})'.\
                    format(self.tres,time_seg))
                print('Returning original Lightcurve')
                return LightcurveList([self])                

            seg_bins = int(time_seg/self.tres)
            n_segs = int(len(self)/seg_bins)           
            #n_segs = int(self.texp/time_seg)

            tmp_meta_data['HISTORY']['SPLITTING_SEG{}'.format(suffix)] = my_cdate()
            tmp_meta_data['SEG_DUR{}'.format(suffix)] = time_seg
            tmp_meta_data['N_SEGS{}'.format(suffix)] = n_segs

            indices = [i*seg_bins for i in range(1,n_segs+1)]
            # np.split performs faster then ad hoc loop
            # The last interval goes from last index to the end of the
            # original array, so it is excluded
            # !!! Time intervals must be contigous to use this!!! 
            time_array = np.split(self.time.to_numpy(),indices)[:-1]
            count_array = np.split(self.counts.to_numpy(),indices)[:-1]

            lcs = []
            for seg_index,(time,counts) in enumerate(zip(time_array,count_array)):
                seg_meta_data = copy.deepcopy(tmp_meta_data)
                seg_meta_data['SEG_INDEX{}'.format(suffix)] = seg_index
                lc = Lightcurve(time_array = time, count_array = counts,
                                low_en = self.low_en, high_en = self.high_en,
                                meta_data = seg_meta_data)
                lcs += [lc]

        return LightcurveList(lcs)

    def rebin(self,factors=1):
        '''
        Linearly rebins the Lightcurve

        PARAMETERS
        ----------
        factors: float, string, list, np.ndarray, or pd.Series (optional)
            Rebinning factor. A value of 2 will increase the time 
            resolution to 2*tres. If factors is a list, rebinning is 
            perfomed to the same Lightcurve multiple times, each time
            with the i-th rebinning factor in the list (default is 1)

        RETURNS
        -------
        saturnx.core.Lightcurve
            Rebinned lightcurve
        '''

        # Checking input
        if not isinstance(factors,(list,np.ndarray,pd.Series)): 
            if isinstance(factors,str):
                if is_number(factors): 
                    factors=eval(factors)
                else:
                    raise TypeError('Rebin factor must be a number')                
            else:
                try:
                    float(factors)
                except Exception:
                    raise TypeError('Rebin factor must be a number')         
            if factors <= 0:
                raise ValueError('Rebin factor cannot be negative')  

            new_factors = [factors] 
        else:
            new_factors = []
            for f in factors:
                if isinstance(f,str):
                    if is_number(f): 
                        f=eval(f)
                    else:
                        raise TypeError('Rebin factor must be a number')                
                else:
                    try:
                        float(f)
                    except Exception:
                        raise TypeError('Rebin factor must be a number')         
                if f <= 0:
                    raise ValueError('Rebin factor cannot be negative') 
                new_factors += [f]                           

        # Initialising meta_data
        meta_data = copy.deepcopy(self.meta_data)
        meta_data['HISTORY']['REBINNING'] = my_cdate()
        meta_data['REBIN_FACTOR'] = new_factors #list

        # Rebinning
        binned_counts = self.counts.to_numpy()
        binned_time = self.time.to_numpy()
        for f in new_factors:
            if f == 1:
                if len(new_factors) == 1:
                    binned_counts = [binned_counts]
            else:  
                binned_time, binned_counts= rebin_arrays(
                    binned_time,rf=f,tres = self.tres,
                    arrays=binned_counts,bin_mode=['sum'],exps=[1])
                print('===>',len(binned_counts),len(binned_time))  

        lc = Lightcurve(time_array = binned_time, count_array = binned_counts[0],
                        low_en = self.low_en, high_en = self.high_en,
                        meta_data = meta_data)
        return lc

    def plot(self,norm=None,ax=None,cr=True,title=False,lfont=16,
             norm_ylabel='',zero_start=True,**kwargs):
        '''
        Plots lightcurve on a new or user defined matplotlib.axes.Axes

        PARAMETERS
        ----------
        norm: value, string, or None (optional)
            If not None, counts or count rates are divided by norm
            (default is None)
        ax: matplotlib.pyplot.axes.Axes or False (optional)
            If not False, Lightcurve will be plotted on specified axis,
            otherwise a new ax will be initialized (default is None)
        cr: Boolean (optional)
            If True, count rates per time bins will be plotted. If False
            counts per time bins will be plotted (default is True)
        title: string or False (optional)
            Title of the plot (defalt is False)
        lfont: int (optional)
            Font of x and y label (default is 16)
        norm_ylabel: str (optional)
            This label is used in case a normalization is specified.
            If empty (default), the value of the normalization will be
            added to the y label
        zero_start: boolean (optional)
            If True (default) the time axis will start from zero, 
            otherwise the original time tag is used. This option is 
            intended to be used (setting it to False) when comparing
            lightcurves from different files, but with time tag 
            referring to the same coordinate system (e.g. solar system 
            barycenter)
        kwargs: dictionary
            Dictionary of keyword arguments for plot()
        '''

        if not 'marker' in kwargs.keys(): kwargs['marker']='o'
        if not 'color' in kwargs.keys(): kwargs['color']='k'

        if ax is None:
            fig, ax = plt.subplots(figsize=(12,6))

        if (title is not False) and (ax is not False):
            ax.set_title(title)

        start = self.time.iloc[0]
        
        # Defining y axis and label
        y = self.counts
        ylabel = 'Counts'
        if cr: 
            y = self.rate
            ylabel = 'Count rate [c/s]'
        if norm is not None:
            if type(norm) == str: 
                norm = eval(norm)
            if norm == 0.:
                raise ValueError('Cannot divide y axis by 0')
            y /= norm
            if norm_ylabel == '':
                ylabel = 'Count rate / {} [c/s]'.format(norm)
            else:
                ylabel = norm_ylabel

        # Defining x axis and label
        x = self.time
        xlabel = 'Time [s]'
        if zero_start: 
            x -=start
            xlabel = 'Time [{} s]'.format(start)

        # Plotting
        if 'label' in kwargs.keys():
            print('WARNING! label argument specified, overwriting default option (energy range)')
            ax.plot(x,y,**kwargs)
        else:
            ax.plot(x,y,label='{}_{}'.format(self.low_en,self.high_en),**kwargs)

        ax.set_xlabel(xlabel,fontsize=lfont)
        ax.set_ylabel(ylabel,fontsize=lfont)
        ax.grid(b=True, which='major', color='grey', linestyle='-')
        ax.legend(title='[keV]')

    @classmethod
    def from_event(cls,events,time_res=1.,user_start_time=0,user_dur=None,
        low_en=0.,high_en=np.inf):
        '''
        Computes a binned lightcurve from an Event object

        PARAMETERS
        ----------
        events: saturnx.core.Event
            Event object (must contain time and pi columns)
        time_res: float or string (optional)
            Time resolution of the binned lightcurve (default is 1)
        user_start_time: float or string (optional)
            User defined start time. If None, first event time will be
            considered (default is None)
        user_dur: float or string (optional)
            User defined duration. If None, interval between max and min
            event time will be considered (default is None)
        low_en: float or string (optional)
            Lower energy [keV] (default is 0). Only events with energy
            larger than this will be considered
        high_en: float or string (optional)
            Upper energy [keV] (default is np.inf). Only events with 
            energy lower than this will be considered
        
        RETURNS
        -------
        saturnx.core.Lightcurve
        '''

        if isinstance(time_res,str): time_res = eval(time_res)
        if isinstance(user_start_time,str): user_start_time = eval(user_start_time)
        if isinstance(user_dur,str): user_dur = eval(user_dur)

        if isinstance(low_en,str): low_en = float(low_en)
        if isinstance(high_en,str): high_en = float(high_en)
    
        if not isinstance(events,(Event,EventList)):
            raise TypeError('Input must be an Event or an EventList object')

        event_flag = False
        if isinstance(events,Event):
            event_flag = True
            event_list = EventList([events])
        else:
            event_list = events

        lc_list = []
        for ev in event_list:

            meta_data = {}
            if len(event_list) == 1:
                meta_data['LC_CRE_MODE'] = 'Lightcurve computed from Event object'
            else:
                meta_data['LC_CRE_MODE'] = 'Lightcurve computed from EventList object'
            # Copying some info from the event file
            keys_to_copy = ['EVT_FILE_NAME','DIR','MISSION','INFO_FROM_HEADER',
                            'GTI_SPLITTING','N_GTIS','GTI_INDEX',
                            'SEG_SPLITTING','N_SEGS','SEG_INDEX',
                            'N_ACT_DET','INACT_DET_LIST',
                            'FILTERING','FILT_EXPR']
            for key in keys_to_copy:
                if key in ev.meta_data.keys():
                    meta_data[key] = ev.meta_data[key]

            local_user_start_time = user_start_time
            local_user_dur = user_dur
            if local_user_start_time <= np.min(ev.time):
                local_user_start_time = np.min(ev.time)
            if local_user_dur is None: 
                local_user_dur = np.max(ev.time)-local_user_start_time
            else:
                if local_user_dur == 0:
                    raise ValueError('Lightcurve duration cannot be zero')
                elif local_user_dur > np.max(ev.time)-local_user_start_time:
                    logging.warning('Lightcurve duration larger than Event duration')
        
            # The following may look messy but it is to ensure that the 
            # lightcurve time resolution is the one specified by the user.
            # Events partially covered by a bin are excluded
            n_bins = int(local_user_dur/time_res)
            # print('n_bins',local_user_dur,time_res,n_bins)
        
            start_time = local_user_start_time
            stop_time = start_time + n_bins*time_res

            # In this way the resolution is exactly the one specified by the user
            time_bins_edges = np.linspace(
                start_time-time_res/2.,stop_time+time_res/2.,n_bins+1,dtype=np.double
                )
            time_bins_center = np.linspace(
                start_time,stop_time,n_bins,dtype=np.double
                )

            # Conversion FROM energy TO channels
            mission = ev.meta_data['MISSION']
            if mission == 'NICER': 
                factor=100.
            elif mission == 'SWIFT':
                factor=100.
            else: 
                factor=1.
            low_ch = low_en*factor
            high_ch = high_en*factor

            # Selecting events according to energy
            mask = (ev.pi >= low_ch) & (ev.pi < high_ch)
            filt_time = ev.time[mask]
                    
            # Computing lightcurve
            counts,dummy = np.histogram(filt_time, bins=time_bins_edges)

            lc = cls(time_array=time_bins_center, count_array=counts,
                     low_en = low_en, high_en = high_en,
                     meta_data = meta_data)
            lc_list += [lc]

        if len(lc_list) == 0:
            return cls()
        elif event_flag:
            return lc_list[0]
        else:
            return LightcurveList(lc_list)

    @classmethod
    def read_fits(cls,fits_file, ext='COUNTS', time_col='TIME', 
                  count_col='COUNTS',rate_col='RATE', keys_to_read=None):
        '''
        Reads lightcurve from FITS file 

        PARAMETERS
        ----------
        ext: string (optional)
            FITS extension to read (default is COUNTS)
        time_col: string (optional)
            time column (default is TIME)
        count_col: string (optional)
            count column (default is COUNTS)
        rate_col: string (optional)
            rate column (default is RATE)
        keys_to_read: string, list, or None (optional)
            keys to be read from the header of the selected extension
            (default is None). The program will anyway try to read 
            certain keys (see get_basic_info)

        RETURNS
        -------
        saturnx.core.Lightcurve
        '''

        if not isinstance(fits_file,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        if isinstance(fits_file,str):
            fits_file = pathlib.Path(fits_file)
        if fits_file.suffix == '':
            fits_file = fits_file.with_suffix('.fits')

        if not fits_file.is_file():
            raise FileNotFoundError('FITS file does not exist')
        
        mission = None
        try:
            mission = getval(fits_file,'TELESCOP',ext)
        except Exception as e:
            print('WARNING: TELESCOP not found while reading Lightcurve from fits')
            try: 
                mission = getval(fits_file,'MISSION',ext) 
            except Exception as e:
                print('WARNING: MISSION not found while reading Lightcurve from fits')
                print('mission will be set to None')  
        try:
            low_en = getval(fits_file,'LOW_EN',ext)
        except:
            low_en = None
        try:
            high_en = getval(fits_file,'HIGH_EN',ext)
        except:
            high_en = None        

        meta_data = {}

        meta_data['LC_CRE_MODE'] = 'Lightcurve read from fits file'
        meta_data['FILE_NAME'] = str(fits_file.name)
        meta_data['DIR'] = str(fits_file.parent)
        meta_data['MISSION'] = mission

        # Reading meaningfull information from event file
        info = get_basic_info(fits_file,ext=ext)
        if keys_to_read is not None:
            if isinstance(keys_to_read,(str,list)): 
                user_info = read_fits_keys(fits_file,keys_to_read,ext)
            else:
                raise TypeError('keys to read must be str or list')
        else: 
            user_info = {}
        total_info = {**info,**user_info}
        meta_data['INFO_FROM_HEADER'] = total_info

        history = {}
        notes = {}
        # Reading the rest of header keywords
        with fits.open(fits_file) as hdu_list:
            header = hdu_list[ext].header
            keys = header.keys()
            for key in keys:
                if 'NOTE' in key: 
                    notes[key.split('_')[1]] = header[key]
                if 'HIST' in key:
                    history[key.split('_')[1]] = header[key]
        meta_data['NOTES'] = notes
        meta_data['HISTORY'] = history

        data = getdata(fits_file,extname=ext,meta_data=False,memmap=True)

        time = data[time_col]
        counts = None
        rate = None
        if count_col in data.columns.names:
            counts = data[count_col]
            print('Reading COUNT column')
        elif count_col.lower().capitalize() in data.columns.names:
            counts = data[count_col.lower().capitalize()]
            print('Reading Count column')            
        elif rate_col in data.columns.names:
            print('Reading RATE column')
            rate = data[rate_col]
            #print(rate)
        elif rate_col.lower().capitalize() in data.columns.names:
            rate = data[rate_col.lower().capitalize()]
            print('Reading Rate column')  
        
        return cls(time_array=time,count_array=counts,rate_array=rate,
                   low_en=low_en,high_en=high_en,
                   meta_data = meta_data)

    def to_fits(self,file_name='lightcurve.fits',fold=pathlib.Path.cwd()):

        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        if isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.fits')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        elif not isinstance(fold,pathlib.Path):
            raise TypeError('fold name must be either a string or a path')
        
        file_name = fold / file_name

        col1 = fits.Column(name='TIME', format='D',array=self.time.to_numpy())
        col2 = fits.Column(name='COUNTS', format='D',array=self.counts.to_numpy()) 
        col3 = fits.Column(name='RATE', format='D',array=self.rate.to_numpy()) 
        hdu = fits.BinTableHDU.from_columns([col1,col2,col3])
        hdu.name = 'LIGHTCURVE'

        for key,item in self.meta_data.items():
            if key == 'INFO_FROM_HEADER':
                for sub_key,sub_item in item.items():
                    hdu.header[sub_key] = sub_item
            elif key not in ['NOTES','HISTORY']:
                hdu.header[key] = item

        hdu.header['LOW_EN'] = self.low_en
        hdu.header['HIGH_EN'] = self.high_en

        for key,item in self.meta_data['HISTORY'].items():
            new_key = 'HIST_'+key
            hdu.header[new_key] = item

        for key,item in self.meta_data['NOTES'].items():
            new_key = 'NOTE_'+key
            hdu.header[new_key] = item

        phdu = fits.PrimaryHDU()
        hdu_list = fits.HDUList([phdu,hdu])
        hdu_list.writeto(file_name)

    def save(self,file_name='lightcurve.pkl',fold=pathlib.Path.cwd()):

        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        if isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        if not isinstance(fold,pathlib.Path):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name
        
        try:
            self.to_pickle(file_name)
            print('Lightcurve saved in {}'.format(file_name))
        except Exception as e:
            print(e)
            print('Could not save Lightcurve')

    @staticmethod
    def load(file_name,fold=pathlib.Path.cwd()):

        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        elif isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        if not isinstance(fold,pathlib.Path):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name

        if not file_name.is_file():
            raise FileNotFoundError(f'{file_name} not found'.format(file_name))
        
        lc = pd.read_pickle(file_name)
        
        return lc

    # >>> ATTRIBUTES <<<
        
    @property
    def tot_counts(self):
        if self.counts.empty or self.counts.isnull().all(): 
            return None
        if len(self.counts) == 0:
            return 0.
        else:
            return self.counts.sum()
       
    @property
    def cr(self):
        if self.counts.empty or self.counts.isnull().all(): 
            return None
        if len(self.counts) == 0:
            return 0.
        else:
            return self.tot_counts/self.texp

    @property
    def count_std(self):
        if self.counts.empty or self.counts.isnull().all():
            return None
        if len(self.counts) == 0:
            return 0.
        else:
            return self.counts.std()

    @property
    def cr_std(self):
        if self.rate.empty or self.rate.isnull().all():
            return None
        if len(self.rate) == 0:
            return 0.
        else:
            return self.rate.std()    

    @property
    def texp(self):
        if len(self.time) > 1:
            return len(self)*self.tres
        else:
            return None 

    @property
    def tres(self):
        # Computing tres if not specified
        if len(self.time) > 1:
            #tres = self.time.iloc[2]-self.time.iloc[1]
            tres = np.median(np.ediff1d(self.time))
            return round_half_up(tres,12)
        else:
            return None

    @property
    def rms(self):
        if self.counts.empty or self.counts.isnull().all(): 
            return None
        if len(self.counts) == 0:
            return 0.
        else:
            return np.sqrt(np.sum(self.counts**2)/len(self.counts))

    @property
    def frac_rms(self):
        if self.counts.empty or self.counts.isnull().all(): 
            return None
        if len(self.counts) == 0:
            return 0
        else:
            return np.sqrt(np.var(self.counts)/np.mean(self.counts)**2)

    @property
    def low_en(self):
        return self._low_en

    @low_en.setter
    def low_en(self,value):
        if value is not None:
            if isinstance(value,str): 
                value = eval(value)
            if value < 0: 
                value = 0
        self._low_en = value

    @property
    def high_en(self):
        return self._high_en

    @high_en.setter
    def high_en(self,value):
        if value is not None and isinstance(value,str): 
            value = eval(value)
        self._high_en = value

    @property
    def meta_data(self):
        return self._meta_data

    @meta_data.setter
    def meta_data(self,value):
        if value is None:
            self._meta_data = {}
        else:
            if not isinstance(value,dict):
                raise TypeError('meta_data must be a dictionary')
            self._meta_data = copy.deepcopy(value)

        if not 'HISTORY' in self.meta_data.keys():
            self._meta_data['HISTORY'] = {}            

        if not 'NOTES' in self.meta_data.keys():
            self._meta_data['NOTES'] = {}   

class LightcurveList(list):
    '''
    A list of Lightcurve objects with extra powers (and responsabilities)

    ATTRIBUTES
    ----------
    tot_counts: float
        Total counts of all the Lightcurves in the LightcurveList
    texp: float
        Sum of all the Lightcurve exposures in the LightcurveList
    
    METHODS
    -------
    comparison operators (<,>,>=,<=,==,!=)
        They return a LightcurveList object with items filtered according 
        to exposure time (texp)
    join(mask=None)
        Joins the elements of an LightcurveList into a single Lightcurve 
    fill_gaps()
        If Lightcurves in LightcurveList are not (time) contigous, it
        fills the time gap with new Lightcurves with zero counts
    split(splitter=16)
        Splits Lightcurves in LightcurveList according either to a Gti
        object or a time segment
    plot(self,norm=None,ax=False,cr=True,title=False,lfont=16,
         xbar=False,ybar=True,
         vlines=False,ndet=True,zero_start=True,**kwargs)
        Plots total number of counts (or the count rate) of each 
        Lightcurve versus the middle time of the lightcurve 
    compare(self,ax=False,mask=None,norm=None,step=None)
        Plots Lightcurve in a LightcurveList on top of each other for
        visual comparison
    info()
        Returns a pandas.DataFrame with basic information about 
        Lightcurves in the LightcurveList
    '''

    # >>> METHODS <<<

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.array([type(lc) == type(Lightcurve()) for lc in self]).all():
            raise TypeError('All the elements must be Lightcurve objects')

    def __setitem__(self, index, lc):
        if type(lc) != type(Lightcurve()):
            raise TypeError('The item must be a Lightcurve object')
        super(LightcurveList, self).__setitem__(index,lc)

    def __lt__(self,value):
        if isinstance(value,str): value = eval(value)
        return LightcurveList([l for l in self if l.texp < value])

    def __le__(self,value):
        if isinstance(value,str): value = eval(value)
        return LightcurveList([l for l in self if l.texp <= value]) 

    def __eq__(self,value):
        if isinstance(value,str): value = eval(value)
        return LightcurveList([l for l in self if l.texp == value])

    def __gt__(self,value):
        if isinstance(value,str): value = eval(value)
        return LightcurveList([l for l in self if l.texp > value])

    def __ge__(self,value):
        if isinstance(value,str): value = eval(value)
        return LightcurveList([l for l in self if l.texp >= value]) 

    def __ne__(self,value):
        if isinstance(value,str): value = eval(value)
        return LightcurveList([l for l in self if l.texp != value])                

    def join(self,mask=None):
        '''
        Joins lightcurve in a LightcurveList along the time axis
        '''

        if mask is None:
            mask = np.ones(len(self),dtype=bool)
        else:
            assert len(mask) == len(self),'Mask must have the same size of LightcurveList'
        
        df_list = []
        for i in range(len(self)):
            if mask[i]: 
                df_list += [pd.DataFrame(self[i])]

        first_valid_index = [i for i in range(len(self)) if mask[i]][0]

        cond1 = len(set([self[i].tres for i in range(len(self)) if mask[i]])) != 1
        if cond1:
            print([self[i].tres for i in range(len(self)) if mask[i]])
            raise ValueError('Cannot concatenate Lightcurves with different time res')
        
        df = pd.concat(df_list,ignore_index=True)

        meta_data = {}
        meta_data['LC_CRE_MODE'] = 'Lightcurve created joining Lightcurves from LightcurveList'
        meta_data['N_ORI_LCS'] = len(self)
        meta_data['N_MASKED_LCS'] = sum(mask)
        if 'MISSION' in self[first_valid_index].meta_data.keys():
            meta_data['MISSION'] = self[first_valid_index].meta_data['MISSION']
        
        return Lightcurve(time_array=df.time,count_array=df.counts,
                          low_en = self[first_valid_index].low_en, 
                          high_en = self[first_valid_index].high_en,
                          meta_data = meta_data)

    def fill_gaps(self):
        '''
        Fills potential gaps between lightcurve time arrays in a 
        LightcurveList with zeros
        '''
        tres_array = np.array([lc.tres for lc in self])
        if not np.all(tres_array==tres_array[0]):
            raise ValueError('Cannot fill gaps if Lightcurves have different time res')

        new_lcs = []
        for i in range(len(self)):
            if i == 0:
                new_lcs += [self[i]]
                continue
            
            lc = self[i]
            prev_lc = self[i-1]
            tres = lc.tres 

            if (lc.time.iloc[0]-prev_lc.time.iloc[-1]) >  tres:
                filler = np.arange(prev_lc.time.iloc[-1]+tres,lc.time.iloc[0],tres)
                filled_lc = Lightcurve(
                    time_array = filler,count_array = np.zeros(len(filler)),
                    low_en = self[0].low_en, high_en = self[0].high_en
                    )
                new_lcs += [filled_lc]
            new_lcs += [self[i]]

        return LightcurveList(new_lcs)

    def split(self,splitter=16):
        '''
        Splits lightcurves in a LightcurveList according to a segment
        duration ora a GTI
        '''

        if isinstance(splitter,Gti):
            gti = splitter
            lc_list = []
            for l in self:
                lc_list += [i for i in l.split(gti)]
            return LightcurveList(lc_list) 
        else:  
            time_seg = splitter
            if isinstance(time_seg,str): time_seg = eval(time_seg)        
            lc_list = []
            for l in self:
                lc_list += [i for i in l.split(time_seg)]
            return LightcurveList(lc_list)        

    def plot(self,norm=None,ax=False,cr=True,title=False,lfont=16,
        xbar=False,ybar=True,
        vlines=False,ndet=True,zero_start=True,**kwargs):
        '''
        PARAMETERS
        ----------
        zero_start: boolean (optional)
            If True (default) the time axis will start from zero, 
            otherwise the original time tag is used. This option is 
            intended to be used (setting it to False) when comparing
            lightcurves from different files, but with time tag 
            belonging to the same system (e.g. solar system barycenter)
        '''

        if not 'marker' in kwargs.keys(): kwargs['marker']='o'
        if not 'color' in kwargs.keys(): kwargs['color']='k'

        if ax is False:
            fig, ax = plt.subplots(figsize=(12,6))

        if (not title is False) and (not ax is False):
            ax.set_title(title)

        start = int(np.min([np.min(t.time) for t in self]))
        if not zero_start: start = 0
        for i in range(len(self)):
            
            y = self[i].tot_counts
            yerr = self[i].count_std
            label = 'Counts'
            if cr: 
                y = self[i].cr
                yerr = self[i].cr_std
                label = 'Count Rate [c/s]'

            if ndet:
                y /= float(self[i].meta_data['N_ACT_DET'])
                yerr /= float(self[i].meta_data['N_ACT_DET'])
                label = 'Count Rate per det [c/s/n_det]'
            if not norm is None:
                if type(norm) == str: norm = eval(norm)
                y /= norm  
                yerr /= norm
                label = label+'/norm'
            
            mid_time = (self[i].time.iloc[-1]+self[i].time.iloc[0])/2.          
            x = mid_time - start
            
            # This is for avoiding printing a label for each
            # Lightcurve
            if i > 0 and 'label'  in kwargs.keys():
                del kwargs['label']

            if ybar:
                capsize=4
                if type(ybar) == int:
                    capsize=ybar
                ax.errorbar(x,y,yerr=yerr,ecolor=kwargs['color'],capsize=capsize)

            if xbar:
                ax.plot([self[i].time.iloc[0]-start,self[i].time.iloc[-1]-start],[y,y],'k--')
            ax.plot(x,y,**kwargs)

            if i > 0 and vlines:
                xline = (self[i].time.iloc[0]+self[i-1].time.iloc[-1])/2. - start
                ax.axvline(xline,color='brown',ls='--',lw=2)

        xlabel = 'Time [s]'
        if zero_start:
            xlabel = 'Time [{} s]'.format(start)
        ax.set_xlabel(xlabel,fontsize=lfont)
        
        ax.set_ylabel(label,fontsize=lfont)
        ax.grid()

    def compare(self,ax=False,mask=None,norm=None,step=None):
        '''
        Plots lightcurves in a LightcurveList on top of each other with
        tunable spacing
        '''

        if norm is None:
            norm = 1
        else:
            if type(norm) == str: norm = eval(norm)
                 
        if mask is None:
            mask = np.ones(len(self),dtype=bool)
        else:
            assert len(mask) == len(self),'Mask must have the same size of LightcurveList'
        
        if ax is False:
            fig, ax =plt.subplots(figsize=(12,8))

        min_count = np.array([i.counts.min() for i in self]).min()
        max_count = np.array([i.counts.max() for i in self]).max()
        if step is None:
            step = (max_count-min_count)/len(self)
        else:
            if type(step) == str: step = eval(step)

        for i in range(len(self)):
            if mask[i]:
                start = self[i].time.iloc[0]
                ax.plot(self[i].time-start,self[i].counts/norm+i*step,label=f'{i}')
        ax.set_xlabel('Time [{} s]'.format(start))
        ax.set_ylabel('Counts')
        ax.legend()
        ax.grid()

    def info(self):
        '''
        Returns a pandas DataFrame relevand information for each Lightcurve
        object in the list
        '''

        columns = ['texp','tres','n_bins','counts','count_rate',
                    'rms','frac_rms',
                    'min_time','max_time',
                    'min_en','max_en','mission']
        info = pd.DataFrame(columns=columns)
        for i,lc in enumerate(self):
            line = {'texp':lc.texp,'tres':lc.tres,'n_bins':len(lc),
                    'counts':lc.tot_counts,'count_rate':lc.cr,
                    'rms':lc.rms,'frac_rms':lc.frac_rms,
                    'min_time':lc.time.iloc[0],'max_time':lc.time.iloc[-1],
                    'min_en':lc.low_en,'max_en':lc.high_en,
                    'mission':lc.meta_data['MISSION']}
            info.loc[i] = pd.Series(line)

        return info

    # >>> ATTRIBUTES <<<

    @property
    def tot_counts(self):
        if len(self) == 0:
            return 0 
        else:
            return np.sum([i.tot_counts for i in self])

    @property
    def texp(self):
        if len(self) == 0:
            return 0
        else:
            return np.sum([i.texp for i in self])

    @property
    def cr(self):
        if len(self) == 0:
            return 0
        else:
            if len(set([i.texp for i in self])) == 1:
                return np.mean([i.cr for i in self])
            else:
                #print([i.tot_counts for i in self])
                return np.sum([i.cr*len(i) for i in self])/np.sum([len(i) for i in self])

    @property
    def count_std(self):
        if len(self) == 0:
            return 0
        else:
            return np.std([lc.counts for lc in self])

    
    @property
    def cr_std(self):
        if len(self) == 0:
            return 0
        else:
            return np.std([lc.cr for lc in self])
    
    def mean(self,mask = None):
        '''
        Computes a Lightcurve having as counts the mean counts of all 
        the lightcurve in the list
        '''
        if mask is None:
            mask = np.ones(len(self),dtype=bool)
        else:
            assert len(mask) == len(self),'Mask must have the same size of LightcurveList'
        first_valid_index = [i for i in range(len(self)) if mask[i]][0]

        meta_data = {}
        meta_data['LC_CRE_MODE'] = 'Mean of Lightcurves in LightcurveList'
        meta_data['N_ORI_LCS'] = len(self)
        meta_data['N_MASKED_LCS'] = sum(mask)
        if 'MISSION' in self[first_valid_index].meta_data.keys():
            meta_data['MISSION'] = self[first_valid_index].meta_data['MISSION']

        if len(self) == 0:
            return 0
        else:
            if len(set([len(i) for i in self])) != 1:
                raise ValueError('Lightcurves have different dimensions')
            if len(set([i.tres for i in self])) != 1:
                raise ValueError('Lightcurves have different time resolution')      
            first_lc = self[first_valid_index]  
            time = first_lc.time - first_lc.time.iloc[0] + first_lc.tres/2.
            counts = np.vstack([self[i].counts.to_numpy() for i in range(len(self)) if mask[i]]).mean(axis=0)
            return Lightcurve(time_array = time, count_array = counts,
                low_en = first_lc.low_en, high_en = first_lc.high_en,
                meta_data = meta_data)

    def save(self,file_name='lightcurve_list.pkl',fold=pathlib.Path.cwd()):

        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        if isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        if not isinstance(fold,pathlib.Path):
            raise TypeError('fold name must be either a string or a path')

        if not str(fold) in str(file_name):
            file_name = fold / file_name
        
        try:
            with open(file_name,'wb') as output:
                pickle.dump(self,output)
            print('LightcurveList saved in {}'.format(file_name))
        except Exception as e:
            print(e)
            print('Could not save LightcurveList')

    @staticmethod
    def load(file_name,fold=pathlib.Path.cwd()):
        
        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        elif isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        if not isinstance(fold,pathlib.Path):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name

        if not file_name.is_file():
            raise FileNotFoundError(f'{file_name} not found'.format(file_name))    
        
        with open(file_name,'rb') as infile:
            lc_list = pickle.load(infile)
        
        return lc_list
    
