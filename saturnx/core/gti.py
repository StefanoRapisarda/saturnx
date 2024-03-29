"""This module contains the definition of Gti and GtiList classes and two
specific functions to clean GTIs and compute the gap between two 
conscutive GTIs.

The Gti object is a container for GTIs specified by start and stop time.
Gti objects are mainly used to determine the full duration of usable data,
to specify specific time intervals to select in an observation, and to
filter other data products (Event and Lightcurve) columns."""

__author__ = 'Stefano Rapisarda'

import os
import copy
import pathlib
import pandas as pd
import numpy as np
from astropy.io.fits import getdata,getval

from saturnx.utils.generic import my_cdate

def clean_gti(start,stop):
    '''
    Re-arrange GTIs in order of starting time and merges
    overlapping GTIs

    PARAMETERS
    ----------
    start: np.array or list
        Array of start times
    stop: np.array or list
        Array of stop times
    
    RETURNS
    -------
    clean_start: np.array or list
        Array of cleaned start times
    clean_stop: np.array or list
        Array of cleaned stop times
    '''

    start = np.asarray(start)
    stop  = np.asarray(stop)

    # Sorting arrays according to increasing starting time
    # !!! essential for this algorithm to work properly !!!
    sorting_indices = np.argsort(start)
    sorted_start = start[sorting_indices]
    sorted_stop = stop[sorting_indices]

    # Checking GTIs
    for t_start,t_stop in zip(sorted_start,sorted_stop):
        if t_start >= t_stop:
            print('GTO start times must always be smaller than stop times')
            raise ValueError

    # Cleaning GTIs
    clean_start = [sorted_start[0]]
    clean_stop = [sorted_stop[0]]

    flag=False
    for i in range(1,len(start)):
        
        if sorted_start[i] <= clean_stop[-1]:
            # Case A, overlapping GTIs
            flag = True
            if sorted_stop[i] <= clean_stop[-1]:
                # Case A1, new GTI included in the old one
                continue
            else:
                # Case A2 , GTI overlap ==> updating GTI stop
                clean_stop[-1] = sorted_stop[i]
        else:
            # Case B
            clean_start += [sorted_start[i]]
            clean_stop  += [sorted_stop[i]]

    if flag: print('Some of the GTIs were overlapping')

    #print(clean_start,clean_stop)

    return np.array(clean_start),np.array(clean_stop)  

def comp_gap(start,stop):
    '''
    Compute gaps between GTIs. Return an array with the same GTI
    dimension with 0 gaps assigned to the first gti
    '''

    gaps = np.array([0]+[start[i]-stop[i-1] for i in range(1,len(start))])
    return gaps

class Gti(pd.DataFrame):
    '''
    Gti object. It stores start and stop times of GTIs (Good Time 
    Intervals). The object extends a pandas.DataFrame

    ATTRIBUTES
    ----------
    meta_data: dictionary
        Container of useful information, user notes (key NOTES), and 
        data reduction history (key HISTORY)

    METHODS
    -------
    comparison operators (<,>,>=,<=,==,!=)
        They return a Gti objects with rows filtered according to 
        duration, i.e. the difference between stop and start
    read_fits(extename=None)
        Reads Gti from a FITS file
    save(file_name='gti.pkl',fold=pathlib.Path.cwd())
        Save Gti to a file via pickle
    load(file_name,fold=pathlib.Path.cwd())
        Read a pickled Gti file
    ''' 

    _metadata = ['meta_data','_meta_data']

    def __init__(self,start_array=None,stop_array=None,
                 clean=True,meta_data=None):
        if type(start_array) == list: start_array = np.array(start_array)
        if type(stop_array) == list: stop_array = np.array(stop_array)   

        self.meta_data = meta_data
        if not 'GTI_CRE_DATE' in self.meta_data['HISTORY'].keys():
            self.meta_data['HISTORY']['GTI_CRE_DATE'] = my_cdate()  

        if start_array is None or len(start_array)==0:
            super().__init__(columns=['start','stop','dur','gap'])
        else:
            if len(start_array) != len(stop_array):
                raise ValueError('start and stop arrays have different dimentions')
            if clean:
                self.meta_data['HISTORY']['GTI_CLEANING'] = my_cdate()
                self.meta_data['N_GTI_ORI'] = len(start_array)
                start,stop = clean_gti(start_array,stop_array)
                self.meta_data['N_GTI_CLEAN'] = len(start) 
            else:
                start = start_array
                stop = stop_array
            columns = {'start':start,'stop':stop,'dur':stop-start,'gap':comp_gap(start,stop)}
            super().__init__(columns)
 
    # Re-inplementing comparison operator to work on GTI duration

    def __lt__(self,value):
        if isinstance(value,str): 
            value = eval(value)
        mask = self.dur < value
        meta_data = self.meta_data
        meta_data['HISTORY']['GTI_FILTERING'] = my_cdate()
        meta_data['FILTERING_EXPR'] = f'<{value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)

    def __le__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur <= value
        meta_data = self.meta_data
        meta_data['HISTORY']['GTI_FILTERING'] = my_cdate()
        meta_data['FILTERING_EXPR'] = f'<={value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)   

    def __eq__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur == value
        meta_data = self.meta_data
        meta_data['HISTORY']['GTI_FILTERING'] = my_cdate()
        meta_data['FILTERING_EXPR'] = f'=={value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)

    def __gt__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur > value
        meta_data = self.meta_data
        meta_data['HISTORY']['GTI_FILTERING'] = my_cdate()
        meta_data['FILTERING_EXPR'] = f'>{value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)

    def __ge__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur >= value
        meta_data = self.meta_data
        meta_data['HISTORY']['GTI_FILTERING'] = my_cdate()
        meta_data['FILTERING_EXPR'] = f'>={value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)      

    def __ne__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur != value
        meta_data = self.meta_data
        meta_data['HISTORY']['GTI_FILTERING'] = my_cdate()
        meta_data['FILTERING_EXPR'] = f'!={value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)  

    @classmethod
    def read_fits(cls,file_name,extname=None):
        '''
        Read GTI (start and stop time) from a fits file
        '''

        if isinstance(file_name,str): file_name = pathlib.Path(file_name)

        mission =  getval(file_name,'telescop',1)

        meta_data = {}

        meta_data['HISTORY'] = {}
        meta_data['HISTORY']['CREATION_DATE'] = my_cdate()
        meta_data['CREATION_MODE'] = 'Gti created from fits file'
        meta_data['FILE_NAME'] = file_name.name
        meta_data['DIR'] = os.path.dirname(file_name)

        if extname is None:
            extname = 'GTI'
            if mission == 'NICER' or mission == 'SWIFT': extname='GTI'
            if mission == 'HXMT': extname='GTI0'
        
        data = getdata(file_name,extname=extname,header=False,memmap=True)
        start,stop = data['START'],data['STOP']
                
        return cls(start_array=start,stop_array=stop,meta_data=meta_data) 

    def save(self,file_name='gti.pkl',fold=pathlib.Path.cwd()):

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
            print('Gti saved in {}'.format(file_name))
        except Exception as e:
            print(e)
            print('Could not save Gti')

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
            raise FileNotFoundError(f'{file_name} not found')
        
        gti = pd.read_pickle(file_name)
        
        return gti

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

class GtiList(list):
    '''
    A list of Gti objects with extra powers (and responsabilities)

    METHODS
    -------
    join(mask=None)
        Joins the elements of an GtiList into a single Gti 
    '''

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.array([isinstance(i,Gti) for i in self]).all():
            raise TypeError('All the elements must be Gti objects')

    def __setitem__(self, index, gti):
        if not isinstance(gti,Gti):
            raise TypeError('The item must be a Gti object')
        self[index] = gti

    def join(self,mask=None):
        '''
        Joints Gtis in a GtiList into a single Gti

        The joining is performed using the pandas method concat

        PARAMETERS
        ----------
        mask: list or np.array
            Array of booleans to select Gtis in a GtiList to join
        '''

        if mask is None:
            mask = np.ones(len(self),dtype=bool)
        else:
            if not len(mask) == len(self):
                print('Mask must have the same size of GtiList')
                raise IndexError
        
        df_list = []
        for i in range(len(self)):
            if mask[i]: 
                df_list += [self[i]]

        df = pd.concat(df_list,ignore_index=True)

        meta_data = {}
        meta_data['GTI_CRE_MODE'] = 'Gti created joining Gtis from GtiList'
        # I specify GTILIST to separate this from N_GTI_ORI in Gti.__init__()
        meta_data['N_GTI_ORI_GTILIST'] = len(self)
        meta_data['N_MASKED_GTIS'] = sum(mask)
        
        return Gti(df.start,df.stop,meta_data=meta_data)
