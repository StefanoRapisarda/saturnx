import numpy as np
from astropy.io import fits
from astropy.io.fits import getdata,getheader,getval
import pandas as pd
import os

from ..functions.my_functions import my_cdate, clean_gti

def read_gti(file_name,clean=True):
    '''
    Read GTI (start and stop time) from a fits file
    '''

    assert os.path.isfile(file_name),'file_name does not exist'
    mission =  getval(file_name,'telescop',1)

    history = {}

    if mission == 'NICER' or mission == 'SWIFT':

        history['CREATION_DATE'] = my_cdate()
        history['CREATION_MODE'] = 'Gti created from fits file'
        history['FILE_NAME'] = os.path.basename(file_name)
        history['DIR'] = os.path.dirname(file_name)

        data = getdata(file_name,extname='GTI',header=False,memmap=True)
        start,stop = data['START'],data['STOP']
            
    return Gti(start_array=start,stop_array=stop,history=history)  

def comp_gap(start,stop):
    '''
    Compute gaps between GTIs. Return an array with the same GTI
    dimension with 0 gaps assigned to the first gti
    '''

    gaps = np.array([0]+[start[i]-stop[i-1] for i in range(1,len(start))])
    return gaps

class Gti(pd.DataFrame):

    _metadata = ['notes','history']

    def __init__(self,start_array=None,stop_array=None,clean=True,
                notes={},history={}):
        if type(start_array) == list: start_array = np.array(start_array)
        if type(stop_array) == list: stop_array = np.array(stop_array)   

        self.notes = notes
        self.history = history

        if start_array is None or len(start_array)==0:
            super().__init__(columns=['start','stop','dur','gap'])
        else:
            assert len(start_array) == len(stop_array),'start and stop array have different dimentions'
            if clean:
                self.history['EVENT_CLEANING'] = my_cdate()
                self.history['N_GTI_ORI'] = len(start_array)
                start,stop = clean_gti(start_array,stop_array)
                self.history['N_GTI_CLEAN'] = len(start) 
            else:
                start = start_array
                stop = stop_array
            columns = {'start':start,'stop':stop,'dur':stop-start,'gap':comp_gap(start,stop)}
            super().__init__(columns)


    def __lt__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur < value
        history = self.history
        history['filtering'] = f'<{value}'
        return Gti(self.start[mask],self.stop[mask],history=history)

    def __le__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur <= value
        history = self.history
        history['filtering'] = f'<={value}'
        return Gti(self.start[mask],self.stop[mask],history=history)   

    def __eq__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur == value
        history = self.history
        history['filtering'] = f'=={value}'
        return Gti(self.start[mask],self.stop[mask],history=history)

    def __gt__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur > value
        history = self.history
        history['filtering'] = f'>{value}'
        return Gti(self.start[mask],self.stop[mask],history=history)

    def __ge__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur >= value
        history = self.history
        history['filtering'] = f'>={value}'
        return Gti(self.start[mask],self.stop[mask],history=history)      

    def __ne__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur != value
        history = self.history
        history['filtering'] = f'!={value}'
        return Gti(self.start[mask],self.stop[mask],history=history)  

class GtiList(list):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.array([isinstance(i,Gti) for i in self]).all():
            raise TypeError('All the elements must be Gti objects')


    def __setitem__(self, index, gti):
        if not isinstance(gti,Gti):
            raise TypeError('The item must be a Gti object')
        self[index] = gti

    def join(self):
        df = pd.concat(self,ignore_index=True)
        history = {}
        history[my_cdate()] = 'Gti created concatenating {} Gtis'.format(len(self))
        
        return Gti(df.start,df.stop,history=history)
