import numpy as np
import pathlib
from astropy.io import fits
from astropy.io.fits import getdata,getval
import pandas as pd
import os

from saturnx.utils.generic import my_cdate

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

def comp_gap(start,stop):
    '''
    Compute gaps between GTIs. Return an array with the same GTI
    dimension with 0 gaps assigned to the first gti
    '''

    gaps = np.array([0]+[start[i]-stop[i-1] for i in range(1,len(start))])
    return gaps

class Gti(pd.DataFrame):

    _metadata = ['notes','meta_data']

    def __init__(self,start_array=None,stop_array=None,clean=True,
                notes={},meta_data={}):
        if type(start_array) == list: start_array = np.array(start_array)
        if type(stop_array) == list: stop_array = np.array(stop_array)   

        self.notes = notes
        self.meta_data = meta_data

        if start_array is None or len(start_array)==0:
            super().__init__(columns=['start','stop','dur','gap'])
        else:
            assert len(start_array) == len(stop_array),'start and stop array have different dimentions'
            if clean:
                self.meta_data['EVENT_CLEANING'] = my_cdate()
                self.meta_data['N_GTI_ORI'] = len(start_array)
                start,stop = clean_gti(start_array,stop_array)
                self.meta_data['N_GTI_CLEAN'] = len(start) 
            else:
                start = start_array
                stop = stop_array
            columns = {'start':start,'stop':stop,'dur':stop-start,'gap':comp_gap(start,stop)}
            super().__init__(columns)


    def __lt__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur < value
        meta_data = self.meta_data
        meta_data['filtering'] = f'<{value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)

    def __le__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur <= value
        meta_data = self.meta_data
        meta_data['filtering'] = f'<={value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)   

    def __eq__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur == value
        meta_data = self.meta_data
        meta_data['filtering'] = f'=={value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)

    def __gt__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur > value
        meta_data = self.meta_data
        meta_data['filtering'] = f'>{value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)

    def __ge__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur >= value
        meta_data = self.meta_data
        meta_data['filtering'] = f'>={value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)      

    def __ne__(self,value):
        if isinstance(value,str): value = eval(value)
        mask = self.dur != value
        meta_data = self.meta_data
        meta_data['filtering'] = f'!={value}'
        return Gti(self.start[mask],self.stop[mask],meta_data=meta_data)  

    @staticmethod
    def read_fits(file_name,extname=None):
        '''
        Read GTI (start and stop time) from a fits file
        '''

        assert os.path.isfile(file_name),'file_name does not exist'
        mission =  getval(file_name,'telescop',1)

        meta_data = {}

        meta_data['CREATION_DATE'] = my_cdate()
        meta_data['CREATION_MODE'] = 'Gti created from fits file'
        meta_data['FILE_NAME'] = os.path.basename(file_name)
        meta_data['DIR'] = os.path.dirname(file_name)

        if extname is None:
            extname = 'GTI'
            if mission == 'NICER' or mission == 'SWIFT': extname='GTI'
            if mission == 'HXMT': extname='GTI0'
        
        data = getdata(file_name,extname=extname,header=False,memmap=True)
        start,stop = data['START'],data['STOP']
                
        return Gti(start_array=start,stop_array=stop,meta_data=meta_data) 

    def save(self,file_name='gti.pkl',fold=pathlib.Path.cwd()):

        if not type(file_name) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        if type(file_name) == str:
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if type(fold) == str:
            fold = pathlib.Path(fold)
        if type(fold) != type(pathlib.Path.cwd()):
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

        if not type(file_name) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        elif type(file_name) == str:
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if type(fold) == str:
            fold = pathlib.Path(fold)
        if type(fold) != type(pathlib.Path.cwd()):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name

        if not file_name.is_file():
            raise FileNotFoundError(f'{file_name} not found')
        
        gti = pd.read_pickle(file_name)
        
        return gti

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
        meta_data = {}
        meta_data[my_cdate()] = 'Gti created concatenating {} Gtis'.format(len(self))
        
        return Gti(df.start,df.stop,meta_data=meta_data)
