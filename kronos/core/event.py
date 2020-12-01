import os

import pandas as pd
import numpy as np

import time
from astropy.io.fits import getdata,getheader,getval

from ..functions.my_functions import my_cdate

def read_event(file_name):
    '''
    Read a fits file and store meaningfull information in an Event object
    '''
    
    assert os.path.isfile(file_name),'file_name does not exist'
    mission =  getval(file_name,'telescop',1)

    history={}

    data = getdata(file_name,extname='EVENTS',header=False,memmap=True)
    now = time.ctime()

    history['CREATION_DATE'] = my_cdate()
    history['CREATION_MODE'] = 'Event created from fits file'
    history['FILE_NAME'] = os.path.basename(file_name)
    history['DIR'] = os.path.dirname(file_name)

    if mission == 'NICER':
        times = data['TIME']
        try:
            det_id = data['DET_ID']
        except Exception as e:
            print('Could not find DET_ID column')
            print(e)
            det_id = np.zeros(len(times))
        times = data['TIME']
        try:
            pi = data['PI']
        except Exception as e:
            print('Could not find PI column')
            print(e)
            pi = np.zeros(len(times))+200

        events = Event(time_array=times,det_array=det_id,pi_array=pi,
                       mission_name=mission,history=history)
    elif mission == 'SWIFT':
        events = Event(time_array=data['TIME'],detx_array=data['DETX'],dety_array=data['DETY'],
                       pi_array=data['PI'],grade_array=data['GRADE'],
                       mission_name=mission,history=history)   

    return events




class Event(pd.DataFrame):

    _metadata = ['_mission','notes','history']

    def __init__(self,time_array=None,pi_array=None,det_array=None,
                detx_array=None,dety_array=None,grade_array=None,
                mission_name=None,notes=None,history=None):
        
        if time_array is None:
            if mission_name == 'NICER':
                super().__init__(columns=['time','pi','det'])
            elif mission_name == 'SWIFT':
                super().__init__(columns=['time','pi','detx','dety','grade'])
            else:
                super().__init__(columns=['time','pi'])
        else:
            if mission_name == 'NICER':
                columns = {'time':time_array,
                        'pi':pi_array,
                        'det':det_array}
                super().__init__(columns)
            elif mission_name == 'SWIFT':
                columns = {'time':time_array,
                        'pi':pi_array,
                        'detx':detx_array,
                        'dety':dety_array,
                        'grade':grade_array}
                super().__init__(columns)   
            else:
                columns = {'time':time_array,
                        'pi':pi_array}
                super().__init__(columns)                             

        self._mission = mission_name

        if notes is None:
            self.notes = {}
        else: self.notes = notes
        if history is None:
            self.history = {}
        else: self.history = history

    def filter(self,expr):
        for col in list(self.columns):
            expr = expr.replace(col,'self.{}'.format(col))
        #print(expr)
        mask = eval(expr)
        kwargs = {}
        for col in list(self.columns):
            kwargs['{}_array'.format(col)] = self[col][mask]
        events = Event(**kwargs)

        filter_index_list = [-1]
        for key in self.history.keys():
            if 'FILTERING' in key:
                filter_index_list = [int(key.split('_')[1])]
        filter_index = max(filter_index_list)+1

        history = self.history.copy()
        history['FILTERING_{}'.format(filter_index)] = my_cdate()
        history['FILTER_EXPR'] = expr

        events = Event(mission_name=self.mission,history=history,**kwargs)

        return events        

    def split(self,gti):

        assert 'kronos.core.gti.Gti' in str(gti.__class__),'Events can be split only according to GTI'
        
        history = self.history.copy()    
        history['GTI_SPLITTING'] = my_cdate()
        history['N_GTIS'] = len(gti)

        events = []
        gti_index = 0
        for start,stop in zip(gti.start,gti.stop):
            mask = (self.time>= start) & (self.time<stop)
            kwargs = {}
            for col in list(self.columns):
                kwargs['{}_array'.format(col)] = self[col][mask]
            history_gti = history.copy()
            history_gti['GTI_INDEX'] = gti_index            
            
            events += [Event(mission_name=self.mission,history=history_gti,**kwargs)]
            gti_index += 1

        return EventList(events)


    @property
    def mission(self):
        return self._mission

    @mission.setter
    def mission(self, name):
        self._mission = name

    @property
    def texp(self):
        if len(self.time) == 0:
            return None
        else:
            return self.time.iloc[-1]-self.time.iloc[0]

    @property
    def info(self):
        info_dic = {}
        info_dic['N_EVENTS'] = len(self.time)
        info_dic['TEXP'] = self.texp

class EventList(list):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.array([isinstance(i,Event) for i in self]).all():
            raise TypeError('All the elements must be Power objects')

    def __setitem__(self, index, event):
        if not isinstance(event,Event):
            raise TypeError('The item must be a Power object')
        self[index] = event

    def join(self,mask=None):
        if mask is None:
            mask = np.ones(len(self),dtype=bool)
        else:
            assert len(mask) == len(self),'Mask must have the same size of EventList'
        df_list = []
        for i in range(len(self)):
            if mask[i]: 
                df_list += [pd.DataFrame(self[i])]
        
        df = pd.concat(df_list,ignore_index=True)
        
        valid_index = [i for i in range(len(self)) if mask[i]][0]

        kwargs = {}
        for col in list(self[valid_index].columns):
            kwargs['{}_array'.format(col)] = df[col]

        history = self[valid_index].history.copy()
        history['CREATION_DATE'] = my_cdate()
        history['CREATION_MODE'] = 'Event created from EventList'
        history['N_EVENT_FILES'] = len(self)
        history['N_MASKED_FILES'] = mask.sum()

        event = Event(mission_name=self,history=history,**kwargs)

        return event


  
