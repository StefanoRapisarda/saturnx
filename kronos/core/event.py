from kronos.utils.generic import clean_expr, is_number, my_cdate
from kronos.utils.fits import get_basic_info
from kronos.core.gti import Gti
import os

import pandas as pd
import numpy as np

import time
from astropy.io.fits import getdata,getval

from ..functions.nicer_functions import all_det

def read_event(file_name,evt_ext_name='EVENTS'):
    '''
    Read a fits file and store meaningfull information in an Event object
    '''
    
    try:
        if os.stat(file_name).st_size == 0:
            print('Event FITS file is Empty')
    except OSError:
        print('File {} does not exist'.format(file_name))
    mission =  getval(file_name,'telescop',1)

    # Reading data
    data = getdata(file_name,extname=evt_ext_name,header=False,memmap=True)

    # Initializing header
    header = {}
    header['CRE_MODE'] = 'Event created from fits file'
    header['EVT_NAME'] = os.path.basename(file_name)
    header['DIR'] = os.path.dirname(file_name)

    # Reading meaningfull information from event file
    info = get_basic_info(file_name)
    header = {**header, **info}

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

        n_act_det = len(np.unique(det_id))
        inact_det_list = np.setdiff1d(all_det, np.unique(det_id))
        header['NACT_DET'] = n_act_det
        header['IDET_DET'] = inact_det_list

        event = Event(time_array=times,det_array=det_id,pi_array=pi,
                       mission=mission,header=header)
    elif mission == 'SWIFT':
        events = Event(time_array=data['TIME'],detx_array=data['DETX'],dety_array=data['DETY'],
                       pi_array=data['PI'],grade_array=data['GRADE'],
                       mission=mission)   

    return event


class Event(pd.DataFrame):
    '''

    NOTES
    -----
    2021 02 19, Stefano Rapisarda (Uppsala)
        !!! It is important that whatever array you will add in the 
        future to the list of already existing arrays, you write the 
        variable in the form <whatever>_array, as methods relies on
        this syntax
    '''

    _metadata = ['header','notes']

    def __init__(self,time_array=None,pi_array=None,det_array=None,
                detx_array=None,dety_array=None,grade_array=None,
                mission=None,header=None,notes=None):
        '''
        Initialise time, pi, and detector arrays according to selected
        mission
        '''
        
        if time_array is None:
            if mission == 'NICER': 
                columns=['time','pi','det']
            elif mission == 'SWIFT':
                columns=['time','pi','detx','dety','grade']
            else:
                columns=['time','pi']
            super().__init__(columns=columns)  
        else:
            if mission == 'NICER':
                columns = {'time':time_array,
                        'pi':pi_array,
                        'det':det_array}
            elif mission == 'SWIFT':
                columns = {'time':time_array,
                        'pi':pi_array,
                        'detx':detx_array,
                        'dety':dety_array,
                        'grade':grade_array}  
            else:
                columns = {'time':time_array,
                        'pi':pi_array}
            super().__init__(columns)  

        if notes is None:
            self.notes = {}
        else: self.notes = notes
        if header is None:
            self.header = {}
        else: self.header = header
        self.header['CRE_DATE'] = my_cdate()
        if not 'MISSION' in  self.header.keys():
            self.header['MISSION'] = mission 
                            

    def filter(self,expr):
        '''
        Evaluate a filtering expression and applies it to the events
        
        NOTES
        -----
        2021 02 18, Stefano Rapisarda (Uppsala)
            To mask pandas Series use &,|,etc are necessary, so that
            and,or,etc are substituted.
        '''

        expr_ori = expr

        # Checking expression
        if type(expr) != str: raise TypeError
        cleaned_expr = clean_expr(expr)
        for piece in cleaned_expr.split():
            if not is_number(piece) and not piece in self.columns:
                raise TypeError('Variables in filter expression must'\
                    ' be event columns')

        # Adapting and evaluating expression
        # !!! It is important that xor is listed BEFORE or
        operators = {'and':'&','xor':'^','or':'|','not':'~'}
        for key,item in operators.items():
            expr = expr.replace(key,item)
        for col in list(self.columns):
            expr = expr.replace(col,'self.{}'.format(col))

        # Applying mask to arrays
        mask = eval(expr)
        kwargs = {}
        for col in list(self.columns):
            kwargs['{}_array'.format(col)] = self[col][mask]
        
        # Copying and updating notes and header
        kwargs['header'] = self.header
        kwargs['notes'] = self.notes
        kwargs['header']['FIL_DATE'] = my_cdate()
        kwargs['header']['FIL_EXPR'] = expr_ori

        # Initializing event object
        events = Event(**kwargs)

        return events        

    def split(self,splitter):
        '''
        Splits Event object in an EventList according to time intervals
        in a Gti object or to a time segment
        '''

        events = []
        header = self.header.copy()   
        notes = self.notes.copy()

        if type(splitter) == type(Gti()):
            gti = splitter

            header['SPLI_GTI'] = my_cdate()
            header['N_GTIS'] = len(gti)

            for gti_index,(start,stop) in enumerate(zip(gti.start,gti.stop)):
                mask = (self.time>= start) & (self.time<stop)
                kwargs = {}
                for col in list(self.columns):
                    kwargs['{}_array'.format(col)] = self[col][mask]
                
                header_gti = header.copy()
                header_gti['GTI_IND'] = gti_index            
                kwargs['header'] = header_gti
                kwargs['notes'] = notes

                events += [Event(**kwargs)]

        elif type(splitter) in [float,int,str,np.double,np.float]:
            
            if type(splitter)==str: 
                if not is_number(splitter):
                    raise TypeError('String in Event.split() must contain only numbers')
                else:
                    time_seg = eval(splitter)
            else:
                time_seg = splitter

            n_segs = int(self.texp//time_seg)
            header['SPLI_SEG'] = my_cdate()
            header['N_SEGS'] = n_segs

            for i in range(n_segs):
                start = i*time_seg
                stop  = (i+1)*time_seg
                mask = (self.time>= start) & (self.time<stop)
                kwargs = {}
                for col in list(self.columns):
                    kwargs['{}_array'.format(col)] = self[col][mask]
                
                header_seg = header.copy()
                header_seg['SEG_IND'] = i            
                kwargs['header'] = header_seg
                kwargs['notes'] = notes

                events += [Event(**kwargs)]
        
        else:
            raise TypeError('Events can be split only according to GTI '\
                'or time segment')

        return EventList(events)

    @property
    def texp(self):
        if len(self.time) == 0:
            return None
        else:
            return max(self.time)-min(self.time)

    @property
    def cr(self):
        if not self.texp is None and (self.texp !=0):
            return len(self.time)/self.texp

class EventList(list):
    '''
    A list of Event with some superpower

    TODO
    ----
    2021 02 19, Stefano Rapisarda (Uppsala)
        Avoid that Events with different columns can populate the same
        list
    '''

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.array([isinstance(i,Event) for i in self]).all():
            raise TypeError('All the elements must be Event objects')

    def __setitem__(self, index, event):
        if not isinstance(event,Event):
            raise TypeError('The item must be an Event object')
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
        
        first_valid_index = [i for i in range(len(self)) if mask[i]][0]

        kwargs = {}
        for col in list(self[first_valid_index].columns):
            kwargs['{}_array'.format(col)] = df[col]

        notes = {}
        header = {}
        header['CRE_MODE'] = 'Event created joining Events from EventList'
        header['EVT_OBJS'] = len(self)
        header['MSK_OBJS'] = mask.sum()
        kwargs['header'] = header
        kwargs['notes'] = notes

        return Event(**kwargs)

    def info(self):
        '''
        Return a pandas DataFrame with the characteristics of each event
        list
        '''

        columns = ['texp','n_events','count_rate',
                    'max_time','min_time',
                    'min_pi','max_pi','mission']
        info = pd.DataFrame(columns=columns)
        for i,event in enumerate(self):
            line = {'texp':event.texp,'n_events':len(event),
                'count_rate':event.cr,
                'min_time':min(event.time),'max_time':max(event.time),
                'min_pi':min(event.pi),'max_pi':max(event.pi),
                'mission':event.header['MISSION']}
            info.loc[i] = pd.Series(line)
        
        return info


  
