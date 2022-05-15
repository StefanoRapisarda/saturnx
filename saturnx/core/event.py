"""This module contains the definition of Event and EventList classes"""

__author__ = 'Stefano Rapisarda'

import os
import pathlib
import pandas as pd
import numpy as np

from astropy.io.fits import getdata,getval
from astropy.io import fits

from saturnx.utils.generic import clean_expr, is_number, my_cdate
from saturnx.utils.fits import get_basic_info, read_fits_keys
from saturnx.utils.nicer_functions import all_det
from saturnx.core.gti import Gti

class Event(pd.DataFrame):
    '''
    Event object. Stores photon time arrival and other observatory-dependend
    parameters.

    HISTORY
    -------
    2020 04 ##, Stefano Rapisarda (Uppsala), creation date

    NOTES
    -----
    2021 02 19, Stefano Rapisarda (Uppsala)
        !!! It is important that whatever array you will add in the 
        future to the list of already existing arrays, you write the 
        variable in the form <whatever>_array, as methods relies on
        this syntax !!!
    '''

    _metadata = ['meta_data','_meta_data']

    def __init__(self,time_array=None,pi_array=None,det_array=None,
                 detx_array=None,dety_array=None,grade_array=None,
                 mission=None,meta_data=None):
        '''
        Initialise time, pi, and detector arrays according to the specified
        mission

        TODO
        ----
        2022 05 12, Stefano Rapisarda, Uppsala
            For NICER data, it reads the events in order to figure out
            the number of active detectors. This could take A LOT of time.
            The last version of NICER software includes this information
            in the FITS file header, you should read the information from
            there.
        '''
        
        # Setting mission-dependend columns
        if mission == 'NICER':
            columns = {
                'time':time_array,
                'pi':pi_array,
                'det':det_array
                }
        elif mission == 'SWIFT':
            columns = {'time':time_array,
                'pi':pi_array,
                'detx':detx_array,
                'dety':dety_array,
                'grade':grade_array
                }  
        else:
            columns = {
                'time':time_array,
                'pi':pi_array
                }

        if time_array is None:
            # columns parameters specifies only the names of the columns
            super().__init__(columns=columns)
        else:
            super().__init__(columns)  

        # Initialiasing meta_data
        if meta_data is None:
            self.meta_data = {}
        else: 
            self.meta_data = meta_data

        if not 'HISTORY' in self.meta_data.keys():
            self.meta_data['HISTORY'] = {}
        self.meta_data['HISTORY']['EVT_CRE_DATE'] = my_cdate()

        if not 'NOTES' in self.meta_data.keys():
            self.meta_data['NOTES'] = {}

        # Mission dependent meta data        
        if not 'MISSION' in  self.meta_data.keys():
            self.meta_data['MISSION'] = mission
 
        if mission == 'NICER' and det_array is not None:
            n_act_det = len(np.unique(self.det))
            inact_det_list = np.setdiff1d(all_det, np.unique(det_array))
            self.meta_data['N_ACT_DET'] = n_act_det
            self.meta_data['INACT_DET_LIST'] = list(inact_det_list)
                            
    def filter(self,expr):
        '''
        Evaluates a filtering expression and applies it to events

        The filtering expression must contains only arithmetical/logical
        operators and column names
        
        NOTE
        ----
        2021 02 18, Stefano Rapisarda (Uppsala)
            To mask pandas Series use &,|,etc in place of and,or,etc.
        '''

        expr_ori = expr

        # Checking expression
        # --------------------------------------------------------------
        if type(expr) != str: raise TypeError

        cleaned_expr = clean_expr(expr) # Removes all operators
        # filtering expression must contain ONLY column names and operators
        for piece in cleaned_expr.split():
            if not is_number(piece) and not piece in self.columns:
                raise TypeError('Variables in filter expression must'\
                    ' be event columns')
        # --------------------------------------------------------------

        # Adapting and evaluating expression
        # !!! It is important that xor is listed BEFORE or !!!
        operators = {'and':'&','xor':'^','or':'|','not':'~'}
        for key,item in operators.items():
            expr = expr.replace(key,item)
        for col in list(self.columns):
            expr = expr.replace(col,'self.{}'.format(col))

        # Defining new arguments for filtered Event
        new_kwargs = {}

        # Applying mask to arrays
        try:
            mask = eval(expr)
        except TypeError:
            print('Something went wrong in evaluating your filtering expression')
            print(f'({expr})')
            print('Check that each condition is wrapped in round brackets')
            raise TypeError

        for col in list(self.columns):
            new_kwargs['{}_array'.format(col)] = self[col][mask]
        
        # Copying and updating notes and meta_data
        new_kwargs['meta_data'] = self.meta_data
        new_kwargs['meta_data']['HISTORY']['FILTERING'] = my_cdate()
        new_kwargs['meta_data']['FILT_EXPR'] = expr_ori

        # Initializing event object
        events = Event(**new_kwargs)

        return events        

    def split(self,splitter):
        '''
        Splits Event object in an EventList according to time intervals
        in a Gti object or to a time segment

        PARAMETERS
        ----------
        splitter: float, int, str, np.double, np.float, or saturnx.core.GTI
            if GTI, the event file will be split according to GTI start 
            and stop time.
            In all the other case, the event file will be split according
            to segment duration

        RETURNS
        -------
        saturnx.core.EventList
        '''

        events = []
        kwargs = {}
        meta_data = self.meta_data.copy()   

        if isinstance(splitter,Gti):
            gti = splitter

            meta_data['HISTORY']['SPLITTING_GTI'] = my_cdate()
            meta_data['N_GTIS'] = len(gti)

            for gti_index,(start,stop) in enumerate(zip(gti.start,gti.stop)):

                mask = (self.time>= start) & (self.time<stop)
                kwargs = {}
                for col in list(self.columns):
                    kwargs['{}_array'.format(col)] = self[col][mask]

                meta_data_gti = meta_data.copy()
                meta_data_gti['GTI_INDEX'] = gti_index            
                kwargs['meta_data'] = meta_data_gti
                kwargs['mission'] = self.meta_data['MISSION']

                events += [Event(**kwargs)]

        elif isinstance(splitter,(float,int,str,np.double,np.float)):
            
            if isinstance(splitter,str): 
                if not is_number(splitter):
                    raise TypeError('String in Event.split() must contain only numbers')
                else:
                    time_seg = eval(splitter)
            else:
                time_seg = splitter

            n_segs = int(self.texp//time_seg)
            meta_data['HISTORY']['SPLITTING_SEG'] = my_cdate()
            meta_data['N_SEGS'] = n_segs

            for i in range(n_segs):

                start = i*time_seg
                stop  = (i+1)*time_seg
                mask = (self.time>= start) & (self.time<stop)
                kwargs = {}
                for col in list(self.columns):
                    kwargs['{}_array'.format(col)] = self[col][mask]

                meta_data_seg = meta_data.copy()
                meta_data_seg['SEG_INDEX'] = i            
                kwargs['meta_data'] = meta_data_seg
                kwargs['mission'] = self.meta_data['MISSION']

                events += [Event(**kwargs)]
        
        else:
            raise TypeError('Events can be split only according to GTI '\
                'or time segment')

        return EventList(events)

    @staticmethod
    def read_fits(file_name,ext='EVENTS',keys_to_read=None):
        '''
        Read a FITS file and store meaningful information in an Event object
        
        PARAMETERS
        ----------
        file_name: str or pathlib.Path()
            Full path of a FITS file 
        evt_ext_name: str, optional
            Name of the FITS file extension to read, default is EVENT
        keys_to_read: str or list, optional
            List or str specifying keys to read from the header of the 
            specified extension. Default is None, in this case a set 
            of standard keywords will be read. Keywords/Values are stored
            in the dictionary Event.meta_data['INFO_FROM_HEADER']

        RETURNS
        -------
        event: saturnx.core.Event
            Event object

        HISTORY
        -------
        2020 04 ##, Stefano Rapisarda (Uppsala)
            Creation date 

        NOTES
        -----
        2021 02 20, Stefano Rapisarda (Uppsala)
            For NICER events, it is important to determine the number of
            active detectors. This is done here and this information should
            be propagated to further timing products (binned lightcurve and
            power spectra)
        '''
        
        if type(file_name) == str: file_name = pathlib.Path(file_name)

        # Checking file existance and size
        try:
            if os.stat(file_name).st_size == 0:
                print('Event FITS file is Empty')
        except OSError:
            print('File {} does not exist'.format(file_name))

        # Reading data
        print('Reading event FITS file')
        hdulist = fits.open(file_name,memmap=True)

        # Initializing meta_data
        mission = hdulist[ext].header['TELESCOP']
        meta_data = {}
        meta_data['EVT_CRE_MODE'] = 'Event created from fits file'
        meta_data['EVT_FILE_NAME'] = file_name.name
        meta_data['DIR'] = file_name.parent

        # Reading meaningfull information from event file
        info = get_basic_info(hdulist,ext=ext)
        if not keys_to_read is None:
            if isinstance(keys_to_read,(str,list)): 
                user_info = read_fits_keys(hdulist,keys_to_read,ext=ext)
            else:
                raise TypeError('keys to read must be str or list')
        else: 
            user_info = {}
        total_info = {**info,**user_info}
        meta_data['INFO_FROM_HEADER'] = total_info

        # Initializing Event object
        if mission == 'NICER':
            times = hdulist[ext].data['TIME']
            try:
                det_id = hdulist[ext].data['DET_ID']
            except Exception as e:
                print('Could not find DET_ID column')
                print(e)
                det_id = None
            times = hdulist[ext].data['TIME']
            try:
                pi = hdulist[ext].data['PI']
            except Exception as e:
                print('Could not find PI column')
                print(e)
                pi = None

            # Further NICER specific info
            n_act_det = len(np.unique(det_id))
            inact_det_list = np.setdiff1d(all_det, np.unique(det_id))
            meta_data['N_ACT_DET'] = n_act_det
            meta_data['N_INACT_DET'] = list(inact_det_list)

            print('Initializing event object')
            event = Event(
                time_array=times,det_array=det_id,pi_array=pi,
                mission=mission,meta_data=meta_data
                )
        elif mission == 'SWIFT':
            event = Event(time_array=hdulist[ext].data['TIME'],
                detx_array=hdulist[ext].data['DETX'],
                dety_array=hdulist[ext].data['DETY'],
                pi_array=hdulist[ext].data['PI'],
                grade_array=hdulist[ext].data['GRADE'],
                mission=mission)  

        hdulist.close()
        del hdulist 

        return event

    @property
    def texp(self):
        if len(self.time) == 0:
            return None
        else:
            return max(self.time)-min(self.time)

    @property
    def cr(self):
        if self.texp is not None and (self.texp !=0):
            return len(self.time)/self.texp
        else:
            return None

class EventList(list):
    '''
    A list of Event with some superpower

    2020 04 ##, Stefano Rapisarda (Uppsala), creation date

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
        '''
        Joins Events in an EventList into a single Event

        The joining is performed using the pandas method concat

        PARAMETERS
        ----------
        mask: list or np.array
            Array of booleans to select Events in the EventList to join
        '''

        if mask is None:
            mask = np.ones(len(self),dtype=bool)
        else:
            if not len(mask) == len(self):
                print('Mask must have the same size of EventList')
                raise IndexError
        
        df_list = []
        for i in range(len(self)):
            if mask[i]: 
                df_list += [self[i]]
        
        df = pd.concat(df_list,ignore_index=True)
        
        first_valid_index = [i for i in range(len(self)) if mask[i]][0]

        kwargs = {}
        for col in list(self[first_valid_index].columns):
            kwargs['{}_array'.format(col)] = df[col]

        meta_data = {}
        meta_data['EVT_CRE_MODE'] = 'Event created joining Events from EventList'
        meta_data['N_ORI_EVTS'] = len(self)
        meta_data['N_MASKED_EVTS'] = sum(mask)
        kwargs['meta_data'] = meta_data
        kwargs['mission'] = self[first_valid_index].meta_data['MISSION']

        return Event(**kwargs)

    def info(self):
        '''
        Returns a pandas DataFrame with relevand information for each Event 
        object in the EventList
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
                'mission':event.meta_data['MISSION']}
            info.loc[i] = pd.Series(line)
        
        return info


  
