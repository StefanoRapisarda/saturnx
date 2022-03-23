
from numpy.testing._private.nosetester import NoseTester
from saturnx.core import Event, EventList, read_event, Gti
from saturnx.utils.time_series import poi_events
from time import ctime
import numpy as np
import pandas as pd
import pytest

class TestEventInit():
    
    def test_empty_event(self):
        '''
        Test initializations of empry columns according to mission,
        meta_data, and notes dictionary
        '''

        # NICER
        nicer_events = Event(mission='NICER')
        assert nicer_events.time.empty
        assert len(nicer_events.time) == 0
        assert len(nicer_events.time.index) == 0
        assert nicer_events.pi.empty
        assert len(nicer_events.pi) == 0
        assert len(nicer_events.pi.index) == 0
        assert nicer_events.det.empty
        assert len(nicer_events.det) == 0
        assert len(nicer_events.det.index) == 0
        assert nicer_events.meta_data['MISSION'] == 'NICER'
        assert len(nicer_events.meta_data) == 2
        assert 'EVT_CRE_DATE' in nicer_events.meta_data.keys()
        assert nicer_events.notes == {}

        # SWIFT
        swift_events = Event(mission='SWIFT')
        assert swift_events.time.empty
        assert len(swift_events.time) == 0
        assert len(swift_events.time.index) == 0
        assert swift_events.pi.empty
        assert len(swift_events.pi) == 0
        assert len(swift_events.pi.index) == 0
        assert swift_events.detx.empty
        assert len(swift_events.detx) == 0
        assert len(swift_events.detx.index) == 0
        assert swift_events.dety.empty
        assert len(swift_events.dety) == 0
        assert len(swift_events.dety.index) == 0
        assert swift_events.grade.empty
        assert len(swift_events.grade) == 0
        assert len(swift_events.grade.index) == 0
        assert swift_events.meta_data['MISSION'] == 'SWIFT'
        assert len(swift_events.meta_data) == 2
        assert 'EVT_CRE_DATE' in swift_events.meta_data.keys()
        assert swift_events.notes == {}

        # Generic mission
        whatever_events = Event()
        assert whatever_events.time.empty
        assert len(whatever_events.time) == 0
        assert len(whatever_events.time.index) == 0
        assert whatever_events.pi.empty
        assert len(whatever_events.pi) == 0
        assert len(whatever_events.pi.index) == 0
        assert whatever_events.meta_data['MISSION'] == None
        assert len(whatever_events.meta_data) == 2
        assert 'EVT_CRE_DATE' in whatever_events.meta_data.keys()
        assert whatever_events.notes == {}

    def test_event(self,fake_nicer_event):
        '''
        Tests correct creation of Event object and its attributes
        '''
        nevents = fake_nicer_event['n_events']
        texp = fake_nicer_event['texp']
        events = fake_nicer_event['event']
        assert type(events) == type(Event())
        assert 'time' in events.columns
        assert 'pi' in events.columns
        assert 'det' in events.columns
        assert events.texp == texp
        assert events.cr == nevents/texp
        assert len(events) == nevents
        for col in events.columns:
            assert not events[col].empty
            assert len(events[col]) == nevents
        assert events.notes['STEF1'] == 'This is a test note'
        assert events.meta_data['EVT_CRE_MODE'] == 'Initialized from fake arrays'
        assert events.meta_data['MISSION'] == 'NICER'
        assert events.meta_data['N_ACT_DET'] == 56
        assert events.meta_data['INACT_DET_LIST'] == []

    def test_wrong_arrays(self):
        '''
        Tests reaction to different size arrays
        '''
        x = np.arange(500)
        y = np.arange(250)
        with pytest.raises(ValueError):
            events = Event(time_array=x,pi_array=y)


class TestEventFilter:

    def test_wrong_expr_type(self,fake_nicer_event):
        with pytest.raises(TypeError):
            new_events = fake_nicer_event['event'].filter(123)

    def test_wrong_expr_columns(self,fake_nicer_event):
        with pytest.raises(TypeError):
            new_events = fake_nicer_event['event'].filter('cazzo>=123')
        with pytest.raises(TypeError):
            new_evnets = fake_nicer_event['event'].filter('detx>20')

    def test_wrong_expr_syntax(self,fake_nicer_event):
        new_events = fake_nicer_event['event'].filter('not ((time >= 5) or (time <= 8)) and '\
            '((pi>=80) xor (pi<=200))')
        
    def test_filtered_event_sintax(self,fake_nicer_event):
        old_meta_data = fake_nicer_event['event'].meta_data.copy()
        old_notes = fake_nicer_event['event'].notes.copy()
        expr = '(time >= 5) & (time <= 8) & '\
            '(pi>=80) & (pi<=200)'
        new_event = fake_nicer_event['event'].filter(expr)
        assert new_event.notes == old_notes
        assert 'FILTERING' in new_event.meta_data.keys()
        assert 'FILT_EXPR' in new_event.meta_data.keys()
        assert new_event.meta_data['FILT_EXPR'] == expr

    def test_filtered_event_arrays(self,fake_nicer_event):
        old_events = fake_nicer_event['event']
        new_events = old_events.filter('((time >= 1) and (time <= 9)) and '\
            '((pi>=80) and (pi<=500))')
        assert not new_events.time.empty
        assert not new_events.pi.empty
        assert len(new_events) <= len(old_events)
        assert (min(new_events.time)>=1) and (max(new_events.time)<=9)
        assert (min(new_events.pi)>=80) and (max(new_events.pi)<=500)

    def test_event_histogram(self,fake_nicer_event):
        nbins = fake_nicer_event['n_bins']
        tres = fake_nicer_event['tres']
        cr = fake_nicer_event['cr']
        time_bins_edges = np.linspace(0-tres/2.,
            nbins*tres+tres/2.,nbins+1,dtype=np.double)
        test_hist, dummy = np.histogram(fake_nicer_event['event'].time,time_bins_edges)
        assert np.round(np.sum(test_hist)/nbins/tres) == cr
        assert len(test_hist) == nbins
        assert np.sum(test_hist) == len(fake_nicer_event['event'])


class TestEventSplit:

    def test_gti_input(self,fake_nicer_event,fake_gti):
        not_a_gti = {'start':[0,1,2,3],'stop':[1,2,3,4]}
        with pytest.raises(TypeError):
            split_event = fake_nicer_event['event'].split(not_a_gti)
        split_event = fake_nicer_event['event'].split(fake_gti)       

    def test_gti_output(self,fake_nicer_event,fake_gti):
        split_event = fake_nicer_event['event'].split(fake_gti)
        
        # Type and size
        assert type(split_event) == type(EventList())
        assert len(split_event) == len(fake_gti)
        for event in split_event:
            assert type(event) == type(Event())
        
        # Columns
        columns = fake_nicer_event['event'].columns
        for col in ['time','pi','det']:
            assert col in columns
        for event in split_event:
            for col in columns: assert col in event.columns

        # Single Event features
        for i,(start,stop) in enumerate(zip(fake_gti.start,fake_gti.stop)):
            assert 'time' in split_event[i].columns
            assert 'det' in split_event[i].columns,f'Event {i}'
            assert 'pi' in split_event[i].columns

            assert min(split_event[i].time) >= start
            assert max(split_event[i].time) <= stop

            assert split_event[i].meta_data['MISSION'] == 'NICER'
            assert split_event[i].meta_data['GTI_IND'] == i
            assert split_event[i].meta_data['N_GTIS'] == len(fake_gti)
            assert 'SPLITTING_GTI' in split_event[i].meta_data.keys()
            assert split_event[i].notes == fake_nicer_event['event'].notes

    def test_seg_input(self,fake_nicer_event):
        wrong_inputs = ['123wer',{},tuple([1,2])]
        with pytest.raises(TypeError):
            for wrong_input in wrong_inputs:
                event_list = fake_nicer_event['event'].split(wrong_input)
        
        right_inputs = ['10',10,10.2]
        for right_input in right_inputs:
            event_list = fake_nicer_event['event'].split(right_input) 

    def test_seg_output(self,fake_nicer_event):
        time_seg = 12
        exp_n_segs = 4
        split_event = fake_nicer_event['event'].split(time_seg) 

        assert type(split_event) == type(EventList())
        assert len(split_event) == exp_n_segs  

        for i, event in enumerate(split_event):
            assert type(event) == type(Event())
            assert event.meta_data['MISSION'] == 'NICER'
            assert event.texp <= time_seg
            assert event.meta_data['SEG_IND'] == i
            assert event.meta_data['N_SEGS'] == exp_n_segs
            assert 'SPLITTING_SEG' in event.meta_data.keys()
            assert event.notes == fake_nicer_event['event'].notes
           

class TestEventListInit:

    def test_wrong_input(self):
        wrong_inputs = ['hello!',132,['1','2'],np.array([1,2,3,4]),{}]
        with pytest.raises(TypeError):
            event_list = EventList(wrong_inputs)

    def test_wrong_setitems(self,fake_nicer_event,fake_gti):
        event_list = fake_nicer_event['event'].split(fake_gti)
        with pytest.raises(TypeError):
            event_list[0] = 'Hello!'

    def test_simple_join(self,fake_nicer_event):
        fake_gti = Gti([0,10,20,100],[10,20,100,1000],clean=False)        
        first_event = fake_nicer_event['event']
        event_list = first_event.split(fake_gti)
        assert len(event_list) == 4
        joined_event = event_list.join()
        assert type(joined_event) == type(Event())
        assert first_event.time.equals(joined_event.time) 
        assert first_event.pi.equals(joined_event.pi)
        assert first_event.det.equals(joined_event.det)
        assert joined_event.meta_data['EVT_CRE_MODE'] == 'Event created joining Events from EventList'
        assert joined_event.meta_data['N_ORI_EVTS'] == len(event_list)
        assert joined_event.meta_data['N_MASKED_EVTS'] == len(event_list)
        assert joined_event.notes == {}

    def test_masked_join(self,fake_nicer_event):
        fake_gti1 = Gti([0,10,20,100],[10,20,100,1000],clean=False)
        fake_gti2 = Gti([0,20],[10,100])
        first_event = fake_nicer_event['event']
        event_list1 = first_event.split(fake_gti1)
        event_list2 = first_event.split(fake_gti2)
        assert 'det' in event_list1[0].columns
        assert 'det' in event_list2[0].columns
        joined_event1 = event_list1.join(mask=[1,0,1,0])
        joined_event2 = event_list2.join()
        assert joined_event2.columns.equals(joined_event2.columns)
        assert joined_event1.time.equals(joined_event2.time) 
        assert joined_event1.pi.equals(joined_event2.pi) 
        assert joined_event1.det.equals(joined_event2.det)

    def test_info(self,fake_nicer_event,fake_gti):
        event_list = fake_nicer_event['event'].split(fake_gti)
        info = event_list.info()
        assert type(info) == type(pd.DataFrame())
        assert len(info) == len(event_list)
        assert list(info.columns) == ['texp','n_events','count_rate',
                                    'max_time','min_time',
                                    'min_pi','max_pi','mission']
        for i in range(len(info)):
            assert info['texp'].iloc[i] == event_list[i].texp 
            assert info['n_events'].iloc[i] == len(event_list[i])
            assert info['count_rate'].iloc[i] == event_list[i].cr
            assert info['min_time'].iloc[i] == min(event_list[i].time) 
            assert info['max_time'].iloc[i] == max(event_list[i].time) 
            assert info['min_pi'].iloc[i] == min(event_list[i].pi) 
            assert info['max_pi'].iloc[i] == max(event_list[i].pi)
            assert info['mission'].iloc[i] == event_list[i].meta_data['MISSION'] 

class TestReadEvent:
    
    def test_missing_file(self):
        with pytest.raises(OSError):
            event = Event.read_fits('NICE_cl.evt.gz')

    def test_user_info_multi(self):
        file_name = 'tests/NICER_cl.evt.gz'
        event = Event.read_fits(file_name,keys_to_read=['XTENSION','DATAMODE'])        
        assert event.meta_data['INFO_FROM_HEADER']['XTENSION'] == 'BINTABLE'
        assert event.meta_data['INFO_FROM_HEADER']['DATAMODE'] == 'PHOTON'

    def test_user_info_single(self):
        file_name = 'tests/NICER_cl.evt.gz'
        event = Event.read_fits(file_name,keys_to_read='EXTNAME')        
        assert event.meta_data['INFO_FROM_HEADER']['EXTNAME'] == 'EVENTS'        

    def test_event_reading(self):
        file_name = 'tests/NICER_cl.evt.gz'
        event = Event.read_fits(file_name)
        assert type(event) == type(Event())
        assert event.meta_data['MISSION'] == 'NICER'
        assert event.notes == {}
        assert not event.time.empty
        assert not event.pi.empty
        assert not event.det.empty
        assert len(event.time) == len(event.pi)
        assert len(event.pi) == len(event.pi)
        assert event.texp
        assert event.cr

        # Checking meta_data
        assert event.meta_data['EVT_CRE_MODE'] == 'Event created from fits file'
        assert event.meta_data['EVT_FILE_NAME'] == 'NICER_cl.evt.gz'
        assert 'N_ACT_DET' in event.meta_data.keys()
        assert 'N_INACT_DET' in event.meta_data.keys()
        assert event.meta_data['N_ACT_DET'] == 50
        assert event.meta_data['N_INACT_DET'] == [11,14,20,22,34,60]

        basic_keys = ['OBJECT','TELESCOP','INSTRUME','OBS_ID','RA_OBJ','DEC_OBJ',
                        'CREATOR','DATE','SOFTVER','CALDBVER','GCALFILE']

        time_keys = ['DATE-OBS','DATE-END','TSTART','TSTOP',
                        'MJDREFI','MJDREFF','TIMEZERO','LEAPINIT','CLOCKAPP',
                        'TIMEZERO','ONTIME','EXPOSURE','NAXIS2','TIMESYS']

        keys = basic_keys + time_keys

        for key in keys:
            assert key in event.meta_data['INFO_FROM_HEADER'].keys()