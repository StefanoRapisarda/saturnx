
from numpy.testing._private.nosetester import NoseTester
from kronos.core import Event, EventList, read_event, Gti
from kronos.utils.functions import poi_events
from time import ctime
import numpy as np
import pandas as pd
import pytest

class TestEventInit():
    
    def test_empty_event(self):
        '''
        Test initializations of empry columns according to mission,
        header, and notes dictionary
        '''

        nicer_events = Event(mission_name='NICER')
        assert nicer_events.time.empty
        assert len(nicer_events.time) == 0
        assert len(nicer_events.time.index) == 0
        assert nicer_events.pi.empty
        assert len(nicer_events.pi) == 0
        assert len(nicer_events.pi.index) == 0
        assert nicer_events.det.empty
        assert len(nicer_events.det) == 0
        assert len(nicer_events.det.index) == 0
        assert nicer_events.header['MISSION'] == 'NICER'
        assert len(nicer_events.header) == 2
        assert 'CRE_DATE' in nicer_events.header.keys()
        assert nicer_events.notes == {}

        swift_events = Event(mission_name='SWIFT')
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
        assert swift_events.header['MISSION'] == 'SWIFT'
        assert len(swift_events.header) == 2
        assert 'CRE_DATE' in swift_events.header.keys()
        assert swift_events.notes == {}

        whatever_events = Event()
        assert whatever_events.time.empty
        assert len(whatever_events.time) == 0
        assert len(whatever_events.time.index) == 0
        assert whatever_events.pi.empty
        assert len(whatever_events.pi) == 0
        assert len(whatever_events.pi.index) == 0
        assert whatever_events.header['MISSION'] == None
        assert len(whatever_events.header) == 2
        assert 'CRE_DATE' in whatever_events.header.keys()
        assert whatever_events.notes == {}

    def test_event(self,fake_nicer_event):
        '''
        Tests correct creation of Event object and its attributes
        '''
        nevents = fake_nicer_event['n_events']
        texp = fake_nicer_event['texp']
        events = fake_nicer_event['event']
        assert type(events) == type(Event())
        assert events.texp == texp
        assert len(events) == nevents
        for col in events.columns:
            assert not events[col].empty
            assert len(events[col]) == nevents
        assert events.notes['STEF1'] == 'This is a test note'
        assert events.header['CRE_MODE'] == 'Initialized from fake arrays'
        assert events.header['MISSION'] == 'NICER'

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
        old_header = fake_nicer_event['event'].header.copy()
        old_notes = fake_nicer_event['event'].notes.copy()
        expr = '(time >= 5) & (time <= 8) & '\
            '(pi>=80) & (pi<=200)'
        new_event = fake_nicer_event['event'].filter(expr)
        assert new_event.notes == old_notes
        assert 'FIL_DATE' in new_event.header.keys()
        assert 'FIL_EXPR' in new_event.header.keys()
        assert new_event.header['FIL_EXPR'] == expr

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
        assert type(split_event) == type(EventList())
        assert len(split_event) == len(fake_gti)
        for i,(start,stop) in enumerate(zip(fake_gti.start,fake_gti.stop)):
            assert type(split_event[i]) == type(Event())
            assert min(split_event[i].time) >= start
            assert max(split_event[i].time) <= stop
            assert split_event[i].header['GTI_IND'] == i
            assert split_event[i].header['N_GTIS'] == len(fake_gti)
            assert 'SPLI_GTI' in split_event[i].header.keys()
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
            assert event.texp <= time_seg
            assert event.header['SEG_IND'] == i
            assert event.header['N_SEGS'] == exp_n_segs
            assert 'SPLI_SEG' in event.header.keys()
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

    def test_simple_join(self,fake_nicer_event,fake_gti):
        pass






# class TestEvent:

#     def setup_class(self):
#         mission = 'NICER'
#         if mission == 'NICER':
#             self.file_name = '/Volumes/Seagate/NICER_data/MAXI_J1807+132/200919-141019/2200840101/xti/event_cl/ni2200840101_0mpu7_cl_bc_bdc.evt.gz'

#             size_mpu = 10000
#             self.size = 7*size_mpu   
#             self.time_array = np.random.rand(self.size)
#             self.time_array[0],self.time_array[-1] = 0,1
#             self.det_array = np.concatenate([np.random.randint(0,8,size_mpu),
#                             np.random.randint(10,18,size_mpu),
#                             np.random.randint(20,28,size_mpu),
#                             np.random.randint(30,38,size_mpu),
#                             np.random.randint(40,48,size_mpu),
#                             np.random.randint(50,58,size_mpu),
#                             np.random.randint(60,68,size_mpu)],axis=0)

#             self.pi_array = np.random.randint(20,1501,self.size)
#             now = ctime()
#             self.history = {now:'CREATION_DATE'}
#             self.notes = {f'Stefano_{now}':'Created for testing purposes'}
#             self.mission = mission

#     def test_init_empty(self):
        
#         events = Event()

#         assert isinstance(events,Event)
#         assert isinstance(events,pd.DataFrame)
        
#         # Empty DataFrame will have column names
#         assert 'time' in events
#         assert 'pi' in events
#         assert 'det' in events

#         assert events.mission == None
#         assert events.notes == {}
#         assert events.history == {}

#     def test_init_full(self):

#         events = Event(self.time_array,self.pi_array,self.det_array,
#         self.mission,self.notes,self.history)

#         assert len(events) == self.size
#         assert 'time' in events
#         assert 'pi' in events
#         assert 'det' in events

#         assert events.mission == 'NICER'   
#         assert isinstance(events.notes,dict)
#         assert isinstance(events.history,dict)

#     def test_read_event(self):

#         events = read_event(self.file_name)

#         assert isinstance(events,Event)
#         assert isinstance(events,pd.DataFrame)
        
#         # Empty DataFrame will have column names
#         assert 'time' in events
#         assert 'pi' in events
#         assert 'det' in events

#         assert events.mission == 'NICER'
#         assert isinstance(events.notes,dict)
#         assert isinstance(events.history,dict)   

#     def test_texp(self):

#         events = Event(self.time_array,self.pi_array,self.det_array,
#         self.mission,self.notes,self.history)

#         assert events.texp == 1

#         events = Event()     

#         assert events.texp is None