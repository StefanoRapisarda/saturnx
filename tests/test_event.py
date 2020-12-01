
from ..event import Event, read_event
from time import ctime
import numpy as np
import pandas as pd


class TestEvent:

    def setup_class(self):
        mission = 'NICER'
        if mission == 'NICER':
            self.file_name = '/Volumes/Seagate/NICER_data/MAXI_J1807+132/200919-141019/2200840101/xti/event_cl/ni2200840101_0mpu7_cl_bc_bdc.evt.gz'

            size_mpu = 10000
            self.size = 7*size_mpu   
            self.time_array = np.random.rand(self.size)
            self.time_array[0],self.time_array[-1] = 0,1
            self.det_array = np.concatenate([np.random.randint(0,8,size_mpu),
                            np.random.randint(10,18,size_mpu),
                            np.random.randint(20,28,size_mpu),
                            np.random.randint(30,38,size_mpu),
                            np.random.randint(40,48,size_mpu),
                            np.random.randint(50,58,size_mpu),
                            np.random.randint(60,68,size_mpu)],axis=0)

            self.pi_array = np.random.randint(20,1501,self.size)
            now = ctime()
            self.history = {now:'CREATION_DATE'}
            self.notes = {f'Stefano_{now}':'Created for testing purposes'}
            self.mission = mission

    def test_init_empty(self):
        
        events = Event()

        assert isinstance(events,Event)
        assert isinstance(events,pd.DataFrame)
        
        # Empty DataFrame will have column names
        assert 'time' in events
        assert 'pi' in events
        assert 'det' in events

        assert events.mission == None
        assert events.notes == {}
        assert events.history == {}

    def test_init_full(self):

        events = Event(self.time_array,self.pi_array,self.det_array,
        self.mission,self.notes,self.history)

        assert len(events) == self.size
        assert 'time' in events
        assert 'pi' in events
        assert 'det' in events

        assert events.mission == 'NICER'   
        assert isinstance(events.notes,dict)
        assert isinstance(events.history,dict)

    def test_read_event(self):

        events = read_event(self.file_name)

        assert isinstance(events,Event)
        assert isinstance(events,pd.DataFrame)
        
        # Empty DataFrame will have column names
        assert 'time' in events
        assert 'pi' in events
        assert 'det' in events

        assert events.mission == 'NICER'
        assert isinstance(events.notes,dict)
        assert isinstance(events.history,dict)   

    def test_texp(self):

        events = Event(self.time_array,self.pi_array,self.det_array,
        self.mission,self.notes,self.history)

        assert events.texp == 1

        events = Event()     

        assert events.texp is None