import pytest
import numpy as np
from kronos.utils.time_series import poi_events
from kronos.core import Event,Gti

@pytest.fixture(scope='class')
def fake_nicer_event(tres=0.01,nbins=5000,cr=5,low_ch=50,high_ch=1000):
    events = poi_events(tres=tres,nbins=nbins,cr=cr)
    texp = max(events)-min(events)
    pi = np.random.uniform(low_ch,high_ch,len(events))
    notes = {}
    notes['STEF1'] = 'This is a test note'
    event_object = Event(time_array=events,
        pi_array=pi,mission='NICER',notes=notes)
    event_object.header['CRE_MODE'] = 'Initialized from fake arrays'
    data = {'event':event_object,'texp':texp,'n_events':len(events),
            'tres':tres,'cr':cr,'n_bins':nbins,'low_ch':50,'high_ch':1000,
            'texp':texp}
    return data

@pytest.fixture(scope='class')
def fake_gti():
    start = np.array([0.,10.,30])
    stop  = np.array([7.,20.,43.])
    gti = Gti(start,stop)
    return gti