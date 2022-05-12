from saturnx.core.lightcurve import Lightcurve
import pytest
import numpy as np

from saturnx.utils.nicer_functions import all_det
from saturnx.utils.time_series import poi_events
from saturnx.core import Event,Gti

@pytest.fixture(scope='class')
def fake_nicer_event(tres=0.01,nbins=10000,cr=5,low_ch=50,high_ch=1000):

    events = poi_events(tres=tres,nbins=nbins,cr=cr)
    texp = max(events)-min(events)
    pi = np.random.uniform(low_ch,high_ch,len(events))
    dets = np.random.choice(all_det,size=len(events))


    notes = {}
    notes['STEF1'] = 'This is a test note'
    meta_data = {}
    meta_data['NOTES'] = notes
    meta_data['EVT_CRE_MODE'] = 'Initialized from fake arrays'
    meta_data['EVT_FILE_NAME'] = 'Test file name'
    meta_data['DIR'] = 'Test dir'
    meta_data['INFO_FROM_HEADER'] = {}
    basic_keys = ['OBJECT','TELESCOP','INSTRUME','OBS_ID','RA_OBJ','DEC_OBJ',
                  'CREATOR','DATE','SOFTVER','CALDBVER','GCALFILE']
    time_keys = ['DATE-OBS','DATE-END','TSTART','TSTOP',
                 'MJDREF','MJDREFI','MJDREFF','TIMEZERO','LEAPINIT','CLOCKAPP',
                 'TIMEZERO','ONTIME','EXPOSURE','NAXIS2','TIMESYS']
    all_header_keys = basic_keys + time_keys
    for key in all_header_keys:
        meta_data['INFO_FROM_HEADER'][key] = key

    event_object = Event(
        time_array=events,det_array=dets,
        pi_array=pi,mission='NICER',meta_data=meta_data
        )

    data = {'event':event_object,'texp':texp,'n_events':len(events),
            'tres':tres,'cr':cr,'n_bins':nbins,'low_ch':50,'high_ch':1000,
            'texp':texp,'header_keys':all_header_keys}
    
    return data

@pytest.fixture(scope='class')
def fake_white_noise_lc(tres=0.01,nbins=5000,cr=5,low_en=0.5,high_en=10):
    events = poi_events(tres=tres,nbins=nbins,cr=cr)
    time_bin_edges = np.linspace(0,nbins*tres,nbins+1,dtype=np.double)
    time_bins_center = np.linspace(0+tres/2.,nbins*tres-tres/2.,nbins,dtype=np.double)
    hist, dummy = np.histogram(events,time_bin_edges)
    notes = {}
    notes['STEF1'] = 'This is a test note'    
    meta_data = {}
    meta_data['MISSION'] = 'NICER'
    lc = Lightcurve(time_array = time_bins_center,count_array = hist,
                    low_en=low_en,high_en=high_en,
                    notes=notes, meta_data = meta_data)
    data = {'lc':lc,'std':np.std(hist),'n_events':len(events),
            'n_bins':len(time_bins_center),
            'cr':len(events)/tres/nbins,'tres':tres,
            'low_en':low_en,'high_en':high_en}
    return data

@pytest.fixture(scope='class')
def fake_gti():
    start = np.array([0.,10.,30])
    stop  = np.array([7.,20.,43.])
    gti = Gti(start,stop)
    return gti