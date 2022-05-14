import numpy as np 
import pandas as pd 
from random import shuffle,random

from saturnx.core.gti import Gti,clean_gti


class TestGti:

    def setup_class(self):

        self.n = 10
        self.start = np.array([i for i in range(self.n)])
        shuffle(self.start)
        self.stop = np.array([round(i+random()*self.n/4.,1) for i in self.start])
        self.dur = self.stop-self.start

    def test_empty_init(self):

        gti = Gti()

        assert isinstance(gti,Gti)
        assert isinstance(gti,pd.DataFrame)

        assert 'start' in gti.columns 
        assert 'stop' in gti.columns 
        assert 'dur' in gti.columns 

        assert len(gti.start) == 0
        assert len(gti.stop) == 0
        assert len(gti.dur) == 0 

        assert len(gti.meta_data) == 2 
        assert 'HISTORY' in gti.meta_data.keys()
        assert 'GTI_CRE_DATE' in  gti.meta_data['HISTORY'].keys()
        assert 'NOTES' in gti.meta_data.keys()
        assert gti.meta_data['NOTES'] == {}

    def test_init(self):

        gti = Gti(self.start,self.stop)
        gti_ori = Gti(self.start,self.stop,clean=False)

        assert isinstance(gti,Gti)
        assert isinstance(gti,pd.DataFrame)

        assert 'start' in gti.columns 
        assert 'stop' in gti.columns 
        assert 'dur' in gti.columns 

        #assert len(gti) == self.n

        print(gti)
        print(gti_ori)
        assert gti.start.iloc[0] == min(self.start) 
        assert len(gti_ori) == self.n 

    def test_comparison_operators(self):

        gti = Gti(self.start,self.stop)
        start,stop = clean_gti(self.start,self.stop)
        dur = stop-start
        value = dur.mean()

        print('value',value)
        print(gti)
        assert len(gti < value) == len(start[dur<value])
        assert len(gti <= value) == len(start[dur<=value])
        assert len(gti == value) == len(start[dur==value])
        assert len(gti > value) == len(start[dur>value])
        assert len(gti >= value) == len(start[dur>=value])
        assert len(gti != value) == len(start[dur!=value])

    def test_clean_gti(self):
        input_raw_gti = [
            (1,5),(2,3),(7,9),(8,10),(12,13),(13,15),(18,19),
            (17,18.5),(17.5,18),(18.5,20),(19,20),(17,20)
            ]
        shuffle(input_raw_gti)
        input_start = [raw_gti[0] for raw_gti in input_raw_gti]
        input_stop  = [raw_gti[1] for raw_gti in input_raw_gti]
        gti = Gti(start_array=input_start,stop_array=input_stop,clean=True)
        expected_gti = Gti(start_array=[1.,7.,12.,17.],stop_array=[5.,10.,15.,20.],clean=True)

        assert len(gti) == len(expected_gti)
        assert gti.start.equals(expected_gti.start)
        assert gti.stop.equals(expected_gti.stop)
        assert gti.dur.equals(expected_gti.dur)
        assert gti.gap.equals(expected_gti.gap)