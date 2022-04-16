from saturnx.core.gti import Gti,clean_gti
import numpy as np 
import pandas as pd 
from random import shuffle,random


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

    def test_comp(self):

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