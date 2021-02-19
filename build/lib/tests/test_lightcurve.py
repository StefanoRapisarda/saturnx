from ..lightcurve import Lightcurve, LightcurveList
import pandas as pd
import numpy as np
from time import ctime
from ..utilities import timmer_koenig
import pytest

class TestLightcurve:

    def setup_class(self):
        # To create an event object with 7 000 events in 1 s
        self.time_bins = 1000
        self.t_res = 0.1
        self.cr = 200
        self.std = 0.5

        self.t_dur = self.time_bins*self.t_res
        self.t,self.counts=timmer_koenig(self.t_dur,self.t_res,0.1,self.cr,self.std)

        self.le = 0.5
        self.he = 10.
      
        self.now = ctime()
        self.history = {self.now:'Creation date'}
        self.notes = {f'Stefano_{self.now}':'Created for testing purposes'}

    def test_init_empty(self):

        lc = Lightcurve()

        assert isinstance(lc,Lightcurve)
        assert isinstance(lc,pd.DataFrame)

        assert 'time' in  lc
        assert 'counts' in lc

        assert lc.low_en == None 
        assert lc.high_en == None

        assert lc.tres == None
        assert lc.texp == None
        assert lc.tot_counts == None
        assert lc.cr == None

        assert lc.notes == {}
        assert lc.history == {}

    def test_init(self):

        lc = Lightcurve(self.t,self.counts,
        self.le,self.he,self.notes,self.history)

        assert isinstance(lc,Lightcurve)
        assert isinstance(lc,pd.DataFrame)

        assert 'time' in  lc
        assert 'counts' in lc       

        assert len(lc) == self.time_bins
        assert lc.time.iloc[0] == self.t[0]
        assert lc.time.iloc[-1] == self.t[-1]

        assert lc.low_en == self.le
        assert lc.high_en == self.he

        assert np.isclose(lc.tres,self.t_res,atol=self.t_res/1000.,rtol=0) 
        assert np.isclose(lc.texp,self.t_dur,atol=self.t_res/100.,rtol=0)
        assert np.round(lc.tot_counts) == np.round(self.cr*self.t_dur)
        assert np.round(lc.cr) == self.cr

        assert isinstance(lc.notes,dict)
        assert isinstance(lc.history,dict)

        assert 'Creation' in lc.history[self.now]
        assert 'Stefano_{}'.format(self.now) in lc.notes

    def test_add(self):

        lc1 = Lightcurve(self.t,self.counts,
        self.le,self.he,self.notes,self.history)
        t2,c2 = timmer_koenig(self.t_dur,self.t_res,0.1,150,self.std)
        lc2 = Lightcurve(t2,c2,self.le,self.he)

        lc3 = lc1 + lc2

        assert isinstance(lc3,Lightcurve)
        assert isinstance(lc3,pd.DataFrame)
        assert len(lc3) == len(lc1)
        assert round(lc3.tot_counts) == round(lc1.tot_counts)+round(lc2.tot_counts)

        assert np.isclose(lc3.time.iloc[0],self.t_res/2.,atol=self.t_res/100.,rtol=0)
        #print(lc3.counts)
        pd.testing.assert_series_equal(lc3.counts,pd.Series(self.counts+c2,name='counts'))

    def test_mul(self):

        lc = Lightcurve(self.t,self.counts,
        self.le,self.he,self.notes,self.history)       
        
        lc2 = lc*3

        assert isinstance(lc2,Lightcurve)
        assert isinstance(lc2,pd.DataFrame)
        assert len(lc2) == len(lc)
        pd.testing.assert_frame_equal(lc*3,3*lc)
        assert round(lc2.tot_counts) == round(lc.tot_counts*3)
        assert round(lc2.cr) == round(lc.cr*3)
        assert lc2.tres == lc.tres
        assert lc2.texp == lc.texp 

        pd.testing.assert_series_equal(lc2.counts,pd.Series(self.counts*3,name='counts'))

    def test_div(self):

        lc = Lightcurve(self.t,self.counts,
        self.le,self.he,self.notes,self.history)       
        
        lc2 = lc/3

        assert isinstance(lc2,Lightcurve)
        assert isinstance(lc2,pd.DataFrame)
        assert len(lc2) == len(lc)
        assert round(lc2.tot_counts) == round(lc.tot_counts/3)
        assert round(lc2.cr) == round(lc.cr/3)
        assert lc2.tres == lc.tres
        assert lc2.texp == lc.texp 

        pd.testing.assert_series_equal(lc2.counts,pd.Series(self.counts/3,name='counts'))

    def test_split(self):

        lc = Lightcurve(self.t,self.counts,
        self.le,self.he,self.notes,self.history) 

        lcl = lc.split()

        assert isinstance(lcl,LightcurveList)
        assert len(lcl) == int(self.t_dur/16)

        for lc in lcl:
            assert np.isclose(lc.texp,16.,atol=self.t_res/4.,rtol=0)
            assert np.isclose(lc.tres,self.t_res,atol=self.t_res/100.,rtol=0)



class TestLightcurveList:

    def setup_class(self):
        self.le = 0.5
        self.he = 10.

        self.time_bins1 = 1000
        self.t_res1 = 0.1
        self.t_dur1 = self.time_bins1*self.t_res1

        t1,c1 = timmer_koenig(self.t_dur1,self.t_res1,0.1,150,0.5)
        lc1 = Lightcurve(t1,c1,self.le,self.he)
        t2,c2 = timmer_koenig(self.t_dur1,self.t_res1,0.1,200,0.7)
        lc2 = Lightcurve(t2,c2,self.le,self.he)
        t3,c3 = timmer_koenig(self.t_dur1,self.t_res1,0.1,350,0.3)
        lc3 = Lightcurve(t3,c3,self.le,self.he)

        self.lcl1 = [lc1,lc2,lc3]
        self.counts1 = [c1,c2,c3]

        self.time_bins2 = 800
        self.t_res2 = 0.8
        self.t_dur2 = self.time_bins2*self.t_res2

        t4,c4 = timmer_koenig(self.t_dur2,self.t_res2,0.1,150,0.5)
        lc4 = Lightcurve(t4,c4,self.le,self.he)              
        t5,c5 = timmer_koenig(self.t_dur2,self.t_res2,0.1,150,0.5)
        lc5 = Lightcurve(t5,c5,self.le,self.he)  

        self.lcl2 = [lc4,lc5]
        self.counts2 = [c4,c5]

    def test_init(self):

        lcl = LightcurveList(self.lcl1+self.lcl2)

        assert isinstance(lcl,LightcurveList)
        assert len(lcl) == len(self.lcl1)+len(self.lcl2)

    def test_bad_init(self):
        with pytest.raises(TypeError, match=".* elements must be .*"):
            lcl = LightcurveList(['ciao'])

    def test_len(self):
        lcl = LightcurveList(self.lcl1+self.lcl2)
        assert len(lcl)==len(self.lcl1)+len(self.lcl2)

    def test_iter(self):
        lcl = LightcurveList(self.lcl1+self.lcl2)
        i = 0
        for lc in lcl:
            assert isinstance(lc,Lightcurve)
            pd.testing.assert_frame_equal(lc,(self.lcl1+self.lcl2)[i])
            i += 1

    def test_tot_counts(self):
        lcl = LightcurveList(self.lcl1+self.lcl2)
        assert lcl.tot_counts == np.array([i.tot_counts for i in (self.lcl1+self.lcl2)]).sum()

    def test_texp(self):
        lcl = LightcurveList(self.lcl1+self.lcl2)
        assert lcl.texp == np.array([i.texp for i in (self.lcl1+self.lcl2)]).sum()   

    def test_texp(self):
        lcl = LightcurveList(self.lcl1+self.lcl2)
        assert lcl.cr == np.array([i.cr for i in (self.lcl1+self.lcl2)]).mean()  

    def test_bad_mean(self):
        lcl = LightcurveList(self.lcl1+self.lcl2)
        with pytest.raises(ValueError,match='Lightcurves have different dimensions'):
            test = lcl.mean 
            
    def test_bad_mean(self):
        lcl1 = LightcurveList(self.lcl1)
        pd.testing.assert_series_equal(lcl1.mean.counts,pd.Series(np.vstack(self.counts1).mean(axis=0),name='counts'))
        lcl2 = LightcurveList(self.lcl2)
        pd.testing.assert_series_equal(lcl2.mean.counts,pd.Series(np.vstack(self.counts2).mean(axis=0),name='counts'))
    
    def test_join(self):
        lcl = LightcurveList(self.lcl1)
        lc = lcl.join()

        assert isinstance(lc,Lightcurve)
        assert isinstance(lc,pd.DataFrame)

        assert np.isclose(lc.tres,self.t_res1,atol=self.t_res1/100.,rtol=0)
        assert len(lc) == self.time_bins1*3

    def test_join_fail(self):
        lcl = LightcurveList(self.lcl1+self.lcl2)      
        with pytest.raises(ValueError,match='Cannot concatenate Lightcurves with different time res'):
            lc = lcl.join()  

    #def test_split(self):
    #    lcl = LightcurveList(self.lcl1+self.lcl2) 



        