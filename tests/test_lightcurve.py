from multiprocessing import Value
from typing import Type
from numpy.core.overrides import array_function_from_dispatcher
import pandas as pd
import numpy as np
import math
import pytest

from kronos.core.lightcurve import Lightcurve, LightcurveList

class TestLightcurveInit:

    def test_empty_lc(self):
        lc = Lightcurve()
        # Columns
        assert 'time' in lc.columns
        assert 'counts' in lc.columns
        assert 'rate' in lc.columns
        assert lc.time.empty
        assert lc.counts.empty
        assert lc.rate.empty
        # Attributes
        assert lc.low_en is None
        assert lc.high_en is None
        assert 'LC_CRE_DATE' in lc.meta_data.keys()
        assert len(lc.meta_data) == 1
        assert lc.notes == {}
        # Properties
        assert lc.tot_counts == 0
        assert lc.cr == 0
        assert lc.cr_std == 0
        assert lc.texp == 0
        assert lc.tres == 0
        assert lc.rms == 0
        assert lc.frac_rms == 0

    def test_lc(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        tres = fake_white_noise_lc['tres']
        std = fake_white_noise_lc['std']
        n_events = fake_white_noise_lc['n_events']
        cr = fake_white_noise_lc['cr']
        # Columns
        assert 'time' in lc.columns
        assert 'counts' in lc.columns
        assert 'rate' in lc.columns
        assert not lc.time.empty
        assert not lc.counts.empty
        assert not lc.rate.empty
        assert np.array_equal(lc.rate,lc.counts/lc.tres)
        assert len(lc) == 5000
        # Attributes
        assert lc.low_en == 0.5
        assert lc.high_en == 10
        assert 'LC_CRE_DATE' in lc.meta_data.keys()
        assert lc.notes['STEF1'] == 'This is a test note'
        # Properties
        assert lc.tot_counts == n_events
        assert math.isclose(np.mean(lc.rate),lc.cr)
        assert lc.texp == np.round(lc.tres*len(lc))
        assert lc.tres == 0.01
        assert np.isclose(lc.rms,
                          np.sqrt(lc.cr_std**2+np.mean(lc.counts)**2),
                          atol=lc.rms/1000,rtol=0)
        assert np.isclose(lc.frac_rms,
                          np.sqrt(lc.cr_std**2/np.mean(lc.counts)**2),
                          atol=lc.frac_rms/1000,rtol=0)


class TestLightcurveAdd:

    def test_bad_input(self,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        lc2 = Lightcurve(np.linspace(0,100),np.linspace(0,100)) 
        lc3 = Lightcurve(np.linspace(0,100,len(lc1)),
                         np.linspace(0,100,len(lc1))) 
        with pytest.raises(TypeError):
            lc4 = lc1 + lc2
        with pytest.raises(TypeError):
            lc4 = lc1 + lc3
        with pytest.raises(TypeError):
            lc4 = lc1 + 'ciao'

    @pytest.mark.parametrize('value',[0,10,'10'],ids=['zero','ten','test_string'])
    def test_add_number(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        if type(value) == str: value = eval(value)
        lc2 = lc1 + value
        assert type(lc2) == type(Lightcurve())
        assert len(lc1) == len(lc2)
        assert lc1.columns.equals(lc2.columns)
        assert lc1.tres == lc2.tres
        assert lc1.texp == lc2.texp
        assert np.isclose(lc2.cr,lc1.cr+(value*len(lc1)/lc1.texp))
        assert np.array_equal(lc1.time,lc2.time)
        assert np.array_equal(lc1.counts+value,lc2.counts)
        assert np.array_equal(lc2.rate[0:3],((lc1.counts+value)/lc1.tres)[0:3]),\
            [lc2.rate[0],lc2.counts[0],
             lc1.rate[0],lc1.counts[0]]
        assert lc1.meta_data == lc2.meta_data
        assert lc1.notes == lc2.notes
        assert lc1.low_en == lc2.low_en
        assert lc1.high_en == lc2.high_en

    def test_add_lc(self,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        lc2 = lc1 + lc1

        assert isinstance(lc2,Lightcurve)
        assert len(lc2) == len(lc1)
        assert lc1.tres == lc2.tres
        assert lc2.cr == 2*lc1.cr
        assert np.array_equal(lc2.counts,lc1.counts*2)
        assert np.array_equal(lc1.time,lc2.time)
        assert lc2.notes['STEF1'] == 'This is a test note'
        assert 'LC_CRE_DATE' in  lc2.meta_data.keys()
        assert lc1.low_en == lc2.low_en
        assert lc1.high_en == lc2.high_en  


class TestLightcurveMul:

    def test_bad_input(self,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        wrong_inputs = ['ciao',{},tuple([])]
        for wrong_input in wrong_inputs:
            with pytest.raises(TypeError):
                lc2 = lc1*wrong_input

    @pytest.mark.parametrize('value',
        [0,1,5,5.5,np.float(5.5),np.double(5.5),np.int(3),'123'],
        ids=['zero','one','int','float','np_float','np.double','np.int','string'])
    def test_single_value_mul(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        if type(value) == str: value = eval(value)
        lc2 = lc1 * value
        assert type(lc2) == type(Lightcurve())
        assert len(lc2) == len(lc1)
        assert lc1.columns.equals(lc2.columns)
        assert lc2.notes == lc1.notes
        assert lc1.meta_data == lc2.meta_data
        assert lc1.low_en == lc2.low_en
        assert lc1.high_en == lc1.high_en
        assert lc1.tres == lc2.tres
        assert lc1.texp == lc2.texp
        assert np.round(lc2.cr) == np.round(lc1.cr*value)
        assert np.array_equal(lc1.time,lc2.time)
        assert np.array_equal(lc2.counts,lc1.counts*value)
        assert np.array_equal(lc2.rate,lc1.rate*value)
    
    @pytest.mark.parametrize('array',[['',2,'ciao']],
        ids = ['list'])
    def test_multi_value_mul_bad_input(self,array,fake_white_noise_lc):    
        lc1 = fake_white_noise_lc['lc']
        with pytest.raises(Exception):
            lc_out = lc1*array   

    @pytest.mark.parametrize('array',
        [[0,1,2,3],np.array([0,1,2,3])],
        ids = ['list','numpy_array'])
    def test_multi_value_mul(self,array,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        lc_out = lc1*array
        assert type(lc_out) == type(LightcurveList())
        assert len(lc_out) == len(array)
        for lc2,value in zip(lc_out,array):
            assert type(lc2) == type(Lightcurve())
            assert len(lc2) == len(lc1)
            assert lc1.columns.equals(lc2.columns)
            assert lc2.notes == lc1.notes
            assert lc1.meta_data == lc2.meta_data
            assert lc1.low_en == lc2.low_en
            assert lc1.high_en == lc1.high_en
            assert lc1.tres == lc2.tres
            assert lc1.texp == lc2.texp
            assert np.round(lc2.cr) == np.round(lc1.cr*value)
            assert np.array_equal(lc1.time,lc2.time)
            assert np.array_equal(lc2.counts,lc1.counts*value)  
            assert np.array_equal(lc2.rate,lc1.rate*value)          
        
    @pytest.mark.parametrize('array',
        [[0,1,2,3],np.array([0,1,2,3])],
        ids = ['list','numpy_array'])
    def test_multi_value_rmul(self,array,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        lc_out = array*lc1
        assert type(lc_out) == type(LightcurveList())
        assert len(lc_out) == len(array)
        for lc2,value in zip(lc_out,array):
            assert type(lc2) == type(Lightcurve())
            assert len(lc2) == len(lc1)
            assert lc1.columns.equals(lc2.columns)
            assert lc2.notes == lc1.notes
            assert lc1.meta_data == lc2.meta_data
            assert lc1.low_en == lc2.low_en
            assert lc1.high_en == lc1.high_en
            assert lc1.tres == lc2.tres
            assert lc1.texp == lc2.texp
            assert np.round(lc2.cr) == np.round(lc1.cr*value)
            assert np.array_equal(lc1.time,lc2.time)
            assert np.array_equal(lc2.counts,lc1.counts*value)  
            assert np.array_equal(lc2.rate,lc1.rate*value)               


class TestLightcurveTruediv:

    @pytest.mark.parametrize('value',[{},'',tuple([1,2])],
        ids = ['dict','string','tuple'])    
    def test_div_bad_input(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        if value == 0:
            with pytest.raises(TypeError):
                lc2 = lc1/value

    @pytest.mark.parametrize('value',[0,1,2,3,'4'],
        ids = ['zero','uno','due','tre','four_string'])
    def test_div(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        if value == 0:
            with pytest.raises(ValueError):
                lc2 = lc1/value
        else:
            if type(value) == str: value = eval(value)
            lc2 = lc1/value
            assert type(lc2) == type(Lightcurve())
            assert len(lc2) == len(lc1)
            assert lc1.columns.equals(lc2.columns)
            assert lc2.notes == lc1.notes
            assert lc1.meta_data == lc2.meta_data
            assert lc1.low_en == lc2.low_en
            assert lc1.high_en == lc1.high_en
            assert lc1.tres == lc2.tres
            assert lc1.texp == lc2.texp
            assert np.round(lc2.cr) == np.round(lc1.cr/value)
            assert np.array_equal(lc1.time,lc2.time)
            assert np.array_equal(lc2.counts,lc1.counts/value)
            assert np.array_equal(lc2.rate[1:10],lc1.rate[1:10]/value)


class TestLightcurveSplit:

    @pytest.mark.parametrize('value',[0,'ciao',{},[],tuple([])],
        ids = ['zero','string','dict','list','tuple'])
    def test_bad_seg(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        if value == 0:
            with pytest.raises(ValueError):
                lc2 = lc1.split(value)
        else:
            with pytest.raises(TypeError):
                lc2 = lc1.split(value)

    def test_split_gti(self,fake_gti,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        tdurs = [fake_gti.dur.iloc[i] for i in range(len(fake_gti))]
        lc_list = lc1.split(fake_gti)

        assert type(lc_list) == type(LightcurveList())
        assert len(lc_list) == len(tdurs)
        for i,lc in enumerate(lc_list):
            assert type(lc) == type(Lightcurve())
            assert np.isclose(lc.texp ,tdurs[i])
            assert np.isclose(lc.tres ,lc1.tres)
            assert lc.notes == lc1.notes
            assert 'SPLITTING_GTI' in  lc.meta_data.keys()
            assert lc.meta_data['N_GTIS'] == len(tdurs)
            assert lc.meta_data['GTI_INDEX'] == i
            assert lc.low_en == lc1.low_en
            assert lc.high_en == lc1.high_en

    @pytest.mark.parametrize('value',[100,0.12,13,11,'10'],
        ids = ['large_seg','0.1','1','10','10str'])
    def test_split_seg(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        lc_list = lc1.split(value)
        
        assert type(lc_list) == type(LightcurveList())
        if value == 100:
            assert len(lc_list) == 1
            assert type(lc_list[0]) == type(Lightcurve())
            assert np.array_equal(lc_list[0].time,lc1.time)
            assert np.array_equal(lc_list[0].counts,lc1.counts)
            assert np.array_equal(lc_list[0].rate,lc1.rate)
            assert lc_list[0].notes == lc1.notes
            assert lc_list[0].meta_data == lc1.meta_data
            assert lc_list[0].low_en == lc1.low_en
            assert lc_list[0].high_en == lc1.high_en
        else:
            if type(value) == str: value=eval(value)
            n_segs = int(lc1.texp/value)
            for i,lc in enumerate(lc_list):
                assert type(lc) == type(Lightcurve())
                assert lc.texp ,value
                assert lc.tres ,lc1.tres
                assert lc.notes == lc1.notes
                assert 'SPLITTING_SEG' in  lc.meta_data.keys()
                assert lc.meta_data['N_SEGS'] == n_segs,lc1.texp
                assert lc.meta_data['SEG_DUR'] == value
                assert lc.meta_data['SEG_INDEX'] == i
                assert lc.low_en == lc1.low_en
                assert lc.high_en == lc1.high_en


class TestLightcurveRebin:

    @pytest.mark.parametrize('value',[-23,0,'ciao',{},tuple([])],
        ids = ['-23','zero','string','dict','tuple'])
    def test_bad_rebin_single_factor(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        if value == 0 or value == -23:
            with pytest.raises(ValueError):
                lc2 = lc1.rebin(value)
        else:
            with pytest.raises(TypeError):
                lc2 = lc1.rebin(value)   

    @pytest.mark.parametrize('value',[1,2,3,'1','2','3'],
        ids = ['one','two','three','one_string','two_string','three_string'])
    def test_lin_rebin_single_value(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        if type(value) == str: value=eval(value)
        lc2 = lc1.rebin(value)

        assert type(lc2) == type(Lightcurve())
        if value == 1:
            assert len(lc1) == len(lc2)
            assert lc1.time.equals(lc2.time)
            assert lc1.counts.equals(lc2.counts)
            assert lc1.rate.equals(lc2.rate)
            assert lc1.tres == lc2.tres
        else:
            assert len(lc2) == int(len(lc1)/value) 
            assert lc2.tres == lc1.tres*value 
            assert lc2.texp == (int(len(lc1)/value))*lc2.tres
        assert lc1.columns.equals(lc2.columns)
        assert lc1.tot_counts == lc2.tot_counts
        assert np.mean(lc1.cr) == np.round(np.mean(lc2.cr),
            int(abs(math.log10(lc1.cr/1000))))
        assert lc1.low_en == lc2.low_en
        assert lc1.high_en == lc2.high_en
        assert lc1.notes == lc2.notes
        assert 'REBINNING' in lc2.meta_data.keys()
        assert lc2.meta_data['REBIN_FACTOR'] == [value]

    @pytest.mark.parametrize('value',
        [[0],['ciao','minchia'],[{},{}],[tuple([]),tuple]],
        ids = ['zero','string','dict','tuple'])
    def test_bad_rebin_list_factor(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        if 0 in value:
            with pytest.raises(ValueError):
                lc2 = lc1.rebin(value)
        else:
            with pytest.raises(TypeError):
                lc2 = lc1.rebin(value)  

    @pytest.mark.parametrize('value',[[1,'2']])
    def test_lin_rebin_list(self,value,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        if type(value) == str: value=eval(value)
        lc2 = lc1.rebin(value)

        assert type(lc2) == type(Lightcurve())
        assert len(lc2) == int(len(lc1)/2)
        assert lc1.columns.equals(lc2.columns)
        assert lc2.tres == lc1.tres*2
        assert lc2.texp == (int(len(lc1)/2))*lc2.tres
        assert lc1.tot_counts == lc2.tot_counts
        assert np.mean(lc1.cr) == np.round(np.mean(lc2.cr),
            int(abs(math.log10(lc2.tres/10))))
        assert lc1.low_en == lc2.low_en
        assert lc1.high_en == lc2.high_en
        assert lc1.notes == lc2.notes
        assert 'REBINNING' in lc2.meta_data.keys()
        assert lc2.meta_data['REBIN_FACTOR'] == [1,2]  

class TestLightcurveFromEvent:

    def test_bad_input(self):
        pass

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



        