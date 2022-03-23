from types import TracebackType
import numpy as np
import pandas as pd
import math
import pytest

from saturnx.core.lightcurve import Lightcurve,LightcurveList
from saturnx.core.power import PowerSpectrum,PowerList

class TestEmptyPower:

    def test_empty_power_type(self):
        power = PowerSpectrum()

        assert type(power) == type(PowerSpectrum())

    def test_empty_power_arrays(self):
        power = PowerSpectrum()

        assert power.freq.empty
        assert power.power.empty
        assert power.spower.empty

    def test_empty_power_attributes(self):
        power = PowerSpectrum()

        assert power.weight == 1
        assert power.low_en == None
        assert power.high_en == None
        assert power.leahy_norm == None
        assert power.rms_norm == None
        assert power.poi_level == None

        assert power.df == None
        assert power.nf == None
        assert power.a0 == None
        assert power.cr == None
    
    def test_empty_power_meta_data(self,mocker):
        mocker.patch('saturnx.core.power.my_cdate',
            return_value='test_current_date')
        power = PowerSpectrum()

        assert len(power.meta_data) == 1
        assert power.meta_data['PW_CRE_DATE'] == 'test_current_date'
        assert power.notes == {}


class TestFromLcSingle:
    
    @pytest.mark.parametrize('wrong_input',
        [123,np.array([1,2,3]),'ciao',{},tuple(),[1,2,3]],
        ids = ['int','numpy array','str','dict','tuple','list'])
    def test_wrong_input(self,wrong_input):
        with pytest.raises(TypeError):
            power = PowerSpectrum.from_lc(wrong_input)

    def test_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)

        assert type(power) == type(PowerSpectrum())
        assert len(power) == len(lc)
        assert not power.freq.empty
        assert not power.power.empty
        assert power.spower.iloc[0] == None

    def test_attributes(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)

        assert power.weight == 1
        assert power.rms_norm == None
        assert power.leahy_norm == None

        assert power.low_en == 0.5
        assert power.high_en == 10.

        assert power.df == 1./(lc.texp)
        assert power.nf == 1./2/lc.tres
        assert power.a0 == lc.tot_counts
        assert power.cr == lc.cr

    def test_meta_data(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)

        assert power.notes == {}
        assert power.meta_data['PW_CRE_MODE'] == 'Power computed from Lightcurve'
        assert power.meta_data['TIME_RES'] == lc.tres
        assert power.meta_data['MISSION'] == 'NICER'

    def test_parcival_theorem(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)

        assert np.isclose(np.sum(lc.counts**2),np.sum(power.power)/len(power))


class TestFromLcMulti:

    def test_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)
        power_list = PowerSpectrum.from_lc(lc_list)

        assert type(power_list) == type(PowerList())
        assert len(power_list) == len(lc_list)
        for power in power_list:
            assert not power.freq.empty
            assert not power.power.empty
            assert power.spower.iloc[0] == None

    def test_attributes(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)
        power_list = PowerSpectrum.from_lc(lc_list)

        for i,power in enumerate(power_list):
            assert power.weight == 1
            assert power.rms_norm == None
            assert power.leahy_norm == None

            assert power.low_en == 0.5
            assert power.high_en == 10.

            assert power.df == 1./10
            assert power.nf == 1./2/lc_list[0].tres
            assert power.a0 == lc_list[i].tot_counts
            assert np.isclose(power.cr,lc_list[i].cr)

    def test_meta_data(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)
        power_list = PowerSpectrum.from_lc(lc_list)

        for i,power in enumerate(power_list):
            assert power.notes == {}
            assert power.meta_data['PW_CRE_MODE'] == 'Power computed from Lightcurve'
            assert power.meta_data['TIME_RES'] == lc_list[0].tres
            assert power.meta_data['MISSION'] == 'NICER'

            assert power.meta_data['SEG_DUR'] == 10
            assert power.meta_data['N_SEGS'] == len(lc_list)
            assert power.meta_data['SEG_INDEX'] == i

    def test_parcival_theorem(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)
        power_list = PowerSpectrum.from_lc(lc_list)

        for i,power in enumerate(power_list):
            assert np.isclose(np.sum(lc_list[i].counts**2),np.sum(power.power)/len(power))


class TestNormalizeLeahy:

    def test_leahy_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        leahy = power.normalize()

        assert type(leahy) == type(PowerSpectrum())
        assert power.freq.equals(leahy.freq)
        assert np.allclose(leahy.power,power.power*2/power.a0)

    def test_leahy_attributes(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        leahy = power.normalize()

        assert leahy.leahy_norm == 2./power.a0
        assert leahy.rms_norm == None
        assert np.isclose(leahy.a0,power.a0)
        assert leahy.low_en == power.low_en
        assert leahy.high_en == power.high_en
        assert leahy.cr == power.cr
        assert leahy.nf == power.nf
        assert leahy.df == power.df

    def test_leahy_meta_data(self,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test date')
        leahy = power.normalize()

        assert leahy.notes == power.notes
        assert leahy.meta_data['NORMALIZING'] == 'Test date'
        assert leahy.meta_data['NORM_MODE'] == 'Leahy'


class TestNormalizeRMS:

    def test_rms_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        rms = power.normalize('rms')

        assert type(rms) == type(PowerSpectrum())
        assert power.freq.equals(rms.freq)
        assert np.allclose(rms.power,power.power*2/power.a0/power.cr)

    def test_rms_attributes(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        rms = power.normalize('rms')

        assert rms.leahy_norm == 2./power.a0
        assert np.isclose(rms.rms_norm,1./power.cr)
        assert np.isclose(rms.a0,power.a0)
        assert rms.low_en == power.low_en
        assert rms.high_en == power.high_en
        assert np.isclose(rms.cr,power.cr)
        assert rms.nf == power.nf
        assert rms.df == power.df

    def test_rms_meta_data(self,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test date')
        rms = power.normalize('rms')

        assert rms.notes == power.notes
        assert rms.meta_data['NORMALIZING'] == 'Test date'
        assert rms.meta_data['NORM_MODE'] == 'FRAC_RMS'


class TestCompFracRms:

    def test_empty_power(self):
        pw = PowerSpectrum()

        rms, srms = pw.comp_frac_rms()

        assert rms == None
        assert srms == None

    def test_freq_input(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)

        freq_bands = [[0,1000],[-34,100],['0','1000'],['-45','254']]
        expected_rms, expected_srms = power.comp_frac_rms()

        for freq_band in freq_bands:
            rms, srms = power.comp_frac_rms(low_freq=freq_band[0],high_freq=freq_band[1])
            assert rms == expected_rms
            assert srms == expected_srms

    def test_freq_error(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)

        with pytest.raises(ValueError):
            power.comp_frac_rms(low_freq=100,high_freq=56)

    def test_pos_only(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        power_zeros = PowerSpectrum.from_lc(lc)
        middle_freq = power.nf/2
        mask = (power.freq <= middle_freq) 
        power_zeros.power[~mask] = 0

        rms, srms = power_zeros.comp_frac_rms()
        expected_rms, expected_srms = power.comp_frac_rms(high_freq=middle_freq)

        assert np.isclose(rms,expected_rms,rms/1000)
        assert srms == expected_srms

    def test_not_norm_pw_rms(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)

        assert power.leahy_norm == None
        assert power.rms_norm == None

        rms,srms = power.comp_frac_rms()

        assert np.isclose(rms,lc.frac_rms)
        assert srms == None

    def test_leahy_norm_pw_rms(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc).normalize('leahy')

        assert power.rms_norm == None

        rms,srms = power.comp_frac_rms()

        assert np.isclose(rms,lc.frac_rms)
        assert srms == None

    def test_rms_norm_pw_rms(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        rms_power = power.normalize('rms')

        rms, srms = rms_power.comp_frac_rms()
        
        assert np.isclose(rms,lc.frac_rms)
        assert srms == None

    @pytest.mark.parametrize('high_freq',[10,20,30,40,50],
        ids = ['10','20','30','40','50'])
    def test_all_norms_high_freq(self,high_freq,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        leahy_power = power.normalize('leahy')
        rms_power = power.normalize('rms')

        rms,srms = power.comp_frac_rms(high_freq = high_freq)
        leahy_rms,leahy_srms = leahy_power.comp_frac_rms(high_freq = high_freq)
        rms_rms,rms_srms = rms_power.comp_frac_rms(high_freq = high_freq)

        assert np.isclose(rms,leahy_rms)
        assert np.isclose(leahy_rms,rms_rms)

        assert srms == None
        assert leahy_srms == None
        assert rms_srms == None

    @pytest.mark.parametrize('low_freq',[10,20,30,40,50],
        ids = ['10','20','30','40','50'])
    def test_all_norms_low_freq(self,low_freq,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        leahy_power = power.normalize('leahy')
        rms_power = power.normalize('rms')

        rms,srms = power.comp_frac_rms(low_freq = low_freq)
        leahy_rms,leahy_srms = leahy_power.comp_frac_rms(low_freq = low_freq)
        rms_rms,rms_srms = rms_power.comp_frac_rms(low_freq = low_freq)

        assert np.isclose(rms,leahy_rms)
        assert np.isclose(leahy_rms,rms_rms)

        assert srms == None
        assert leahy_srms == None
        assert rms_srms == None


class TestSubPoi:

    def test_value_none_low_freq_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        middle_freq = power.nf/2
        poi_power = power.sub_poi(low_freq=middle_freq)

        poi_level = np.mean(power.power[power.freq>=middle_freq])

        current = poi_power.power[power.freq>0]
        expected = power.power[power.freq>0]-poi_level
        assert np.allclose(current,expected)
        assert poi_power.freq.equals(power.freq)
        assert poi_power.power.iloc[0] == power.power.iloc[0]
        assert not np.any(poi_power.spower)

    def test_value_none_low_freq_attributes(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        middle_freq = power.nf/2
        poi_power = power.sub_poi(low_freq=middle_freq)
        poi_level = np.mean(power.power[power.freq>=middle_freq])

        assert poi_power.df == power.df
        assert poi_power.nf == power.nf
        assert poi_power.a0 == power.a0
        assert poi_power.cr == power.cr

        assert poi_power.leahy_norm == None
        assert poi_power.rms_norm == None
        assert poi_power.poi_level == poi_level
        assert poi_power.low_en == power.low_en
        assert poi_power.high_en == power.high_en
        assert poi_power.weight == power.weight

    def test_value_none_low_freq_meta_data(self,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        middle_freq = power.nf/2
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test creation date')
        poi_power = power.sub_poi(low_freq=middle_freq)

        assert poi_power.meta_data['SUBTRACTING_POI'] == 'Test creation date'
        assert poi_power.meta_data['POI_RANGE'] == f'{middle_freq}-{poi_power.nf}'
        assert poi_power.notes == power.notes

    def test_value_none_high_freq_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        middle_freq = power.nf/2
        poi_power = power.sub_poi(high_freq=middle_freq)

        mask =(power.freq>0)&(power.freq<middle_freq)
        poi_level = np.mean(power.power[mask])

        current = poi_power.power[power.freq>0]
        expected = power.power[power.freq>0]-poi_level
        assert np.allclose(current,expected)
        assert poi_power.freq.equals(power.freq)
        assert poi_power.power.iloc[0] == power.power.iloc[0]
        assert not np.any(poi_power.spower)

    def test_value_none_high_freq_attributes(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        middle_freq = power.nf/2
        poi_power = power.sub_poi(high_freq=middle_freq)
        mask =(power.freq>0)&(power.freq<middle_freq)
        poi_level = np.mean(power.power[mask])

        assert poi_power.df == power.df
        assert poi_power.nf == power.nf
        assert poi_power.a0 == power.a0
        assert poi_power.cr == power.cr

        assert poi_power.leahy_norm == None
        assert poi_power.rms_norm == None
        assert poi_power.poi_level == poi_level
        assert poi_power.low_en == power.low_en
        assert poi_power.high_en == power.high_en
        assert poi_power.weight == power.weight

    def test_value_none_high_freq_meta_data(self,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        middle_freq = power.nf/2
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test creation date')
        poi_power = power.sub_poi(high_freq=middle_freq)

        assert poi_power.meta_data['SUBTRACTING_POI'] == 'Test creation date'
        assert poi_power.meta_data['POI_RANGE'] == f'{0}-{middle_freq}'
        assert poi_power.notes == power.notes

    @pytest.mark.parametrize('value',[0.5,0.236,1.245,6.345])
    def test_value_arrays(self,value,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        mocker.patch('kronos.core.power.my_cdate',return_value='Test creation date')
        poi_power = power.sub_poi(value = value)

        current = poi_power.power[power.freq>0]
        expected = power.power[power.freq>0]-value
        assert np.allclose(current,expected)
        assert poi_power.freq.equals(power.freq)
        assert poi_power.power.iloc[0] == power.power.iloc[0]
        assert not np.any(poi_power.spower)   

    @pytest.mark.parametrize('value',[0.5,0.236,1.245,6.345])
    def test_value_attributes(self,value,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        mocker.patch('kronos.core.power.my_cdate',return_value='Test creation date')
        poi_power = power.sub_poi(value = value)  

        assert poi_power.df == power.df
        assert poi_power.nf == power.nf
        assert poi_power.a0 == power.a0
        assert poi_power.cr == power.cr

        assert poi_power.leahy_norm == None
        assert poi_power.rms_norm == None
        assert poi_power.poi_level == value
        assert poi_power.low_en == power.low_en
        assert poi_power.high_en == power.high_en
        assert poi_power.weight == power.weight 

    def test_array_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        poi_list = [0.345 for i in range(len(power))]
        inputs = [poi_list,np.array(poi_list),pd.Series(poi_list)]

        expected = power.power[power.freq>0]-0.345

        for poi_array in inputs:
            poi_power = power.sub_poi(value = poi_array)

            current = poi_power.power[power.freq>0]
            assert np.allclose(current,expected)
            assert poi_power.freq.equals(power.freq)
            assert poi_power.power.iloc[0] == power.power.iloc[0]
            assert not np.any(poi_power.spower) 

    def test_array_attributes(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        poi_list = [0.345 for i in range(len(power))]
        inputs = [poi_list,np.array(poi_list),pd.Series(poi_list)] 

        for poi_array in inputs:
            poi_power = power.sub_poi(value = poi_array)

            assert poi_power.df == power.df
            assert poi_power.nf == power.nf
            assert poi_power.a0 == power.a0
            assert poi_power.cr == power.cr

            assert poi_power.leahy_norm == None
            assert poi_power.rms_norm == None
            if type(poi_array) == list:
                assert poi_power.poi_level == poi_array
            else:
                assert np.array_equal(poi_power.poi_level,poi_array)
            assert poi_power.low_en == power.low_en
            assert poi_power.high_en == power.high_en
            assert poi_power.weight == power.weight       

    @pytest.mark.parametrize('array',
        [[i for i in range(23)],np.linspace(0,34),pd.Series(np.arange(0,1,00.1))],
        ids = ['list','numpy array','pandas array'])
    def test_array_bad_inputs(self,array,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)

        with pytest.raises(ValueError):
            poi_power = power.sub_poi(value = array)


class TestRebin:

    def test_linear_rebin_one_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        rebin = power.rebin(1)
        mask = power.freq >= 0

        assert type(rebin) == type(PowerSpectrum())
        assert type(rebin.power) == pd.Series

        assert len(power[mask]) == len(rebin)
        assert np.array_equal(power.power[mask],rebin.power)
        assert np.array_equal(power.freq[mask],rebin.freq)

    def test_linear_rebin_one_attributes(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        rebin = power.rebin(1)
        mask = power.freq >= 0

        assert type(rebin) == type(PowerSpectrum())
        assert type(rebin.power) == pd.Series

        assert rebin.df == power.df
        assert rebin.nf == power.nf
        assert rebin.a0 == power.a0
        assert rebin.cr == power.cr

        assert rebin.leahy_norm == None
        assert rebin.rms_norm == None
        assert rebin.poi_level == 0
        assert rebin.low_en == power.low_en
        assert rebin.high_en == power.high_en
        assert rebin.weight == power.weight 

    @pytest.mark.parametrize('rf',[-1,1,4,-5,[1,2,-3]],
        ids = ['-1','1','4','-5','list'])
    def test_linear_rebin_one_meta_data(self,rf,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        mocker.patch('kronos.core.power.my_cdate',return_value='Test creation date')
        rebin = power.rebin(rf)

        if type(rf) == list: 
            expected_rf = rf
        else:
            expected_rf = [rf]
        assert rebin.meta_data['REBINNING'] == 'Test creation date'
        assert rebin.meta_data['REBIN_FACTOR'] == expected_rf

    def test_linear_rebin_two_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        rebin = power.rebin(2)
        mask = power.freq >= 0

        assert type(rebin) == type(PowerSpectrum())
        assert type(rebin.power) == pd.Series

        assert len(power[mask])//2 + 1 == len(rebin)


class TestPowerList:

    @pytest.mark.parametrize('bad_inputs',
        [[1,45,''],[PowerSpectrum(),{}],[tuple([])]])
    def test_bad_inputs(self,bad_inputs):
        with pytest.raises(TypeError):
            power_list = PowerList(bad_inputs)

    def test_averate_leahy_all(self,mocker,fake_white_noise_lc):
            lc = fake_white_noise_lc['lc']
            lc_split = lc.split(10)
            power_list = PowerSpectrum.from_lc(lc_split)
            power = power_list.average_leahy() 
            mask = power.freq>0 

            assert np.round(power.power[mask].mean()) == 2

class BohPower:

    def setup_class(self,t_res=0.01,time_bins=1701,cr=20000,std=0.5):

        self.time_bins = time_bins
        self.cr = cr
        self.std = std
        self.t_res = t_res

        self.t_dur = self.time_bins*self.t_res
        self.t,self.counts=timmer_koenig(self.t_dur,self.t_res,0.1,self.cr)
      
        self.history_key = 'CREATION_DATE'
        self.history_value = ctime()
        self.history = {self.history_key:self.history_value}
        self.notes_key = ctime()
        self.notes_value = 'Created for testing purposes'
        self.notes = {self.notes_key:self.notes_value}

        self.lc = Lightcurve(self.t,self.counts,0.5,10,
        self.notes,self.history)

        self.nlc = 50
        lc_list = []
        for i in range(self.nlc):
            t,counts=timmer_koenig(self.t_dur,self.t_res,0.1,self.cr,self.std)
            t += self.t_dur*i+self.t_res
            self.lc = Lightcurve(t,counts,0.5,10,
            self.notes,self.history)
            lc_list += [Lightcurve(t,counts,0.5,10,
        self.notes,self.history)]
        self.lcs = LightcurveList(lc_list)

        lc_list = []
        for i in range(self.nlc):
            t,counts=timmer_koenig(self.t_dur,self.t_res,0.1,(np.random.random()+1)*200,self.std)
            t += self.t_dur*i+self.t_res
            self.lc = Lightcurve(t,counts,0.5,10,
            self.notes,self.history)
            lc_list += [Lightcurve(t,counts,0.5,10,
        self.notes,self.history)]
        self.lcs2 = LightcurveList(lc_list)       

    def test_make_power_single(self):
        power = PowerSpectrum.from_lc(self.lc)

        # It must return an instance of the PowerSpectrum class
        assert isinstance(power,PowerSpectrum)
        assert isinstance(power,pd.DataFrame)

        # Frequency bins and time bins must be the same
        assert len(power.freq) == self.time_bins

        # Total counts must be the same   
        print('a0',power.a0,'tot_counts',self.lc.tot_counts)     
        assert np.isclose(power.a0,self.lc.tot_counts,atol=self.lc.tot_counts/1000.,rtol=0)

        # Frequency resolution must be the same
        print('df',power.df,'1/T',1./self.lc.texp)
        assert np.isclose(power.df,1./self.lc.texp,atol=power.df/1000,rtol=0)

        # Nyquist frequency myst be the same
        if len(self.lc)%2!=0:
            print('fn2',power.freq[power.freq>0].iloc[-1],'1/2dt',1./(2*self.lc.tres))
            assert np.isclose(power.freq[power.freq>0].iloc[-1],1./(2*self.lc.tres),atol=power.df,rtol=0)

        # Powers must be not NaN in all possible normalizations
        assert not power.freq.isnull().values.any()
        assert not power.power.isnull().values.any()

        leahy = power.leahy()
        rms = leahy.rms()
        assert not leahy.freq.isnull().values.any()
        assert not leahy.power.isnull().values.any()
        assert not rms.freq.isnull().values.any()
        assert not rms.power.isnull().values.any()


        # Parseval theorem (about fft computation)
        assert np.isclose(np.sum(self.lc.counts**2),np.sum(power.power)/len(self.lc))

    def test_make_power_multi(self):
        powers = PowerSpectrum.from_lc(self.lcs)

        assert isinstance(powers,PowerList)
        assert isinstance(powers,list)
        for p in powers:
            assert isinstance(p,PowerSpectrum)
            assert isinstance(p,pd.DataFrame)
            assert 'freq' in p.columns 
            assert 'power' in p.columns 
            assert len(p.freq) != 0
            assert len(p.power) != 0 

        power = powers.average_leahy()

        # It must return an instance of the PowerSpectrum class
        assert isinstance(power,PowerSpectrum)

        # Frequency bins and time bins must be the same
        assert len(power.freq) == self.time_bins

        # Total counts must be the same   
        print('a0',power.a0,'tot_counts',self.lcs.mean.tot_counts)     
        assert np.isclose(power.a0,self.lcs.mean.tot_counts,atol=self.lcs.mean.tot_counts/1000.,rtol=0)

        # Frequency resolution must be the same
        print('df',power.df,'1/T',1./self.lcs[0].texp)
        assert np.isclose(power.df,1./self.lcs[0].texp,atol=power.df/1000,rtol=0)

        # Nyquist frequency myst be the same
        if len(self.lc)%2!=0:
            print('fn2',power.freq[power.freq>0].iloc[-1],'1/2dt',1./(2*self.lcs[0].tres))
            assert np.isclose(power.freq[power.freq>0].iloc[-1],1./(2*self.lcs[0].tres),atol=power.df,rtol=0)

        # Parseval theorem (about fft computation)
        sumt2 = np.sum(self.lcs.mean.counts**2)
        suma2 = np.sum(power.power*self.lcs.mean.tot_counts/2.)
        print('sumt2',sumt2,'suma2/N',suma2/len(self.lcs[0]))
        assert np.isclose(sumt2,suma2/len(self.lcs[0]))

    def test_make_power_multi2(self):
        powers = PowerSpectrum.from_lc(self.lcs2)

        assert isinstance(powers,PowerList)

        power = powers.average_leahy()

        # It must return an instance of the PowerSpectrum class
        assert isinstance(power,PowerSpectrum)

        # Frequency bins and time bins must be the same
        assert len(power.freq) == self.time_bins

        # Total counts must be the same   
        print('a0_diff',power.a0,'tot_counts_diff',self.lcs2.mean.tot_counts)     
        assert np.isclose(power.a0,self.lcs2.mean.tot_counts,atol=self.lcs.mean.tot_counts/1000.,rtol=0)

        # Frequency resolution must be the same
        print('df_diff',power.df,'1/T_diff',1./self.lcs2[0].texp)
        assert np.isclose(power.df,1./self.lcs2[0].texp,atol=power.df/1000,rtol=0)

        # Nyquist frequency myst be the same
        if len(self.lc)%2!=0:
            print('fn2_diff',power.freq[power.freq>0].iloc[-1],'1/2dt_diff',1./(2*self.lcs2[0].tres))
            assert np.isclose(power.freq[power.freq>0].iloc[-1],1./(2*self.lcs2[0].tres),atol=power.df,rtol=0)

        # Parseval theorem (about fft computation)
        sumt2 = np.sum(self.lcs2.mean.counts**2)
        suma2 = np.sum(power.power*self.lcs2.mean.tot_counts/2.)
        print('sumt2_diff',sumt2,'suma2/N_diff',suma2/len(self.lcs2[0]))
        assert np.isclose(sumt2,suma2/len(self.lcs2[0]),atol=sumt2/10.,rtol=0.)
    
    def test_sub_level(self):
        power = PowerSpectrum.from_lc(self.lc) 
        assert not power.a0 is None
        assert not power.sub_poi(2).a0 is None
        assert power.a0 == power.sub_poi(2).a0

    def test_leahy(self):
        power = PowerSpectrum.from_lc(self.lc)
        leahy = power.leahy()
        assert isinstance(leahy,PowerSpectrum)
        assert not leahy.leahy_norm is None
        assert np.array_equal(leahy.freq,power.freq)

        assert np.array_equal(leahy.power,leahy.leahy().power)
    
    def test_rms(self):
        power = PowerSpectrum.from_lc(self.lc)
        assert not power.power is None
        leahy = power.leahy()
        assert not leahy.power is None
        rms = leahy.rms()
        rms2 = power.rms()
        assert isinstance(rms,PowerSpectrum)
        assert not rms.rms_norm is None
        assert np.array_equal(rms.freq,leahy.freq)
        #assert np.array_equal(rms.rms(self.lc.cr).power,rms.power)

    def test_rebin(self):
        power = PowerSpectrum.from_lc(self.lc)
        rf = 2
        rebin = power.rebin(rf)
        assert isinstance(rebin,PowerSpectrum)  
        pos_freq = power.freq[power.freq>0]
        assert rebin.a0 == power.a0
        if len(pos_freq)%rf==0:
            assert len(rebin.freq) == int(len(pos_freq)/2)+1
        else:
            assert len(rebin.freq) == int(len(pos_freq)/2)+1+1

    #def test_plot(self):
    #    power = PowerSpectrum.from_lc(self.lc)
    #    rebin = power.rebin(2)
    #    rebin.plot()   
    
    def test_comp_frac_rms(self):
        power = PowerSpectrum.from_lc(self.lc).leahy().rms()
        leahy = PowerSpectrum.from_lc(self.lc).leahy()
        rms, srms = power.comp_frac_rms()
        rms2 = (rms/100.)**2
        rms2corr = rms2+1

        # Fractional rms must be equal to standard deviation devided by mean
        assert np.isclose(rms/100.,self.lc.counts.std()/self.lc.counts.mean(),atol=rms/100./100.,rtol=0.)

        # Total rms squared divided by the mean should be equal to the fractional RMS squared + 1 
        assert np.isclose(rms2+1,(self.lc.rms/100)**2/(self.lc.counts.mean()**2),atol=rms2/100.,rtol=0)

        # Total rms squared must be euqal  to a certain linear comination of Leahy power
        if len(self.lc) %2 ==0:
            lin_comb = leahy.a0/(len(self.lc)**2)*(leahy.power[leahy.freq>0].sum()+
            leahy.power[0]*0.5+leahy.power[leahy.freq>0].iloc[-1]*0.5)
        else:
            lin_comb = leahy.a0/(len(self.lc)**2)*(leahy.power[leahy.freq>0].sum()+
            leahy.power[0]*0.5)
        assert np.isclose((self.lc.rms/100)**2,lin_comb,atol=lin_comb/100.,rtol=0)

        print((self.lc.rms/100.)**2.,(self.lc.tot_counts/len(self.lc))**2.*rms2corr)
        print('-'*50)
        print((self.lc.rms/100.)**2.,self.lc.counts.var(),self.lc.counts.mean()**2)
        print((self.lc.tot_counts/len(self.lc))**2.*rms2,(self.lc.tot_counts/len(self.lc))**2)
        print('-'*50)
        print(rms/100.,self.lc.counts.std()/self.lc.counts.mean())
        #assert False

    def test_save_load(self):
        leahy = PowerSpectrum.from_lc(self.lc).leahy()
        leahy.save()

        power = PowerSpectrum.load()
        assert isinstance(power,PowerSpectrum().__class__)
        assert not power.leahy_norm is None
        assert power.rms_norm is None
        assert power.history['FILE_NAME'] == 'power_spectrum.pkl'

class fde:

    def setup_class(self):
        self.time_bins = 1000
        self.cr = 200
        self.t_res = 0.01
        self.t_dur = self.time_bins*self.t_res
        self.t,self.counts=timmer_koenig(self.t_dur,self.t_res,0.1,self.cr)
      
        self.history_key = 'CREATION_DATE'
        self.history_value = ctime()
        self.history = {self.history_key:self.history_value}
        self.notes_key = ctime()
        self.notes_value = 'Created for testing purposes'
        self.notes = {self.notes_key:self.notes_value}

        self.lc = Lightcurve(self.t,self.counts,0.5,10,
        self.notes,self.history)

        self.nlc = 20
        lc_list = []
        for i in range(self.nlc):
            self.t,self.counts=timmer_koenig(self.t_dur,self.t_res,0.1,self.cr)
            self.lc = Lightcurve(self.t,self.counts,0.5,10,
            self.notes,self.history)
            lc_list += [Lightcurve(self.t,self.counts,0.5,10,
        self.notes,self.history)]
        self.lcs = LightcurveList(lc_list)
        self.pl = PowerSpectrum.from_lc(self.lcs)             

    def test_init(self):
        assert isinstance(self.pl,PowerList)

    def test_getitem(self):
        assert isinstance(self.pl[3],PowerSpectrum)

    def test_ave_leahy(self):
        ave_leahy = self.pl.average_leahy()
        assert isinstance(ave_leahy,PowerSpectrum)
        assert len(ave_leahy.freq)==len(self.pl[0].leahy().freq)
        assert ave_leahy.weight == len(self.pl)