from types import TracebackType
import numpy as np
import pandas as pd
import math
import pytest

from saturnx.core.lightcurve import Lightcurve,LightcurveList
from saturnx.core.power import PowerSpectrum,PowerList
from saturnx.utils.time_series import timmer_koenig2

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

        assert power.fres == None
        assert power.nyqf == None
        assert power.a0 == None
        assert power.cr == None
    
    def test_empty_power_meta_data(self,mocker):
        mocker.patch('saturnx.core.power.my_cdate',
            return_value='test_current_date')
        power = PowerSpectrum()

        assert len(power.meta_data) == 2
        assert power.meta_data['HISTORY']['PW_CRE_DATE'] == 'test_current_date'
        assert power.meta_data['NOTES'] == {}


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

        assert power.fres == 1./(lc.texp)
        assert power.nyqf == 1./2/lc.tres
        assert power.a0 == lc.tot_counts
        assert power.cr == lc.cr

    def test_meta_data(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)

        assert power.meta_data['NOTES'] == {}
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

            assert power.fres == 1./10
            assert power.nyqf == 1./2/lc_list[0].tres
            assert power.a0 == lc_list[i].tot_counts
            assert np.isclose(power.cr,lc_list[i].cr)

    def test_meta_data(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)
        power_list = PowerSpectrum.from_lc(lc_list)

        for i,power in enumerate(power_list):
            assert power.meta_data['NOTES'] == {}
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
        assert leahy.nyqf == power.nyqf
        assert leahy.fres == power.fres

    def test_leahy_meta_data(self,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test date')
        leahy = power.normalize()

        assert leahy.meta_data['NOTES'] == power.meta_data['NOTES']
        assert leahy.meta_data['HISTORY']['NORMALIZING'] == 'Test date'
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
        assert rms.fres == power.fres
        assert rms.nyqf == power.nyqf

    def test_rms_meta_data(self,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test date')
        rms = power.normalize('rms')

        assert rms.meta_data['NOTES'] == power.meta_data['NOTES']
        assert rms.meta_data['HISTORY']['NORMALIZING'] == 'Test date'
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
        middle_freq = power.nyqf/2
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
        middle_freq = power.nyqf/2
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
        middle_freq = power.nyqf/2
        poi_power = power.sub_poi(low_freq=middle_freq)
        poi_level = np.mean(power.power[power.freq>=middle_freq])

        assert poi_power.fres == power.fres
        assert poi_power.nyqf == power.nyqf
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
        middle_freq = power.nyqf/2
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test creation date')
        poi_power = power.sub_poi(low_freq=middle_freq)

        assert poi_power.meta_data['HISTORY']['SUBTRACTING_POI'] == 'Test creation date'
        assert poi_power.meta_data['POI_RANGE'] == f'{middle_freq}-{poi_power.nyqf}'
        assert poi_power.meta_data['NOTES'] == power.meta_data['NOTES']

    def test_value_none_high_freq_arrays(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        middle_freq = power.nyqf/2
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
        middle_freq = power.nyqf/2
        poi_power = power.sub_poi(high_freq=middle_freq)
        mask =(power.freq>0)&(power.freq<middle_freq)
        poi_level = np.mean(power.power[mask])

        assert poi_power.fres == power.fres
        assert poi_power.nyqf == power.nyqf
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
        middle_freq = power.nyqf/2
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test creation date')
        poi_power = power.sub_poi(high_freq=middle_freq)

        assert poi_power.meta_data['HISTORY']['SUBTRACTING_POI'] == 'Test creation date'
        assert poi_power.meta_data['POI_RANGE'] == f'{0}-{middle_freq}'
        assert poi_power.meta_data['NOTES'] == power.meta_data['NOTES']

    @pytest.mark.parametrize('value',[0.5,0.236,1.245,6.345])
    def test_value_arrays(self,value,mocker,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        power = PowerSpectrum.from_lc(lc)
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test creation date')
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
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test creation date')
        poi_power = power.sub_poi(value = value)  

        assert poi_power.fres == power.fres
        assert poi_power.nyqf == power.nyqf
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

            assert poi_power.fres == power.fres
            assert poi_power.nyqf == power.nyqf
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

        assert rebin.fres == power.fres
        assert np.isclose(rebin.nyqf,power.nyqf,power.fres/10)
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
        mocker.patch('saturnx.core.power.my_cdate',return_value='Test creation date')
        rebin = power.rebin(rf)

        if type(rf) == list: 
            expected_rf = rf
        else:
            expected_rf = [rf]
        assert rebin.meta_data['HISTORY']['REBINNING'] == 'Test creation date'
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
        power = power_list.average() 
        mask = power.freq>0 

        assert np.round(power.power[mask].mean()) == 2