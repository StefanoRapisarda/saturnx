import pandas as pd
import numpy as np
import math
import pytest

from astropy.io import fits

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
        assert lc.tres == tres
        assert np.isclose(lc.count_std,std,int(abs(math.log10(std/1000))))
        assert lc.cr == cr
        assert np.isclose(lc.count_std,lc.rate.std()*lc.tres)
        assert np.isclose(lc.rms,
                          np.sqrt(lc.count_std**2+np.mean(lc.counts)**2),
                          atol=lc.rms/1000,rtol=0)
        assert np.isclose(lc.frac_rms,
                          np.sqrt(lc.count_std**2/np.mean(lc.counts)**2),
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

    @pytest.mark.parametrize('events',[[],{},str,12.3],
    ids=['list','dict','string','float'])
    def test_bad_input(self,events):
        with pytest.raises(TypeError):
            lc = Lightcurve.from_event(events)

    def test_lc_meta_data(self,fake_nicer_event):
        events = fake_nicer_event['event']
        header_keys = fake_nicer_event['header_keys']
        lc = Lightcurve.from_event(events)

        assert type(lc) == type(Lightcurve())

        assert 'LC_CRE_DATE' in lc.meta_data.keys()
        assert lc.meta_data['LC_CRE_MODE'] == 'Lightcurve computed from Event object'
        assert lc.meta_data['EVT_FILE_NAME'] == 'Test file name'
        assert lc.meta_data['DIR'] == 'Test dir'
        assert lc.meta_data['MISSION'] == 'NICER'
        info = lc.meta_data['INFO_FROM_HEADER']
        for key in header_keys:
            assert info[key] == key

    def test_start_time(self,fake_nicer_event):
        events = fake_nicer_event['event']
        lc1 = Lightcurve.from_event(events,user_start_time=10)
        lc2 = Lightcurve.from_event(events,user_start_time=-3)
        lc3 = Lightcurve.from_event(events,user_start_time='5')
        lc4 = Lightcurve.from_event(events)
        assert lc1.time.iloc[0] == 10
        assert lc2.time.iloc[0] == events.time.iloc[0]
        assert lc3.time.iloc[0] == 5
        assert lc4.time.iloc[0] == events.time.iloc[0]

    def test_zero_dur(self,fake_nicer_event):
        events = fake_nicer_event['event'] 
        with pytest.raises(ValueError):
            lc = Lightcurve.from_event(events,user_dur=0)
    
    @pytest.mark.parametrize('tres',['0.01',0.07,0.1,'1',5.34,'5.34'],
        ids = ['0.01_string','0.07','0.1','1_string','5.34','5.34_string'])
    def test_dur(self,tres,fake_nicer_event):
        if type(tres) == str: tres = eval(tres)
        events = fake_nicer_event['event']
        lc = Lightcurve.from_event(events,time_res=tres) 
        pre = abs(math.log10(lc.tres/1000))
        assert np.isclose(lc.texp,events.texp,pre)
        assert np.isclose(lc.tres,tres,pre)
        # The histofram process looses at most 1 photon1
        assert lc.tot_counts <= len(events)
        assert lc.tot_counts >= len(events)-1

    @pytest.mark.parametrize('dur',[1,10,23.7,35.6,'10',100],
     ids = ['1','10','23.7','35.6','10_string','100'])
    def test_dur(self,dur,fake_nicer_event):
        events = fake_nicer_event['event'] 
        start_time = 5   
        lc = Lightcurve.from_event(events,time_res=0.1,user_dur=dur,
            user_start_time=start_time)
        pre = abs(math.log10(lc.tres/1000))
        assert np.isclose(lc.texp,events.texp,pre)
        assert np.isclose(lc.tres,0.1,pre)

    @pytest.mark.parametrize('low_en',[0.7,1.,1.5,2.,3.5,7.],
        ids = ['0.7','1.','1.5','2.','3.5','7.'])
    def test_low_en(self,low_en,fake_nicer_event):
        events = fake_nicer_event['event']
        lc = Lightcurve.from_event(events,low_en=low_en)
        assert lc.low_en == low_en
        
    @pytest.mark.parametrize('high_en',[0.7,1.,1.5,2.,3.5,7.],
        ids = ['0.7','1.','1.5','2.','3.5','7.'])
    def test_low_en(self,high_en,fake_nicer_event):
        events = fake_nicer_event['event']
        lc = Lightcurve.from_event(events,high_en=high_en)
        assert lc.high_en == high_en


class TestReadFromFits:

    def test_no_input(self):
        with pytest.raises(FileNotFoundError):
            lc = Lightcurve.read_from_fits('prova')

    def test_read_from_fits_counts(self,tmp_path,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']

        # Making fits
        hdu1 = fits.PrimaryHDU()
        
        col1 = fits.Column(name='TIME', format='D',array=lc.time.to_numpy())
        col2 = fits.Column(name='COUNTS_TEST', format='D',array=lc.counts.to_numpy())
        hdu2 = fits.BinTableHDU.from_columns([col1,col2])
        hdu2.name = 'EXT2'
        hdu2.header['TELESCOP'] = 'NICER'
        keys_to_read = []
        values = []
        for i in range(10):
            key = f'TEST{i}'
            keys_to_read += [key]
            values += [f'VALUE{i}']
            hdu2.header[key] = f'VALUE{i}'
        
        hdu3 = fits.TableHDU()
        hdu3.name = 'GTI'

        hdu_list = fits.HDUList([hdu1,hdu2,hdu3])

        # Writing fits
        d = tmp_path/'sub'
        d.mkdir()
        file_name = d/'lc.fits'
        hdu_list.writeto(file_name)

        # Testing
        lcr = Lightcurve.read_from_fits(file_name,ext='EXT2',
            count_col='COUNTS_TEST',keys_to_read=keys_to_read)
        assert np.array_equal(lc.time,lcr.time)
        assert np.array_equal(lc.counts,lcr.counts)
        assert np.array_equal(lc.rate,lcr.rate)
        assert lc.tres == lcr.tres
        assert lc.texp == lcr.texp
        assert lcr.low_en is None
        assert lcr.high_en is None
        assert lcr.notes == {}
        assert 'LC_CRE_DATE' in lcr.meta_data.keys()
        assert lcr.meta_data['LC_CRE_MODE'] == 'Gti created from fits file'
        assert lcr.meta_data['FILE_NAME'] == 'lc.fits'
        assert lcr.meta_data['MISSION'] == 'NICER'
        for key,value in zip(keys_to_read,values):
            assert lcr.meta_data['INFO_FROM_HEADER'][key] == value

        
    def test_read_from_fits_rate(self,tmp_path,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']

        # Making fits
        hdu1 = fits.PrimaryHDU()
        
        col1 = fits.Column(name='TIME', format='D',array=lc.time.to_numpy())
        col2 = fits.Column(name='RATE_TEST', format='D',
            array=lc.counts.to_numpy()/lc.tres)
        hdu2 = fits.BinTableHDU.from_columns([col1,col2])
        hdu2.name = 'EXT2'
        hdu2.header['TELESCOP'] = 'NICER'
        keys_to_read = []
        values = []
        for i in range(10):
            key = f'TEST{i}'
            keys_to_read += [key]
            values += [f'VALUE{i}']
            hdu2.header[key] = f'VALUE{i}'
        
        hdu3 = fits.TableHDU()
        hdu3.name = 'GTI'

        hdu_list = fits.HDUList([hdu1,hdu2,hdu3])

        # Writing fits
        d = tmp_path/'sub'
        d.mkdir()
        file_name = d/'lc.fits'
        hdu_list.writeto(file_name)

        # Testing
        lcr = Lightcurve.read_from_fits(file_name,ext='EXT2',
            rate_col='RATE_TEST',keys_to_read=keys_to_read)
        assert np.array_equal(lc.time,lcr.time)
        assert np.array_equal(lc.counts,lcr.counts)
        assert np.array_equal(lc.rate,lcr.rate)
        assert lc.tres == lcr.tres
        assert lc.texp == lcr.texp
        assert lcr.low_en is None
        assert lcr.high_en is None
        assert lcr.notes == {}
        assert 'LC_CRE_DATE' in lcr.meta_data.keys()
        assert lcr.meta_data['LC_CRE_MODE'] == 'Gti created from fits file'
        assert lcr.meta_data['FILE_NAME'] == 'lc.fits'
        assert lcr.meta_data['MISSION'] == 'NICER'
        for key,value in zip(keys_to_read,values):
            assert lcr.meta_data['INFO_FROM_HEADER'][key] == value

class TestLightcurveList:

    @pytest.mark.parametrize('wrong_input',['hello!',132,['1','2'],np.array([1,2,3,4]),[{}]],
        ids = ['str','int','list','np_array','dict'])
    def test_wrong_input(self,wrong_input):
        with pytest.raises(TypeError):
            lc_list = LightcurveList(wrong_input)

    @pytest.mark.parametrize('wrong_input',['hello!',132,['1','2'],np.array([1,2,3,4]),[{}]],
        ids = ['str','int','list','np_array','dict'])
    def test_wrong_setitem(self,wrong_input):
        lc_list = LightcurveList()
        with pytest.raises(TypeError):
            lc_list[0] = wrong_input            

    def test_setitem(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = LightcurveList([Lightcurve() for i in range(5)])
        
        i = 2
        lc_list[2] = lc

        assert lc_list[i].time.equals(lc.time)
        assert lc_list[i].counts.equals(lc.counts)
        assert lc_list[i].rate.equals(lc.rate)
        assert lc_list[i].texp == lc.texp
        assert lc_list[i].tres == lc.tres
        assert lc_list[i].low_en == lc.low_en
        assert lc_list[i].high_en == lc.high_en
        assert lc_list[i].meta_data == lc.meta_data
        assert lc_list[i].notes == lc.notes

    def test_join_diff_tres(self,fake_white_noise_lc):
        lc1 = fake_white_noise_lc['lc']
        lc2 = lc1.rebin(2)
        lc_list = LightcurveList([lc1,lc2])

        with pytest.raises(ValueError):
            lcj = lc_list.join()

    def test_join_no_mask(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)

        assert type(lc_list) == type(LightcurveList())
        for lc_i in lc_list: assert type(lc_i) == type(Lightcurve())
        assert len(lc_list) == 5

        lcj = lc_list.join()

        assert len(lcj.time) == len(lc.time)
        assert lcj.time.equals(lc.time)
        assert lcj.counts.equals(lc.counts)
        assert lcj.rate.equals(lc.rate)
        assert lcj.tres == lc.tres
        assert lcj.texp == lc.texp
        assert lcj.low_en == lc.low_en
        assert lcj.high_en == lc.high_en
        assert 'LC_CRE_DATE' in  lcj.meta_data.keys()
        assert lcj.meta_data['EVT_CRE_MODE'] == \
            'Lightcurve created joining Lightcurves from LightcurveList'
        assert lcj.meta_data['N_ORI_LCS'] == 5
        assert lcj.meta_data['N_MASKED_LCS'] == 5
        assert lcj.meta_data['MISSION'] == 'NICER'
        assert lcj.notes == {}     

    def test_join_mask(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)

        mask = np.array([0,1,0,1,1])

        assert type(lc_list) == type(LightcurveList())
        for lc_i in lc_list: assert type(lc_i) == type(Lightcurve())
        assert len(lc_list) == 5

        lcj = lc_list.join(mask=mask)
        lcj2 = LightcurveList([lc_list[1],lc_list[3],lc_list[4]]).join()

        assert len(lcj.time) == len(lcj2)
        assert lcj.time.equals(lcj2.time)
        assert lcj.counts.equals(lcj2.counts)
        assert lcj.rate.equals(lcj2.rate)
        assert lcj.tres == lc.tres
        assert lcj.texp == lcj2.texp
        assert lcj.low_en == lc.low_en
        assert lcj.high_en == lc.high_en
        assert 'LC_CRE_DATE' in  lcj.meta_data.keys()
        assert lcj.meta_data['EVT_CRE_MODE'] == \
            'Lightcurve created joining Lightcurves from LightcurveList'
        assert lcj.meta_data['N_ORI_LCS'] == 5
        assert lcj.meta_data['N_MASKED_LCS'] == 3
        assert lcj.meta_data['MISSION'] == 'NICER'
        assert lcj.notes == {}      


class TestLightcurveListFillGaps:

    def test_fill(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list10 = lc.split(10) 
        lc_list10_b = lc.split(10)
        lc_list10_b[1] = lc_list10_b[1]*0.
        lc_list10_b[3] = lc_list10_b[3]*0.
        lc_list = LightcurveList([lc_list10[i] for i in [0,2,4]])
        
        lcfg = lc_list.fill_gaps().join()
        lcfg_b = lc_list10_b.join()

        assert len(lcfg) == len(lcfg_b)
        assert lcfg.tres == lcfg_b.tres
        assert np.allclose(lcfg.time,lcfg_b.time)
        assert lcfg.counts.equals(lcfg_b.counts)
        assert lcfg.rate.equals(lcfg_b.rate)


class TestLightcurveListSplit:

    def test_split_seg(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_protolist = lc.split(10)
        lc_list = lc_protolist.split(5)

        assert type(lc_list) == type(LightcurveList())
        assert len(lc_list) == 10
        for i,lci in enumerate(lc_list):
            assert type(lci) == type(Lightcurve())
            assert lci.tres == lc.tres
            assert lci.texp == 5
            assert lci.low_en == lc.low_en
            assert lci.high_en == lc.high_en
            assert 'SPLITTING_SEG' in lci.meta_data.keys()
            assert 'SPLITTING_SEG_1' in lci.meta_data.keys()
            assert lci.meta_data['SEG_DUR'] == 10
            assert lci.meta_data['N_SEGS'] == 5
            assert lci.meta_data['SEG_DUR_1'] == 5
            assert lci.meta_data['N_SEGS_1'] == 2 
            assert lci.meta_data['SEG_INDEX'] == int(i/2)
            assert lci.meta_data['SEG_INDEX_1'] == i%2     
            assert lci.notes == lc.notes


class TestLightcurveListInfo:

    def test_info(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)
        info = lc_list.info()
        assert type(info) == type(pd.DataFrame())
        assert len(info) == len(lc_list)
        assert list(info.columns) == ['texp','tres','n_bins','counts','count_rate',
                    'rms','frac_rms',
                    'max_time','min_time',
                    'min_en','max_en','mission']
        for i in range(len(info)):
            assert info['texp'].iloc[i] == lc_list[i].texp 
            assert info['tres'].iloc[i] == lc_list[i].tres
            assert info['n_bins'].iloc[i] == len(lc_list[i])
            assert info['counts'].iloc[i] == lc_list[i].tot_counts
            assert info['count_rate'].iloc[i] == lc_list[i].cr
            assert info['rms'].iloc[i] == lc_list[i].rms
            assert info['frac_rms'].iloc[i] == lc_list[i].frac_rms
            assert info['min_time'].iloc[i] == min(lc_list[i].time) 
            assert info['max_time'].iloc[i] == max(lc_list[i].time) 
            assert info['min_en'].iloc[i] == lc_list[i].low_en 
            assert info['max_en'].iloc[i] == lc_list[i].high_en
            assert info['mission'].iloc[i] == lc_list[i].meta_data['MISSION']


class TestLightcurveListAttributes:

    def test_tot_counts(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)

        assert lc.tot_counts == lc_list.tot_counts

    def test_texp(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)

        assert lc.texp == lc_list.texp

    def test_cr(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)

        assert np.isclose(lc.cr,lc_list.cr)

    def test_cr_std(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)

        pre = int(abs(math.log10(lc.cr_std/1000)))
        assert np.isclose(lc.cr_std,lc_list.cr_std,pre)

    def test_count_std(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)

        pre = int(abs(math.log10(lc.count_std/1000)))
        assert np.isclose(lc.count_std,lc_list.count_std,pre)

    def test_mean(self,fake_white_noise_lc):
        lc = fake_white_noise_lc['lc']
        lc_list = lc.split(10)
        lc0 = lc_list[0]

        lc2 = lc_list.mean()
        assert type(lc2) == type(Lightcurve())
        assert np.array_equal(lc0.time,lc2.time)
        assert np.allclose(lc0.counts,lc2.counts,1e+9)
        assert np.allclose(lc0.rate,lc2.rate,1e+9)
        assert lc2.meta_data['LC_CRE_MODE'] == 'Mean of Lightcurves in LightcurveList'
        assert lc2.meta_data['N_ORI_LCS'] == len(lc_list)
        assert lc2.meta_data['N_MASKED_LCS'] == len(lc_list)
        assert lc2.meta_data['MISSION'] == 'NICER'

class boh:
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



        