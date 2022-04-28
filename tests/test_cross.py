from multiprocessing.sharedctypes import Value
import numpy as np
import pytest

from saturnx.core.cross import CrossSpectrum

class TestEmptyCrossSpectrum:

    def test_empty_cross_type(self):
        cross = CrossSpectrum()

        assert isinstance(cross,CrossSpectrum)

    def test_empty_cross_columns(self):
        cross = CrossSpectrum()

        # empty id a pandas feature checking if a column is empty
        assert cross.freq.empty
        assert cross.cross.empty
        assert cross.scross.empty

    def test_empty_cross_attributes(self):
        cross = CrossSpectrum()

        assert cross.weight == 1
        assert len(cross.en_range) == 0
        assert cross.en_range == []
        assert cross.leahy_norm == None
        assert cross.rms_norm == None
        assert cross.poi_level == None

        assert cross.fres == None
        assert cross.nyqf == None
        # count rate and a0
    
    def test_empty_cross_meta_data(self,mocker):
        mocker.patch('saturnx.core.cross.my_cdate',
            return_value='test_current_date')
        cross = CrossSpectrum()

        # meta_data is supposed to contain only HISTORY and NOTES
        assert len(cross.meta_data) == 2
        assert cross.meta_data['HISTORY']['PW_CRE_DATE'] == 'test_current_date'
        assert cross.meta_data['NOTES'] == {}  

    @pytest.mark.parametrize('weight',[-3,3.2,-3.2,'-3','3.2','-3.2','ciao'])
    def test_empty_cross_wrong_weight_setter(self,weight):
        message = 'weight must be a positive integer'
        with pytest.raises(ValueError) as e_info:
            cross = CrossSpectrum(weight=weight)
        assert str(e_info.value) == message

    def test_empty_cross_with_freq(self):
        freqs = np.linspace(0,10,1000)

        cross = CrossSpectrum(freq_array=freqs)

class TestEnergyInitialization:

    @pytest.mark.parametrize('en_range',[3,'ciao',np.array([0,2])])
    def test_wrong_energy_range_type_single(self,en_range):
        exp_message = 'Energy range must be either a tuple or a list of tuples'
        with pytest.raises(TypeError) as e_info:
            cross = CrossSpectrum(en_range=en_range)
        assert str(e_info.value) == exp_message

    @pytest.mark.parametrize('en_range',[[3],['ciao',3],[np.array([0,2]),[1,2]],[[1,2],[3,4]]])
    def test_wrong_energy_range_type_list(self,en_range):
        exp_message = 'Energy ranges must be tuples'
        with pytest.raises(TypeError) as e_info:
            cross = CrossSpectrum(en_range=en_range)
        assert str(e_info.value) == exp_message  

    @pytest.mark.parametrize('en_range',[(1,),(1,2,3)],ids=['len1 tuple','len3 tuple'])
    def test_wrong_energy_range_tuple_size(self, en_range):
        print(en_range,type(en_range))
        exp_message = 'Energy range must contain low and high energy'
        with pytest.raises(ValueError) as e_info:
            cross = CrossSpectrum(en_range=en_range)
        assert str(e_info.value) == exp_message

    @pytest.mark.parametrize('en_range',[(3,2),(10,4),(5.4,5.4)])
    def test_wrong_energy_range_tuple_boundaries(self,en_range):
        exp_message = 'Low energy must be (ghess...?) lower than high energy'
        with pytest.raises(ValueError) as e_info:
            cross = CrossSpectrum(en_range=en_range)
        assert str(e_info.value) == exp_message
        
