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

    @pytest.mark.parametrize('en_range',[[2],[3,2],['2'],['3','2']])
    def test_wrong_energy_range_without_high_energy(self,en_range):
        with pytest.raises(ValueError):
            cross = CrossSpectrum(en_range=en_range)

        
