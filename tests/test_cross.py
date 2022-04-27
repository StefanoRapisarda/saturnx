import numpy as np

from mock import mocker
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
        assert cross.low_en == None
        assert cross.high_en == None
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

    def test_empty_cross_with_freq(self):
        freqs = np.linspace(0,10,1000)

        cross = CrossSpectrum(freq_array=freqs)
