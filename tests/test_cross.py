from saturnx.core.cross import CrossSpectrum

class TestEmptyCrossSpectrum:

    def test_empty_cross_type(self):
        cross = CrossSpectrum()

        assert isinstance(cross,CrossSpectrum)

    def test_empty_cross_columns(self):
        cross = CrossSpectrum()

        assert cross.freq.empty
        assert cross.cross.empty
        assert cross.scross.empty

    def test_empty_cross_attributed(self):
        cross = CrossSpectrum()

        assert cross.weight == 1
        assert cross.low_en == None
        assert cross.high_en == None
        assert cross.leahy_norm == None
        assert cross.rms_norm == None
        assert cross.poi_level == None

        assert cross.df == None
        assert cross.nf == None
        assert cross.a0 == None
        assert cross.cr == None
    
    def test_empty_cross_meta_data(self,mocker):
        mocker.patch('saturnx.core.cross.my_cdate',
            return_value='test_current_date')
        cross = CrossSpectrum()

        assert len(cross.meta_data) == 1
        assert cross.meta_data['HISTORY']['PW_CRE_DATE'] == 'test_current_date'
        assert cross.notes == {}        
