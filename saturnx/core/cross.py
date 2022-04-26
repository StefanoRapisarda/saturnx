import numpy as np
import pandas as pd

from saturnx.utils.generic import my_cdate, round_half_up

class CrossSpectrum(pd.DataFrame):

    _metadata = [
        '_weight','_high_en','_low_en',
        '_leahy_norm','rms_norm','_poi_level',
        'meta_data']

    def __init__(
        self,freq_array=np.array([]),cross_array=None,scross_array=None,
        weight=1,low_en=None,high_en=None,
        leahy_norm=None,rms_norm=None,poi_level=None,
        smart_index=True,
        meta_data=None
        ):

        # Initialisation
        column_dict = {'freq':freq_array,'cross':cross_array,'scross':scross_array}
        if len(freq_array) != 0:
            n = len(freq_array)
            if n % 2 == 0:
                index_array = np.concatenate(([i for i in range(int(n/2)+1)],
                                              [i for i in range(int(1-n/2),0)]))
            else:
                index_array = np.concatenate(([i for i in range(int((n-1)/2)+1)],
                                              [i for i in range(int(-(n-1)/2),0)]))                
            if smart_index:
                super().__init__(column_dict,index=index_array)
        super().__init__(column_dict)

        self._weight = weight

        self._leahy_norm = leahy_norm
        self._rms_norm = rms_norm
        self._poi_level = poi_level

        # Energy range
        if not low_en is None and type(low_en) == str: 
            low_en = eval(low_en)
        if not low_en is None and type(high_en) == str: 
            high_en = eval(high_en)
        if not low_en is None and low_en < 0: low_en = 0
        self._low_en = low_en
        self._high_en = high_en

        if meta_data is None:
            self.meta_data = {}
        else: 
            self.meta_data = meta_data

        if not 'HISTORY' in self.meta_data.keys():
            self.meta_data['HISTORY'] = {}
        self.meta_data['HISTORY']['PW_CRE_DATE'] = my_cdate()

    @property
    def fres(self):
        if len(self.freq) == 0: return None
        fres = np.median(np.ediff1d(self.freq[self.freq>0]))
        #fres = np.round(df,abs(int(math.log10(df/1000))))
        return round_half_up(fres,12)