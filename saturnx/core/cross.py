import numpy as np
import pandas as pd

from saturnx.utils.generic import my_cdate, round_half_up

class CrossSpectrum(pd.DataFrame):

    _metadata = [
        'weight','en_range',
        'leahy_norm','rms_norm','poi_level',
        'meta_data']

    def __init__(
        self,freq_array=np.array([]),cross_array=None,scross_array=None,
        weight=1,en_range=[],
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
            else:
                super().__init__(column_dict)
        else:
            super().__init__(column_dict)

        self.leahy_norm = leahy_norm
        self.rms_norm = rms_norm
        self.poi_level = poi_level

        self.weight = weight
        self.en_range = en_range

        # Initializing meta data
        if meta_data is None:
            self.meta_data = {}
        else: 
            self.meta_data = meta_data

        if not 'HISTORY' in self.meta_data.keys():
            self.meta_data['HISTORY'] = {}
        self.meta_data['HISTORY']['PW_CRE_DATE'] = my_cdate()

        if not 'NOTES' in self.meta_data.keys():
            self.meta_data['NOTES'] = {}

    @property
    def fres(self):
        if len(self.freq) == 0: return None
        fres = np.median(np.ediff1d(self.freq[self.freq>0]))
        #fres = np.round(df,abs(int(math.log10(df/1000))))
        return round_half_up(fres,12)

    @property
    def nyqf(self):
        if len(self.freq) == 0: return None
        if np.all(self.freq >= 0):
            if len(self)%2==0:
                nyq = (len(self)-1)*self.fres
            else: 
                nyq = len(self)*self.fres
        else:
            nyq = len(self)*self.fres/2.
        return nyq

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self,weight_value):
        if (not isinstance(weight_value,int) or weight_value < 1):
            raise ValueError('weight must be a positive integer')
        self._weight = weight_value

    @property
    def en_range(self):
        return self._en_range

    @en_range.setter
    def en_range(self,en_range):
        if not type(en_range) in [list,tuple]:
            raise TypeError('Energy range must be either a tuple or a list of tuples')
        
        if isinstance(en_range,tuple):
            en_range = [en_range]
        
        for range in en_range:
            if not isinstance(range,tuple):
                raise TypeError('Energy ranges must be tuples')
            if len(range)!=2:
                raise ValueError('Energy range must contain low and high energy')
            if range[0] >= range[1]:
                raise ValueError('Low energy must be (ghess...?) lower than high energy')
       
        #en_range = clean_en_range(en_range)
        self._en_range = en_range


