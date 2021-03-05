import os
import numpy as np
import pandas as pd
import pickle
import pathlib
import math
from scipy.fft import fft,fftfreq

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scipy.fftpack import fftfreq, fft
from astropy.io.fits import getdata

from kronos.core.lightcurve import Lightcurve, LightcurveList
from kronos.utils.time_series import rebin_xy, rebin_arrays
from kronos.utils.fits import read_fits_keys, get_basic_info
from kronos.utils.generic import is_number, my_cdate



class PowerSpectrum(pd.DataFrame):

    _metadata = ['_weight','_high_en','_low_en',
                 '_leahy_norm','_rms_norm','_poi_level',
                 'notes','meta_data']

    def __init__(self,freq_array=np.array([]),power_array=None,spower_array=None,
                 weight=1,low_en=None,high_en=None,
                 leahy_norm=None,rms_norm=None,poi_level=None,
                 smart_index=True,
                 notes=None,meta_data=None):

        # Initialisation  
        column_dict = {'freq':freq_array,'power':power_array,'spower':spower_array}
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

        # Meta_data
        if notes is None:
            self.notes = {}
        else: self.notes = notes

        if meta_data is None:
            self.meta_data = {}
        else: self.meta_data = meta_data
        self.meta_data['PW_CRE_DATE'] = my_cdate()
        
    @property
    def df(self):
        if len(self.freq) == 0: return None
        df = np.median(np.ediff1d(self.freq[self.freq>0]))
        df = np.round(df,abs(int(math.log10(df/1000))))
        return df

    @property
    def nf(self):
        if len(self.freq) == 0: return None
        return len(self)*self.df/2.

    @property
    def a0(self):
        if self.power.empty: return None
        if self.power.iloc[0] == None: return None

        if (self._leahy_norm is None) and (self.freq.iloc[0] == 0):
            a0 = np.sqrt(self.power.iloc[0])
        elif not self._leahy_norm is None:
            a0 = 2/self._leahy_norm
        elif (self.freq.iloc[0] == 0) and self._rms_norm is None:
            # if _leahy_norm is not None, the powerspectrum
            # is Leahy normalized
            a0 = self.power.iloc[0]/2
        elif (self.freq.iloc[0] == 0):
            a0 = self.power.iloc[0]/self._rms_norm/2
        else:
            a0 = None

        return a0

    @property
    def cr(self):
        if not self.a0 is None:
            return self.a0*self.df
        else: return None

    def comp_frac_rms(self,low_freq=0,high_freq=np.inf,pos_only=False):

        if not self.power.any():
            return None, None

        if type(low_freq) == str: low_freq = eval(low_freq)
        if type(high_freq) == str: high_freq = eval(high_freq)
        if low_freq < 0: low_freq = 0
        if high_freq > self.nf: high_freq = self.nf 
        if low_freq > high_freq:
            raise ValueError('low_freq must be lower than high_freq')

        mask = (self.freq > low_freq) & (self.freq < high_freq)
        if pos_only:
            mask = mask & (self.power>0)
        nyq_power = np.double(self.power[self.freq == min(self.freq)])
        if len(self)%2 != 0 or high_freq < self.nf: nyq_power = 0

        if (self.leahy_norm is None) and (self.rms_norm is None):
            rms2 = (2*np.sum(self.power[mask]) + nyq_power)/self.a0**2
        elif self.rms_norm is None:
            rms2 = (np.sum(self.power[mask]) + nyq_power/2)/self.a0
        else:
            rms2 = (np.sum(self.power[mask]) + nyq_power/2) * self.df

        rms = np.sqrt(rms2)

        if self.spower.any():
            srms2_term1 = 1./4/rms
            if (self.leahy_norm is None) and (self.rms_norm is None):
                srms2_term2 = 4*np.sum(self.spower[mask]**2)
            elif self.rms_norm is None:
                srms2_term2 = self.a0**2 * np.sum(self.spower[mask]**2)/len(self)**4
            else:
                srms2_term2 = np.sum(self.spower[mask]**2) * self.df**2           
            
            srms2 = srms2_term1 * srms2_term2
            srms = np.sqrt(srms2)
        else:
            srms = None

        return rms,srms

    def sub_poi(self,value=None,low_freq=0,high_freq=np.inf):
        '''
        Subtracts poisson level from the PowerSpectrum power

        PARAMETERS
        ----------
        value: float, numpyp.array, list, pandas.Series, or None (optional)
            value to be subtracted. If a list or an array with the same
            length of power, all its elements are subtracted from power.
            If None (default), value is estimated averaging power level
            between low_freq and high_freq
        low_freq: float (optional)
            default is 0
        high_freq: float (optional)
            default is equal to the Nyquist frequency

        RETURNS
        -------
        pw_poi: kronos.core.PowerSpectrum
            PowerSpectrum with subtracted power and updated meta_data

        '''
        print(f'Subtracting {value} to power')

        meta_data= self.meta_data.copy()
        meta_data['SUBTRACTING_POI'] = my_cdate()

        if value is None:
            if low_freq < 0: low_freq = 0
            if low_freq > self.nf: low_freq = self.nf
            if high_freq > self.nf: high_freq = self.nf
            if low_freq > high_freq:
                raise ValueError('low_freq must be lower than high_freq')
            mask = (self.freq>=low_freq) & (self.freq<high_freq) * (self.freq>0)
            value = self.power[mask].mean()
            meta_data['POI_RANGE'] = f'{low_freq}-{high_freq}'
        elif type(value) in [list,np.ndarray,pd.Series]:
            if len(value) != len(self):
                raise ValueError('values must have the same dimension of power')
    
        # Keeping the original DC (a0) component
        if type(value) in [list,np.ndarray]:
            value[0] = 0
        elif type(value) == pd.Series:
            value.iloc[0] = 0        
        power = np.subtract(self.power,value)
        if not type(value) in [list,np.ndarray,pd.Series]:
            power.iloc[0] += value

        if not 'spower' in self.columns :
            poi_pw = PowerSpectrum(freq_array = self.freq, power_array = power,
                                weight = self._weight, low_en = self._low_en, high_en = self._high_en,
                                leahy_norm = self._leahy_norm, rms_norm = self._rms_norm, poi_level = value,
                                notes = {},meta_data = meta_data)
        else:
            poi_pw = PowerSpectrum(freq_array = self.freq, power_array = power, spower_array = self.spower,
                                weight = self._weight,low_en = self._low_en, high_en = self._high_en,
                                leahy_norm = self._leahy_norm, rms_norm = self._rms_norm, poi_level = value,
                                notes = {}, meta_data = meta_data)        

        return poi_pw

    def normalize(self,norm='leahy',bkg_cr=0):

        meta_data = self.meta_data.copy()

        if norm == 'leahy':
            if (self._leahy_norm is None) and (self._rms_norm is None):
                norm = 2./self.a0
                norm_leahy = norm
                norm_rms = None
                meta_data['NORMALIZING'] = my_cdate()
                meta_data['NORM_MODE'] = 'Leahy'
            else:
                print('The power spectrum is already either Leahy or RMS normalized')
                norm = 1
        elif norm == 'rms':
            if (self._leahy_norm is None) and (self._rms_norm is None):
                norm_leahy = (2./self.a0)
                norm_rms = self.cr/( (self.cr-bkg_cr)**2 )
                norm = norm_leahy*norm_rms
                meta_data['NORMALIZING'] = my_cdate()
                meta_data['NORM_MODE'] = 'FRAC_RMS'
            elif (self._rms_norm is None) and (not self._leahy_norm is None):
                norm = self.cr/( (self.cr-bkg_cr)**2 )
                norm_rms = norm
                meta_data['NORMALIZING'] = my_cdate()
                meta_data['NORM_MODE'] = 'FRAC_RMS'
            elif not self._rms_norm is None:     
                print('The power spectrum is already RMS normalized')  
        else:
            if type(norm) == str: norm = eval(norm)   
            meta_data['NORMALIZING'] = my_cdate()
            meta_data['NORM_MODE'] = 'number'
            meta_data['NORM_VALUE'] = norm   
            norm_leahy = None
            norm_rms = None  


        if not self.spower.any() :
            power = PowerSpectrum(freq_array=self.freq,power_array=self.power*norm,
                                weight = self._weight,low_en=self._low_en,high_en=self._high_en,
                                leahy_norm=norm_leahy,rms_norm=norm_rms,poi_level=self._poi_level,
                                notes={},meta_data=meta_data)    
        else:
            power = PowerSpectrum(freq_array=self.freq,power_array=self.power*norm,spower_array=self.spower*norm,
                                weight = self._weight,low_en=self._low_en,high_en=self._high_en,
                                leahy_norm=norm_leahy,rms_norm=norm_rms,poi_level=self._poi_level,
                                notes={},meta_data=meta_data)             

        return power
                                                   
    def rebin(self,factors=-30):

        if type(factors) != list: factors=[factors]
        
        mask = self.freq > 0
        binned_freq = self.freq[mask]
        
        # Poisson level is reintroduced, the array is rebinned, and 
        # the Poisson level is subtracted again
        if not self._poi_level is None:
            poi_level = self._poi_level
        else:
            poi_level = 0.
        if type(self._poi_level) in [list,np.ndarray,pd.Series]:
            poi_level = self._poi_level[mask]
        
        binned_power = np.add(self.power[mask],poi_level)
        binned_poi = poi_level
        dc_power = self.power.iloc[0]

        if self.spower.any():
            binned_spower = self.spower[mask]
            #print('Before:',len(binned_power),len(binned_spower))
            for f in factors:
                binned_freq,binned_power,dummy,binned_spower=rebin_xy(
                    binned_freq,binned_power,ye=binned_spower,rf = f)
                if type(binned_poi) in [list,np.ndarray,pd.Series]:
                    d,binned_poi,d,d = rebin_xy(binned_freq,binned_poi,rf = f)

            #print('After:',len(binned_power),len(binned_spower))
            pw = PowerSpectrum(freq_array = np.append(0,binned_freq),
                power_array = np.append(dc_power,binned_power-binned_poi),
                spower_array = np.append(0,binned_spower),
                weight = self._weight, low_en = self._low_en, high_en = self._high_en,
                smart_index = False,
                leahy_norm = self._leahy_norm, rms_norm = self._rms_norm,
                poi_level = binned_poi,
                notes = self.notes, meta_data = self.meta_data)
        else:
            for f in factors:
                binned_freq,binned_power,dummy,dummy=rebin_xy(
                    binned_freq,binned_power,rf = f)
                if type(binned_poi) in [list,np.ndarray,pd.Series]:
                    d,binned_poi,d,d = rebin_xy(binned_freq,binned_poi,rf = f)

            pw = PowerSpectrum(freq_array = np.append(0,binned_freq),
                power_array = np.append(dc_power,binned_power-binned_poi),
                weight = self._weight, low_en = self._low_en, high_en = self._high_en,
                smart_index = False,
                leahy_norm = self._leahy_norm, rms_norm = self._rms_norm,
                poi_level = binned_poi,
                notes = self.notes, meta_data = self.meta_data)

        return pw
            
    def plot(self,ax=False,xy=False,title=False,lfont=16,**kwargs):
        
        if not 'color' in kwargs.keys(): kwargs['color'] = 'k'
        if not 'marker' in kwargs.keys(): kwargs['marker']='o'

        if ax is False:
            fig, ax = plt.subplots(figsize=(6,6))

        if (not title is False) and (not ax is False):
            ax.set_title(title)

        mask = self.freq > 0

        if 'spower' in self.columns:
            if xy:
                ax.errorbar(self.freq[mask],self.power[mask]*self.freq[mask],
                                            self.spower[mask]*self.freq[mask],
                                            **kwargs)
            else:
                ax.errorbar(self.freq[mask],self.power[mask],self.spower[mask],
                            **kwargs)
        else:
            if xy:
                ax.plot(self.freq[mask],self.power[mask]*self.freq[mask],
                        **kwargs) 
            else:
                ax.plot(self.freq[mask],self.power[mask],
                        **kwargs)           

        ax.set_xlabel('Frequency [Hz]',fontsize=lfont)
        if not self._rms_norm is None:
            if xy:
                ax.set_ylabel('Power [(rms/mean)$^2$',fontsize=lfont)
            else:
                ax.set_ylabel('Power [(rms/mean)$^2$/Freq.',fontsize=lfont)
        elif not self.leahy_norm is None:
            if xy:
                ax.set_ylabel('Leahy Power * Freq.',fontsize=lfont)
            else:
                ax.set_ylabel('Leahy Power',fontsize=lfont)
        else:
            if xy:
                ax.set_ylabel('Power * Freq.',fontsize=lfont)
            else:
                ax.set_ylabel('Power',fontsize=lfont)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()

    @staticmethod
    def from_lc(lightcurve):

        # I want the information contained in these keyword to propagate
        # in the power spectrum
        target_keys = ['N_GTIS','GTI_INDEX','N_SEGS','SEG_INDEX','MISSION']

        meta_data = {}
        meta_data['PW_CRE_MODE'] = 'Power computed from Lightcurve'

        if type(lightcurve) == type(LightcurveList()):

            meta_data['TIME_RES'] = lightcurve[0].tres
            if 'SEG_DUR' in lightcurve[0].meta_data.keys():
                meta_data['SEG_DUR'] = lightcurve[0].meta_data['SEG_DUR']

            powers = []
            for l in lightcurve:
                if (not l.tot_counts is None) and (l.tot_counts > 0):

                    power_meta_data = meta_data.copy()
                    for key in target_keys:
                        if key in l.meta_data.keys(): power_meta_data[key]=l.meta_data[key]

                    freq = fftfreq(len(l),np.double(l.tres))
                    amp = fft(l.counts)
                    powers += [PowerSpectrum(freq_array = freq, power_array = np.multiply(amp, np.conj(amp)).real,
                               low_en = l.low_en, high_en = l.high_en, weight = 1,
                               meta_data = power_meta_data)]

            if len(powers) != 0:
                return PowerList(powers)
            else:
                print('WARNING: Empty PowerList')
                return PowerList()
                
        elif type(lightcurve) == type(Lightcurve()):

            meta_data['TIME_RES'] = lightcurve.tres
            if 'SEG_DUR' in lightcurve.meta_data.keys():
                meta_data['SEG_DUR'] = lightcurve.meta_data['SEG_DUR']
            for key in target_keys:
                if key in lightcurve.meta_data.keys(): meta_data[key]=lightcurve.meta_data[key]

            if (not lightcurve.counts is None) and (lightcurve.tot_counts > 0):
                
                freq = fftfreq(len(lightcurve),np.double(lightcurve.tres))
                amp = fft(lightcurve.counts)
                power = PowerSpectrum(freq_array = freq, power_array = np.multiply(amp, np.conj(amp)).real,
                                low_en = lightcurve.low_en, high_en = lightcurve.high_en,
                                weight = 1, meta_data = meta_data)
            else:
                power = PowerSpectrum()
            
            return power 

        else:
            raise TypeError('You can compute Power Spectrum only from lightcurve')

    @staticmethod
    def read_from_fits(fits_file, extname, freq_col, power_col, spower_col):

        data = getdata(fits_file,extname=extname,meta_data=False,memmap=True)        
        freq = data[freq_col]
        power = data[power_col]
        spower = data[spower_col]

        return PowerSpectrum(freq,power,spower)   
       
    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self,weight_value):
        self._weight = weight_value

    @property
    def low_en(self):
        return self._low_en

    @low_en.setter
    def low_en(self,low_en_value):
        self._low_en = low_en_value
        
    @property
    def high_en(self):
        return self._high_en

    @high_en.setter
    def high_en(self,high_en_value):
        self._high_en = high_en_value 

    @property
    def leahy_norm(self):
        return self._leahy_norm 

    @leahy_norm.setter
    def leahy_norm(self,value):
        self._leahy_norm = value

    @property
    def rms_norm(self):
        return self._rms_norm

    @rms_norm.setter
    def rms_norm(self,value) :
        self._rms_norm = value

    @property
    def poi_level(self):
        return self._poi_level

    @poi_level.setter
    def poi_level(self,value):
        self._poi_level = value 

        
class PowerList(list):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.array(['kronos.core.power.PowerSpectrum' in str(i.__class__) for i in self]).all():
            raise TypeError('All the elements must be Power objects')

    def __setitem__(self, index, power):
        if not 'kronos.core.power.PowerSpectrum' in str(power.__class__):
            raise TypeError('The item must be a Power object')
        self[index] = power

    def average_leahy(self, exc=[]):
        '''
        Method for averaging Leahy powers in the list
        '''

        if len(self) != 0:
            num = []
            den = 0
            counter = 1
            for i in range(len(self)):
                if i in exc: 
                    counter += 1
                    continue
                else:
                    counter = 1
                if i > 0:
                    assert self[i].freq.equals(self[i-counter].freq),'Frequency array do not correspond, impossible to average'

                num += [self[i].leahy().power*self[i].weight]
                den += self[i].weight
            new_weight = den
            num = np.array(num).sum(axis=0)

            power = num/den
            spower = power/np.sqrt(new_weight)

            meta_data={}
            meta_data['PW_CRE_DATE'] = my_cdate()
            meta_data['PW_CRE_MODE'] = 'Average of Leahy power spectra'
            meta_data['N_PWA']=len(self)
            meta_data['SEG_DUR'] = self[0].meta_data['SEG_DUR']
            meta_data['TIME_RES'] = self[0].meta_data['TIME_RES']

            return PowerSpectrum(time_array = self[0].freq,
                                 power_array = power,
                                 spower_array = spower,
                                 weight = new_weight,
                                 leahy_norm = 1,
                                 low_en = self[0].low_en, high_en = self[0].high_en,
                                 notes = {}, meta_data = meta_data)
        else:
            print('WARNING: PowerList is empty, returning empty PowerSpectrum')
            return PowerSpectrum()

    def save(self,file_name='power_list.pkl',fold=os.getcwd()):
        try:
            with open(os.path.join(fold,file_name),'wb') as output:
                pickle.dump(self,output)
            print('PowerList saved in {}'.format(os.path.join(fold,file_name)))
        except Exception as e:
            print(e)
            print('Could not save PowerList')

    @staticmethod
    def load(file_name):
        assert os.path.isfile(file_name),f'{file_name} not found'
        with open(file_name,'rb') as infile:
            power_list = pickle.load(infile)
        return power_list