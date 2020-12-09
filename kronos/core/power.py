import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.fft import fft,fftfreq

from ..functions.my_functions import my_cdate, my_rebin

from datetime import datetime
import pickle
import os

class PowerSpectrum(pd.DataFrame):

    _metadata = ['_weight','_high_en','_low_en',
                 '_leahy_norm','_rms_norm','_poi_level',
                 'notes','history']

    def __init__(self,freq_array=None,power_array=None,spower_array=None,
                 weight_value=None,low_en_value=None,high_en_value=None,
                 leahy_norm_value=None,rms_norm_value=None,poi_level_value=None,
                 smart_index=True,
                 notes={},history={}):

        if freq_array is None and power_array is None:
            super().__init__(columns=['freq','power'])
        else:

            assert len(freq_array) == len(power_array),'Frequency and Powers do not correspond '
            n = len(freq_array)
            if n % 2 == 0:
                index_array = np.concatenate(([i for i in range(int(n/2)+1)],
                                              [i for i in range(int(1-n/2),0)]))
            else:
                index_array = np.concatenate(([i for i in range(int((n-1)/2)+1)],
                                              [i for i in range(int(-(n-1)/2),0)]))                
            if spower_array is None:
                columns = {'freq':freq_array,'power':power_array}
            else:
                assert len(power_array) == len(spower_array),'Powers and errors do not correspond (len(power)={},len(spower)={})'.\
                    format(len(power_array),len(spower_array))
                columns = {'freq':freq_array,'power':power_array,'spower':spower_array}
            if smart_index:
                super().__init__(columns,index=index_array)
            else:
                super().__init__(columns)

        self._weight = weight_value
        self._low_en = low_en_value
        self._high_en = high_en_value

        self._leahy_norm = leahy_norm_value
        self._rms_norm = rms_norm_value
        self._poi_level = poi_level_value

        self.notes=notes
        self.history=history
        
    @property
    def df(self):
        if len(self.freq) == 0: return None
        return np.round(np.median(np.ediff1d(self.freq[self.index>0])),9)

    @property
    def nf(self):
        if len(self.freq) == 0: return None
        return len(self)*self.df/2.

    @property
    def a0(self):
        if len(self.power) == 0: return None
        if (self._leahy_norm is None) and (self._rms_norm is None):
            return np.sqrt(self.power.iloc[0])
        elif not self._leahy_norm is None:
            return self.power[0]/2.
        else:
            return None

    @property
    def cr(self):
        if len(self.power) == 0: return None
        return self.a0*self.df

    def comp_frac_rms(self,low_freq=0,high_freq=np.inf,pos_only=False):

        assert not self.rms_norm is None,'You need to first normalize the power by rms'
        if np.isinf(high_freq) and not self.nf==None and len(self.freq%2==0):
            mask = (self.freq > 0.) & (self.freq > low_freq) & (self.freq < self.nf) 
            integral = self.power[self.freq>0].iloc[-1]*self.df/2.
        else:
            mask = (self.freq > 0.) & (self.freq > low_freq) & (self.freq < high_freq)
            integral = 0.
        if pos_only:
            mask = mask & (self.power>0)
        #mask = (self._freq >= low_freq) & (self._freq<high_freq)
        integral += (self.power[mask]).sum()*self.df
        rms = 100.*np.sqrt(integral)
        if 'spower' in self.columns:
            sintegral = np.sqrt( (self.spower[mask]**2).sum() )*self.df
            srms = 50./np.sqrt(integral)*sintegral
        else:
            srms = None
        return rms,srms

    def sub_poi(self,value=None,low_freq=0,high_freq=np.inf):

        if value is None:
            mask = (self.freq>=low_freq) & (self.freq<high_freq)
            value = self.power[mask].mean()
        
        print(f'Subtracting {value} to power')
        power = self.power-value
        power.iloc[0] += value

        changes = self.history.copy()
        changes['POI_SUB']=my_cdate()
        changes['POI_RANGE']=f'{low_freq}-{high_freq}'

        if not 'spower' in self.columns :
            return PowerSpectrum(freq_array=self.freq,power_array=power,
                                weight_value = self._weight,low_en_value=self._low_en,high_en_value=self._high_en,
                                leahy_norm_value=self._leahy_norm,rms_norm_value=self._rms_norm,poi_level_value=value,
                                notes={},history=changes)
        else:
            return PowerSpectrum(freq_array=self.freq,power_array=power,spower_array=self.spower,
                                weight_value = self._weight,low_en_value=self._low_en,high_en_value=self._high_en,
                                leahy_norm_value=self._leahy_norm,rms_norm_value=self._rms_norm,poi_level_value=value,
                                notes={},history=changes)        



    def leahy(self):
        if (self._leahy_norm is None) and (self._rms_norm is None):
            norm = 2./self.a0
            #print(np.sqrt(self._power[self._freq==0][0]), 'a0')

            changes = self.history.copy()
            changes['LEAHY_NORM']=my_cdate()

            if not 'spower' in self.columns :
                return PowerSpectrum(freq_array=self.freq,power_array=self.power*norm,
                                    weight_value = self._weight,low_en_value=self._low_en,high_en_value=self._high_en,
                                    leahy_norm_value=norm,rms_norm_value=self._rms_norm,poi_level_value=self._poi_level,
                                    notes={},history=changes)    
            else:
                return PowerSpectrum(freq_array=self.freq,power_array=self.power*norm,spower_array=self.spower*norm,
                                    weight_value = self._weight,low_en_value=self._low_en,high_en_value=self._high_en,
                                    leahy_norm_value=norm,rms_norm_value=self._rms_norm,poi_level_value=self._poi_level,
                                    notes={},history=changes)             
        else:
            print('The power spectrum is already either Leahy or RMS normalized')
            return self
        
    def rms(self,bkg=0):
        changes = self.history.copy()
        if (self._leahy_norm is None) and (self._rms_norm is None):
            norm = (2./self.a0)*self.cr/( (self.cr-bkg)**2 )
            changes['RMS_NORM'] = my_cdate()
        elif (self._rms_norm is None) and (not self._leahy_norm is None):
            norm = self.cr/( (self.cr-bkg)**2 )
            changes['RMS_NORM'] = my_cdate()
        elif not self._rms_norm is None:     
            print('The power spectrum is already RMS normalized')

        if not 'spower' in self.columns:
            return PowerSpectrum(freq_array=self.freq,power_array=self.power*norm,
                                weight_value = self._weight,low_en_value=self._low_en,high_en_value=self._high_en,
                                leahy_norm_value=self._leahy_norm,rms_norm_value=norm,poi_level_value=self._poi_level,
                                notes={},history=changes)                                          
        else:
            return PowerSpectrum(freq_array=self.freq,power_array=self.power*norm,spower_array=self.spower*norm,
                                weight_value = self._weight,low_en_value=self._low_en,high_en_value=self._high_en,
                                leahy_norm_value=self._leahy_norm,rms_norm_value=norm,poi_level_value=self._poi_level,
                                notes={},history=changes)                                                        


    def rebin(self,factors=-30):
        if type(factors) != list: factors=[factors]
        mask = self.freq > 0
        binned_freq = self.freq[mask].to_numpy()
        if not self._poi_level is None:
            poi_level = self._poi_level
        else:
            poi_level = 0.
        binned_power = self.power[mask].to_numpy()+poi_level
        dc_power = self.power[0]
        if 'spower' in self.columns:
            binned_spower = self.spower[mask]
            #print('Before:',len(binned_power),len(binned_spower))
            for f in factors:
                binned_freq,binned_power,dummy,binned_spower=my_rebin(
                    binned_freq,binned_power,ye=binned_spower,rf = f)
            #print('After:',len(binned_power),len(binned_spower))
            return PowerSpectrum(np.append(0,binned_freq),np.append(dc_power,binned_power-poi_level),np.append(0,binned_spower),
            weight_value=self.weight,low_en_value=self.low_en,high_en_value=self.high_en,smart_index=False,
            leahy_norm_value=self._leahy_norm, rms_norm_value=self._rms_norm,poi_level_value=self._poi_level,
            notes=self.notes,history=self.history)
        else:
            for f in factors:
                binned_freq,binned_power,dummy,dummy=my_rebin(binned_freq,binned_power,rf = f)
            return PowerSpectrum(np.append(0,binned_freq),np.append(dc_power,binned_power-poi_level),
            weight_value=self._weight,low_en_value=self._low_en,high_en_value=self._high_en,smart_index=False,
            leahy_norm_value=self._leahy_norm, rms_norm_value=self._rms_norm,poi_level_value=self._poi_level,
            notes=self.notes,history=self.history)
            
    def plot(self,ax=False,xy=False,title=False,lfont=16,**kwargs):
        
        if not 'color' in kwargs.keys(): kwargs['color'] = 'k'

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
        target_keys = ['N_GTIS','GTI_INDEX','N_SEGS','SEG_INDEX']

        history = {}
        history['CREATION_DATE'] = my_cdate()

        if 'List' in str(lightcurve.__class__) or type(lightcurve) == list:

            history['CREATION_MODE'] = 'Power computed from LightcurveList'
            history['T_RES'] = lightcurve[0].tres
            if 'SEG_DUR' in lightcurve[0].history.keys():
                history['SEG_DUR'] = lightcurve[0].history['SEG_DUR']

            powers = []
            for l in lightcurve:
                if l.counts.sum() > 0:

                    power_history = history.copy()
                    for key in target_keys:
                        if key in l.history.keys(): power_history[key]=l.history[key]
                    #assert isinstance(l,Lightcurve),'Object must be Lightcurve'
                    freq = fftfreq(len(l),np.double(l.tres))
                    #print(l.counts.shape,type(l.counts))
                    amp = fft(l.counts.to_numpy())
                    powers += [PowerSpectrum(freq,np.multiply(amp, np.conj(amp)).real,
                                low_en_value=l.low_en,high_en_value=l.high_en,weight_value=1,
                                history=power_history)]

            if len(powers) != 0:
                return PowerList(powers)
            else:
                print('WARNING: Empty PowerList')
                return PowerList()
                
        else:

            history['CREATION_MODE'] = 'Power computed from Lightcurve'
            history['T_RES'] = lightcurve.tres
            if 'SEG_DUR' in lightcurve.history.keys():
                history['SEG_DUR'] = lightcurve.history['SEG_DUR']
            for key in target_keys:
                if key in lightcurve.history.keys(): history[key]=lightcurve.history[key]

            if lightcurve.counts.sum() > 0:
                #assert isinstance(lightcurve,Lightcurve),'Object myst be Lightcurve'
                freq = fftfreq(len(lightcurve),np.double(lightcurve.tres))
                amp = fft(lightcurve.counts.to_numpy())
                #print(len(freq),len(fft))
                power = PowerSpectrum(freq,np.multiply(amp, np.conj(amp)).real,
                                low_en_value=lightcurve.low_en,high_en_value=lightcurve.high_en,
                                weight_value=1,history=history)
            else:
                power = PowerSpectrum()
            
            return power 
            
       
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

            changes={}
            changes['CREATION_DATE']=my_cdate()
            changes['CREATION_MODE']='Average of Leahy power spectra'
            changes['N_SPECTRA']=len(self)
            changes['SEG_DUR'] = self[0].history['SEG_DUR']
            changes['T_RES'] = self[0].history['T_RES']

            return PowerSpectrum(self[0].freq,power,spower,weight_value=new_weight,
                                leahy_norm_value = 1,
                                low_en_value=self[0].low_en,high_en_value=self[0].high_en,
                                history=changes)
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