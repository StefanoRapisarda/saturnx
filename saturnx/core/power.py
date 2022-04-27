import os
import numpy as np
import pandas as pd
import pickle
import pathlib
import math
from collections.abc import Iterable
from scipy.fft import fft,fftfreq

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scipy.fftpack import fftfreq, fft

from astropy.io import fits
from astropy.io.fits import getdata, getval

from saturnx.core.lightcurve import Lightcurve, LightcurveList
from saturnx.utils.time_series import rebin_xy, rebin_arrays
from saturnx.utils.fits import read_fits_keys, get_basic_info
from saturnx.utils.generic import is_number, my_cdate, round_half_up



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
            else:
                super().__init__(column_dict)
        else:
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
    def fres(self):
        if len(self.freq) == 0: return None
        fres = np.median(np.ediff1d(self.freq[self.freq>0]))
        # fres = np.round(fres,abs(int(math.log10(fres/1000))))
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
    def a0(self):
        if not self.power.any(): return None

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
            return self.a0*self.fres
        else: return None

    def __mul__(self,value):
        
        if type(value) in [list,np.ndarray,pd.Series]:
            if len(value) == 0: return self
            
            powers = []
            for item in value:
                if type(item) == str:
                    if is_number(item): 
                        value=eval(item)
                    else:
                        raise TypeError('Cannot multiply string to PowerSpectrum')                
                else:
                    try:
                        float(item)
                    except Exception:
                        raise TypeError('Array items must be numbers')

                power = self.power*item
                if not self.spower is None:
                    spower = self.spower*item
                else:
                    spower = None

                powers += [PowerSpectrum(freq_array= self.freq,
                    power_array=power,spower_array=spower,
                    weight=self.weight,low_en=self.low_en,high_en=self.high_en,
                    leahy_norm=self.leahy_norm,rms_norm=self.rms_norm,
                    poi_level=self.poi_level,smart_index=True,
                    notes=self.notes,meta_data=self.meta_data)]
            
            return PowerList(powers)

        else:
            if type(value) == str:
                if is_number(value): 
                    value=eval(value)
                else:
                    raise TypeError('Cannot multiply string to PowerSpectrum')                
            else:
                try:
                    float(value)
                except Exception:
                    raise TypeError('Value must be a number')            
            
            power = self.power*value 
            if not self.spower is None:
                spower = self.spower*value
            else:
                spower = None   

            return PowerSpectrum(freq_array= self.freq,
                power_array=power,spower_array=spower,
                weight=self.weight,low_en=self.low_en,high_en=self.high_en,
                leahy_norm=self.leahy_norm,rms_norm=self.rms_norm,
                poi_level=self.poi_level,smart_index=True,
                notes=self.notes,meta_data=self.meta_data)

    def __rmul__(self,value):
        return self*value


    def comp_frac_rms(self,low_freq=0,high_freq=np.inf,pos_only=False):

        if not self.power.any():
            return None, None

        if type(low_freq) == str: low_freq = eval(low_freq)
        if type(high_freq) == str: high_freq = eval(high_freq)
        if low_freq < 0: low_freq = 0
        if high_freq > self.nyq: high_freq = self.nyq 
        if low_freq > high_freq:
            raise ValueError('low_freq must be lower than high_freq')

        mask = (self.freq > low_freq) & (self.freq < high_freq)
        if pos_only:
            mask = mask & (self.power>0)
        nyq_power = np.double(self.power[self.freq == min(self.freq)])
        if len(self)%2 != 0 or high_freq < self.nyq: nyq_power = 0

        if (self.leahy_norm is None) and (self.rms_norm is None):
            rms2 = (2*np.sum(self.power[mask]) + nyq_power)/self.a0**2
        elif self.rms_norm is None:
            rms2 = (np.sum(self.power[mask]) + nyq_power/2)/self.a0
        else:
            rms2 = (np.sum(self.power[mask]) + nyq_power/2) * self.fres

        rms = np.sqrt(rms2)

        if self.spower.any():
            srms2_term1 = 1./4/rms2
            if (self.leahy_norm is None) and (self.rms_norm is None):
                srms2_term2 = 4*np.sum(self.spower[mask]**2)/self.a0**4
            elif self.rms_norm is None:
                srms2_term2 = np.sum(self.spower[mask]**2)/self.a0**2
            else:
                srms2_term2 = np.sum(self.spower[mask]**2) * self.fres**2           
            
            srms2 = srms2_term1 * srms2_term2
            srms = np.sqrt(srms2)
        else:
            srms = None

        return rms,srms

    def sub_poi(self,value=None,low_freq=0,high_freq=np.inf,print_value=False):
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
        pw_poi: saturnx.core.PowerSpectrum
            PowerSpectrum with subtracted power and updated meta_data

        '''

        meta_data= self.meta_data.copy()
        meta_data['SUBTRACTING_POI'] = my_cdate()

        if value is None:
            if low_freq < 0: low_freq = 0
            if low_freq > self.nyq: low_freq = self.nyq
            if high_freq > self.nyq: high_freq = self.nyq
            if low_freq > high_freq:
                raise ValueError('low_freq must be lower than high_freq')
            mask = (self.freq>=low_freq) & (self.freq<high_freq) * (self.freq>0)
            value = self.power[mask].mean()
            meta_data['POI_RANGE'] = f'{low_freq}-{high_freq}'
        elif type(value) in [list,np.ndarray,pd.Series]:
            if len(value) != len(self):
                raise ValueError('values must have the same dimension of power')
    
        if print_value:
            print('Poisson level: {}'.format(value))

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

        if self.cr-bkg_cr < 0:
            print('Warning!!! background cr larger than source cr')

        if norm == 'leahy':
            if (self._leahy_norm is None) and (self._rms_norm is None):
                norm = 2./self.a0
                norm_leahy = norm
                norm_rms = None
                meta_data['NORMALIZING'] = my_cdate()
                meta_data['NORM_MODE'] = 'Leahy'
            elif (self._rms_norm is None):
                print('The power spectrum is already either Leahy or RMS normalized')
                norm = 1
                norm_leahy = self._leahy_norm
                norm_rms = self._rms_norm
        elif norm == 'rms':
            if (self._leahy_norm is None) and (self._rms_norm is None):
                norm_leahy = (2./self.a0)
                norm_rms = self.cr/( (self.cr-bkg_cr)**2 )
                norm = norm_leahy*norm_rms
                meta_data['NORMALIZING'] = my_cdate()
                meta_data['NORM_MODE'] = 'FRAC_RMS'
            elif (self._rms_norm is None) and (not self._leahy_norm is None):
                norm = self.cr/( (self.cr-bkg_cr)**2 )
                norm_leahy = self.leahy_norm
                norm_rms = norm
                meta_data['NORMALIZING'] = my_cdate()
                meta_data['NORM_MODE'] = 'FRAC_RMS'
            elif not self._rms_norm is None:   
                norm = 1
                norm_leahy = self._leahy_norm
                norm_rms = self._rms_norm  
                print('The power spectrum is already RMS normalized') 
        elif norm is None:
            return self 
        else:
            if type(norm) == str: norm = eval(norm)   
            meta_data['NORMALIZING'] = my_cdate()
            meta_data['NORM_MODE'] = 'number'
            meta_data['NORM_VALUE'] = norm   
            norm_leahy = None
            norm_rms = None  


        if not self.spower.any() :
            #print('Power without errors')
            power = PowerSpectrum(freq_array=self.freq,power_array=self.power*norm,
                                weight = self._weight,low_en=self._low_en,high_en=self._high_en,
                                leahy_norm=norm_leahy,rms_norm=norm_rms,poi_level=self._poi_level,
                                notes={},meta_data=meta_data)    
        else:
            #print('Power with errors')
            power = PowerSpectrum(freq_array=self.freq,power_array=self.power*norm,spower_array=self.spower*norm,
                                weight = self._weight,low_en=self._low_en,high_en=self._high_en,
                                leahy_norm=norm_leahy,rms_norm=norm_rms,poi_level=self._poi_level,
                                notes={},meta_data=meta_data)             

        return power
                                                   
    def rebin(self,factors=-30):

        if type(factors) != list: factors=[factors]

        meta_data = self.meta_data.copy()
        meta_data['REBINNING'] = my_cdate()
        meta_data['REBIN_FACTOR'] = factors
        
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
                notes = self.notes, meta_data = meta_data)
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
                notes = self.notes, meta_data = meta_data)

        return pw
            
    def plot(self,ax=False,xy=False,title=False,lfont=16,**kwargs):
        
        if not 'color' in kwargs.keys(): kwargs['color'] = 'k'
        #if not 'marker' in kwargs.keys(): kwargs['marker']='o'

        if ax is False:
            fig, ax = plt.subplots(figsize=(6,6))

        if (not title is False) and (not ax is False):
            ax.set_title(title)

        mask = self.freq > 0

        if self.spower.any():
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
        if not self.rms_norm is None:
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
        target_keys = ['EVT_FILE_NAME','DIR','MISSION','INFO_FROM_HEADER',
                'GTI_SPLITTING','N_GTIS','GTI_INDEX',
                'SEG_SPLITTING','N_SEGS','SEG_INDEX',
                'N_ACT_DET','INACT_DET_LIST',
                'FILTERING','FILT_EXPR']

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
                    amp = fft(l.counts.to_numpy())
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
                amp = fft(lightcurve.counts.to_numpy())
                power = PowerSpectrum(freq_array = freq, power_array = np.multiply(amp, np.conj(amp)).real,
                                low_en = lightcurve.low_en, high_en = lightcurve.high_en,
                                weight = 1, meta_data = meta_data)
            else:
                power = PowerSpectrum()
            
            return power 

        else:
            raise TypeError('You can compute Power Spectrum only from lightcurve')

    @staticmethod
    def read_fits(fits_file, ext='POWER_SPECTRUM', freq_col='FREQ', power_col='POWER', 
        spower_col='POWER_ERR',keys_to_read=None):

        if not type(fits_file) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        if type(fits_file) == str:
            fits_file = pathlib.Path(fits_file)
        if fits_file.suffix == '':
            fits_file = fits_file.with_suffix('.fits')

        if not fits_file.is_file():
            raise FileNotFoundError('FITS file does not exist')

        mission = None
        try:
            mission = getval(fits_file,'TELESCOP',ext)
        except Exception as e:
            print('Warning: TELESCOP not found while reading PowerSpectrum from fits')
        try:
            low_en = getval(fits_file,'LOW_EN',ext)
        except:
            low_en = None
        try:
            high_en = getval(fits_file,'HIGH_EN',ext)
        except:
            high_en = None        

        meta_data = {}

        meta_data['PW_CRE_MODE'] = 'Lightcurve read from fits file'
        meta_data['FILE_NAME'] = str(fits_file.name)
        meta_data['DIR'] = str(fits_file.parent)
        meta_data['MISSION'] = mission 

        # Reading meaningfull information from event file
        info = get_basic_info(fits_file)
        if not keys_to_read is None:
            if type(keys_to_read) in [str,list]: 
                user_info = read_fits_keys(fits_file,keys_to_read,ext)
            else:
                raise TypeError('keys to read must be str or list')
        else: user_info = {}
        total_info = {**info,**user_info}
        meta_data['INFO_FROM_HEADER'] = total_info       

        data = getdata(fits_file,extname=ext,meta_data=False,memmap=True)        
        freq = data[freq_col]
        if power_col in data.columns.names:
            power = data[power_col]
        if spower_col in data.columns.names:
            spower = data[spower_col]

        return PowerSpectrum(freq_array = freq, power_array = power, spower_array = spower,
            low_en = low_en, high_en = high_en, weight = 1,
            meta_data = meta_data, notes = {})

    def to_fits(self,file_name='power_spectrum.fits',fold=pathlib.Path.cwd()):

        if not type(file_name) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        if type(file_name) == str:
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.fits')

        if type(fold) == str:
            fold = pathlib.Path(fold)
        elif type(fold) != type(pathlib.Path.cwd()):
            raise TypeError('fold name must be either a string or a path')
        
        file_name = fold / file_name

        cols = []
        cols += [fits.Column(name='FREQ', format='D',array=self.freq.to_numpy())]
        cols += [fits.Column(name='POWER', format='D',array=self.power.to_numpy())]
        if self.spower.any(): 
            cols += [fits.Column(name='POWER_ERR', format='D',array=self.rate.to_numpy())]
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.name = 'POWER_SPECTRUM'

        for key,item in self.meta_data.items():
            if key != 'INFO_FROM_HEADER':
                hdu.header[key] = item
            else:
                for sub_key,sub_item in item.items():
                    hdu.header[sub_key] = sub_item
        hdu.header['LOW_EN'] = self.low_en
        hdu.header['HIGH_EN'] = self.high_en

        for key,item in self.notes.items():
            new_key = 'NOTE_'+key
            hdu.header[new_key] = item

        phdu = fits.PrimaryHDU()
        hdu_list = fits.HDUList([phdu,hdu])
        hdu_list.writeto(file_name)

    def save(self,file_name='power_spectrum.pkl',fold=pathlib.Path.cwd()):

        if not type(file_name) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        if type(file_name) == str:
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if type(fold) == str:
            fold = pathlib.Path(fold)
        if type(fold) != type(pathlib.Path.cwd()):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name
        
        try:
            self.to_pickle(file_name)
            print('PowerSpectrum saved in {}'.format(file_name))
        except Exception as e:
            print(e)
            print('Could not save PowerSpectrum')

    @staticmethod
    def load(file_name,fold=pathlib.Path.cwd()):

        if not type(file_name) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        elif type(file_name) == str:
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if type(fold) == str:
            fold = pathlib.Path(fold)
        if type(fold) != type(pathlib.Path.cwd()):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name

        if not file_name.is_file():
            raise FileNotFoundError(f'{file_name} not found'.format(file_name))
        
        lc = pd.read_pickle(file_name)
        
        return lc
       
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
        if not np.array([type(i) == type(PowerSpectrum()) for i in self]).all():
            raise TypeError('All the elements must be Power objects')

    def __setitem__(self, index, power):
        if not type(power) != type(PowerSpectrum()):
            raise TypeError('The item must be a Power object')
        super(PowerList, self).__setitem__(index,power)

    def average(self, norm='leahy',exc=[]):
        '''
        Method for averaging Leahy powers in the list
        '''

        if len(self) != 0:
            num = []
            den = 0
            counter = 1
            a0 = 0
            for i in range(len(self)):
                if i in exc: 
                    counter += 1
                    continue
                else:
                    counter = 1
                if i > 0:
                    assert self[i].freq.equals(self[i-counter].freq),\
                        'Frequency array do not correspond, impossible to average'

                a0 += self[i].a0*self[i].weight
                num += [self[i].normalize(norm).power*self[i].weight]
                den += self[i].weight
            
            new_weight = den
            num = np.array(num).sum(axis=0)
            a0 /= den

            power = num/den
            spower = power/np.sqrt(new_weight)

            if norm is None:
                power[0] = a0**2
                leahy_norm = None
                rms_norm = None
            elif norm == 'leahy':
                leahy_norm = 2./a0
                rms_norm = None

            meta_data={}
            meta_data['PW_CRE_DATE'] = my_cdate()
            meta_data['PW_CRE_MODE'] = 'Average of Leahy power spectra'
            meta_data['N_PWA']=len(self)
            try:
                meta_data['SEG_DUR'] = self[0].meta_data['SEG_DUR']
            except:
                meta_data['SEG_DUR'] = None
            try:
                meta_data['TIME_RES'] = self[0].meta_data['TIME_RES']
            except:
                meta_data['TIME_RES'] = None
            try:
                meta_data['INFO_FROM_HEADER'] = self[0].meta_data['INFO_FROM_HEADER']
            except:
                meta_data['INFO_FROM_HEADER'] = None

            return PowerSpectrum(freq_array = self[0].freq,
                                 power_array = power,
                                 spower_array = spower,
                                 weight = new_weight,
                                 leahy_norm = leahy_norm, rms_norm = rms_norm,
                                 low_en = self[0].low_en, high_en = self[0].high_en,
                                 notes = {}, meta_data = meta_data)
        else:
            print('WARNING: PowerList is empty, returning empty PowerSpectrum')
            return PowerSpectrum()

    def info(self):
        '''
        Returns a pandas DataFrame relevand information for each PowerSpectrum
        object in the list
        '''

        columns = ['fres','nyq','n_bins','a0','count_rate',
                    'frac_rms','frac_rms_err',
                    'leahy_norm','rms_norm','weight',
                    'min_en','max_en','mission']
        info = pd.DataFrame(columns=columns)
        for i,pw in enumerate(self):
            if isinstance(pw.poi_level,Iterable):
                poi_level = 'array'
            else:
                poi_level = pw.poi_level
            line = {'fres':pw.fres,'nyq':pw.nyq,'n_bins':len(pw),
                'a0':pw.a0,'count_rate':pw.cr,
                'leahy_norm':pw.leahy_norm,'rms_norm':pw.rms_norm,
                'weight':pw.weight,'poi_level':poi_level,
                'frac_rms':pw.comp_frac_rms()[0],'frac_rms_err':pw.comp_frac_rms()[1],
                'min_en':pw.low_en,'max_en':pw.high_en,
                'mission':pw.meta_data['MISSION']}
            info.loc[i] = pd.Series(line)

        return info

    def save(self,file_name='power_list.pkl',fold=pathlib.Path.cwd()):

        if not type(file_name) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        if type(file_name) == str:
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if type(fold) == str:
            fold = pathlib.Path(fold)
        if type(fold) != type(pathlib.Path.cwd()):
            raise TypeError('fold name must be either a string or a path')

        if not str(fold) in str(file_name):
            file_name = fold / file_name      

        try:
            with open(file_name,'wb') as output:
                pickle.dump(self,output)
            print('PowerList saved in {}'.format(file_name))
        except Exception as e:
            print(e)
            print('Could not save PowerList')

    @staticmethod
    def load(file_name,fold=pathlib.Path.cwd()):

        if not type(file_name) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        elif type(file_name) == str:
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if type(fold) == str:
            fold = pathlib.Path(fold)
        if type(fold) != type(pathlib.Path.cwd()):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name

        if not file_name.is_file():
            raise FileNotFoundError(f'{file_name} not found'.format(file_name))   

        with open(file_name,'rb') as infile:
            power_list = pickle.load(infile)
        
        return power_list