'''This module contains the definition of Power and PowerList 
classes'''

import numpy as np
import pandas as pd
import pickle
import pathlib
import copy
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
    '''
    PowerSpectrum object. It computes/stores Fourier power and 
    frequencies with specific normalization

    ATTRIBUTES
    ----------
    weight: integer
        weight of each power. If a power has been the result of average
        of other power, it weight will be the number of averaged powers
    low_en: float, string, or None
        Lower energy in keV
    high_en: float, string, or None
        Upper energy in keV
    leahy_norm: float or None
        It is either initialized by the user or by the method normalize.
        In the latter case, the Leahy normalization is equal to 2/a0,
        where a0 is the zero-frequency Fourier amplitude (DC).
        When not None, it also indicates that the power has been Leahy
        normalized.
    rms_norm: float or None
        It is either initialized by the user or by the method normalize.
        In the latter case, the RMS normalization is equal to cr/(cr-bkg),
        where cr is the total count rate and bkg is the background count
        rate. When bkg is not zero, this normalization is ofter referred
        to as source RMS normalization.
        When not None, it also indicates that the power has been RMS
        normalized.
    poi_level: float or None 
        It is either initialized by the user or by the method sub_poi.
        When not None, it indicates that the Poisson level has been 
        subtracted from the Power.
    fres: float or None
        Frequency resolution computed as the median of the difference of 
        consecutive frequency bins rounded up to the 12th order of 
        magnitude (picoHz). If the frequency array is empty, None is 
        returned.
    nyqf: float or None
        Value of the last frequency bin. It may not correspond to the
        theoretical value 1/(2dt), where dt is the time resolution. 
        If the frequency array is empty, nyqf = None
    a0: float or None
        Zero-frequency, or direct current (DC), Fourier component. It is
        equal to the total number of photons. If the power has been 
        normalized, it is retrieved from the normalization, otherwise
        it is just the square root of the power at zero frequency.
    cr: float or None
        Count rate of the Lightcurve used to compute the power spectrum,
        it is computed as a0*fres
    meta_data: dictionary
        Container of useful information, user notes (key NOTES), and 
        data reduction history (key HISTORY)

    METHODS
    -------
    __mul__(self, value)
        - If value is a list, a numpy.ndarray, or a pandas.Series...
        Returns a PowerList where each PowerSpectrum bin power is
        multiplied by the i-th array element.
        - If value object is a number...
        Multiplies counts and value and returns the modified PowerSpectrum
    
    comp_frac_rms(low_freq=0,high_freq=np.inf,pos_only=False)
        Computes fractional RMS between two specified frequencies

    sub_poi(value=None,low_freq=0,high_freq=np.inf,print_value=False)
        Subtract the Poisson noise from the power according either to
        a specified value or averaging power between low_freq and 
        high_freq

    normalize(norm='leahy',bkg_cr=0)
        Normalizes power spectrum according to specified normalization
        (Leahy, RMS, or simple value)

    plot(ax=False,xy=False,title=False,lfont=16,**kwargs)
        Plots Power Spectrum to a new or existing matplotlib.Axes.axis

    from_lc(lightcurve)
        Computed PowerSpectrum from a Lightcurve or LightcurveList
        object, in the latter case it returns a PowerList

    read_fits(fits_file, ext='POWER_SPECTRUM', freq_col='FREQ', 
              power_col='POWER', spower_col='POWER_ERR',keys_to_read=None)
        Reads a power spectrum from a FITS file

    to_fits(file_name='power_spectrum.fits',fold=pathlib.Path.cwd())
        Write a PowerSpectrum object into a FITS file

    save(file_name='power_spectrum.pkl',fold=pathlib.Path.cwd())
        Saves Power Spectrum in a pickle file

    load(fold=pathlib.Path.cwd())
        Loads Power Spectrum from a pickle file
    '''

    _metadata = ['_weight','weight','_high_en','high_en','_low_en',
                 'low_en','_leahy_norm','leahy_norm','_rms_norm',
                 'rms_norm','_poi_level','poi_level','meta_data',
                 '_meta_data']

    def __init__(self,freq_array=np.array([]),power_array=None,spower_array=None,
                 weight=1,low_en=None,high_en=None,
                 leahy_norm=None,rms_norm=None,poi_level=None,
                 smart_index=True,meta_data=None):

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

        self.weight = weight

        self.leahy_norm = leahy_norm
        self.rms_norm = rms_norm
        self.poi_level = poi_level

        self.low_en = low_en
        self.high_en = high_en

        self.meta_data = meta_data
        if not 'PW_CRE_DATE' in self.meta_data['HISTORY'].keys():
            self.meta_data['HISTORY']['PW_CRE_DATE'] = my_cdate() 

    def __mul__(self,value):
        
        if isinstance(value,(list,np.ndarray,pd.Series)):
     
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
                if self.spower is not None:
                    spower = self.spower*item
                else:
                    spower = None

                powers += [PowerSpectrum(
                    freq_array= self.freq,power_array=power,spower_array=spower,
                    weight=self.weight,low_en=self.low_en,high_en=self.high_en,
                    leahy_norm=self.leahy_norm,rms_norm=self.rms_norm,
                    poi_level=self.poi_level,smart_index=True,
                    meta_data=self.meta_data
                    )]
            
            return PowerList(powers)

        else:
            if isinstance(value,str):
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

            return PowerSpectrum(
                freq_array= self.freq,power_array=power,spower_array=spower,
                weight=self.weight,low_en=self.low_en,high_en=self.high_en,
                leahy_norm=self.leahy_norm,rms_norm=self.rms_norm,
                poi_level=self.poi_level,smart_index=True,
                meta_data=self.meta_data
                )

    def __rmul__(self,value):
        return self*value

    def comp_frac_rms(self,low_freq=0,high_freq=np.inf,pos_only=False):

        if not self.power.any():
            return None, None

        if type(low_freq) == str: low_freq = eval(low_freq)
        if type(high_freq) == str: high_freq = eval(high_freq)
        if low_freq < 0: low_freq = 0
        if high_freq > self.nyqf: high_freq = self.nyqf
        if low_freq > high_freq:
            raise ValueError('low_freq must be lower than high_freq')

        mask = (self.freq > low_freq) & (self.freq < high_freq)
        if pos_only:
            mask = mask & (self.power>0)
        nyq_power = np.double(self.power[self.freq == min(self.freq)])
        if len(self)%2 != 0 or high_freq < self.nyqf: nyq_power = 0

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

        meta_data= copy.deepcopy(self.meta_data)
        meta_data['HISTORY']['SUBTRACTING_POI'] = my_cdate()

        if value is None:
            if low_freq < 0: low_freq = 0
            if low_freq > self.nyqf: low_freq = self.nyqf
            if high_freq > self.nyqf: high_freq = self.nyqf
            if low_freq > high_freq:
                raise ValueError('low_freq must be lower than high_freq')
            mask = (self.freq>=low_freq) & (self.freq<high_freq) * (self.freq>0)
            value = self.power[mask].mean()
            meta_data['POI_RANGE'] = f'{low_freq}-{high_freq}'
            meta_data['POI_LEVEL'] = value
        elif isinstance(value,(list,np.ndarray,pd.Series)):
            if len(value) != len(self):
                raise ValueError('values must have the same dimension of power')
    
        if print_value:
            print('Poisson level: {}'.format(value))

        # Keeping the original DC (a0) component
        if isinstance(value,(list,np.ndarray)):
            value[0] = 0
        elif isinstance(value,pd.Series):
            value.iloc[0] = 0        
        power = np.subtract(self.power,value)
        if not isinstance(value,(list,np.ndarray,pd.Series)):
            power.iloc[0] += value

        if not 'spower' in self.columns :
            poi_pw = PowerSpectrum(freq_array = self.freq, power_array = power,
                                weight = self._weight, low_en = self._low_en, high_en = self._high_en,
                                leahy_norm = self._leahy_norm, rms_norm = self._rms_norm, poi_level = value,
                                meta_data = meta_data)
        else:
            poi_pw = PowerSpectrum(freq_array = self.freq, power_array = power, spower_array = self.spower,
                                weight = self._weight,low_en = self._low_en, high_en = self._high_en,
                                leahy_norm = self._leahy_norm, rms_norm = self._rms_norm, poi_level = value,
                                meta_data = meta_data)        

        return poi_pw

    def normalize(self,norm='leahy',bkg_cr=0):

        meta_data = copy.deepcopy(self.meta_data)

        if self.cr-bkg_cr < 0:
            print('Warning!!! background cr larger than source cr')

        if norm is None:
            return self
        
        if norm.upper() == 'LEAHY':
            if (self._leahy_norm is None) and (self._rms_norm is None):
                norm = 2./self.a0
                norm_leahy = norm
                norm_rms = None
                meta_data['HISTORY']['NORMALIZING'] = my_cdate()
                meta_data['NORM_MODE'] = 'Leahy'
            elif (self._rms_norm is None):
                print('The power spectrum is already Leahy normalized')
                norm = 1
                norm_leahy = self._leahy_norm
                norm_rms = self._rms_norm
        elif norm.upper() == 'RMS':
            if (self._leahy_norm is None) and (self._rms_norm is None):
                norm_leahy = (2./self.a0)
                norm_rms = self.cr/( (self.cr-bkg_cr)**2 )
                norm = norm_leahy*norm_rms
                meta_data['HISTORY']['NORMALIZING'] = my_cdate()
                meta_data['NORM_MODE'] = 'FRAC_RMS'
            elif (self._rms_norm is None) and (not self._leahy_norm is None):
                norm = self.cr/( (self.cr-bkg_cr)**2 )
                norm_leahy = self.leahy_norm
                norm_rms = norm
                meta_data['HISTORY']['NORMALIZING'] = my_cdate()
                meta_data['NORM_MODE'] = 'FRAC_RMS'
            elif not self._rms_norm is None:   
                norm = 1
                norm_leahy = self._leahy_norm
                norm_rms = self._rms_norm  
                print('The power spectrum is already RMS normalized') 
        else:
            if isinstance(norm,str): 
                norm = eval(norm)   
            meta_data['HISTORY']['NORMALIZING'] = my_cdate()
            meta_data['NORM_MODE'] = 'number'
            meta_data['NORM_VALUE'] = norm   
            norm_leahy = None
            norm_rms = None  

        if not self.spower.any() :
            #print('Power without errors')
            power = PowerSpectrum(
                freq_array=self.freq,power_array=self.power*norm,
                weight = self.weight,low_en=self.low_en,high_en=self.high_en,
                leahy_norm=norm_leahy,rms_norm=norm_rms,poi_level=self.poi_level,
                meta_data=meta_data
                )    
        else:
            #print('Power with errors')
            power = PowerSpectrum(
                freq_array=self.freq,power_array=self.power*norm,
                spower_array=self.spower*norm,weight = self.weight,
                low_en=self.low_en,high_en=self.high_en,
                leahy_norm=norm_leahy,rms_norm=norm_rms,poi_level=self.poi_level,
                meta_data=meta_data
                )             

        return power
                                                   
    def rebin(self,factors=-30):

        if type(factors) != list: factors=[factors]

        meta_data = copy.deepcopy(self.meta_data)
        meta_data['HISTORY']['REBINNING'] = my_cdate()
        meta_data['REBIN_FACTOR'] = factors
        
        mask = self.freq > 0
        binned_freq = self.freq[mask]
        
        # Poisson level is reintroduced, the array is rebinned, and 
        # the Poisson level is subtracted again
        if self.poi_level is not None:
            poi_level = self.poi_level
        else:
            poi_level = 0.
        if isinstance(self.poi_level,(list,np.ndarray,pd.Series)):
            poi_level = self.poi_level[mask]
        
        binned_power = np.add(self.power[mask],poi_level)
        binned_poi = poi_level
        dc_power = self.power.iloc[0]

        if self.spower.any():
            binned_spower = self.spower[mask]
            #print('Before:',len(binned_power),len(binned_spower))
            for f in factors:
                binned_freq,binned_power,dummy,binned_spower=rebin_xy(
                    binned_freq,binned_power,ye=binned_spower,rf = f)
                if isinstance(binned_poi,(list,np.ndarray,pd.Series)):
                    d,binned_poi,d,d = rebin_xy(binned_freq,binned_poi,rf = f)

            #print('After:',len(binned_power),len(binned_spower))
            pw = PowerSpectrum(
                freq_array = np.append(0,binned_freq),
                power_array = np.append(dc_power,binned_power-binned_poi),
                spower_array = np.append(0,binned_spower),
                weight = self.weight, low_en = self.low_en, high_en = self.high_en,
                smart_index = False,
                leahy_norm = self.leahy_norm, rms_norm = self.rms_norm,
                poi_level = binned_poi,
                meta_data = meta_data)
        else:
            for f in factors:
                binned_freq,binned_power,dummy,dummy=rebin_xy(
                    binned_freq,binned_power,rf = f)
                if type(binned_poi) in [list,np.ndarray,pd.Series]:
                    d,binned_poi,d,d = rebin_xy(binned_freq,binned_poi,rf = f)

            pw = PowerSpectrum(
                freq_array = np.append(0,binned_freq),
                power_array = np.append(dc_power,binned_power-binned_poi),
                weight = self.weight, low_en = self.low_en, high_en = self.high_en,
                smart_index = False,
                leahy_norm = self.leahy_norm, rms_norm = self.rms_norm,
                poi_level = binned_poi,
                meta_data = meta_data
                )

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

    @classmethod
    def from_lc(cls,lightcurve):

        # I want the information contained in these keyword to propagate
        # in the power spectrum
        target_keys = ['EVT_FILE_NAME','DIR','MISSION','INFO_FROM_HEADER',
                       'N_GTIS','GTI_INDEX',
                       'N_SEGS','SEG_INDEX',
                       'N_ACT_DET','INACT_DET_LIST',
                       'FILTERING','FILT_EXPR']

        meta_data = {}
        meta_data['PW_CRE_MODE'] = 'Power computed from Lightcurve'

        if isinstance(lightcurve,LightcurveList):

            meta_data['TIME_RES'] = lightcurve[0].tres
            if 'SEG_DUR' in lightcurve[0].meta_data.keys():
                meta_data['SEG_DUR'] = lightcurve[0].meta_data['SEG_DUR']

            powers = []
            for l in lightcurve:
                if (l.tot_counts is not None) and (l.tot_counts > 0):

                    power_meta_data = copy.deepcopy(meta_data)
                    for key in target_keys:
                        if key in l.meta_data.keys(): 
                            power_meta_data[key]=l.meta_data[key]

                    freq = fftfreq(len(l),np.double(l.tres))
                    amp = fft(l.counts.to_numpy())
                    powers += [cls(
                        freq_array = freq,
                        power_array = np.multiply(amp, np.conj(amp)).real,
                        low_en = l.low_en, high_en = l.high_en, weight = 1,
                        meta_data = power_meta_data
                        )]

            if len(powers) != 0:
                return PowerList(powers)
            else:
                print('WARNING: Empty PowerList')
                return cls()
                
        elif isinstance(lightcurve,Lightcurve):

            meta_data['TIME_RES'] = lightcurve.tres
            if 'SEG_DUR' in lightcurve.meta_data.keys():
                meta_data['SEG_DUR'] = lightcurve.meta_data['SEG_DUR']
            for key in target_keys:
                if key in lightcurve.meta_data.keys(): meta_data[key]=lightcurve.meta_data[key]

            if (lightcurve.counts is not None) and (lightcurve.tot_counts > 0):
                
                freq = fftfreq(len(lightcurve),np.double(lightcurve.tres))
                amp = fft(lightcurve.counts.to_numpy())
                power = cls(
                    freq_array = freq, 
                    power_array = np.multiply(amp, np.conj(amp)).real,
                    low_en = lightcurve.low_en, high_en = lightcurve.high_en,
                    weight = 1, meta_data = meta_data)
            else:
                power = cls()
            
            return power 

        else:
            raise TypeError('You can compute Power Spectrum only from lightcurve')

    @classmethod
    def read_fits(cls,fits_file, ext='POWER_SPECTRUM', freq_col='FREQ', 
                  power_col='POWER',spower_col='POWER_ERR',keys_to_read=None):

        if not isinstance(fits_file,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        if isinstance(fits_file,str):
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
                mission = getval(fits_file,'MISSION',ext) 
            except Exception as e:
                print('WARNING: MISSION not found while reading Lightcurve from fits')
                print('mission will be set to None')  
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
            if isinstance(keys_to_read,(str,list)): 
                user_info = read_fits_keys(fits_file,keys_to_read,ext)
            else:
                raise TypeError('keys to read must be str or list')
        else: user_info = {}
        total_info = {**info,**user_info}
        meta_data['INFO_FROM_HEADER'] = total_info      

        history = {}
        notes = {}
        # Reading the rest of header keywords
        with fits.open(fits_file) as hdu_list:
            header = hdu_list[ext].header
            keys = header.keys()
            for key in keys:
                if 'NOTE' in key: 
                    notes[key.split('_')[1]] = header[key]
                if 'HIST' in key:
                    history[key.split('_')[1]] = header[key]
        meta_data['NOTES'] = notes
        meta_data['HISTORY'] = history

        data = getdata(fits_file,extname=ext,meta_data=False,memmap=True)        
        
        freq = data[freq_col]
        if power_col in data.columns.names:
            power = data[power_col]
        if spower_col in data.columns.names:
            spower = data[spower_col]

        return cls(freq_array = freq, power_array = power, spower_array = spower,
                   low_en = low_en, high_en = high_en, weight = 1,
                   meta_data = meta_data)

    def to_fits(self,file_name='power_spectrum.fits',fold=pathlib.Path.cwd()):

        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        if isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.fits')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        elif not isinstance(fold,pathlib.Path):
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
            if key == 'INFO_FROM_HEADER':
                for sub_key,sub_item in item.items():
                    hdu.header[sub_key] = sub_item
            elif key not in ['NOTES','HISTORY']:
                hdu.header[key] = item

        hdu.header['LOW_EN'] = self.low_en
        hdu.header['HIGH_EN'] = self.high_en

        for key,item in self.meta_data['HISTORY'].items():
            new_key = 'HIST_'+key
            hdu.header[new_key] = item

        for key,item in self.meta_data['NOTES'].items():
            new_key = 'NOTE_'+key
            hdu.header[new_key] = item

        phdu = fits.PrimaryHDU()
        hdu_list = fits.HDUList([phdu,hdu])
        hdu_list.writeto(file_name)

    def save(self,file_name='power_spectrum.pkl',fold=pathlib.Path.cwd()):

        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        if isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        if not isinstance(fold,pathlib.Path):
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

        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        elif isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        if not isinstance(fold,pathlib.Path):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name

        if not file_name.is_file():
            raise FileNotFoundError(f'{file_name} not found'.format(file_name))
        
        lc = pd.read_pickle(file_name)
        
        return lc

    # >>> ATTRIBUTES <<<
       
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
    def low_en(self,value):
        if value is not None:
            if isinstance(value,str): 
                value = eval(value)
            if value < 0: 
                value = 0
        self._low_en = value

    @property
    def high_en(self):
        return self._high_en

    @high_en.setter
    def high_en(self,value):
        if value is not None and isinstance(value,str): 
            value = eval(value)
        self._high_en = value

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

    @property
    def fres(self):
        if len(self.freq) > 1:
            fres = np.median(np.ediff1d(self.freq[self.freq>0]))
            # fres = np.round(fres,abs(int(math.log10(fres/1000))))
            return round_half_up(fres,12)
        else:
            return None

    @property
    def nyqf(self):
        if len(self.freq) == 0: 
            return None
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
        if not self.power.any(): 
            return None

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
        else: 
            return None

    @property
    def meta_data(self):
        return self._meta_data

    @meta_data.setter
    def meta_data(self,value):
        if value is None:
            self._meta_data = {}
        else:
            if not isinstance(value,dict):
                raise TypeError('meta_data must be a dictionary')
            self._meta_data = copy.deepcopy(value)

        if not 'HISTORY' in self.meta_data.keys():
            self._meta_data['HISTORY'] = {}            

        if not 'NOTES' in self.meta_data.keys():
            self._meta_data['NOTES'] = {}   

        
class PowerList(list):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.array([isinstance(i,PowerSpectrum) for i in self]).all():
            raise TypeError('All the elements must be Power objects')

    def __setitem__(self, index, power):
        if not isinstance(power,PowerSpectrum):
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
                                 meta_data = meta_data)
        else:
            print('WARNING: PowerList is empty, returning empty PowerSpectrum')
            return PowerSpectrum()

    def info(self):
        '''
        Returns a pandas DataFrame relevand information for each PowerSpectrum
        object in the list
        '''

        columns = ['fres','nyqf','n_bins','a0','count_rate',
                    'frac_rms','frac_rms_err',
                    'leahy_norm','rms_norm','weight',
                    'min_en','max_en','mission']
        info = pd.DataFrame(columns=columns)
        for i,pw in enumerate(self):
            if isinstance(pw.poi_level,Iterable):
                poi_level = 'array'
            else:
                poi_level = pw.poi_level
            line = {'fres':pw.fres,'nyqf':pw.nyqf,'n_bins':len(pw),
                    'a0':pw.a0,'count_rate':pw.cr,
                    'leahy_norm':pw.leahy_norm,'rms_norm':pw.rms_norm,
                    'weight':pw.weight,'poi_level':poi_level,
                    'frac_rms':pw.comp_frac_rms()[0],'frac_rms_err':pw.comp_frac_rms()[1],
                    'min_en':pw.low_en,'max_en':pw.high_en,
                    'mission':pw.meta_data['MISSION']}
            info.loc[i] = pd.Series(line)

        return info

    def save(self,file_name='power_list.pkl',fold=pathlib.Path.cwd()):

        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        if isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        if not isinstance(fold,pathlib.Path):
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

        if not isinstance(file_name,(pathlib.Path,str)):
            raise TypeError('file_name must be a string or a Path')
        elif isinstance(file_name,str):
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if isinstance(fold,str):
            fold = pathlib.Path(fold)
        if not isinstance(fold,pathlib.Path):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name

        if not file_name.is_file():
            raise FileNotFoundError(f'{file_name} not found'.format(file_name))   

        with open(file_name,'rb') as infile:
            power_list = pickle.load(infile)
        
        return power_list