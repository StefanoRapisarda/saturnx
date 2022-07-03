import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import pathlib
import math
import pickle

from scipy.stats import chi2
from scipy.signal import convolve, wavelets
from scipy.fftpack import fft, fftfreq

from saturnx.core.lightcurve import Lightcurve
from saturnx.utils.generic import my_cdate



class BaseWavelet:
    '''
    Base wavelet object

    ATTRIBUTES
    ----------
    x: numpy ndarray or None
        x array (initialized here)

    y: numpy ndarray or None
        y array (initialized by subclass)

    family: str
        wavelet family acronym (6 characters)
        default is ''

    scale: float
        wavelet scale, default is 1

    shift: float
        shift parameter (it is subtracted to the x axis)

    var: float
        variance of the wavelet

    tres: float
        time resolution of the wavelet

    tdur: float
        time duration of the wavelet

    fc: float
        wavelet characteristic frequency, computed as the frequency
        where the maximum power occurs

    energy: float
        sum of the squared wavelet amplitude

    METHODS
    -------
    fourier()
        return the fourier transform of the wavelet in the format 
        (fft_values, freq)

    power(only_pos=False)
        return fourier amplitudes squared of the wavelet if only_pos 
        is True, only positive frequencies are returned
    
    plot(ax=None, title=None, lfont=16, color='k', label='', 
         xlabel='Time', to_plot='real')
        plot the wavelet. If ax is None, it initializes a figure and an 
        axis, plotting on those. Otherwise, it plots on an existing 
        ax. 


    '''

    def __init__(self, x=None, y=None, family='', scale=1, shift=0, coi_comp='peak',
        *args, **kwargs):
        
        assert isinstance(family,str),'family must be a string'
        
        if x is None:
            # The time interval goes from -5*scale to +5*scale
            # the wavelet time resolution depends on the scale
            # but the number of bins is fixed
            dur = 10*scale
            n = int(2**12)
            self.x = np.linspace(0,dur,n)-dur/2.
        elif isinstance(x,int) or (isinstance(x,float) and x.is_integer()):
            # The time interval goes from -(n-1)/2 to +n/2
            # the wavelet time resolution is fixed to 1
            n = x
            self.x = np.arange(0,n+1)-(n-1.0)/2
        elif isinstance(x,list) or isinstance(x,np.ndarray):
            # The time array is specified by the user
            self.x = np.asarray(x)-(x[-1]-x[0])/2

        self.y = y
        if not self.y is None:
            assert len(self.x) == len(self.y),'x and y must have sam dimension'

        self.family = family
        self.scale = scale
        assert self.scale != 0.,'scale cannot be zero'

        self.shift = shift
        #if not self.x is None: self.x -= self.shift

        self.coi_comp = coi_comp


    def fourier(self):
        freq = fftfreq(len(self.y),self.tres)
        fft_values = fft(self.y)

        return fft_values, freq

    def power(self, only_pos=False):
        freq = fftfreq(len(self.y),self.tres)
        fft_values = fft(self.y)
        fft_squared = (fft_values*np.conj(fft_values)).real

        if only_pos:
            mask = freq>0
        else:
            mask = freq!=np.nan

        return fft_squared[mask], freq[mask]

    def plot(self, ax=None, title=None, lfont=16, xlabel='Time',
            to_plot='real',yshift=0,**kwargs):

        if not 'color' in kwargs.keys(): kwargs['color'] = 'k'

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        if (not title is None) and (not ax is None):
            ax.set_title(title)

        if np.iscomplexobj(self.y):
            if to_plot == 'imag':
                y = self.y.imag
                y_label = 'Imaginary part'
            elif to_plot == 'real':
                y = self.y.real
                y_label = 'Real part'
        else:
            y = self.y
            y_label = 'Amplitude'
        
        ax.plot(self.x, y+yshift,'-', **kwargs)
        ax.set_xlabel(xlabel, fontsize=lfont)
        ax.set_ylabel(y_label, fontsize=lfont)

        ax.grid()

    @property
    def var(self):
        return self.y.var()

    @property
    def tres(self):
        return self.x[1]-self.x[0]

    @property
    def tdur(self):
        return self.x[-1]-self.x[0]

    @property
    def fp(self):
        '''
        Central frequency computed as maximum of power
        
        2021 11 17, Stefano Rapisarda (Uppsala)
            It seems that this estimate is always half frequency  
            bin lower then the characteristic f and the analytical 
            value. I therefore added half frequency bin to the 
            returned value'''

        power,freq = self.power(only_pos=True)
        #factor = (freq[2]-freq[1])*0.517
        factor = 0
        return freq[np.argmax(power)]+factor

    @property
    def fc(self):
        '''
        Passband centre defined as the second moment of area
        of the energy spectrum
        '''

        power,freq = self.power(only_pos=True)
        num = np.sum(freq**2*power)
        den = np.sum(power)
        return np.sqrt(num/den)

    @property
    def coi(self):
        if self.coi_comp != 'anal':
            prec = 10
            signal = np.zeros(len(self.x))
            start_i = 0
            signal[start_i] = 1
            wavelet_data = np.conj(self.y[::-1])
            output = convolve(signal, wavelet_data, mode='same')
            power = abs(output)**2
            target_power = power[start_i]/(math.e)**2

            i = np.argmin(abs(power[start_i:]-target_power))
            if power[i]<target_power and i !=0:
                tp = (i-1)*self.tres
                tf = i*self.tres
                pp = power[i-1]
                pf = power[i]
            elif power[i]>target_power:
                tp = i*self.tres
                tf = (i+1)*self.tres
                pp = power[i]
                pf = power[i+1]
            else:
                tp = i*self.tres
                tf,tp = 1,1
                pf,pp = 2,1
            
            result = tp+(target_power-pp)*(tf-tp)/(pf-pp)

        else:
            if self.family.upper() in ['MORLET','MEXHAT']:
                result = np.sqrt(2)*self.scale
        return result

    @property
    def energy(self):
        if np.iscomplexobj(self.y):
            return np.sum((self.y*np.conj(self.y)).real)
        else:
            return np.sum(self.y**2)

    def __add__(self, other):
        if isinstance(other,np.ndarray):
            assert len(other) == len(self.y), 'Array must have wavelet dimension'
        elif isinstance(other,list):
            if len(other) == 1:
                other = other[0]
            else:
                assert len(other) == len(self.y), 'Array must have wavelet dimension'
                other = np.array(other)
        elif isinstance(other,str):
            other = eval(other)
        
        return Wavelet(self.x, self.y+other, self.family)

    def __sub__(self, other):
        if isinstance(other,np.ndarray):
            assert len(other) == len(self.y), 'Array must have wavelet dimension'
        elif isinstance(other,list):
            if len(other) == 1:
                other = other[0]
            else:
                assert len(other) == len(self.y), 'Array must have wavelet dimension'
                other = np.array(other)
        elif isinstance(other,str):
            other = eval(other)
        
        return Wavelet(self.x, self.y-other, self.family)

    def __mul__(self, other):
        if isinstance(other,np.ndarray):
            assert len(other) == len(self.y), 'Array must have wavelet dimension'
        elif isinstance(other,list):
            if len(other) == 1:
                other = other[0]
            else:
                assert len(other) == len(self.y), 'Array must have wavelet dimension'
                other = np.array(other)
        elif isinstance(other,str):
            other = eval(other)
        
        return Wavelet(self.x, self.y*other, self.family)   

    def __rmul__(self, other):
        if isinstance(other,np.ndarray):
            assert len(other) == len(self.y), 'Array must have wavelet dimension'
        elif isinstance(other,list):
            if len(other) == 1:
                other = other[0]
            else:
                assert len(other) == len(self.y), 'Array must have wavelet dimension'
                other = np.array(other)
        elif isinstance(other,str):
            other = eval(other)
        
        return Wavelet(self.x, self.y*other, self.family)    

    def __truediv__(self, other):
        if isinstance(other,np.ndarray):
            assert len(other) == len(self.y), 'Array must have wavelet dimension'
        elif isinstance(other,list):
            if len(other) == 1:
                other = other[0]
            else:
                assert len(other) == len(self.y), 'Array must have wavelet dimension'
                other = np.array(other)
        elif isinstance(other,str):
            other = eval(other)

        assert other != 0, 'Dude, you cannot divide by zero'
        
        return Wavelet(self.x, self.y/other, self.family)   

    def __pow__(self, other):
        return Wavelet(self.x, self.y**other, self.family)

    def __len__(self):
        return len(self.y)


class Wavelet(BaseWavelet):
    '''
    Wavelet object, istance of a base wavelet object. This initializes
    the wavelet amplitude according to the specified family
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        if self.family.upper() == 'MORLET':

            try:
                self.f0 = kwargs['f0']
            except Exception:
                #print('Morlet central frequency not specified, default = 0.8')
                self.f0 = 0.7957747154594768 # so that omega0 = 5, like in scipy

            norm = np.sqrt(self.tres/self.scale) # to have energy equal to 1
            t_scaled_shifted = (self.x-self.shift)/self.scale
            
            norm2 = np.pi**(-0.25) # usually norm2 belongs to specific wavelet
            omega0 = 2*np.pi*self.f0
            exp1 = np.exp(1j*omega0*t_scaled_shifted)
            exp2 = np.exp(-0.5*omega0**2)
            exp3 = np.exp(-0.5*t_scaled_shifted**2)

            self.y = np.asarray(norm*norm2*(exp1-exp2)*exp3)

        elif self.family.upper() == 'MEXHAT':

            norm = np.sqrt(self.tres/self.scale) # to have energy equal to 1
            t_scaled_shifted = (self.x-self.shift)/self.scale            

            norm2 = np.pi**(-0.25)*2./np.sqrt(3)
            exp1  = np.exp(-0.5*t_scaled_shifted**2)

            self.y = np.asarray(norm*norm2*(1-t_scaled_shifted**2)*exp1)

        elif self.family.upper() == 'CSHANN':

            norm = np.sqrt(self.tres/self.scale) # to have energy equal to 1
            t_scaled_shifted = (self.x-self.shift)/self.scale 

            exp1 = np.exp(-2*np.pi*1j*t_scaled_shifted)
            sinc = np.sin(np.pi*t_scaled_shifted)/np.pi/t_scaled_shifted
        
            self.y = norm*sinc*exp1
        else:

            print('Specified wavelet is not available')


class WaveletTransform:

    def __init__(self,scales=np.array([]),freqs=np.array([]),
                time=np.array([]),counts=np.array([]),
                wt=np.array([]),coi=np.array([]),
                family='mexhat',
                notes=None,meta_data=None):

        self.scales = scales
        self.freqs = freqs
        self.time = time
        self.counts = counts
        self.coi = coi
        self.wt = wt

        self._family = family

        # Meta_data
        if notes is None:
            self.notes = {}
        else: self.notes = notes

        if meta_data is None:
            self.meta_data = {}
        else: self.meta_data = meta_data
        self.meta_data['WT_CRE_DATE'] = my_cdate()

    @property
    def texp(self):
        if len(self.time) == 0: return None
        return self.time[-1]-self.time[0]

    @property
    def df(self):
        if len(self.time) == 0: return None
        return 1./self.texp

    @property
    def nf(self):
        if len(self.time) == 0: return None
        return 1/2/self.tres

    @property
    def tres(self):
        if len(self.time) > 1:
            #return self.time.iloc[2]-self.time.iloc[1]
            tres = np.median(np.ediff1d(self.time))
            return tres
        elif len(self.time) == 0:
            return None
        else:
            return 0

    @property
    def power(self):
        if len(self.wt) == 0: return None
        return (abs(self.wt))**2

    def global_power(self,norm=None):
        if len(self.wt) == 0: return None
        return self.norm_power(norm).mean(axis=1)

    @property
    def family(self):
        return self._family

    def norm_power(self,norm='leahy'):
        if len(self.time) == 0 or len(self.wt) == 0: 
            return None
        
        if norm is None:
            return self.power

        if norm.upper() == 'LEAHY':
            #result = result/np.sum(self.counts)
            return self.power/np.var(self.counts)*2
        elif norm.upper() == 'RMS':
            #result = result*len(self.time)*self.tres/np.sum(self.counts)**2
            return self.power/np.var(self.counts)*2*len(self.time)*self.tres/np.sum(self.counts)
        else:
            raise ValueError('Selected normalization option does not exist')
            return None

    @staticmethod
    def from_lc(lightcurve,scales=None,dt=1,s_min=None,s_max=None,dj=None,family='mexhat',
        method='fft',pad=True,cfreq='cf',**kwargs):

        meta_data = {}

        if type(lightcurve) == type(Lightcurve()):
            print('Input is a lightcurve object')
            dt = lightcurve.tres
            tdur = lightcurve.texp
            counts = lightcurve.counts.to_numpy()
            time = lightcurve.time.to_numpy()

            meta_data['WT_CRE_MODE'] = 'Wavelet transform computed from Lightcurve object'
            for key,item in lightcurve.meta_data.items():
                meta_data[key] = item
        else:
            if type(lightcurve) == list:
                lightcurve = np.array(lightcurve)
            tdur = dt*len(lightcurve)
            time = np.arange(0,tdur,dt)
            if len(time) != len(lightcurve):
                raise ValueError('time and lightcurve arrays do not have the same dimension')
            counts = lightcurve

            meta_data['WT_CRE_MODE'] = 'Wavelet transform computed from array'

        if scales is None:
            # Computing scales
            # The default minimum scale is 2 times the time resolution
            if s_min is None: s_min = 2*dt
            # The default maximum scales is 1/4 of the duration of the input
            # lightcurve
            if s_max is None: s_max = tdur/4.
            if dj is None: dj = 0.05 

            scales = comp_scales(s_min, s_max, dj=dj)

        coef, freqs, coi = cwt(counts,dt=dt,scales=scales,
            family=family,method=method,pad=pad,cfreq=cfreq,**kwargs)

        result = WaveletTransform(scales=scales,freqs=freqs,
            time=time,counts=counts,
            wt=coef,coi=coi,
            meta_data = meta_data)

        return result

    def plot(self,ax=None,time=False,freqs=True,logs=True,
        norm='maxpxs',sigma_norm=True,power=True,
        xrange=[],yrange=[],conf_levels=[],power_level=None,
        cmap=plt.cm.jet,cf_colors=['white'],
        mini_signal=None,coi=True,
        ylabel='counts',title=False,lfont=16,**kwargs):
        '''
        Method for plotting the wavelet transform or power

        PARAMETERS
        ----------
        ax: matplotlib.Axes.axes (optional) 
            This should be the external matplotlib axes 
            provided by the user (default is False)
        time: boolean (optional)
            if True, time series is plot on top of the wavelet products
            (default is False)
        freqs: boolean (optional)
            if True, Frequency are plotted instead of scales
            (default is True)
        logs: boolean(optional)
            if True, scales/frequencies are plotted in logarithmic
            scale
        norm: float, str, or None (optional)
            - if float, the plotted wavelet products are divided by the 
            float
            - if equal to 'maxpxs' (default), wavelet products are divided
            by the maximum of the products at each scale
            - if None, no normalization is applied
        sigma_norm: boolean (optional)
            if True, wavelet power is divided by the variance of 
            the original time series (default is True)
        power: boolean (optional)
            if True, the wavelet power is plotted (default),
            if False, the wavelet transform is plotted (real part)
        xrange: list (optional)
            x axis range (starting from zero), if empty (default), the 
            entire array is plotted
        conf_levels: list (optional)
            list of confidence levels as a fraction of 1 (default is [])
        power_level: numpy.array (optional)
            If not None (None is default), the confidence level is 
            computed according to this level. The specified power_level
            must have the same scale/frequency dimension
        cmap: matplotlib color map
            default is plt.cm.jet
        mini_signal: float or None (optional)
            If float, it will be printed a mini signal (one full 
            oscillation) with the specified frequency (float).
            Default is None
        coi: boolean (optional)
            If True (default) the cone of influence is plotted
        ylabel: str (optional)
            Label for the time series plot (default is "counts")
        title: str or None (optional)
            Title of the plot
        lfont: int (optional)
            xy label font (default is 16)
        '''
        if not 'color' in kwargs.keys(): kwargs['color'] = 'k'

        flag_bar = False
        if ax is None:
            if time == False:
                fig, ax = plt.subplots(figsize=(12,6))
            else:
                fig, (axt,ax) = plt.subplots(2,figsize=(12,10))
                plt.subplots_adjust(hspace=0)
            cbar_ax = fig.add_axes([0.95,0.15,0.03,.7])
            flag_bar = True

        if title:
            try:
                axt.set_title(title)
            except:
                ax.set_title(title)  
                

        start_time = int(np.min(self.time))
        time_array = self.time-start_time

        if not time == False:
            axt.plot(time_array,self.counts,'k')
            axt.set_ylabel(ylabel,fontsize=lfont)
            axt.set_xlim([np.min(time_array),np.max(time_array)])
            if len(xrange)!=0:
                axt.set_xlim(xrange)
            axt.set_xticklabels([])
            axt.grid()

        y = np.flip(self.scales)
        z = self.wt
        if power: 
            z = self.power
        if freqs: 
            y = self.freqs
        
        if sigma_norm: 
            z = z/np.var(self.counts)*2
        if not norm is None:
            if type(norm) in [int,float]:
                z = z/norm
            elif type(norm) == str and norm=='maxpxs':
                maxes = np.transpose(np.tile(np.max(z,axis=1),(len(time_array),1)))
                z = z/maxes

        if len(conf_levels)!=0:
            #dof=1
            #if not np.iscomplexobj(self.wt): dof=1
            power_levels = chi2.ppf(conf_levels,df=2)

            if not power_level is None:
                assert len(power_level) == len(self.scales), 'Background power must have same shape of scales/freqs'
                ps_extended = np.transpose(np.tile(np.flip(power_level),(len(self.time),1)))
                power_to_check = self.norm_power('leahy')/ps_extended
            else:
                power_to_check = self.norm_power('leahy')

            ax.contour(
                time_array,y,power_to_check,
                levels=power_levels,colors=cf_colors,
                linestyles=['-'],linewidths=[1],zorder=2)

        im = ax.contourf(time_array,y,z,extend='both',cmap=cmap,zorder=1)

        if coi:
            coi_mask = self.coi<=time_array[len(time_array)//2]
            ax.plot(time_array[0]+self.coi[coi_mask],y[coi_mask],'r')
            ax.plot(time_array[-1]-self.coi[coi_mask],y[coi_mask],'r')

        ax.set_xlim([np.min(time_array),np.max(time_array)])
        if len(xrange)!=0:
            ax.set_xlim(xrange)
        if len(yrange)!=0:
            ax.set_ylim(yrange)
        if logs: ax.set_yscale('log')
        ax.set_ylabel('Scales [s]',fontsize=lfont)
        if freqs: ax.set_ylabel('Frequency [Hz]',fontsize=lfont)
        time_label = 'Time [s]'
        if float(start_time) != 0: time_label = 'Time [s] since {}'.format(start_time)
        ax.set_xlabel(time_label,fontsize=lfont)
        ax.grid()
        
        if flag_bar:
            # !!! fig.colorbar must be added after the axes
            fig.colorbar(im, cax = cbar_ax, orientation='vertical')

        if not mini_signal is None:
            ax.axhline(mini_signal,color='white',ls='--')

            # the x coords of this transformation are data, and the y coord are axes
            trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            signal_period = 1/mini_signal
            step = (ax.get_xlim()[1]-ax.get_xlim()[0])/30
            start = ax.get_xlim()[0]+step
            amp = ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])/100
            mini_x = np.arange(start,start+signal_period,self.tres)
            signal = np.sin(2*np.pi*mini_signal*mini_x)/20+0.15
            ax.plot([mini_x[0],mini_x[-1]],[0.05,0.05],color='white',lw=2,ls='--',transform=trans,zorder=3)
            ax.text(mini_x[-1]+step,0.05,s='{} Hz'.format(mini_signal),color='white',transform=trans,zorder=3)
            ax.plot(mini_x,signal,color='white',lw=2,transform=trans,zorder=3)

        return ax, im


    def save(self,file_name='wavelet_transform.pkl',fold=pathlib.Path.cwd()):

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
                pickle.dump(self,output, protocol=4)
            print('WaveletTransform saved in {}'.format(file_name))
        except Exception as e:
            print(e)
            print('Could not save WaveletTransform')

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
            wt = pickle.load(infile)
        
        return wt   

from .functions import comp_scales,cwt,scale2freq