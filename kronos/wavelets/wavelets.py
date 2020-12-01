import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq

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

    def __init__(self, x=None, y=None, family='', scale=1, shift=0):
        
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
        if not self.x is None: self.x -= self.shift


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

    def plot(self, ax=None, title=None, lfont=16, color='k', label='', xlabel='Time',
            to_plot='real'):

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        if not title is None and not ax is None:
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
        
        ax.plot(self.x, y,'-', color=color, label=label)
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
    def fc(self):
        '''
        Central frequency computed as maximum of power
        '''

        power,freq = self.power(only_pos=True)
        return freq[np.argmax(power)]

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

        else:

            print('Specified wavelet is not available')




