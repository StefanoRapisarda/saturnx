import numpy as np

from sherpa.models import model

__all__ = ('Fmax_lorentzian1D','F0_lorentzian1D')

def _lor1d(pars,x,opt):
    '''Evaluate Lorentzian either using f0 (standard definition)
    or fmax
    
    PARAMETERS
    ----------
    pars: sequence of 3 numbers
        The order is amplitude, width (corresponding to q or FWHM
        in case of fmax and f0, respectively), and frequency (corre-
        sponding to fmax or f0 in case of fmax and f0, respectively).
    x: sequence of numbers
        Thegrid on which to evaluate the model
        
    RETURNS
    -------
    y: sequence of numbers
        The model evaluated on the input grid
        
    NOTES
    -----
    Lorentzian expressions according to the definition of 
    Belloni, Psaltis, and van der Klis 2002 and 
    van Straaten et al. 2001.
    The general formula is

    lor = amp * delta / (lor_int) / (delta**2 + (f-f0)**2)

    where delta is the HWHM, lor_int is the integral between 0 and inf 
    of the Lorentzian, equal to pi/2 + arctan(f0/delta). Dividing the 
    formula by this term the integral of the Lorentzian is 1. 
    To get the fmax rapresentation:
    - Q = f0/2/delta;
    - delta = f_max / sqrt(1+4*Q**2) or f_max**2 = delta**2 * (4*Q**2+1);
    - f_max**2 = f0**2 + delta**2

    HISTORY
    -------
    2020 01 18, Stefano Rapisarda (Uppsala), creation date    
    '''

    (amplitude,width,freq) = pars

    if opt == 'fmax':
        q = width
        fmax = freq

        # Delta is the half width at half maximum
        delta = fmax/np.sqrt(1+4*q**2)

        # norm is the integral of the Lorentzian between 0 and +inf
        norm = 1./( np.pi/2 + np.arctan(2*q) )

        den = delta**2 + (x-2*delta*q)**2
    
    elif opt == 'f0':
        gamma = width
        f0 = freq

        # Delta is the half width at half maximum
        delta = gamma/2.

        # norm is the integral of the Lorentzian between 0 and +inf
        norm = 1./( np.pi/2 + np.arctan(f0/delta) )

        den = delta**2 + (x-f0)**2

    lor = amplitude*norm*delta/den

    return lor

class Fmax_lorentzian1D(model.RegriddableModel1D):
    '''
    A onde dimensional Lorentzian

    The model parameters are:
    ampl
        The amplitude of the Lorentzian
    q
        The coherence factor (governing width)
    fmax
        The frequency where most power is concentrated, corresponding
        to central frequency for large Q

    HISTORY
    -------
    2020 01 18, Stefano Rapisarda (Uppsala), creation date  
    '''

    def __init__(self, name='fmax_lor1d'):

        par_values=[1,10,1]
        par_names = ['ampl','q','fmax']

        self.ampl = model.Parameter(name,par_names[0],par_values[0],min=0, hard_min=0)
        self.q = model.Parameter(name,par_names[1],par_values[1],min=0, hard_min=0) 
        self.fmax = model.Parameter(name,par_names[2],par_values[2],min=0, hard_min=0)

        model.RegriddableModel1D.__init__(self, name,
            (self.ampl,self.q,self.fmax)) 

    def calc(self,pars,x,*args,**kwargs):
        '''Evaluate the model'''
        
        # If given an integrated data set, use the center of the bin
        if len(args) == 1:
            x = (x+args[0])/2

        return _lor1d(pars,x,'fmax')

class F0_lorentzian1D(model.RegriddableModel1D):
    '''
    A onde dimensional Lorentzian

    The model parameters are:
    ampl
        The amplitude of the Lorentzian
    gamma
        Full width at half maximum
    f0
        Centroid of the Lorentzian

    HISTORY
    -------
    2020 01 18, Stefano Rapisarda (Uppsala), creation date  
    '''

    def __init__(self, name='f0_lor1d'):

        par_values=[1,10,1]
        par_names = ['ampl','gamma','f0']

        self.ampl = model.Parameter(name,par_names[0],par_values[0],min=0, hard_min=0)
        self.gamma = model.Parameter(name,par_names[1],par_values[1],min=0, hard_min=0) 
        self.f0 = model.Parameter(name,par_names[2],par_values[2],min=0, hard_min=0)

        model.RegriddableModel1D.__init__(self, name,
            (self.ampl,self.gamma,self.f0)) 

    def calc(self,pars,x,*args,**kwargs):
        '''Evaluate the model'''
        
        # If given an integrated data set, use the center of the bin
        if len(args) == 1:
            x = (x+args[0])/2

        return _lor1d(pars,x,'f0')
