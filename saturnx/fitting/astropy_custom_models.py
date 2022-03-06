import numpy as np

from astropy.modeling import Fittable1DModel, Parameter

__all__ = ['fmax_lorentzian','f0_lorentzian']

class fmax_lorentzian(Fittable1DModel):
    '''
    Lorentzian model according to the definition of 
    Belloni, Psaltis, and van der Klis 2002 and 
    van Straaten et al. 2001
    '''

    n_inputs = 1
    n_outputs = 1

    amplitude = Parameter()
    q = Parameter()
    fmax = Parameter()

    @staticmethod
    def evaluate(x, amplitude, q, fmax):
        
        # Delta is the half width at half maximum
        delta = fmax/np.sqrt(1+4*q**2)

        # norm is the integral of the Lorentzian between 0 and +inf
        norm = 1./( np.pi/2 + np.arctan(2*q) )

        den = delta**2 + (x-2*delta*q)**2

        lor = amplitude*norm*delta/den
        return lor

    @staticmethod
    def fit_deriv(x, amplitude, q, fmax):

        # Delta is the half width at half maximum
        delta = fmax/np.sqrt(1+4*q**2)

        # norm is the integral of the Lorentzian between 0 and +inf
        norm = 1./( np.pi/2 + np.arctan(2*q) )

        den = delta**2 + (x-2*delta*q)**2

        d_amplitude = norm*delta/den
        d_q = amplitude*delta*norm/den*( 2/(1+4*q**2)*norm -
                    4*delta*(x-2*delta*q)/den )
        d_delta_d_fmax = 1./np.sqrt(1+4*q**2)
        d_fmax = d_delta_d_fmax*amplitude*norm/den*(1+ 
            (2*delta**2 - 4*delta*q*(x-2*delta*q))/den)

        return [d_amplitude,d_q,d_fmax]

class f0_lorentzian(Fittable1DModel):
    '''
    Lorentzian model according to the definition of 
    Belloni, Psaltis, and van der Klis 2002 and 
    van Straaten et al. 2001
    '''

    n_inputs = 1
    n_outputs = 1

    amplitude = Parameter()
    gamma = Parameter()
    f0 = Parameter()

    @staticmethod
    def evaluate(x, amplitude, gamma, f0):
        
        # Delta is the half width at half maximum
        delta = gamma/2.

        # norm is the integral of the Lorentzian between 0 and +inf
        norm = 1./( np.pi/2 + np.arctan(f0/delta) )

        den = delta**2 + (x-f0)**2

        lor = amplitude*norm*delta/den
        return lor

    @staticmethod
    def fit_deriv(x, amplitude, gamma, f0):

        # Delta is the half width at half maximum
        delta = gamma/2.

        # norm is the integral of the Lorentzian between 0 and +inf
        norm = 1./( np.pi/2 + np.arctan(f0/delta) )

        den = delta**2 + (x-f0)**2

        d_amplitude = norm*delta/den
        d_gamma = 0.5 * amplitude*norm/den*(1-
            norm*f0/delta/(1+(f0/delta)**2) + 2*delta*norm)
        d_f0 = amplitude*delta*norm/den( 1/(delta*(1+(f0/delta)**2))*norm -
            2*(x-f0)/(delta**2 + (x-f0)**2 ) )

        return [d_amplitude,d_gamma,d_f0]