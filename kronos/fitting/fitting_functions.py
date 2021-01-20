import numpy as np 
import functools

__all__ = ['lorentzian']

class attach_name:

    def __init__(self,name=''):
        self.name = name

    def __call__(self,f):

        @functools.wraps(f)
        def wrapped_f(*args):
            return f(*args)
            
        wrapped_f.name = self.name
        return wrapped_f

#@attach_name('Lorentzian')
def lorentzian(x,amp,q,freq):
    nu_max=freq
    delta = nu_max/np.sqrt(1+4*q**2)
    nu_0 = np.sqrt(nu_max**2-delta**2) 
    r2 = (0.5 + np.arctan(2*q)/np.pi)
    lor = amp*delta/np.pi/r2/(delta**2+(x-nu_0)**2)
    return lor
