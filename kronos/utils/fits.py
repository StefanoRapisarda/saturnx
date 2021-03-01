import os
import warnings
from astropy.io.fits import getval

def read_fits_keys(file_name,keys,ext=0):
    '''
    Return a dictionary of key,value for the selected
    extension of the FITS file

    HISTORY
    -------
    2020, Stefano Rapisarda (Uppsala),creation date
    '''

    if isinstance(keys,str):
        keys = [keys]
    
    info = {}
    for key in keys:
        try:
            value = getval(file_name,key,ext)
        except Exception as e:
            print(e)
            warnings.warn(f'Could not read key {key} in ext {ext} of {file_name}')
            value = 'None'
        info[key] = value
    return info


def get_basic_info(fits_file):
    '''
    Return a dictionary with basic informations collected 
    from a fits_file (supposed to be event file)
    '''
    
    assert os.path.isfile(fits_file),'fits_file does not exist'

    basic_keys = ['OBJECT','TELESCOP','INSTRUME','OBS_ID','RA_OBJ','DEC_OBJ',
                  'CREATOR','DATE','SOFTVER','CALDBVER','GCALFILE']

    time_keys = ['DATE-OBS','DATE-END','TSTART','TSTOP',
                 'MJDREF','MJDREFI','MJDREFF','TIMEZERO','LEAPINIT','CLOCKAPP',
                 'TIMEZERO','ONTIME','EXPOSURE','NAXIS2','TIMESYS']
    
    keys = basic_keys + time_keys
    info = read_fits_keys(fits_file,keys)

    return info