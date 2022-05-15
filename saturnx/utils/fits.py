import os
import logging
import pathlib
from .my_logging import LoggingWrapper
from astropy.io.fits import getval
from astropy.io import fits

def read_fits_keys(input_object,keys,ext=0):
    '''
    Return a dictionary of key,value for the selected extension of the 
    FITS file

    PARAMETERS
    ----------
    input_object: str, pathlib.Path, or astropy.io.fits.hdu_list

    HISTORY
    -------
    2020 xx xx, Stefano Rapisarda (Uppsala),creation date
    2022 03 15, Stefano Rapisarda (Uppsala)
        Updated with pathlib and Logging Wrapper.
        Now it also works with an opened hdu_list, and instead of calling
        get_val for each keyword, it opens the file once and it reads all
        the keys.
    '''

    mylogging = LoggingWrapper()

    if isinstance(keys,str):
        keys = [keys]

    info = {}

    close_flag = False
    if isinstance(input_object,fits.hdu.hdulist.HDUList):
        hdu_list = input_object
    else:
        if isinstance(input_object,str): 
            input_object = pathlib.Path(input_object)
        if not input_object.is_file():
            mylogging.error(f'FITS file {input_object} does not exist')
            return {}

        try:
            hdu_list = fits.open(input_object)  
            close_flag = True
        except:
            mylogging.error(f'Could not open FITS file ({input_object})')
            return {}

    for key in keys:
        try:
            value = hdu_list[ext].header[key]
        except Exception as e:
            mylogging.warning(f'Could not read key {key} in ext {ext}')
            value = None
        info[key] = value

    if close_flag:
        hdu_list.close()
        del hdu_list

    return info


def get_basic_info(input_object,ext=0):
    '''
    Return a dictionary with basic informations collected 
    from a fits_file (supposed to be event file)

    HISTORY
    -------
    2022 03 15, Stefano Rapisarda, Uppsala
        Greatly simplified. Now this is just a wrapper for read_fits_keys
        with standard keys.
    '''

    basic_keys = ['OBJECT','TELESCOP','INSTRUME','OBS_ID','RA_OBJ','DEC_OBJ',
                'CREATOR','DATE','SOFTVER','CALDBVER','GCALFILE']

    time_keys = ['DATE-OBS','DATE-END','TSTART','TSTOP',
                'MJDREF','MJDREFI','MJDREFF','TIMEZERO','LEAPINIT','CLOCKAPP',
                'TIMEZERO','ONTIME','EXPOSURE','NAXIS2','TIMESYS']

    info = read_fits_keys(input_object,basic_keys+time_keys,ext=ext)

    return info