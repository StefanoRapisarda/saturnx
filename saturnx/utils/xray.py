import os
import sys
import numpy as np
import pandas as pd
import pathlib
import logging
from datetime import datetime
from astropy.io import fits

from .my_logging import LoggingWrapper

def convert_time(time_info,time=False,time_sys_out='UTC'):
    '''
    Convert NICER times as recorded by the spacecraft into absolute
    time in MJD

    PARAMETERS
    ----------
    time_info: dictionary or pandas Series
        dictionary with time information
        ('MJDREFI','MJDREFF','TIMEZERO','LEAPINIT','TIMESYS')
    time: value, list, or array (optional)
        If True, time is converted directly
        If False, conversion factors are returned, you then
        need to add such factors to your values
    time_sys_out: string (optional)
        Output time scale. It can be euther UTC or TT. 
    
    RETURN
    ------
    mjd_out: float
        Absolute time or correction factor in the time format 
        specified by the user. In case the data are barycentric 
        corrected, the output time format will be automatically
        TDB.
        
    HISTORY
    -------
    2020      , Stefano Rapisarda (Uppsala), creation date
    2020 10 15, Stefano Rapisarda (Uppsala)
        Updated after reading 
        https://hera.gsfc.nasa.gov/docs/nicer/analysis_threads/time/
        In the formulas it was +leapinit, I changed it to -leapinit
        That was from RXTE conversion formula, I guess. I need to be
        carefull as the conversion formula may change according to the 
        mission, even if the time is still stored in MET.
        Furthermore, now it checks the time format of the input file
        (TT or TDB) and used different expressions accordingly.
        Also, now the function return a single value (time value or 
        time stamp to add to time), the output timesys is specified by
        user.
        
    TO DO
    -----
    - make it general for all  X-ray missions
    '''

    if isinstance(time_info,pd.Series):
        time_info = time_info.to_dict()

    
    assert 'MJDREFI'  in time_info,'MJDREFI is missing'
    assert 'MJDREFF'  in time_info,'MJDREFF is missing'
    assert 'TIMEZERO' in time_info,'TIMEZERO is missing'
    assert 'LEAPINIT' in time_info,'LEAPINIT is missing'
    assert 'TIMESYS'  in time_info,'TIMESYS is missing'
    assert 'TELESCOP' in time_info,'TELESCOP is missing'
    assert (time_sys_out == 'TT' or time_sys_out == 'UTC'),\
            'Output time format must be either TT or UTC'
    mjdrefi  = time_info['MJDREFI']
    mjdreff  = time_info['MJDREFF']
    timezero = time_info['TIMEZERO']
    leapinit = time_info['LEAPINIT']
    timesys  = time_info['TIMESYS']
    mission  = time_info['TELESCOP']
    
    if mission == 'NICER':

        if not time is False:
            if isinstance(time,list):
                time = np.array(time)
            if timesys == 'TT':
                print('Data Time is in TT')
                if time_sys_out == 'TT':
                    mjd_out = (mjdrefi+mjdreff)+(timezero+time)/86400.
                elif time_sys_out == 'UTC':
                    mjd_out = mjdrefi + (timezero+time-leapinit)/86400.
            elif timesys == 'TDB':
                print('Data Time is TDB')
                # In this case mjd_out is in TDB
                print('Data Time is TDB')
                mjd_out = (mjdrefi+mjdreff)+(timezero+time)/86400.
            else:
                print('Unknown input time system')
                sys.exit()
        else:
            if timesys == 'TT':
                print('Data Time is in TT')
                if time_sys_out == 'TT':
                    mjd_out = (mjdrefi+mjdreff)+(timezero)/86400.
                elif time_sys_out == 'UTC':
                    mjd_out = mjdrefi + (timezero-leapinit)/86400.
            elif timesys == 'TDB':
                mjd_out = (mjdrefi+mjdreff)+(timezero)/86400.
            else:
                print('Unknown input time system')
                sys.exit()            
            print('WARNING!!!')
            print('In order to obtain the absolute time you need to')
            print('convert your time in MJD dividing by 86400 and add')
            print('the values returned by this function\n')
            
    else:
        
        print('Mission not recognized.')
        sys.exit()

    return mjd_out

def get_cr(spec_file, low_en=0.5, high_en=10., mission='NICER',instrument='HE'):
    '''
    Return count rate and corresponding error in selected energy range
    read an energy spectrum in FITS file.

    PARAMETERS
    ----------
    spec_file: str
        Full path of a .pha file (the one that you would feed to XSPEC)
    low_en: float
        Lowest energy channel in keV
    high_en: float
        Highest energy channel in keV
    mission: str
        Mission identifier

    HISTORY
    -------
    2020 10 16, Stefano Rapisarda (Uppsala), creation date
    2020 11 30, Stefano Rapisarda (Uppsala)
        changed channels to energy and introduced the mission parameter
    2021 05 01, Stefano Rapisarda (Uppsala)
        name changed into get cr and multi-mission option introduced
    2021 12 10, Stefano Rapisarda (Uppsala)
        Adopted the LogginWrapper approach and corrected but about rate
        (so far it was only looking for RATE keyword in the spectrum,
        now if it does not find it it looks for counts and exposure in the
        header)

    TODO:
    - 2020 10 16, Stefano Rapisarda (Uppsala)
        this should be moved to xray functions, for now is here as it
        works only with python
    '''

    mylogging = LoggingWrapper()

    if isinstance(low_en,str): low_en = eval(low_en)
    if isinstance(high_en,str): high_en = eval(high_en)

    factor = 1
    if mission == 'NICER':
        factor = 100
        corr = 0
    elif mission == 'HXMT':
        if instrument == 'HE':
            factor = 256/370.
            corr = 15
        elif instrument == 'ME':
            factor = 1024/60.
            corr = 3
        elif instrument == 'LE':
            factor = 1536/13.
            corr = 0.1       

    low_cha = (low_en-corr)*factor
    high_cha = (high_en-corr)*factor

    with fits.open(str(spec_file)) as hdul:
        data = hdul[1].data
        col_names = data.names

        if not 'CHANNEL' in col_names:
            mylogging.error(f'CHANNEL key not found in {spec_file.name}')
            return None,None
        else:
            cha = data['CHANNEL']

        flag_rate = False
        if 'RATE' in col_names:
            flag_rate = True
            rate = data['RATE']
        else:
            mylogging.warning(f'RATE key not in {spec_file.name}. I will read counts.')
            counts = data['COUNTS']
            exp = hdul[1].header['ONTIME']

        if 'STAT_ERR' in col_names:
            err = data['STAT_ERR']
        else:
            mylogging.warning(f'STAT_ERR key not in {spec_file.name}. I will set errors to zero')
            err = np.zeros(len(cha))
    
    mask = (cha >= low_cha) & (cha <= high_cha)
    if flag_rate:
        cr = rate[mask].sum()
        cr_err = np.sqrt(((err[mask])**2).sum())
    else:
        cr = counts[mask].sum()/float(exp)
        cr_err = np.sqrt(((err[mask])**2).sum())/float(exp)

    return cr,cr_err

def run_xselect(cl_event_file_fp,binsize=16,
    outfile='spectrum.pha',overwrite=False):
    '''
    Runs xselect with specified options

    PARAMETERS
    ----------
    cl_event_file: str or pathlib.Path()
        Full path of the cleaned event file
    binsize: int (optional)
        Binsize (default is 16)
    outfile: str (optional)
        Output name for the energy spectrum or the lightcurve
        According to the extension of this file (.pha or .lc),
        the script will automatically select if extracting an 
        energy spectrum (.pha) or a lightcurve (.lc)
        (default is spectrum.pha)
    overwrite: boolean (optional)
        If True and an outfile is found, the existing outfile is first
        deleted (default is False)

    RETURNS
    -------
        True if everything went smoothly, False otherwise

    HISTORY
    -------
    2021 10 15, Stefano Rapisarda (Uppsala), creation date
    2021 11 03, Stefano Rapisarda (Uppsala)
        Introduced the wrapper LoggingWrapper and removed the
        (now useless) function argument mylogging_on
    2022 02 01, Stefano Rapisarda (Uppsala)
        Bug on saving lightcurve corrected (xselect wanted to read 
        "curve" not lightcurve when saving...). 
        Overwrite option added
    '''

    mylogging = LoggingWrapper()

    if type(outfile) == str:
        outfile = pathlib.Path(outfile)

    if outfile.is_file():
        mylogging.info('run_xselect: Output file already exists')
        if overwrite:
            mylogging.info('run_xselect: Overwrite option is True, removing it...')
            os.system('rm -f {}'.format(outfile))
        else:
            mylogging.info('run_xselect: Overwrite option is False, skipping operations')
            return True

    # Recording date and time
    now = datetime.now()
    date = ('%d_%d_%d') % (now.day,now.month,now.year)
    time = ('%d_%d') % (now.hour,now.minute)

    # Checking if the event file exists
    if type(cl_event_file_fp) == str:
        cl_event_file_fp = pathlib.Path(cl_event_file_fp)
    if not cl_event_file_fp.exists():
        message = 'run_xselect: cleaned event file does not exists'
        mylogging.error(message)
        mylogging.error(cl_event_file_fp)
        return False
    else:
        cl_event_file = cl_event_file_fp.name
        cl_event_file_dir = cl_event_file_fp.parent

    # Getting extention from file name
    ext = pathlib.Path(outfile).suffix
    if ext == '.gz': ext = pathlib.Path(outfile.stem).suffix

    # Automatically select 
    # option
    # -----------------------------------------------------------------
    if ext == '.pha':
        opt = 'spectrum'
    elif ext == '.lc':
        opt = 'curve'
    else:
        mylogging.error('run_xselect: wrong option')
        return False
    # -----------------------------------------------------------------

    # Moving to working directory
    home = pathlib.Path.cwd()
    os.chdir(cl_event_file_dir)

    # Writing infile for xselect
    # -----------------------------------------------------------------
    infile_name = f'xselect_{ext[1:]}_D{date}_T{time}.in'
    log_name = f'xselect_{ext[1:]}_D{date}_T{time}.log'

    lines = [f'D{date}_T{time}',
             f'read events {cl_event_file}',
             '.','Y',f'set binsize {binsize}',
             f'extract {opt}',
             f'save {opt} {outfile}',
             'exit','N']

    with open(infile_name,'w') as output:
        for line in lines:
            output.write(line+'\n')
    # -----------------------------------------------------------------

    # Running xselect
    # -----------------------------------------------------------------
    try:
        os.system(f'xselect < {infile_name} > {log_name}')
    except:
        message = 'run_xselect: Something went wrong running xselect'
        mylogging.error(message)
        return False

    # Checking if file was created
    if not outfile.is_file():
        message = 'run_xselect: output file not created'
        mylogging.error(message)
        return False        
    # -----------------------------------------------------------------

    # Going back home
    os.chdir(home)

    return True


