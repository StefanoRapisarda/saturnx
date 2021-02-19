import sys
import numpy as np
import pandas as pd

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
    assert (time_sys_out is 'TT' or time_sys_out is 'UTC'),\
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