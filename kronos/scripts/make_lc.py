import os
import pathlib
import logging

from kronos.utils.logging import make_logger
from kronos.core.event import Event,EventList
from kronos.core.gti import Gti,GtiList
from kronos.core.lightcurve import Lightcurve,LightcurveList

def make_nicer_lc(event_file,tres=1.,en_bands=[[0.5,10.]],
            split_event = False,destination=pathlib.Path.cwd(),
            output_suffix='',override=False,log_name=None):
    '''
    It makes a lightcurve list from a NICER event file

    The result of this is always a Lightcurve list containing 
    lightcurves per GTI. If split_event is True, the event file
    is split according to GTI and then lightcurves are computed 
    on every event segment. If split_event is False, Lightcurve is
    computed on the full event file and then split according to
    GTI. The former should be less time consuming.
    The gti file is also saved as gti.gti    

    PARAMETERS
    ----------
    event_file: string
        full path of a NICER event file
    tres: float (optional)
        time resolution (default=1)
    en_bands: list (optional)
        list of energy bands (default is [[0.5,10]])
    split_event: boolean (optional)
        if True, it splits the event file into GTIs. GTIs are read
        from the event_file itself (default=False)
    destination: pathlib.Path (optional)
        full path of the output folder (default=current folder) 
    output_suffix: string
        string to attache at the end of the output file (default='')
    override: boolean (optional)
        if True, files are rewritten (default is False)
    log_name: string or None (optional)
        name of the log file (defaule=None)

    RETURNS
    -------
    1 if successfull, 0 if not

    NOTES
    -----
    2021 01 03, Stefano Rapisarda (Uppsala)
        The result of this is always a Lightcurve list containing 
        lightcurves per GTI. If split_event is True, the event file
        is split according to GTI and then lightcurves are computed 
        on every event segment. If split_event is False, Lightcurve is
        computed on the full event file and then split according to
        GTI. The former should be less time consuming

    HISTORY
    -------
    2020 01 01, Stefano Rapisarda (Uppsala), creation date
    2021 02 03, Stefano Rapisarda (Uppsala)
        Cleaned up
    2021 03 06, Stefano Rapisarda (Uppsala)
        Updated path to pathlib.Path
    2021 03 15, Stefano Rapisarda (Uppsala)
        Instead of working on a single energy band, not it works on a
        list of them, in this way it opens the Event object only once
    '''

    if type(destination) == str:
        destination = pathlib.Path(destination)

    # Logging
    if log_name is None:
        log_name = make_logger('make_lc',outdir=destination)

    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'make_nicer_lc','*'*24))
    logging.info('*'*72)

    # Making folders
    # -----------------------------------------------------------------
    logging.info('Creating analysis folder...')
    an = destination/'analysis'
    if not an.is_dir():
        os.mkdir(an)

    if type(event_file) == str:
        event_file = pathlib.Path(event_file)
    if not event_file.is_file():
        logging.info('Event file does not exist')
        return 0

    if not 'mpu' in str(event_file):
        logging.info('This is not a NICER event file.')
        return 0
    else:
        obs_id = str(event_file.name).split('_')[0].\
            replace('ni','')

    obs_id_dir = an/obs_id
    if not obs_id_dir.is_dir():
        os.mkdir(obs_id_dir)
        logging.info('Creating obs_ID folder...')
    # -----------------------------------------------------------------

    # Printing some info
    # -----------------------------------------------------------------
    logging.info('')
    logging.info('Obs ID: {}'.format(obs_id))
    logging.info('Settings:')
    logging.info('-'*60)
    logging.info('Selected energy band: {}-{} keV'.format(low_en,high_en))
    logging.info('Selected time resolution: {} s'.format(tres)) 
    logging.info('Split events: {}'.format('yes' if split_event else 'no'))
    logging.info('Log file name: {}'.format(log_name))
    logging.info('-'*60)
    logging.info('')
    # -----------------------------------------------------------------

    # Output names
    # -----------------------------------------------------------------
    lc_list_names = []
    gti_names = []
    lc_list_files = []
    for en_band in en_bands:
        low_en = en_band[0]
        high_en = en_band[1]
        if output_suffix != '':
            lc_list_name = 'lc_list_E{}_{}_T{}_{}.pkl'.\
                format(low_en,high_en,tres,output_suffix)
        else:
            lc_list_name = 'lc_list_E{}_{}_T{}.pkl'.\
                format(low_en,high_en,tres)    
        gti_name = 'gti_E{}_{}.gti'.format(low_en,high_en)

        lc_list_file = obs_id_dir/lc_list_name

        lc_list_names += [lc_list_name]
        gti_names += [gti_name]
        lc_list_files += [lc_list_file]
    # -----------------------------------------------------------------

    # Checking file existance
    no_lc_files = False
    for lc_list_file,lc_list_name in zip(lc_list_files,lc_list_names):
        if not lc_list_file.is_file():
            logging.info('Lightcurve list file {} already exists.'.\
                format(lc_list_name))            
            no_lc_files = True

    # Computing lightcurve
    # -----------------------------------------------------------------
    if no_lc_files or override:
        logging.info('Reading event file')
        try:
            events = Event.read_fits(event_file)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not open event file.')
            logging.info(e)
            return 0

        logging.info('Reading GTI from event file')
        try:
            gti = Gti.read_fits(event_file)
            logging.info('Done!')
            logging.info('N. GTIs: {}'.format(len(gti)))
        except Exception as e:
            logging.info('Could not read GTI.')
            logging.info(e)
            return 0   

        if split_event:
            logging.info('Splitting events according to GTIs')
            try:
                event_list = events.split(gti)
                logging.info('Done!')
            except Exception as e:
                logging.info('Could not split event file.')
                logging.info(e)
                return 0   

        for i,en_band in enumerate(en_bands):
            low_en = en_band[0]
            high_en = en_band[1]  

            logging.info('Computing lightcurve list')
            logging.info('Energy band {}-{} keV'.format(low_en,high_en))
            try:
                if split_event:
                    lcs = Lightcurve.from_event(event_list,time_res=tres,
                        low_en=low_en,high_en=high_en)     
                else:
                    lightcurve = Lightcurve.from_event(events,time_res=tres,low_en=low_en,high_en=high_en)
                    logging.info('Done!')
                    lcs = lightcurve.split(gti)
                logging.info('Done!')
            except Exception as e:
                logging.info('Could not compute lightcurve list')
                logging.info(e)
                return 0 

            if len(lcs)==0:
                logging.warning('Lightcurve list (after gti split) is empty!')
                return 1

            # Saving Lightcurve list and gti
            logging.info('Saving Lightcurve in the event folder')
            lcs.save(lc_list_names[i],fold=obs_id_dir)
            logging.info('Done!')

            logging.info('Saving Gti in the event folder')
            gti.save(gti_names[i],fold=obs_id_dir)
            logging.info('Done!') 
        
    # -----------------------------------------------------------------       

    return 1

# TODO:
# Here I will write a main to run the script from terminal with parameters
# according to each mission