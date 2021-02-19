import os
import sys
from datetime import datetime
import logging

from ..functions.my_functions import initialize_logger,make_logger,my_cdate
from ..core.event import read_event,Event,EventList
from ..core.gti import read_gti, Gti, GtiList
from ..core.lightcurve import Lightcurve,LightcurveList

def make_nicer_lc(event_file,tres=1.,low_en=0.5,high_en=10.,
            split_event = False,destination=os.getcwd(),
            output_suffix='',log_name=None):
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
    tres: float
        time resolution (default=1)
    low_en: float
        low energy in keV (default=0.5)
    high_en: float
        high energy in keV (default=10.)
    split_event: boolean
        if True, it splits the event file into GTIs. GTIs are read
        from the event_file itself (default=False)
    destination: string
        full path of the output folder (default=current folder) 
    output_suffix: string
        string to attache at the end of the output file (default='')
    log_name: string or None
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
        on every event segment. If slit_event is False, Lightcurve is
        computed on the full event file and then split according to
        GTI. The former should be less time consuming

    HISTORY
    -------
    2020 01 01, Stefano Rapisarda (Uppsala), creation date
    2021 02 03, Stefano Rapisarda (Uppsala)
        Cleaned up
    '''

    # Logging
    if log_name is None:
        log_name = make_logger('make_lc',outdir=destination)

    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'make_lc','*'*24))
    logging.info('*'*72)

    # Making folders
    # -----------------------------------------------------------------
    logging.info('Creating analysis folder...')
    an = os.path.join(destination,'analysis')
    if not os.path.isdir(an):
        os.mkdir(an)

    if not os.path.isfile(event_file):
        logging.info('Event file does not exist')
        return 0

    if not 'mpu' in event_file:
        logging.info('This is not a NICER event file.')
        return 0
    else:
        obs_id = os.path.basename(event_file).split('_')[0].\
            replace('ni','')

    obs_id_dir = os.path.join(an,obs_id)
    if not os.path.isdir(obs_id_dir):
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
    if output_suffix != '':
        lc_list_name = 'lc_list_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,tres,output_suffix)
    else:
        lc_list_name = 'lc_list_E{}_{}_T{}.pkl'.\
            format(low_en,high_en,tres)    
    gti_name = 'gti.gti'

    lc_list_file = os.path.join(obs_id_dir,lc_list_name)
    gti_file = os.path.join(obs_id_dir,gti_name)
    # -----------------------------------------------------------------
    
    # Computing lightcurve
    # -----------------------------------------------------------------
    if os.path.isfile(lc_list_file):
        logging.info('Lightcurve list file {} already exists.'.\
            format(lc_list_name))
    else:
        logging.info('Reading event file')
        try:
            events = read_event(event_file)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not open event file.')
            logging.info(e)
            return 0

        logging.info('Reading GTI from event file')
        try:
            gti = read_gti(event_file)
            logging.info('Done!')
            logging.info('N. GTIs: {}'.format(len(gti)))
        except Exception as e:
            logging.info('Could not read GTI.')
            logging.info(e)
            return 0   

        if split_event:
            logging.info('Splitting event according to GTIs')
            try:
                event_list = events.split(gti)
                logging.info('Done!')
            except Exception as e:
                logging.info('Could not split event file.')
                logging.info(e)
                return 0     

        logging.info('Computing lightcurve list')
        try:
            if split_event:
                lcs = LightcurveList(
                [Lightcurve.from_event(e,time_res=tres,low_en=low_en,high_en=high_en)
                for e in event_list])     
                # When you split events according to GTI, GTI information
                # is lost when creating the lightcurve, so I need to 
                # manually specify it
                for i,lc in enumerate(lcs):
                    lc.history['GTI_SPLITTING'] = my_cdate()
                    lc.history['SPLITTING_NOTE'] = 'Event Splitting'
                    lc.history['N_GTIS'] = len(lcs)  
                    lc.history['GTI_INDEX'] = i
            else:
                lightcurve = Lightcurve.from_event(events,time_res=tres,low_en=low_en,high_en=high_en)
                logging.info('Done!')
                lcs = lightcurve.split(gti)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not compute lightcurve')
            logging.info(e)
            return 0 

        if len(lcs)==0:
            logging.warning('Lightcurve list (after gti split) is empty!')
            return 1

        # Saving Lightcurve list and gti
        logging.info('Saving Lightcurve in the event folder')
        lcs.save(lc_list_name,fold=obs_id_dir)
        logging.info('Done!')

        logging.info('Saving Gti in the event folder')
        gti.to_pickle(gti_file)
        logging.info('Done!') 
        
    # -----------------------------------------------------------------       

    return 1

# TODO:
# Here I will write a main to run the script from terminal with parameters
# according to each mission