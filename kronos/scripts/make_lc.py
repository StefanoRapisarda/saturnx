import os
import sys
from datetime import datetime
import logging

from ..functions.my_functions import initialize_logger
from ..core.event import read_event,Event,EventList
from ..core.gti import read_gti, Gti, GtiList
from ..core.lightcurve import Lightcurve,LightcurveList

def make_nicer_lc(event_file,destination=os.getcwd(),
            tres=1.,low_en=0.5,high_en=10.,
            output_suffix='',drama=False,log_dir=None,log_name=None):


    if log_name is None:

        external_log = False

        # For logging purposes
        # -------------------------------------------------------------
        now = datetime.now()
        date = ('%d_%d_%d') % (now.day,now.month,now.year)
        time = ('%d_%d') % (now.hour,now.minute)
        log_name = os.path.basename('make_lc_D{}_T{}'.\
                                    format(date,time))
        initialize_logger(log_name)

        logging.info('Creating log folder...')
        log_dir = os.path.join(destination,'logs')
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        # -------------------------------------------------------------
    else:
        external_log = True

    def move_log():
        os.system(f'mv {log_name}.log {log_dir}')

    # Making folders
    # -----------------------------------------------------------------
    logging.info('Creating analysis folder...')
    an = os.path.join(destination,'analysis')
    if not os.path.isdir(an):
        os.mkdir(an)

    if not os.path.isfile(event_file):
        logging.info('Event file does not exist')
        if not external_log: move_log()
        return 0

    if not 'mpu' in event_file:
        logging.info('This is not a NICER event file.')
        if not external_log: move_log()
        return 0
    else:
        obs_id = os.path.basename(event_file).split('_')[0].\
            replace('ni','')


    obs_id_dir = os.path.join(an,obs_id)
    if not os.path.isdir(obs_id_dir):
        os.mkdir(obs_id_dir)
        logging.info('Creating obs_ID folder...')

    logging.info('')
    logging.info('Obs ID: {}'.format(obs_id))
    logging.info('Settings:')
    logging.info('-'*60)
    logging.info('Selected energy band: {}-{} keV'.format(low_en,high_en))
    logging.info('Selected time resolution: {} s'.format(tres)) 
    logging.info('Log file name: {}'.format(log_name))
    logging.info('-'*60)
    logging.info('')
    # -----------------------------------------------------------------

    # Output names
    if output_suffix != '':
        lc_name = 'lc_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,tres,output_suffix)
    else:
        lc_name = 'lc_E{}_{}_T{}.pkl'.\
            format(low_en,high_en,tres)        
    gti_name = 'gti.gti'

    logging.info('Processing event file: {}'.format(event_file))
    
    if os.path.isfile(os.path.join(obs_id_dir,lc_name)):
        logging.info('Lightcurve list file {} already exists.')
        return 1
    else:
        logging.info('Reading event file')
        try:
            events = read_event(event_file)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not open event file.')
            logging.info(e)
            if not external_log: move_log()
            return 0

        logging.info('Reading GTI from event file')
        try:
            gti = read_gti(event_file)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not read GTI.')
            logging.info(e)
            if not external_log: move_log()
            return 0       

        logging.info('Computing lightcurve')
        try:
            lightcurve = Lightcurve.from_event(events,time_res=tres,low_en=low_en,high_en=high_en)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not compute lightcurve')
            logging.info(e)
            if not external_log: move_log()
            return 0 

        logging.info('Saving Lightcurve in the event folder')
        lightcurve.to_pickle(os.path.join(obs_id_dir,lc_name))
        logging.info('Done!')

        logging.info('Saving Gti in the event folder')
        gti.to_pickle(os.path.join(obs_id_dir,gti_name))
        logging.info('Done!')        

    return 1


# Here I will write a main to run the script from terminal with parameters
# according to each mission