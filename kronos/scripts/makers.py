import os
import sys
from datetime import datetime
import logging

from ..functions.my_functions import initialize_logger
from ..core.event import read_event,Event,EventList
from ..core.gti import read_gti, Gti, GtiList
from ..core.lightcurve import Lightcurve,LightcurveList
from ..core.power import PowerList,PowerSpectrum


def make_power_nicer(event_file,destination=os.getcwd(),
              tres=1.,tseg=16.,gti_dur=16,low_en=0.5,high_en=10.,
              split_event=False,drama=False,log_dir=None,log_name=None):


    if log_name is None:

        external_log = False

        # For logging purposes
        # -------------------------------------------------------------
        now = datetime.now()
        date = ('%d_%d_%d') % (now.day,now.month,now.year)
        time = ('%d_%d') % (now.hour,now.minute)
        log_name = os.path.basename('make_power_D{}_T{}'.\
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

    if 'mpu' in event_file:
        event_file_name = os.path.basename(event_file)
        obs_id = event_file_name[2:12]
        mission = 'NICER'
        logging.info(f'Reducing {mission} data')
    else:
        logging.info('Could not recognize event file mission.')
        if not external_log: move_log()
        return 0

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
    logging.info('Selected time segment: {} s'.format(tseg))
    logging.info('Time bins: {}'.format(int(tseg/tres)))
    logging.info('Selected GTI minimum duration: {} s'.format(gti_dur))
    if split_event:
        logging.info('Splitting event in GTIs? Yes')  
    else:
        logging.info('Splitting event in GTIs? No')   
    logging.info('Log file name: {}'.format(log_name))
    logging.info('-'*60)
    logging.info('')
    # -----------------------------------------------------------------

    # Output names
    lc_list_name = 'lc_list_E{}_{}_T{}_{}.pkl'.format(low_en,high_en,tres,tseg)
    power_list_name = 'power_list_E{}_{}_T{}_{}.pkl'.format(low_en,high_en,tres,tseg)
    power_name = 'power_E{}_{}_T{}_{}.pkl'.format(low_en,high_en,tres,tseg)
    lc_list_file = os.path.join(obs_id_dir,lc_list_name)
    power_list_file = os.path.join(obs_id_dir,power_list_name)
    power_file = os.path.join(obs_id_dir,power_name)

    logging.info('Processing event file: {}'.format(event_file))
    
    if os.path.isfile(lc_list_file):
        logging.info('Lightcurve list file {} already exists. Loading...'.format(lc_list_file))
        if not os.path.isfile(power_file):
            lcs2 = LightcurveList.load(lc_list_file)
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

        if split_event:
            logging.info('Splitting event according to GTIs')
            try:
                event_list = events.split(gti>=gti_dur)
            except Exception as e:
                logging.info('Could not split event file.')
                logging.info(e)
                if not external_log: move_log()
                return 0           

        logging.info('Computing lightcurve')
        try:
            if split_event:
                lcs = LightcurveList(
                [Lightcurve.from_event(e,time_res=tres,low_en=low_en,high_en=high_en)
                for e in event_list])       
            else:
                lightcurve = Lightcurve.from_event(events,time_res=tres,low_en=low_en,high_en=high_en)
                logging.info('Splitting lightcurve according to GTI')
                lcs = lightcurve.split(gti>=gti_dur)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not compute lightcurve')
            logging.info(e)
            if not external_log: move_log()
            return 0 

        if len(lcs)==0:
            logging.info('Lightcurve list (after gti split) is empty!')
            if not external_log: move_log()
            return 0

        logging.info('Splitting lightcurve list according to segments')
        try:
            lcs2 = lcs.split(tseg)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not split the lightcurve')
            logging.info(e)
            if not external_log: move_log()
            return 0 

        if len(lcs2)==0:
            logging.info('Lightcurve list (after seg split) is empty!')
            if not external_log: move_log()
            return 0

        logging.info('Saving LightcurveList in the event folder')
        lcs2.save(file_name=lc_list_name,fold=obs_id_dir)

    if os.path.isfile(power_file):
        logging.info('Power file {} already exists'.format(power_file))
    else:
        try:
            logging.info('Computing power spectrum list')
            powers = PowerSpectrum.from_lc(lcs2)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not compute power list')
            logging.info(e)
            if not external_log: move_log()
            return 0   

        logging.info('Saving power list')
        powers.save(file_name=power_list_file,fold=obs_id_dir)
        logging.info('Done!')

        try:
            logging.info('Computing average power')
            power = powers.average_leahy()
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not average power')
            logging.info(e)
            if not external_log: move_log()
            return 0 

        logging.info('Saving average power')
        power.to_pickle(power_file)
        logging.info('Done!')

    return 1




