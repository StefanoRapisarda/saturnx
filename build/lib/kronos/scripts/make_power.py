import os
import sys
from datetime import datetime
import logging

from ..functions.my_functions import initialize_logger, my_cdate, make_logger
from ..core.event import read_event,Event,EventList
from ..core.gti import read_gti, Gti, GtiList
from ..core.lightcurve import Lightcurve,LightcurveList
from ..core.power import PowerList,PowerSpectrum

def make_power(lc_list_file,destination=os.getcwd(),
              tseg=16.,log_name=None):

    if log_name is None:
        log_name = make_logger('make_power',outdir=destination)

    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'make_power','*'*24))
    logging.info('*'*72)
    logging.info('')
    logging.info('Settings:')
    logging.info('-'*60)
    logging.info('Lightcurve list file: {}'.\
        format(os.path.basename(lc_list_file)))
    logging.info('Selected time segment: {} s'.format(tseg))
    logging.info('Log file name: {}'.format(log_name))
    logging.info('-'*60)
    logging.info('')
    # -----------------------------------------------------------------

    # Reading parameters from Lightcurve list file
    lc_list_name = os.path.basename(lc_list_file)
    obs_id_dir = os.path.dirname(lc_list_file)
    cleaned = lc_list_name.replace('lc_list_','').replace('.pkl','')
    div = cleaned.split('_')
    low_en = div[0].replace('E','')
    high_en = div[1]
    tres = div[2].replace('T','')
    if len(div) > 3:
        output_suffix = div[3]
    else:
        output_suffix = ''

    # Output names
    if output_suffix != '':
        lc2_list_name = 'lc_list_E{}_{}_T{}_{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg,output_suffix)
        power_list_name = 'power_list_E{}_{}_T{}_{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg,output_suffix)
        power_name = 'power_E{}_{}_T{}_{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg,output_suffix)                         
    else:
        lc2_list_name = 'lc_list_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg)
        power_list_name = 'power_list_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg)
        power_name = 'power_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg)

    lc2_list_file = os.path.join(obs_id_dir,lc2_list_name)
    power_list_file = os.path.join(obs_id_dir,power_list_name)
    power_file = os.path.join(obs_id_dir,power_name)

    logging.info('Processing lightcurve: {}'.format(lc_list_file))
    
    if os.path.isfile(lc2_list_file):
        logging.info('Lightcurve list file {} already exists.'.\
            format(os.path.basename(lc2_list_name)))
        lcs2 = LightcurveList.load(lc2_list_file)  
    else:
        lcs = LightcurveList.load(lc_list_file)
        lcs_filt = LightcurveList([lc for lc in lcs if lc.texp >= tseg])
        for i,lc in enumerate(lcs_filt):
            lc.history['GTI_FILTERING'] = my_cdate()
            lc.history['FILTERING_NOTE'] = 'gti longer than {}s'.format(tseg)
            lc.history['N_GTIS'] = len(lcs_filt)  
            lc.history['GTI_INDEX'] = i        

        logging.info('Splitting lightcurve list according to segments')
        try:
            lcs2 = lcs_filt.split(tseg)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not split the lightcurve into segments')
            logging.info(e)
            return 0 

        if len(lcs2)==0:
            logging.info('Lightcurve list (after seg split) is empty!')
            return 0

        logging.info('Saving LightcurveList in the event folder')
        lcs2.save(file_name=lc2_list_name,fold=obs_id_dir)

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
            return 0 

        logging.info('Saving average power')
        power.to_pickle(power_file)
        logging.info('Done!')

    return 1
