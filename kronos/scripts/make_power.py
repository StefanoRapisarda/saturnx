import os
import pathlib
from datetime import datetime
import logging

from kronos.utils.logging import make_logger
from kronos.utils.generic import my_cdate
from kronos.core.event import Event,EventList
from kronos.core.gti import Gti, GtiList
from kronos.core.lightcurve import Lightcurve,LightcurveList
from kronos.core.power import PowerList,PowerSpectrum

def make_power(lc_list_file,destination=pathlib.Path.cwd(),
              tseg=16.,log_name=None, override=False):

    if type(lc_list_file) == str:
        lc_list_file = pathlib.Path(lc_list_file)

    if type(destination) == str:
        destination = pathlib.Path(destination)

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

    # Reading parameters from Lightcurve list file name
    # -----------------------------------------------------------------
    lc_list_name = lc_list_file.name
    obs_id_dir = lc_list_file.parent
    cleaned = str(lc_list_name).replace('lc_list_','').replace('.pkl','')
    div = cleaned.split('_')
    low_en = div[0].replace('E','')
    high_en = div[1]
    tres = div[2].replace('T','')
    if len(div) > 3:
        output_suffix = div[3]
    else:
        output_suffix = ''
    # -----------------------------------------------------------------

    # Output names
    # -----------------------------------------------------------------
    # lc2_list_file will be splitted according to segment
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

    lc2_list_file = obs_id_dir/lc2_list_name
    power_list_file = obs_id_dir/power_list_name
    power_file = obs_id_dir/power_name
    # -----------------------------------------------------------------

    logging.info('Processing lightcurve: {}'.format(lc_list_file))
    
    if lc2_list_file.is_file() and not override:
        logging.info('Lightcurve list file {} already exists.'.\
            format(lc2_list_name))
        lcs2 = LightcurveList.load(lc2_list_name,obs_id_dir)  
    else:
        lcs = LightcurveList.load(lc_list_name,obs_id_dir)
        if len(lcs) == 0:
            logging.info('Lightcurve list file is empty, skipping.')
            logging.info(lc_list_name)
            return 0
        lcs_filt = LightcurveList([lc for lc in lcs if (not lc.texp is None) and (lc.texp>= tseg)])
        # lcs is the original lightcurve filtered according to GTIs
        # meta_data in lcs store N_GTIS and GTI_INDEX according to the 
        # original GTIs
        # Here, I am updating these two keywords according to this filtering
        for i,lc in enumerate(lcs_filt):
            lc.meta_data['FILTERING_GTI'] = my_cdate()
            lc.meta_data['FILTERING_MODE'] = 'texp longer than {}s'.format(tseg) 
            lc.meta_data['ORI_N_GTIS'] = len(lcs)
            lc.meta_data['N_GTIS'] = len(lcs_filt)
            lc.meta_data['GTI_INDEX'] = i   

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

    if os.path.isfile(power_file) and not override:
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
            power = powers.average('leahy')
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not average power')
            logging.info(e)
            return 0 

        logging.info('Saving average power')
        power.save(power_name,obs_id_dir)
        logging.info('Done!')

    return 1
