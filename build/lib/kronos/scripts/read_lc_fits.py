import os
import logging
from astropy.io.fits import getval

from ..functions.my_functions import initialize_logger, make_logger
from ..core.lightcurve import Lightcurve
from ..core.gti import read_gti

def read_lc(fits_file, destination=os.getcwd(),mission='HXMT',
    log_name=None):
    '''
    Reads a binned lightcurve in a FITS file and stores it in a 
    lightcurve list

    NOTES
    -----
    2021 01 27, Stefano Rapisarda (Uppsala)
        Lightcurves are supposed to be created with ftool and are 
        supposed to be stored in the same folder of raw data
    '''

    if log_name is None:
        log_name = make_logger('make_power',outdir=destination)

    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'read_lc','*'*24))
    logging.info('*'*72)

    logging.info(f'Reading {mission} lightcurve FITS file')

    # Making folders
    # -----------------------------------------------------------------
    logging.info('Creating analysis folder...')
    an = os.path.join(destination,'analysis')
    if not os.path.isdir(an):
        os.mkdir(an)

    if not os.path.isfile(fits_file):
        logging.info('FITS file does not exist')
        if not external_log: move_log()
        return 0

    rmission = getval(fits_file,'telescop',1)
    if rmission != mission:
        logging.info(f'This is not a {mission} FITS file')
        if not external_log: move_log()
        return 0
    else:
        obs_id = os.path.basename(fits_file).split('_')[0]

    obs_id_dir = os.path.join(an,obs_id)
    if not os.path.isdir(obs_id_dir):
        os.mkdir(obs_id_dir)
        logging.info('Creating obs_ID folder...')
    # -----------------------------------------------------------------

    # Defining PI mission-dependent mul. and correction factors 
    # -----------------------------------------------------------------
    if mission == 'HXMT':
        if 'HE' in fits_file:
            factor = 370./256
            corr = 15
        elif 'ME' in fits_file:
            factor = 60./1024
            corr = 3
        elif 'LE' in fits_file:
            factor = 13./1536
            corr = 0.1
    elif mission == 'NICER' or 'SWIFT':
        factor = 100
        corr = 0 
    else:
        factor = 1
        corr = 0
    # -----------------------------------------------------------------

    # Reading energy channels
    # -----------------------------------------------------------------
    try:
        minpi = getval(fits_file,'minPI',1)
        maxpi = getval(fits_file,'maxPI',1)
        low_en = round(float(minpi)*factor+corr,1)
        high_en = round(float(maxpi)*factor+corr,1)
    except:
        logging.info('Could not determine energy bands from FITS file')
        low_en = None
        high_en = None
    # -----------------------------------------------------------------

    # Printing info
    # -----------------------------------------------------------------
    logging.info('')
    logging.info('Obs ID: {}'.format(obs_id))
    logging.info('Settings:')
    logging.info('-'*60)
    logging.info('Selected energy band: {}-{} keV'.format(low_en,high_en))
    logging.info('Selected time resolution: {} s'.format(lc.tres)) 
    logging.info('Log file name: {}'.format(log_name))
    logging.info('-'*60)
    logging.info('')
    # -----------------------------------------------------------------

    # Initializing output names
    # -----------------------------------------------------------------
    lc_name = 'lc_E{}_{}_T{}.pkl'.format(low_en,high_en,lc.tres) 
    lc_list_name = 'lc_list_E{}_{}_T{}.pkl'.format(low_en,high_en,lc.tres)       
    gti_name = 'gti_E{}_{}_T{}.gti'.format(low_en,high_en,lc.tres) 
    # -----------------------------------------------------------------

    if os.path.isfile(os.path.join(obs_id_dir,lc_list_name)):
        logging.info('Lightcurve list file {} already exists.'.\
            format(lc_list_name))
    else:
        # Computing lightcurve
        # -------------------------------------------------------------
        logging.info('Reading FITS file')
        try:
            lc = Lightcurve.read_from_fits(fits_file)
            lc.low_en = low_en
            lc.high_en = high_en
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not read FITS file')
            logging.info(e)
            return 0
        # -------------------------------------------------------------

        logging.info('Reading GTI from event file')
        try:
            gti = read_gti(fits_file)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not read GTI.')
            logging.info(e)
            return 0   

        logging.info('Saving Lightcurve in the event folder')
        lc.to_pickle(os.path.join(obs_id_dir,lc_name))
        logging.info('Done!')

        logging.info('Saving Gti in the event folder')
        gti.to_pickle(os.path.join(obs_id_dir,gti_name))
        logging.info('Done!')   

        # Computing lightcurve list
        # -------------------------------------------------------------
        logging.info('Splitting lightcurve according to GTI')
        try:
            lcs = lc.split(gti)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not split lightcurve')
            logging.info(e)
            return 0
        # -------------------------------------------------------------

        logging.info('Saving Lightcurve list in the event folder')
        lcs.save(lc_list_name,obs_id_dir)
        logging.info('Done!')
     
    return 1           