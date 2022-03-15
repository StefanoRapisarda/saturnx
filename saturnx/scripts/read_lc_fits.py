import os
import logging
import pathlib
from astropy.io.fits import getval

from saturnx.utils.my_logging import make_logger
from saturnx.core.lightcurve import Lightcurve
from saturnx.core.gti import Gti

def read_lc(fits_file, ext='COUNTS',destination=pathlib.Path.cwd(),
    mission='HXMT',output_suffix='',single_gti = True, log_name=None):
    '''
    Reads a binned lightcurve in a FITS file and stores it in a 
    lightcurve list.

    This is supposed to work as make_lc, but instead of making it 
    reading an Event object (or FITS file), it reads an already
    computed one from a FITS file

    NOTES
    -----
    2021 01 27, Stefano Rapisarda (Uppsala)
        Lightcurves are supposed to be created with ftool and are 
        supposed to be stored in the same folder of raw data
    '''

    if type(destination) == str:
        destination = pathlib.Path(destination)

    if log_name is None:
        log_name = make_logger('make_power',outdir=destination)

    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'read_lc','*'*24))
    logging.info('*'*72)

    logging.info(f'Reading {mission} lightcurve FITS file')

    # Making folders
    # -----------------------------------------------------------------
    logging.info('Creating analysis folder...')
    an = destination/'analysis'
    if not an.is_dir():
        os.mkdir(an)

    if type(fits_file) == str: fits_file = pathlib.Path(fits_file)
    if not fits_file.is_file():
        logging.info('FITS file does not exist')
        return 0

    bkg_flag = False
    if ('BKG' in str(fits_file) or 'bkg' in str(fits_file)) and mission =='HXMT':
        ext = 'RATE'
        bkg_flag = True
        logging.info('This is a background HXMT FITS file')

    rmission = getval(fits_file,'telescop',ext)
    if rmission != mission:
        logging.info(f'This is not a {mission} FITS file')
        return 0
    
    if mission == 'HXMT':
        try:
            obs_id = getval(fits_file,'EXP_ID',ext)
        except:
            file_name = fits_file.name
            obs_id = file_name.split('_')[0]
    elif mission == 'NICER':
        obs_id = getval(fits_file,'OBS_ID',ext)
    else:
        obs_id = str(fits_file.name).split('_')[0]

    obs_id_dir = an/obs_id
    if not obs_id_dir.is_dir():
        os.mkdir(obs_id_dir)
        logging.info('Creating obs_ID folder...')
    # -----------------------------------------------------------------

    # Defining PI mission-dependent mul. and correction factors 
    # -----------------------------------------------------------------
    if mission == 'HXMT':
        instrument = getval(fits_file,'INSTRUME',ext)
        if instrument == 'HE':
            factor = 370./256
            corr = 15
        elif instrument == 'ME':
            factor = 60./1024
            corr = 3
        elif instrument == 'LE':
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
    except KeyError:
        logging.info('Could not determine energy bands from FITS file')
        if 'ch' in str(fits_file.name):
            file_name = str(fits_file.name)
            target = file_name.split('_')[4].replace('ch','')
            minpi = int(target.split('-')[0])
            maxpi = int(target.split('-')[1])
            low_en = round(float(minpi)*factor+corr,1)
            high_en = round(float(maxpi)*factor+corr,1)
        else:
            logging.info('Could not determine energy bands from FITS file name')
            low_en = None
            high_en = None
    # -----------------------------------------------------------------

    # Computing lightcurve
    # -----------------------------------------------------------------
    logging.info('Reading FITS file')
    try:
        lc = Lightcurve.read_fits(fits_file,ext=ext)
        lc.low_en = low_en
        lc.high_en = high_en
        logging.info('Done!')
    except Exception as e:
        logging.info('Could not read FITS file')
        logging.info(e)
        return 0
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
    if output_suffix != '':
        lc_name = 'lc_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,lc.tres,output_suffix) 
        lc_list_name = 'lc_list_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,lc.tres,output_suffix)       
        gti_name = 'gti_E{}_{}_T{}_{}.gti'.\
            format(low_en,high_en,lc.tres,output_suffix)    
    else:
        lc_name = 'lc_E{}_{}_T{}.pkl'.format(low_en,high_en,lc.tres) 
        lc_list_name = 'lc_list_E{}_{}_T{}.pkl'.format(low_en,high_en,lc.tres)       
        gti_name = 'gti_E{}_{}.gti'.format(low_en,high_en) 
    # -----------------------------------------------------------------

    logging.info('Saving Lightcurve in the event folder')
    lc.save(lc_name,obs_id_dir)
    logging.info('Done!')

    gti_flag = False
    if not bkg_flag: # <--- source lightcurve
        logging.info('Reading GTI from event file')
        try:
            gti = Gti.read_fits(fits_file)
            gti_flag = True
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not read GTI.')
            logging.info(e)  

        if gti_flag:
            logging.info('Saving Gti in the event folder')
            gti.save(gti_name,obs_id_dir)
            logging.info('Done!')   
    else: # <--- background lightcurve
        try:
            if single_gti:
                gti_name = 'gti_E{}_{}.gti'.format(low_en,high_en) 
            gti = Gti.load(gti_name,obs_id_dir)
            gti_flag = True
        except FileNotFoundError:
            logging.warning('I did not find a GTI file, I will NOT compute lightcurve list')


    if not bkg_flag or gti_flag:
        # Computing lightcurve list
        # -----------------------------------------------------------------
        logging.info('Splitting lightcurve according to GTI')
        try:
            lcs = lc.split(gti)
            logging.info('Done!')
        except Exception as e:
            logging.info('Could not split lightcurve')
            logging.info(e)
            return 0
        # -----------------------------------------------------------------

        logging.info('Saving Lightcurve list in the event folder')
        lcs.save(lc_list_name,obs_id_dir)
        logging.info('Done!')
     
    return 1           