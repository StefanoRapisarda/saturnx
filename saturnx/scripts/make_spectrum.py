import os
import pathlib
import logging

from saturnx.utils.my_logging import make_logger

def make_nicer_spectrum(event_file, low_en = 0.2, high_en=12.,
    destination=pathlib.Path.cwd(),override=False,log_name=None):
    
    if type(destination) == str:
        destination = pathlib.Path(destination)

    # Logging
    if log_name is None:
        log_name = make_logger('make_lc',outdir=destination)

    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'make_lc','*'*24))
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
    logging.info('Selected energy band: {}-{} keV'.format(low_en,high_en))))
    logging.info('Log file name: {}'.format(log_name))
    logging.info('-'*60)
    logging.info('')
    # -----------------------------------------------------------------

    spec = f'{obs_id}_tot_spec.pi'
    bkg_spec = f'{obs_id}_bkg_spec.pi'

    
    wf = os.path.join(df,'{}/xti/event_cl'.format(obs_ID))
    cl_files = mf.list_items(wf,opt=2,include_all = ['cl','evt','bc'], exclude = 'bdc')
    if cl_files:
        if len(cl_files) > 1:
            logging.warning('You selected more than one cleaned event file.')
            sys.exit()
        else:
            cl_file_ori = cl_files[0]
    else:
        logging.warning('I did not find any event file')
        logging.info('='*80+'\n')
        

    # Moving to working directory
    os.chdir(wf)                                                                # <=== !!!MOVING!!!

    # Excluding noisy detectors (This is supposed to be done by the reduction script)
    cl_file = cl_file_ori.replace('_bc.','_bc_bdc.')
    if not os.path.isfile(cl_file):
        logging.info('Removing noisy and inactive detectors')   
        cmd = 'fselect {} {} "(DET_ID != 11) && (DET_ID != 14)  && (DET_ID != 20)  && (DET_ID != 22)  && (DET_ID != 34)  && (DET_ID != 60)"'.format(cl_file_ori,cl_file)
        os.system(cmd)
        # Checking if the file was properly created
        if not os.path.isfile(cl_file):
            logging.warning('Cleaned event file was not created.Quitting')
            logging.info('='*80+'\n')
            sys.exit()

    if not for_all:
        opts = ['Delete the files',
                'Move it to the graveyard',
                'Skip observations with already created cleaned event files']
        file_list = next(os.walk(wf))[2]
        found = False
        for f in file_list:
            if '.pha' in f:
                file_name = f
                found = True
        if found:
            logging.info('I found an energy spectrum: {}'.format(file_name))
            opt = mf.ask_opts(opts)
            for_all = mf.yesno('Do you want to keep your choice for all the obs_ID?')

            if opt == 0:
                logging.info('Removing old .pha files')
                cmd = 'rm -f {}/*.pha'.format(wf)
                os.system(cmd)        
            elif opt == 1:
                graveyard = mf.create_dir('graveyard',wf)
                old = mf.create_dir('spectra_D{}_T{}'.format(date,time),graveyard)
                logging.info('Moving old .pha files to graveyard')
                cmd = 'mv {}/*.pha {}'.format(wf,old)
                os.system(cmd)
            elif opt ==2:
                logging.info('I found an energy spectrum: {}'.format(file_name))
                logging.info('Skipping this observation')
                logging.info('='*80 + '\n')
                continue
            
    # Running xselect
    # -----------------------------------------------------------------------
    xselect = ['xsel',
            'set mission nicer',
            'read event {}'.format(cl_file),
            './',
            'filter time file {}/{}'.format(wf,cl_file),
            'extract spectrum',
            'save spectrum {}.pha'.format(obs_ID),
            'exit',
            'no']

    with open('xselect.in','w') as outfile:
        for i in xselect:
            outfile.write(i+'\n')
            
    os.system('xselect < {} > xselect.log'.format('xselect.in'))
    # -----------------------------------------------------------------------

    # Computing the background
    # -----------------------------------------------------------------------
    logging.info('Computing the background')
    spectrum = '{}.pha'.format(obs_ID)
    
    # Running niprefilter2
    mkf2 = '{0}/{1}/auxil/ni{1}.mkf2'.format(df,obs_ID)
    cmd = 'niprefilter2 indir={0}/{1} infile={0}/{1}/auxil/ni{1}.mkf \
           outfile={0}/{1}/auxil/ni{1}.mkf2 clobber=YES'.format(df,obs_ID)
    os.system(cmd)
    # Checcking that the file was properly created
    if not os.path.isfile(mkf2):
        logging.warning('.mkf2 file not created. Quitting...')
        logging.info('='*80 + '\n')
        sys.exit()

    # Updating the .mkf2 file in order to include the KP values
    kp_fits= '/Users/stefano/science/Software/background/NICER/kp.fits'
    #kp_fits = '/var/science_products/background/NICER/xti/kp.fits'
    status = bk.add_kp(mkf2,kpfile=kp_fits)
    mkf3 = '{0}/{1}/auxil/ni{1}.mkf3'.format(df,obs_ID)
    if os.path.isfile(mkf3):
        logging.info('mkf3 file successfully created')
    else:
        logging.warning('mkf3 file not created. Quitting')
        sys.exit()
    
    
    # Counting the number of active detectors
    #obs = nc.nob(cl_file)
    #ndet = 56 - len(obs.inact_det_list) -2
    ndet = 50

    # Creating background spectrum
    #bkg_evt = '/var/science_products/background/NICER/xti/30nov18targskc_enhanced.evt'
    bkg_evt = '/Users/stefano/science/Software/background/NICER/30nov18targskc_enhanced.evt'
    bkg_chan, bkgspec_tot, btotexpo = bk.mk_bkg_spec_evt(wf+'/'+spectrum,mkf3file=mkf3,
                                                 bevt=bkg_evt,numfpms=ndet)
    bkg = '{}_bkg.pha'.format(obs_ID)
    bkg_flag = True
    if os.path.isfile(bkg):
        logging.info('bkg file successfully created')
    else:
        logging.warning('bkg file not created. <=============')
        bkg_flag = False
    # -----------------------------------------------------------------------
    
    # GRPPHA is an interactive command driven taks to define (or redefine) and/or
    # display the grouping (binning) & quality flags, and the fractional systematic
    # error associated with channels in a FITS PHA file.
    # The necessary grouping, quality & systematic error information for each channel
    # is written alongside the PHA dataset to be picked up by subsequent GRPPHA
    # commands and downstream software (e.g. EXSPEC). In other words, the original file
    # is not modified
    # In this specific case energy bins are grouped in order to have a minimum of
    # 30 counts
    # -----------------------------------------------------------------------
    logging.info('Grouping energy spectrum bins')
    spectrum_grp = '{}_grp30.pha'.format(obs_ID)
    grppha=['{}.pha'.format(obs_ID),
            spectrum_grp,
            'group min 30',
            'exit']

    with open('grppha.in','w') as outfile:
        for i in grppha:
            outfile.write(i+'\n')
            
    os.system('grppha < {} > grppha.log'.format('grppha.in'))
    # -----------------------------------------------------------------------

    # Printing energy spectrum on column file for plotting with python
    # -----------------------------------------------------------------------
    logging.info('Running XSPEC and creating energy spectrum object')
    ldat = '{}_0.2-12.dat'.format(obs_ID)
    xspec=['data {}'.format(spectrum_grp),
           'response {}'.format(nf.rmf),
           'arf {}'.format(nf.arf),
           'backgr {}'.format(bkg),
           'ignore **-0.2 12.0-**',
           'ignore bad',
           'setplot energy',
           'plot ldata',
           'iplot',
           'wdata {}'.format(ldat),
           'exit',
           'exit']
    if not bkg_flag:
        xspec.remove('backgr {}'.format(bkg))
    
    with open('xspec.in','w') as outfile:
        for i in xspec:
            outfile.write(i+'\n')
            
    os.system('xspec < {} > xspec.log'.format('xspec.in'))
    # -----------------------------------------------------------------------

    # There is the possibility that XSPEC did not load the data (exposure time = 0)
    

    # Reading the just created column data and making an object out from 
    # -----------------------------------------------------------------------
    if os.path.isfile('{}'.format(ldat)):
        array = np.loadtxt(wf+'/'+ldat,skiprows=3)
        dictob = {'OBS_ID':obs_ID,'CRE_DATE':'D{}_T{}'.format(date,time),
                  'DATA':array,'PHA':spectrum_grp}
        with open('{}_energy_spectrum.pkl'.format(obs_ID),'wb') as outfile:
            pickle.dump(dictob,outfile)
    # -----------------------------------------------------------------------
    logging.info('='*80+'\n')
    
    # Going back home
    os.chdir(home)