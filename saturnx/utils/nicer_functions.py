'''
Contain functions specific to NICER data 
'''

import os
from re import LOCALE
from astropy.io import fits
from .my_functions import get_nn_var,list_items
from .my_logging import LoggingWrapper
import numpy as np
import pathlib

all_det = np.array([0,1,2,3,4,5,6,
                10,11,12,13,14,15,7,
                20,21,22,23,24,25,16,
                30,31,32,33,34,26,17,
                40,41,42,43,44,35,27,
                50,51,52,53,45,36,37,
                60,62,64,66,54,46,47,
                61,63,65,67,55,56,57])

def check_nicerl2_io(obs_id_dir):
    '''
    Checks if nicerl2 has everything it needs

    HISTORY
    -------
    2021 11 02, Stefano Rapisarda (Uppsala), creation date
        I made this to clean up the execution flow in NICER_pipeline
    '''

    if type(obs_id_dir) == str:
        obs_id_dir = pathlib.Path(obs_id_dir)

    mylogging = LoggingWrapper() 

    # Folders
    # -----------------------------------------------------------------
    folders = {
        'auxil':'auxil',
        'hk':'xti/hk',
        'event_uf':'xti/event_uf',
        'event_cl':'xti/event_cl'
        }    
    for folder_name,full_path in folders.items():
        to_check = obs_id_dir/full_path
        if not to_check.exists():
            mylogging.error(f'{folder_name} does not exist')
            return False 
    # -----------------------------------------------------------------

    # Files
    # -----------------------------------------------------------------
    # 7 Uncalibrated files (one per MPU)
    uf_files = list_items(obs_id_dir/folders['event_uf'],itype='file',
        include_and=['_uf.evt'])
    if len(uf_files) != 7:
        mylogging.error('The uncalibrated files are not 7')
        return False

    # Make filter file (also unzipping if zipped)
    mkf_files = list_items(obs_id_dir/folders['auxil'],itype='file',include_and=['.mkf'],
        exclude_or=['mkf2','mkf3'])
    if len(mkf_files) == 0:
        mylogging.error('The make filter file (.mkf) does not exist')
        return False
    elif len(mkf_files) > 1:
        mylogging.error('There is more than one make filter file (.mkf), check your auxil folder.')
        return False

    # Orbit file
    orbit_file = list_items(obs_id_dir/folders['auxil'],itype='file',include_and=['.orb'])             
    if len(orbit_file) != 1:
        mylogging.error('The orbit file (.orb) does not exist')
        return False

    # Attitude file
    att_files = list_items(obs_id_dir/folders['auxil'],itype='file',include_and=['.att'])
    if len(att_files) != 1:
        mylogging.error('The attitude file (.att) does not exist')
        return False
    # -----------------------------------------------------------------

    return True
        

def run_nicerl2(obs_id_dir,tasks="ALL",filtcolumns="NICERV3,3C50",
        niprefilter2="YES",saafilt="NO",nicersaafilt="YES",
        ang_dist="0.015",elv="15",br_earth="30",cor_range="*-*",
        underonly_range="0-500",overonly_range="0-1.5",min_fpm='1',
        overonly_expr="1.52*OVERONLY_NORM*COR_SAX**(-0.633)",clobber="yes",
        log_file='nicerl2.log'):
    '''
    Runs HEASoft nicerl2 with specified options

    RETURNS
    -------
    result: boolean
        True if everything went smoothly

    HISTORY
    -------
    2021 10 13, Stefano Rapisarda (Uppsala), creation date
    2021 11 02, Stefano Rapisarda (Uppsala)
        Moved some instructions here from the main script (NICER_pipeline)
        Now it prints its own logging command and it checks if the output
        was properly created
    '''

    mylogging = LoggingWrapper() 

    if type(obs_id_dir) == str:
        obs_id_dir = pathlib.Path(obs_id_dir)

    command = f'nicerl2 indir="{str(obs_id_dir)}" tasks="{tasks}" \
        filtcolumns="{filtcolumns}" niprefilter2="{niprefilter2}" \
        saafilt="{saafilt}" nicersaafilt="{nicersaafilt}" \
        ang_dist="{ang_dist}" elv="{elv}" br_earth="{br_earth}" \
        cor_range="{cor_range}" underonly_range="{underonly_range}" \
        overonly_range="{overonly_range}" overonly_expr="{overonly_expr}" \
        min_fpm="{min_fpm}" \
        clobber="{clobber}"> {log_file}'
    
    mylogging.info('Running nicerl2 ...')
    try:
        os.system(command)
        mylogging.info('... and done!\n')
    except:
        mylogging.error('Something went wrong running nicerl2, check log file\n' +\
            f'({nicerl2_log_name})')
        return False

    # Checking nicerl2 output files
    # -----------------------------------------------------------------
    event_cl_dir = obs_id_dir/'xti/event_cl'

    ufa_files = list_items(event_cl_dir,itype='file',
        include_and=['_ufa.evt'])
    if len(ufa_files) != 8:
        mylogging.error('ufa files not created correctly')
        return False

    cl_file = list_items(event_cl_dir,itype='file',
        include_and=['_0mpu7_cl.evt'])
    if len(cl_file) != 1:
        mylogging.error('Cleaned event file not created correctly')
        return False   
    # -----------------------------------------------------------------   

    return cl_file[0]


def run_nifpmsel(cl_file,det_list,clobber='yes',log_file='nifpmsel.log'):
    '''
    PARAMETERS
    ----------
    cl_file: str or pathlib.Path()
        Full path of the cleaned event file
    det_list: str
        Detectors to exclude separated by coma
        (e.g. -11,-34)
    log_file: str or pathlib.Path()
        Name, with full path, of the log file for nifmpsel

    HISTORY
    -------
    2021 11 02, Stefano Rapisarda (Uppsala), creation date
        I wrote this to clean up the main program flow
    '''

    mylogging = LoggingWrapper()  

    if type(cl_file) == str:
        cl_file = pathlib.Path(cl_file)

    event_cl_dir = cl_file.parent

    if cl_file.suffix == '.gz':
        mylogging.info('run_nifpmsel: unzipping cleaned event file')
        os.system(f'gunzip {cl_file}')
        input_evt_file = pathlib.Path(cl_file.stem)
    else:
        input_evt_file = cl_file

    name_cl_bdc = str(input_evt_file.name).replace('cl.evt','cl_bdc.evt')
    
    cl_bdc_file = event_cl_dir/name_cl_bdc     
    
    cmd = f'nifpmsel infile="{input_evt_file}" outfile="{cl_bdc_file}"  \
        detlist="launch,{det_list}" clobber={clobber} > {log_file}'   
    
    mylogging.info('Excluding noisy detectors ...')
    try:
        os.system(cmd)
        mylogging.info('... and done!')
    except:
        mylogging.error('Something went wrong running nifpmsel, check log file\n' +\
            f'({log_file})')
        return False       

    # Checking if the file was created
    if not cl_bdc_file.exists():
        mylogging.warning('nifpmsel file not created')
        return False

    return cl_bdc_file


def run_barycorr(evt_file,orbit_file,refframe='ICRS',
    ephem='JPLEPH.430',clobber='yes',log_file='barycorr.log'):

    mylogging = LoggingWrapper()  

    if type(evt_file) == str:
        evt_file = pathlib.Path(evt_file)

    evt_dir = evt_file.parent

    # Barycorr does not like if the input file is zipped
    if evt_file.suffix == '.gz':
        mylogging.info('barycorr: unzipping cleaned event file')
        os.system(f'gunzip {evt_file}')
        input_evt_file = pathlib.Path(evt_file.stem)
    else:
        input_evt_file = evt_file

    name_evt_bc = str(input_evt_file.name).replace('.evt','_bc.evt')
    bc_file = evt_dir/name_evt_bc

    cmd = f'barycorr infile="{input_evt_file}" \
            outfile="{bc_file}" \
            orbitfiles="{orbit_file}" \
            refframe={refframe} ephem={ephem} \
            clobber={clobber} > {log_file}'    
    mylogging.info('Applying barycentering correction ...')
    try:
        os.system(cmd)
        mylogging.info('... and done!')
    except:
        mylogging.error('Something went wrong running barycorr, check log file\n' +\
            f'({log_file})')
        return False   

    # Checking if the file was created
    if not bc_file.exists():
        mylogging.warning('Barycentric corrected file not created!')
        return False

    return bc_file

def check_nicer_data(obs_id_dir,files_to_check={}):
    '''
    Checks the existance of certain files. 
    
    This can be used either to 
    check if NICER observations contain all the files needed for 
    running specific ftools or to check the correct creation of data
    analysis products.

    PARAMETERS
    ----------
    obs_id_dir: str or pathlib.Path
        Full path of a NICER obs id directory
    files_to_check: dict (optional)
        Dictionary containing the files to check
        The format of the dictionary MUST be:
        {
            <user_key>:[<file_str_identifier>,<relative_location>]
        }
        where:
        - user_key is an arbitrary str identifying the file;
        - file_str_identifier is a str contained in the name of the
            file to check and selected in a way that ALONE AND ONLY
            the file to check contains this string and can be 
            uniquely by it;
        - relative_location is a str containing the location of the 
            file to check relative to the obs ID folder.
        If no dictionary is specified ({}, default), then files_to_check
        will be initialized according to my criteria (corresponding to 
        the output files of my current NICER pipeline)
    
    RETURNS
    -------
    are_files_there, files: (dict,dict)
        are_files_there is a dict containing a boolean for each of the
        user_key in files_to_check. The boolean indicates if the file 
        has been found to not
        files is a dict containing either a full path or an empty str
        for each of the user_key in files_to_check in case the file has 
        been found or not.

    HISTORY
    -------
    2020 .. .., Stefano Rapisarda (Uppsala), creation date
    2021 11 03, Stefano Rapisarda (Uppsala)
        Updated with pathlibPath, removed dir check (as I think it was
        redundant at this point, as checking files with their full paths
        should be enough), updated files_to_check with files created 
        by the new nicer_pipeline, and adopted the LoggingWrapper 
        philosophy.
    '''

    mylogging = LoggingWrapper()

    if type(obs_id_dir) == str: obs_id_dir = pathlib.Path(obs_id_dir)
    
    # Checking obs ID
    obs_id = obs_id_dir.name
    if not obs_id.isdigit():
        mylogging(f'check_nicer_data: Something is wrong with the obs. ID name ({obs_id})')

    if len(files_to_check) == 0:
        files_to_check = {
            'att':['.att','auxil'],
            'orb':['.orb','auxil'],
            'cat':['.cat','auxil'],
            'mkf':['.mkf','auxil'],
            'mkf2':['.mkf2','auxil'],
            'mkf3':['.mkf3','auxil'],
            'uf':['_uf.evt','xti/event_uf'],
            'ufa':['_ufa.evt','xti/event_cl'],
            'cl':['_cl.evt','xti/event_cl'],
            'cl_bdc':['_cl_bdc.evt','xti/event_cl'],
            'cl_bdc_bc':['_cl_bdc_bc.evt','xti/event_cl'],
            'spectrum':['_bdc.pha','xti/event_cl'],
            'bkg_spectrum':['_bdc_bkg.pha','xti/event_cl'],
            'grp_spectrum':['_bdc_grp25.pha','xti/event_cl'],
            '3C50_spectrum':['3C50_tot','xti/event_cl'],
            '3C50_bkg':['3C50_bkg','xti/event_cl'],
            'arf':['arf_bdc.arf','xti/event_cl'],
            'rmf':['rmf_bdc.rmf','xti/event_cl'],
            'lis':['arf_bdc.lis','xti/event_cl'],
            }

    # Checking files
    # =================================================================
    are_files_there = {}
    files = {}
    for key,item in files_to_check.items():
        are_files_there[key] = False
        files[key] = ''
        target_dir = obs_id_dir/item[1]
        if key == 'mkf':
            ffiles = list_items(target_dir,itype='file',include_and=[item[0]],exclude_and=['mkf2','mkf3'])
        else:
            ffiles = list_items(target_dir,itype='file',include_and=[item[0]])
        
        if key == 'uf': 
            if len(ffiles) == 7:
                are_files_there[key] = True
                files[key] = ffiles
        elif key == 'ufa': 
            if len(ffiles) == 8:
                are_files_there[key] = True
                files[key] = ffiles
        else:
            if len(ffiles) != 0:
                are_files_there[key] = True
                files[key] = ffiles[0]
    # =================================================================
    
    return are_files_there, files

def check_nicer_filtering(filter_file,
    filter_expr = {
        'nicersaafilt':'YES',
        'saafilt':'NO',
        'trackfilt':'YES',
        'ang_dist':0.015,
        'st_valid':'YES',
        'elv':15,
        'br_earth':30,
        'cor_range':'*-*',
        'min_fpm':7,
        'underonly_range':'0-500',
        'overonly_range':'0-1.5',
        'overonly_expr':'1.52*COR_SAX**(-0.633)'
        },
    show_filt_expr=False):
    '''
    Return a dictionary with the percent of photon filtered per 
    filtering criterium using a certain filter expression for NICERL2

    DESCRIPTION
    -----------
    The filter file is a file having a time column with time bins 
    (rows) of 1 second. For each time bin (and so, for each row),
    we have values of different parameters related to the satellite
    oparating condition (like pointing, elevetation, satellite 
    altitude, etc)


    '''
    # (More user friendy description)           Corresponding values
    # nicersaafilt=YES                          --> NICER_SAA==0
    # saafilt=NO                                --> SAA==0
    # trackfilt=YES                             --> ATT_MODE==1 && ATT_SUBMODE_AZ==2 && ATT_SUBMODE_EL==2
    # ang_dist=DIST=0.015                       --> ANG_DIST < DIST
    # st_valid=YES                              --> ST_VALID==1
    # elv=MINELV=15                             --> ELV > MINELV
    # br_earth=MIN_BR_EARTH=30                  --> BR_EARTH > MIN_BR_EARTH
    # min_fpm=MIN_FPM=7                         --> NUM_FPM_ON > MIN_FPM
    # underlonly_range=A-B=0-200                --> FPM_UNDERONLY_COUNT > A && FPM_UNDERONLY_COUNT < B
    # overonly_range=A-B=0-1.0                  --> FPM_OVERONLY_COUNT > A && FPM_OVERONLY_COUNT < B
    # overonly_expr=EXPR=1.52*COR_SAX**(-0.633) --> FPM_OVERONLY_COUNT < EXPR

    mylogging = LoggingWrapper()

    my_ori = {
            'time':'TIME',
            'nicer_saa':'NICER_SAA',
            'saa':'SAA',
            'att_mode':'ATT_MODE',
            'att_mode_az':'ATT_SUBMODE_AZ',
            'att_mode_el':'ATT_SUBMODE_EL',
            'ang_dist':'ANG_DIST',
            'cor_sax':'COR_SAX',
            'elv':'ELV',
            'br_earth':'BR_EARTH',
            'min_fpm_on':'NUM_FPM_ON',
            'st_valid':'ST_VALID',
            'underonly':'FPM_UNDERONLY_COUNT',
            'overonly':'FPM_OVERONLY_COUNT',
            'pointing_ra':'RA',
            'pointing_dec':'DEC'
            }

    if show_filt_expr:
        mylogging.info('Current filtering criteria')
        for key,value in filter_expr.items():
            mylogging.info(f'{key}={value}')

    # Reading make filter file (.mkf, it's a FITS) and storing selected
    # columns in the pars dictionary
    # -----------------------------------------------------------------
    pars = {}
    try:
        with fits.open(filter_file) as hdu_list:
            hdu1_data = hdu_list[1].data

            for key,item in my_ori.items():
                try:
                    pars[key] = hdu1_data[item]
                except:
                    mylogging.info(f'Key {key} does not exist')
                    pars[key] = False
    except:
        mylogging.warning('I could not open the filter file')
        return
    # -----------------------------------------------------------------

    # Extracting time column
    if not pars['time'] is False: 
        time = pars['time']
        before = len(time)
        mylogging.info('There are {} 1s rows (time bins)'.format(before))
    else:
        mylogging.error('I could not find the time column in the filter file')
        return

    # Mask and percent 
    masks = []
    per ={}

    # For each filtering criterium...
    # 1) Check if the corresponding column is the make filter file
    # 2) Check if the filtering criterium is the specified nicerl2
    #    filtering expression
    # 3) Select rows corresponding to the specified filtering value 
    #    making a mask for the time array
    # 4) Appending the mask to a list of masks in order to evaluate
    #    total filtering percentage
    # 5) Compute percent of filtered rows
    # -----------------------------------------------------------------
    if not pars['nicer_saa'] is False:
        key = 'nicersaafilt'
        if key in filter_expr:
            if filter_expr[key] == 'YES':
                mask = pars['nicer_saa'] == 0
                masks += [mask]
                after = len(time[mask])
                per[key] = round(100*(before-after)/before)
            else:
                per[key] = None

    if  not pars['saa'] is False:
        key = 'saafilt'
        if key in filter_expr:
            if filter_expr[key] == 'YES':
                mask = pars['saa'] == 0
                masks += [mask]
                after = len(time[mask])
                per[key] = round(100*(before-after)/before)
            else:
                per[key] = None   

    if (not pars['att_mode'] is False) and \
       (not pars['att_mode_az'] is False) and \
       (not pars['att_mode_el'] is False):
        key = 'trackfilt'
        if key in filter_expr:
            if filter_expr[key] == 'YES':
                mask = (pars['att_mode']==1) & (pars['att_mode_az']==2) \
                    & (pars['att_mode_el']==2)
                masks += [mask]
                after = len(time[mask])
                per[key] = round(100*(before-after)/before)
            else:
                per[key] = None

    if not pars['ang_dist'] is False:
        key = 'ang_dist'
        if key in filter_expr:
            mask = pars['ang_dist'] < filter_expr[key]
            masks += [mask]
            after = len(time[mask])
            per[key] = round(100*(before-after)/before)

    if not pars['st_valid'] is False: 
        key = 'st_valid'
        if key in filter_expr:
            if filter_expr[key] == 'YES':
                mask = pars['st_valid']==1
                masks += [mask]
                after = len(time[mask])
                per[key] = round(100*(before-after)/before)
            else:
                per[key] = None   

    if not pars['elv'] is False:
        key = 'elv'
        if key in filter_expr:
            mask = pars['elv'] > filter_expr[key]
            masks += [mask]
            after = len(time[mask])
            per[key] = round(100*(before-after)/before)

    if not pars['br_earth'] is False:
        key = 'br_earth'
        if key in filter_expr:
            mask = pars['br_earth'] > filter_expr[key]
            masks += [mask]
            after = len(time[mask])
            per[key] = round(100*(before-after)/before)   

    if not pars['min_fpm_on'] is False:
        key = 'min_fpm'
        if key in filter_expr:
            mask = pars['min_fpm_on'] > filter_expr[key]
            masks += [mask]
            after = len(time[mask])
            per[key] = round(100*(before-after)/before)  

    if not pars['cor_sax'] is False:
        key = 'cor_range'
        if key in filter_expr:
            a = filter_expr[key].split('-')[0]
            b = filter_expr[key].split('-')[1]
            if a == '*':
                a = -np.inf
            else:
                a = np.double(a)
            if b == '*':
                b = np.inf
            else:
                b = np.double(b)
            mask = ((pars['cor_sax'] > a) & (pars['cor_sax'] < b)) 
            masks += [mask]
            after = len(time[mask])
            per[key] = round(100*(before-after)/before)             

    if not pars['underonly'] is False:
        key = 'underonly_range'
        if key in filter_expr:
            a = np.double(filter_expr[key].split('-')[0])
            b = np.double(filter_expr[key].split('-')[1])
            mask = ((pars['underonly'] > a) & (pars['underonly'] < b)) 
            masks += [mask]
            after = len(time[mask])
            per[key] = round(100*(before-after)/before)         

    if not pars['overonly'] is False:
        key = 'overonly_range'
        if key in filter_expr:
            a = np.double(filter_expr[key].split('-')[0])
            b = np.double(filter_expr[key].split('-')[1])
            mask = ((pars['overonly'] > a) & (pars['overonly'] < b)) 
            masks += [mask]
            after = len(time[mask])
            per[key] = round(100*(before-after)/before)  

    if not pars['overonly'] is False:
        key = 'overonly_expr'
        if key in filter_expr:
            expr = filter_expr[key]
            vars = get_nn_var(expr)
            if len(vars) != 0:
                with fits.open(filter_file) as hdu_list:
                    hdu1_data = hdu_list[1].data
                    new_vars = []
                    for var in vars:
                        #print('Variable name',var)
                        #print('Done')
                        new_vars += [hdu1_data[var]] 

                new_expr = expr
                for i in range(len(vars)):
                    new_expr = new_expr.replace(vars[i],f'new_vars[{i}]')
                mask = pars['overonly'] < eval(new_expr)
            else:
                mask = pars['overonly'] < eval(expr)

        masks += [mask]
        after = len(time[mask])
        per[key] = round(100*(before-after)/before)  
    # -----------------------------------------------------------------
    
    # Computing total percentage of filtered time
    # -----------------------------------------------------------------
    if len(masks) != 0:
        tot_mask = masks[0]
        for i in range(1,len(masks)):
            tot_mask = tot_mask & masks[i]

        after = len(time[tot_mask])
        per['total'] = round(100*(before-after)/before) 
    # -----------------------------------------------------------------

    if show_filt_expr:
        print('Percent of filtering per criterium:')
        for key,value in per.items():
            print(f'{key}: {value}%')

    return per


