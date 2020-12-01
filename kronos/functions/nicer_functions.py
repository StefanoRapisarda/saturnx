'''
Contain functions specific to NICER data 
'''

import os
from astropy.io import fits
from .my_functions import get_nn_var
import numpy as np

def check_nicer_data(obs_id,home=os.getcwd()):
    '''
    Check NICER folder data structure
    '''

    files = {}

    # Folder structure
    check_dir = {}
    obs_id_dir = os.path.join(home,obs_id)
    check_dir['obs_id_dir'] = os.path.isdir(obs_id_dir)
    auxil_dir = os.path.join(obs_id_dir,'auxil')
    check_dir['auxil'] = os.path.isdir(auxil_dir)
    xti_dir = os.path.join(obs_id_dir,'xti')
    check_dir['xti'] = os.path.isdir(xti_dir)

    event_cl_dir = os.path.join(xti_dir,'event_cl')
    check_dir['event_cl'] = os.path.isdir(event_cl_dir)
    event_uf_dir = os.path.join(xti_dir,'event_uf')
    check_dir['event_uf'] = os.path.isdir(event_uf_dir)    
    hk_dir = os.path.join(xti_dir,'hk')
    check_dir['hk_dir'] = os.path.isdir(hk_dir)  

    public_key = True

    # Files
    check_files = {'att':False,'orb':False,'mkf':False,'mkf2':False,'mkf3':False}
    try:
        auxil_files = next(os.walk(auxil_dir))[2]
        for key in check_files.keys():
            for f in auxil_files:
                if key in f:
                    check_files[key]= True
                    files[key] = os.path.join(auxil_dir,f)
                    continue
    except StopIteration as e:
        print(e)
    
    check_files['hk'] = True
    for i in range(7):
        if not os.path.isfile(os.path.join(hk_dir,f'ni{obs_id}_0mpu{i}.hk.gz')):
            check_files['hk'] = False
    
    if not check_files['hk']:
        test = [fname.endswith('.gpg') for fname in os.listdir(hk_dir)]
        if sum(test) != 0: public_key=False

    check_files['uf'] = True
    first = True
    for i in range(7):
        uf = os.path.join(event_uf_dir,f'ni{obs_id}_0mpu{i}_uf.evt.gz')
        if not os.path.isfile(uf):
            check_files['uf'] = False
        else:
            if first:
                first = False 
                files['uf'] = [uf]
            else:
                files['uf'] += [uf]

    if not check_files['uf']:
        test = [fname.endswith('.gpg') for fname in os.listdir(event_uf_dir)]
        if sum(test) != 0: public_key=False
 

    ufa = os.path.join(event_cl_dir,f'ni{obs_id}_0mpu7_ufa.evt.gz')
    check_files['ufa'] = os.path.isfile(ufa)
    if check_files['ufa']:
        files['ufa'] = ufa
    else:
        files['ufa'] = None

    if not check_files['ufa']:
        test = [fname.endswith('.gpg') for fname in os.listdir(event_cl_dir)]
        if sum(test) != 0: public_key=False

    cl = os.path.join(event_cl_dir,f'ni{obs_id}_0mpu7_cl.evt.gz')
    check_files['cl'] = os.path.isfile(cl)
    if check_files['cl']:
        files['cl'] = cl
    else:
        files['cl'] = None

    if not check_files['cl']:
        test = [fname.endswith('.gpg') for fname in os.listdir(event_cl_dir)]
        if sum(test) != 0: public_key=False

    check_files['public'] = public_key

    
    return check_dir,check_files,files

def check_nicer_filtering(filter_file,
                          filter_expr = {
                                        'nicersaafilt':'YES',
                                        'saafilt':'NO',
                                        'trackfilt':'YES',
                                        'ang_dist':0.015,
                                        'st_valid':'YES',
                                        'elv':13,
                                        'br_earth':30,
                                        'min_fpm':7,
                                        'underonly_range':'0-200',
                                        'overonly_range':'0-1.0',
                                        'overonly_expr':'1.52*COR_SAX**(-0.633)'
                                        },
                          show_filt_expr=False):
    '''
    Return a dictionary with the percent of photon filtered per filtering criterium
    using a certain filter expression for NICERL2
    '''
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
        print('Current filtering criteria')
        for key,value in filter_expr.items():
            print(f'{key}={value}')
        print()

    pars = {}
    try:
        with fits.open(filter_file) as hdu_list:
            hdu1_data = hdu_list[1].data

            for key,item in my_ori.items():
                try:
                    pars[key] = hdu1_data[item]
                except:
                    print(f'Key {key} does not exist')
                    pars[key] = False
    except:
        print('Could not open the header')
        for key,item in my_ori.items():
            pars[key] = False


    if not pars['time'] is False: 
        time = pars['time']
        before = len(time)
        print('There are {} 1s rows'.format(before))
    
    masks = []
    per ={}

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
    
    if len(masks) != 0:
        tot_mask = masks[0]
        for i in range(1,len(masks)):
            tot_mask = tot_mask & masks[i]

        after = len(time[tot_mask])
        per['total'] = round(100*(before-after)/before) 

    if show_filt_expr:
        print('Percent of filtering per criterium:')
        for key,value in per.items():
            print(f'{key}: {value}%')

    return per


def get_counts(spec_file, low_en=0.5, high_en=10., mission='NICER'):
    '''
    Return count rate and corresponding error in selected energy range
    read an energy spectrum in FITS file.

    PARAMETERS
    ----------
    spec_file: str
        Full path of a .pha file (the one that you would feed to XSPEC)
    low_en: float
        Lowest energy channel in keV
    high_en: float
        Highest energy channel in keV
    mission: str
        Mission identifier

    HISTORY
    -------
    2020 10 16, Stefano Rapisarda (Uppsala), creation date
    2020 11 30, Stefano Rapisarda (Uppsala)
        changed channels to energy and introduced the mission parameter

    TODO:
    - 2020 10 16, Stefano Rapisarda (Uppsala)
        this should be moved to xray functions, for now is here as it
        works only with python
    '''

    factor = 1
    if mission == 'NICER':
        factor = 100

    low_cha = low_en*factor
    high_cha = high_en*factor

    with fits.open(spec_file) as hdul:
        data = hdul[1].data
        cha = data['CHANNEL']
        rate = data['RATE']
        err = data['STAT_ERR']
        
    mask = (cha >= low_cha) & (cha <= high_cha)
    cr = rate[mask].sum()
    cr_err = np.sqrt(((err[mask])**2).sum())

    return cr,cr_err

