'''
Contain functions specific to NICER data 
'''

import os
from astropy.io import fits
from .my_functions import get_nn_var
import numpy as np

all_det = np.array([0,1,2,3,4,5,6,
                10,11,12,13,14,15,7,
                20,21,22,23,24,25,16,
                30,31,32,33,34,26,17,
                40,41,42,43,44,35,27,
                50,51,52,53,45,36,37,
                60,62,64,66,54,46,47,
                61,63,65,67,55,56,57])

def run_nicerl2(obs_id_dir,tasks="ALL",filtcolumns="NICERV3,3C50",
        niprefilter2="YES",saafilt="NO",nicersaafilt="YES",
        ang_dist="0.015",elv="15",br_earth="30",cor_range="*-*",
        underonly_range="0-500",overonly_range="0-1.5",
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
    '''

    command = f'nicerl2 indir="{str(obs_id_dir)}" tasks="{tasks}" \
        filtcolumns="{filtcolumns}" niprefilter2="{niprefilter2}" \
        saafilt="{saafilt}" nicersaafilt="{nicersaafilt}" \
        ang_dist="{ang_dist}" elv="{elv}" br_earth="{br_earth}" \
        cor_range="{cor_range}" underonly_range="{underonly_range}" \
        overonly_range="{overonly_range}" overonly_expr="{overonly_expr}" \
        clobber="{clobber}"> {log_file}'
    
    result = True
    try:
        os.system(command)
    except:
        retult = False
    
    return result
        

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


