import os
from astropy.io import fits
from .my_functions import clean_gti
import logging

def list_modes(obs_id_dir=os.getcwd()):
    '''
    List all the available observing modes in an obs ID

    HISTORY
    -------
    2017 10 13, Stefano Rapisarda (Amsterdam), creation date
    2019 05 27, Stefano Rapisarda (Amsterdam), cleaning up
    2020 11 16, Stefano Rapisarda (Uppsala)
        Cleaned up, corrected comments, added some security asserts,
        removed the printing on a file option.
        Now the function returns a dictionary with mode names (as they
        appear on the FMI file) as keys and another dictionary as item.
        This last dictionary contains the mode specific name, its 
        corresponding file (assuming the obs ID is the root folder),
        and start and stop time of CLEANED GTIs (so GTIs sorted 
        according to start and stop times and merged, if necessary)
    '''

    obs_id = os.path.basename(obs_id_dir)

    # Finding the name of the PCA_Index_File, the fits file listing
    # all start and stop times for every observing mode
    fmi = os.path.join(obs_id_dir,'FMI')
    assert os.path.isfile(fmi),'FMI file not found in {}'.format(obs_id)
    try:
        with fits.open(fmi) as hdu_list:
            pca_index_name = hdu_list[1].data['PCA_Index_File'][0]
    except Exception as e:
        print('Could not open FMI file')
        print(e)
        return {}
    pca_index_file = os.path.join(obs_id_dir,pca_index_name)
    assert os.path.isfile(pca_index_file),\
        'PCA_Index_File does not exist in {}'.format(obs_id)

    # Opening PCA index file
    with fits.open(pca_index_file) as hdu_list:
        tbdata = hdu_list[1].data

    # The first column (index 0) containes the row index
    # Each observational mode has its own column. For each column, rows
    # contain GTIs. If column, so an observational mode, exists, the 
    # number of non empty rows corresponds to the number of GTIs

    # RXTE can observe a maximum of 7 modes at once, their names (in 
    # the PCS Index File) in an index from 1 to 7

    modes = {}
    for i in range(1,8):

        modeID = 'EA' + str(i) + 'ModeId' 
        modename = 'EA' + str(i) + 'ModeNm' # Name of the mode
        modefile =  'EA' + str(i) + 'Data'  # Mode file in the obsID folder

        try: 
            tbdata[modename] 
        except KeyError:
            logging.warning(f'I could not find mode {modename} '\
                            'maybe it does not exist')
            modes[modename] = None
            continue

        # The column EA<i>ModeId is not zero only when the mode is active
        active_mode_mask = tbdata[modeID] != 0
        if active_mode_mask.sum() != 0:
            active_modename = tbdata[modename][active_mode_mask]
            assert len(set(active_modename))==1, 'Active mode names'\
                ' in a single column are different...that is warring'
            active_modefile = tbdata[modefile][active_mode_mask]
            assert len(set(active_modefile))==1, 'Active mode files'\
                ' in a single column are different...that is warring'           
            active_mode_start = tbdata['BeginMET'][active_mode_mask]
            active_mode_stop = tbdata['EndMET'][active_mode_mask]
        else:
            logging.warning('Mode {} does not have GTIs'.format(modename))   

        # GTIs can be overlapping or adjacent, so...
        clean_mode_start, clean_mode_stop = clean_gti(
                active_mode_start,active_mode_stop)

        info = {'modename':active_modename[0],
                'modefile':active_modefile[0],
                'mode_start':clean_mode_start,
                'mode_stop':clean_mode_stop}

        modes[modename] = info

    return modes

