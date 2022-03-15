import os
import math
from os import path
import pathlib
import logging
import pickle
import numpy as np
import pandas as pd
import gc

from astropy.time import Time

from PyPDF2 import PdfFileMerger, PdfFileReader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib import pyplot

from saturnx.core.gti import Gti
from saturnx.core.lightcurve import LightcurveList
from saturnx.core.power import PowerList
from saturnx.utils.my_logging import make_logger,get_logger_name,LoggingWrapper
from saturnx.utils.generic import str_title
from saturnx.wavelets.wavelets import WaveletTransform
from saturnx.wavelets.functions import comp_scales
from saturnx.utils.pdf import pdf_page

def make_wlt_single(obs_id_dir,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['0.5','10.0'], 
    wt_families = ['mexhat','morlet'],
    rebin=10,min_scale = None, max_scale = None, dj = 0.05,
    save_all = False):
    '''
    Computes mexhat and morlet wavelet transform from a lightcurve list 
    object per lightcurve in the list

    The lightcurve list must be named as 
    lc_list_E{low_en}_{high_en}_T{tres}_{tseg}

    HISTORY
    -------
    21 04 2021, Stefano Rapisarda (Uppsala), creation date
        I used the same model of NICER standard products, with the 
        difference that here products (wavelet transform), plots (jpeg),
        and PDF pages are created per segment inside the loop.   
    '''

    mylogging = LoggingWrapper()

    logging.info('*'*72)
    logging.info(str_title('make_wt_single'))
    logging.info('*'*72+'\n')
    
    if type(obs_id_dir) == str: obs_id_dir = pathlib.Path(obs_id_dir)
    obs_id = obs_id_dir.name
    
    # Making folders (1 for plots, 1 for products)
    # -----------------------------------------------------------------
    wt_plot_dir = obs_id_dir/'wt_plots'
    if not wt_plot_dir.is_dir():
        mylogging.info('wt_plots does not exist, creating one...')
        os.mkdir(wt_plot_dir)
    else:
        mylogging.info('wt_plots already exists.')
        
    wt_prod_dir = obs_id_dir/'wt_prods'
    if not wt_prod_dir.is_dir():
        mylogging.info('wts does not exist, creating one...')
        os.mkdir(wt_prod_dir)
    else:
        mylogging.info('wts already exists.')   
    # -----------------------------------------------------------------
    
    # Defining names of files to read (LightcurveList file)
    # -----------------------------------------------------------------
    main_prod_name = 'E{}_{}_T{}_{}'.\
        format(main_en_band[0],main_en_band[1],tres,tseg)
    main_lc_list_file = obs_id_dir/'lc_list_{}.pkl'.\
        format(main_prod_name)
    # -----------------------------------------------------------------
    
    # Printing some info
    # -----------------------------------------------------------------
    mylogging.info('')
    mylogging.info('Obs ID: {}'.format(obs_id))
    mylogging.info('Settings:')
    mylogging.info('-'*60)
    mylogging.info('Selected main energy band: {}-{} keV'.\
        format(main_en_band[0],main_en_band[1]))     
    mylogging.info('Selected time resolution: {} s'.format(tres)) 
    mylogging.info('Selected time segment: {} s'.format(tseg)) 
    mylogging.info('Selected wavelets:')
    for wt_family in wt_families:
        mylogging.info('- {}'.format(wt_family))
    mylogging.info('Selected LightcurveList file: {}'.\
        format(main_lc_list_file))
    mylogging.info('-'*60)
    mylogging.info('')
    # -----------------------------------------------------------------

    # Checking if the pain is worth
    if not main_lc_list_file.is_file():
        mylogging.error('main_lc_list_file {} does not exist'.\
            format(main_lc_list_file))
        return
    
    # Computing the Wavelet Transform per segment, plotting, and 
    # printing
    # =================================================================

    # Initializing single PDF file for the entire obs_ID
    merger = PdfFileMerger()

    # Loading LightcurveList file
    main_lc_list = LightcurveList.load(main_lc_list_file)

    # In this lightcurve list file each lightcurve is supposed to have 
    # the same length
    # LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOP ==> over lc START!
    for i,lc in enumerate(main_lc_list):

        mylogging.info('Processing lc segment {}/{}\n'.\
            format(i+1,len(main_lc_list)))
        
        # Extracting segment infomation for printing
        # -------------------------------------------------------------
        n_gtis = lc.meta_data['N_GTIS']
        gti_i = lc.meta_data['GTI_INDEX']+1
        n_segs = lc.meta_data['N_SEGS']
        seg_i = lc.meta_data['SEG_INDEX']+1

        local_bookmark_text = 'GTI_{}_SEG_{}'.format(gti_i,seg_i)
        # -------------------------------------------------------------

        newtres = lc.tres*rebin

        # Defining wavelet scale
        # -------------------------------------------------------------
        if min_scale is None: min_scale = newtres*4
        if max_scale is None: max_scale = np.round(lc.texp/2,1)
        nscales = len(comp_scales(min_scale,max_scale,dj))
        # -------------------------------------------------------------

        # Defining upper text for pdf page
        # -------------------------------------------------------------
        text1 = 'obs_ID: {}, GTI: {}/{}, seg_index: {}/{}'.\
            format(obs_id, gti_i,n_gtis,seg_i,n_segs)
        text2 = 't_res: {}, t_exp: {}, min_s: {}, max_s: {}'.\
            format(newtres, np.round(lc.texp,1), min_scale, max_scale)
        # -------------------------------------------------------------

        wlt_root_name = '{}_S{}_{}_{}'.\
            format(main_prod_name,min_scale,max_scale,nscales)
        
        # Creating local wt plot folders specific for these settings
        # -------------------------------------------------------------
        if not (wt_plot_dir/wlt_root_name).is_dir():
            mylogging.info('wt single plot folder does not exist, creating one...')
            os.mkdir(wt_plot_dir/wlt_root_name) 
        else:
            mylogging.info('wt single plot folder already exists.')   
        # -------------------------------------------------------------

        # Initializing temporary PDF page
        pdf = pdf_page(margins=[7,20,7,0])

        # LOOOOOOOOOOOOOOOOOOOOOOOOOOOP ==> over wavelet families START!
        for wt_family in wt_families:

            wlt_plot_name = 'wlt_{}_{}.jpeg'.format(wt_family,i+1)
            
            if not (wt_plot_dir/wlt_root_name/wlt_plot_name).is_file():

                # Rebinning lightcurve
                lc_rebin = lc.rebin(rebin)
                
                # Computing wavelet or loading
                # -----------------------------------------------------
                wlt_file_name = 'wlt_{}_S{}_{}_{}_{}_{}.pkl'.\
                    format(main_prod_name,min_scale,max_scale,nscales,wt_family,i+1)
                if not (wt_prod_dir/wlt_root_name/wlt_file_name).is_file():
                    mylogging.info('Computing {} wavelet ...'.format(wt_family))
                    wltmh = WaveletTransform.\
                        from_lc(lc_rebin,s_min=min_scale,s_max=max_scale,dj=dj,family=wt_family)

                    # Saving
                    if save_all:
                        mylogging.info('Saving ...')
                        wltmh.save(file_name=wlt_file_name,fold=wt_prod_dir/wlt_root_name)
                        mylogging.info('... and done!')
                else:
                    mylogging.info('Loading {} wavelet ...'.format(wt_family))
                    wltmh = WaveletTransform.load(file_name=wlt_file_name,fold=wt_prod_dir/wlt_root_name)
                    mylogging.info('... and done!')
                # -----------------------------------------------------
            
                # Plotting
                # -----------------------------------------------------
                # Plotting wavelet
                fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(8.25*1.5,11.75*1.4))
                plt.subplots_adjust(hspace=0.15)

                # Standard wavelet power
                ax1.set_title('{}'.format(wt_family))
                ax1,im1=wltmh.plot(ax=ax1,norm=None,cmap=plt.cm.plasma)
                fig.colorbar(im1, ax=ax1)
                ax1.set_xlabel('')

                # Standard wavelet power with binary color map
                ax2,im2=wltmh.plot(ax=ax2,norm=None,cmap=plt.cm.binary)
                fig.colorbar(im2, ax=ax2)
                ax2.set_xlabel('')

                # Normalized wavelet power
                ax3.set_title('{} normalized'.format(wt_family))
                ax3,im3=wltmh.plot(ax=ax3,cmap=plt.cm.plasma)
                fig.colorbar(im3, ax=ax3)
                ax3.set_xlabel('')

                # Normalized wavelet power with binary color map
                ax4,im4=wltmh.plot(ax=ax4,cmap=plt.cm.binary)
                fig.colorbar(im4, ax=ax4)
                
                # Saving
                fig.savefig(str(wt_plot_dir/wlt_root_name/wlt_plot_name), 
                    dpi=300, bbox_inches = 'tight',pad_inches = 0)
                
                # Cleaning
                pyplot.clf() # This is supposed to clean the figure
                plt.close(fig)
                gc.collect()
                # -----------------------------------------------------
           
            # Printing plot in the temporary pdf page
            # ---------------------------------------------------------
            pdf.add_page()
            coors = pdf.get_grid_coors(grid=[1,1,0,0])
            pdf.print_text(text=text1,xy=(5,5),fontsize=12)
            pdf.print_text(text=text2,xy=(5,10),fontsize=12)
            pdf.image(str(wt_plot_dir/wlt_root_name/wlt_plot_name),
                x=coors[0],y=coors[1],w=coors[2]-coors[0])
            # ---------------------------------------------------------
        # LOOOOOOOOOOOOOOOOOOOOOOOOOOOOP ==> over wavelet families STOP!

        # Saving the temporary PDF file
        pdf_name = wt_plot_dir/wlt_root_name/'tmp.pdf'
        pdf.output(str(pdf_name),'F')

        # Adding temporary PDF file to main PDF file
        # This allows to make parent and child bookmarks
        # ---------------------------------------------------------------
        with open(pdf_name,'rb') as infile:
            merger.append(PdfFileReader(infile))
            if i == 0:
                parent = merger.addBookmark(obs_id,0)
                merger.addBookmark(local_bookmark_text,int(i*2),parent=parent)
            else:
                merger.addBookmark(local_bookmark_text,int(i*2),parent=parent)
        # ---------------------------------------------------------------

        # Removing temporary pdf file
        os.system('rm {}'.format(pdf_name))

    # LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOP ==> over lc STOP!
    
    # Writing main PDF file
    pdf_file_name = wt_plot_dir/wlt_root_name/'{}_wlt_{}_S_{}_{}_{}.pdf'.\
            format(obs_id,main_prod_name,min_scale,max_scale,nscales)
    merger.write(str(pdf_file_name))

    return pdf_file_name

def make_wlt_prods(obs_id_dirs,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['0.5','10.0'], 
    wt_families = ['mexhat','morlet'],
    rebin=10,min_scale = None, max_scale = None, dj = 0.05,
    save_all = False):

    mylogging = LoggingWrapper()

    first_obs_id_dir = obs_id_dirs[0]
    if type(first_obs_id_dir) == str:
        first_obs_id_dir = pathlib.Path(first_obs_id_dir)
    an_dir = first_obs_id_dir.parent
 
    for o,obs_id_dir in enumerate(obs_id_dirs):

        obs_id = obs_id_dir.name

        mylogging.info('Processing obs. ID: {} ({}/{})'.\
            format(obs_id,o+1,len(obs_id_dirs)))

        if isinstance(obs_id_dir,str):
            obs_id_dir = pathlib.Path(obs_id_dir)

        make_wlt_single(obs_id_dir,tres=tres,tseg=tseg,
            main_en_band=main_en_band,wt_families=wt_families,
            rebin=rebin,min_scale=min_scale,max_scale=max_scale,dj=dj,
            save_all=save_all)

        

        