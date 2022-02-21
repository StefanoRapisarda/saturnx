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

from kronos.core.gti import Gti
from kronos.core.lightcurve import LightcurveList
from kronos.core.power import PowerList
from kronos.utils.my_logging import make_logger,get_logger_name,LoggingWrapper
from kronos.utils.generic import str_title
from kronos.wavelets.wavelets import WaveletTransform
from kronos.wavelets.functions import comp_scales
from kronos.utils.pdf import pdf_page

def make_wt_single(obs_id_dir,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['0.5','10.0'], 
    wt_families = ['mexhat','morlet'],
    rebin=10,min_scale = None, max_scale = None, dj = 0.05):
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
    
    if type(obs_id_dir) == str: obs_id_dir = pathlib.Path(obs_id_dir)
  
    logging.info('*'*72)
    logging.info(str_title('make_wt_single'))
    logging.info('*'*72+'\n')

    obs_id = obs_id_dir.name
    
    # Making folders 
    # -----------------------------------------------------
    wt_plot_dir = obs_id_dir/'wt_plots'
    if not wt_plot_dir.is_dir():
        mylogging.info('wt_plots does not exist, creating one...')
        os.mkdir(wt_plot_dir)
    else:
        mylogging.info('wt_plots already exists.')
        
    wt_prod_dir = obs_id_dir/'wts'
    if not wt_prod_dir.is_dir():
        mylogging.info('wts does not exist, creating one...')
        os.mkdir(wt_prod_dir)
    else:
        mylogging.info('wts already exists.')   
    # -----------------------------------------------------
    
    # Defining names of files to read
    # -----------------------------------------------------
    main_prod_name = 'E{}_{}_T{}_{}'.format(main_en_band[0],main_en_band[1],tres,tseg)
    main_lc_list_file = obs_id_dir/'lc_list_{}.pkl'.format(main_prod_name)
    # -----------------------------------------------------
    
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
    mylogging.info('-'*60)
    mylogging.info('')
    # -----------------------------------------------------------------
    
    # Computing the Wavelet Transform per segment, plotting, and printing
    # -----------------------------------------------------------------
    
    # Setting PDF page
    pdf = pdf_page(margins=[7,20,7,0])

    if not main_lc_list_file.is_file():
        mylogging.error('main_lc_list_file {} does not exist'.format(main_lc_list_file))
        return
        
    main_lc_list = LightcurveList.load(main_lc_list_file)
    # In this lightcurve list file each lightcurve is supposed to have the same length
    for i,lc in enumerate(main_lc_list):

        mylogging.info('Processing lc segment {}/{}\n'.format(i+1,len(main_lc_list)))
        
        # Extracting segment infomation for printing
        n_gtis = lc.meta_data['N_GTIS']
        gti_i = lc.meta_data['GTI_INDEX']+1
        n_segs = lc.meta_data['N_SEGS']
        seg_i = lc.meta_data['SEG_INDEX']+1

        newtres = lc.tres*rebin
        # Defining wavelet scale
        if min_scale is None: min_scale = newtres*4
        if max_scale is None: max_scale = np.round(lc.texp/2,1)

        nscales = len(comp_scales(min_scale,max_scale,dj))

        # Defining upper text for pdf page
        text1 = 'obs_ID: {}, GTI: {}/{}, seg_index: {}/{}'.\
            format(obs_id, gti_i,n_gtis,seg_i,n_segs)
        text2 = 't_res: {}, t_exp: {}, min_s: {}, max_s: {}'.\
            format(newtres, np.round(lc.texp,1), min_scale, max_scale)

        wlt_gen_name = '{}_S{}_{}_{}'.format(main_prod_name,min_scale,max_scale,nscales)
        
        if not (wt_plot_dir/wlt_gen_name).is_dir():
            mylogging.info('wt single plot folder does not exist, creating one...')
            os.mkdir(wt_plot_dir/wlt_gen_name) 
        else:
            mylogging.info('wt single plot folder already exists.')   

        for wt_family in wt_families:

            wlt_plot_name = 'wlt_{}_{}.jpeg'.format(wt_family,i+1)
            
            if not (wt_plot_dir/wlt_gen_name/wlt_plot_name).is_file():

                # Rebinning lightcurve
                lc_rebin = lc.rebin(rebin)
                
                # Computing wavelet
                wlt_file_name = 'wlt_{}_S{}_{}_{}_{}_{}.pkl'.\
                    format(main_prod_name,min_scale,max_scale,nscales,wt_family,i+1)
                if not (wt_prod_dir/wlt_gen_name/wlt_file_name).is_file():
                    mylogging.info('Loading {} wavelet ...'.format(wt_family))
                    wltmh = WaveletTransform.\
                        from_lc(lc_rebin,s_min=min_scale,s_max=max_scale,dj=dj,family=wt_family)
                    mylogging.info('Saving ...')
                    wltmh.save(file_name=wlt_file_name,fold=wt_prod_dir/wlt_gen_name)
                    mylogging.info('... and done!')
                else:
                    mylogging.info('Loading {} wavelet ...'.format(wt_family))
                    wltmh = WaveletTransform.load(file_name=wlt_file_name,fold=wt_prod_dir/wlt_gen_name)
                    mylogging.info('... and done!')
            
                # Plotting
                # -------------------------------------------------------------
                # Plotting wavelet
                fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(8.25*1.5,11.75*1.4))
                plt.subplots_adjust(hspace=0.15)

                ax1.set_title('{}'.format(wt_family))
                ax1,im1=wltmh.plot(ax=ax1,norm=None,cmap=plt.cm.plasma)
                fig.colorbar(im1, ax=ax1)
                ax1.set_xlabel('')

                ax2,im2=wltmh.plot(ax=ax2,norm=None,cmap=plt.cm.binary)
                fig.colorbar(im2, ax=ax2)
                ax2.set_xlabel('')

                ax3.set_title('{} normalized'.format(wt_family))
                ax3,im3=wltmh.plot(ax=ax3,cmap=plt.cm.plasma)
                fig.colorbar(im3, ax=ax3)
                ax3.set_xlabel('')

                ax4,im4=wltmh.plot(ax=ax4,cmap=plt.cm.binary)
                fig.colorbar(im4, ax=ax4)
                
                fig.savefig(str(wt_plot_dir/wlt_gen_name/wlt_plot_name), dpi=300, bbox_inches = 'tight',
                    pad_inches = 0)

                pyplot.clf() # This is supposed to clean the figure
                plt.close(fig)
                gc.collect()
                # -------------------------------------------------------------
           
                # Printing plot in a pdf page
                pdf.add_page()
                coors = pdf.get_grid_coors(grid=[1,1,0,0])
                pdf.print_text(text=text1,xy=(5,5),fontsize=12)
                pdf.print_text(text=text2,xy=(5,10),fontsize=12)
                pdf.image(str(wt_plot_dir/wlt_gen_name/wlt_plot_name),x=coors[0],y=coors[1],w=coors[2]-coors[0])
    # -----------------------------------------------------------------

    # Saving the PDF file
    pdf_name = wt_plot_dir/wlt_gen_name/'{}_wlt_{}_S_{}_{}_{}.pdf'.\
        format(obs_id,main_prod_name,min_scale,max_scale,nscales)
    pdf.output(str(pdf_name),'F')