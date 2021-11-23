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
from kronos.utils.my_logging import make_logger,get_logger_name
from kronos.wavelets.wavelets import WaveletTransform
from kronos.wavelets.functions import comp_scales
from kronos.utils.pdf import pdf_page

def make_wt_single(obs_id_dir,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['0.5','10.0'],
    rebin=10,min_scale = None, max_scale = None, dj = 0.05,
    log_name=None):
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
    
    if type(obs_id_dir) == str: obs_id_dir = pathlib.Path(obs_id_dir)

    # Logging
    if log_name is None:
        log_name = get_logger_name('make_nicer_wt_single')
        make_logger(log_name,outdir=obs_id_dir)
  
    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'make_nicer_wt_single','*'*24))
    logging.info('*'*72)

    obs_id = obs_id_dir.name
    
    # Making folders 
    # -----------------------------------------------------
    wt_plot_dir = obs_id_dir/'wt_plots'
    if not wt_plot_dir.is_dir():
        logging.info('wt_plots does not exist, creating one...')
        os.mkdir(wt_plot_dir)
    else:
        logging.info('wt_plots already exists.')
        
    wt_prod_dir = obs_id_dir/'wts'
    if not wt_prod_dir.is_dir():
        logging.info('wts does not exist, creating one...')
        os.mkdir(wt_prod_dir)
    else:
        logging.info('wts already exists.')   
    # -----------------------------------------------------
    
    # Defining names of files to read
    # -----------------------------------------------------
    main_prod_name = 'E{}_{}_T{}_{}'.format(main_en_band[0],main_en_band[1],tres,tseg)
    main_lc_list_file = obs_id_dir/'lc_list_{}.pkl'.format(main_prod_name)
    # -----------------------------------------------------
    
    # Printing some info
    # -----------------------------------------------------------------
    logging.info('')
    logging.info('Obs ID: {}'.format(obs_id))
    logging.info('Settings:')
    logging.info('-'*60)
    logging.info('Selected main energy band: {}-{} keV'.\
        format(main_en_band[0],main_en_band[1]))     
    logging.info('Selected time resolution: {} s'.format(tres)) 
    logging.info('Selected time segment: {} s'.format(tseg)) 
    logging.info('Log file name: {}'.format(log_name))
    logging.info('-'*60)
    logging.info('')
    # -----------------------------------------------------------------
    
    # Computing the Wavelet Transform per segment, plotting, and printing
    # -----------------------------------------------------------------
    
    # Setting PDF page
    pdf = pdf_page(margins=[7,20,7,0])

    if not main_lc_list_file.is_file():
        logging.info('main_lc_list_file {} does not exist'.format(main_lc_list_file))
        return
        
    main_lc_list = LightcurveList.load(main_lc_list_file)
    # In this lightcurve list file each lightcurve is supposed to have the same length
    for i,lc in enumerate(main_lc_list):
        
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
        wlt_plot_name = 'wlt_{}_{}.jpeg'.format(wlt_gen_name,i+1)

        if not (wt_plot_dir/wlt_gen_name).is_dir():
            logging.info('wt single plot folder does not exist, creating one...')
            os.mkdir(wt_plot_dir/wlt_gen_name)    

        if not (wt_prod_dir/wlt_gen_name).is_dir():
            logging.info('wt single prod folder does not exist, creating one...')
            os.mkdir(wt_prod_dir/wlt_gen_name)    

        if not (wt_plot_dir/wlt_gen_name/wlt_plot_name).is_file():

            # Rebinning lightcurve
            lc_rebin = lc.rebin(rebin)
            
            # Computing Mexican hat wavelet
            wltmh_file_name = 'wlt_{}_S{}_{}_{}_mexhat_{}.pkl'.\
                format(main_prod_name,min_scale,max_scale,nscales,i+1)
            if not (wt_prod_dir/wlt_gen_name/wltmh_file_name).is_file():
                wltmh = WaveletTransform.\
                    from_lc(lc_rebin,s_min=min_scale,s_max=max_scale,dj=dj,family='mexhat')
                wltmh.save(file_name=wltmh_file_name,fold=wt_prod_dir/wlt_gen_name)
            else:
                wltmh = WaveletTransform.load(file_name=wltmh_file_name,fold=wt_prod_dir/wlt_gen_name)
                
            # Computing Morlet wavelet
            wltmo_file_name = 'wlt_{}_S{}_{}_{}_morlet_{}.pkl'.\
                format(main_prod_name,min_scale,max_scale,nscales,i+1)
            if not (wt_prod_dir/wlt_gen_name/wltmo_file_name).is_file():
                wltmo = WaveletTransform.\
                    from_lc(lc_rebin,s_min=min_scale,s_max=max_scale,dj=dj,family='morlet')
                wltmo.save(file_name=wltmo_file_name,fold=wt_prod_dir/wlt_gen_name)
            else:
                wltmo = WaveletTransform.load(file_name=wltmh_file_name,fold=wt_prod_dir/wlt_gen_name) 

            # Plotting
            # -------------------------------------------------------------
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(8.25*1.5,11.75*1.4))
            plt.subplots_adjust(hspace=0.15)

            ax1.set_title('Mexican hat normalized')
            im1=wltmh.plot(ax=ax1)
            fig.colorbar(im1, ax=ax1)
            ax1.set_xlabel('')

            ax2.set_title('Mexican hat original')
            im2=wltmh.plot(ax=ax2,norm=None)
            fig.colorbar(im2, ax=ax2)
            ax2.set_xlabel('')

            ax3.set_title('Morlet normalized')
            im3=wltmo.plot(ax=ax3)
            fig.colorbar(im3, ax=ax3)
            ax3.set_xlabel('')

            ax4.set_title('Morlet original')
            im4=wltmo.plot(ax=ax4,norm=None)
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