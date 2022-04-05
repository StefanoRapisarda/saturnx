import os
import math
from os import path
import pathlib
import logging
import pickle
import numpy as np
import pandas as pd

from astropy.time import Time
from datetime import datetime

from PyPDF2 import PdfFileMerger, PdfFileReader

from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import xspec
from sherpa.astro import ui

from saturnx.core.gti import Gti
from saturnx.core.event import Event
from saturnx.core.lightcurve import Lightcurve, LightcurveList
from saturnx.core.power import PowerList, PowerSpectrum
from saturnx.utils.my_logging import make_logger, LoggingWrapper
from saturnx.utils.generic import chunks, my_cdate, str_title
from saturnx.utils.pdf import pdf_page
from saturnx.utils.nicer_functions import all_det
from saturnx.utils.xray import get_cr
from saturnx.utils.my_functions import list_items

def make_hxmt_std_prod_single(obs_id_dir,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['52.6','188.4'], spec_en_ch = ['26','120'], en_bands = [],
    rebin=-30,data_dir=pathlib.Path.cwd()):

    mylogging = LoggingWrapper()

    colors = ['red','blue','green','orange','brown']
    markers = ['s','^','p','H','X']
    if len(en_bands) > len(colors):
        raise ValueError('Energy bands cannot be more than 5')

    if type(obs_id_dir) == str: obs_id_dir = pathlib.Path(obs_id_dir)
    if type(data_dir) == str: data_dir = pathlib.Path(data_dir)

    mylogging.info('*'*72)
    mylogging.info(str_title('make_hxmt_std_prod_single'))
    mylogging.info('*'*72+'\n')

    obs_id = obs_id_dir.name

    # Making directories
    # -----------------------------------------------------    
    std_plot_dir = obs_id_dir/'std_plots'
    if not std_plot_dir.is_dir():
        logging.info('std_plots directory does not exist, creating one...')
        os.mkdir(std_plot_dir)
    else:
        logging.info('std_plots directory already exists.')
    # -----------------------------------------------------

    # Defining names of files to read
    # -----------------------------------------------------
    
    # Main files (Full energy band or energy band to highlight)
    main_prod_name = 'E{}_{}_T{}_{}'.format(main_en_band[0],main_en_band[1],tres,tseg)
    bkg_prod_name = 'E{}_{}_T{}'.format(main_en_band[0],main_en_band[1],tres)

    main_gti_name = 'gti_E{}_{}.gti'.format(main_en_band[0],main_en_band[1])
    gti_lc_list_file = obs_id_dir/'lc_list_E{}_{}_T{}.pkl'.\
        format(main_en_band[0],main_en_band[1],tres)
    main_lc_list_file = obs_id_dir/'lc_list_{}.pkl'.format(main_prod_name)
    bkg_lc_list_file = obs_id_dir/'lc_list_{}_bkg.pkl'.format(bkg_prod_name)
    main_pw_list_file = obs_id_dir/'power_list_{}.pkl'.format(main_prod_name)
    
    # Other energy bands
    lc_list_files = []
    gti_lc_list_files = []
    power_list_files = []
    bkg_list_files = []
    for en_band in en_bands:
        low_en, high_en = en_band[0], en_band[1]
        lc_list_files += [obs_id_dir/'lc_list_E{}_{}_T{}_{}.pkl'.\
                                        format(low_en,high_en,tres,tseg)]
        gti_lc_list_files += [obs_id_dir/'lc_list_E{}_{}_T{}.pkl'.\
                                        format(low_en,high_en,tres)]
        power_list_files += [obs_id_dir/'power_list_E{}_{}_T{}_{}.pkl'.\
                                        format(low_en,high_en,tres,tseg)]
        bkg_list_files += [obs_id_dir/'lc_list_E{}_{}_T{}_{}_bkg.pkl'.\
                                        format(low_en,high_en,tres,tseg)]
    
    # Energy spectra
    ch1, ch2 = spec_en_ch[0],spec_en_ch[1]
    spec = data_dir/obs_id/'total_spectrum_CH{}_{}.pi'.format(ch1,ch2)
    bkg_spec = data_dir/obs_id/'total_spectrum_bkg_CH{}_{}.pi'.format(ch1,ch2)
    rsp = data_dir/obs_id/'total_spectrum_rsp_CH{}_{}.pi'.format(ch1,ch2)
    # ------------------------------------------------------

    # Printing some info
    # -----------------------------------------------------------------
    logging.info('')
    logging.info('Obs ID: {}'.format(obs_id))
    logging.info('Settings:')
    logging.info('-'*72)
    logging.info('Selected main energy band: {}-{} keV'.\
        format(main_en_band[0],main_en_band[1]))
    for i,en_band in enumerate(en_bands):
        logging.info('Selected energy band {}: {}-{} keV'.\
            format(i,en_band[0],en_band[1]))        
    logging.info('Selected time resolution: {} s'.format(tres)) 
    logging.info('Selected time segment: {} s'.format(tseg)) 
    logging.info('Selected channels for energy spectrum: {}-{}'.\
        format(ch1,ch2))
    logging.info('Log file name: {}'.format(log_name))
    logging.info('-'*72)
    logging.info('')
    # -----------------------------------------------------------------

    # Plotting
    # =======================================================
    
    plots = []
    plt.tight_layout()
    # I create the figure anyway, then, if the file does not exists, 
    # the figure will be empty


    # Plot1: count rate per GTI
    # ------------------------------------------------------
    logging.info('Plotting count rate per GTI')
    fig,ax = plt.subplots(figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Count rate per GTI',fontsize=14)
    
    # Full energy band count rate
    if gti_lc_list_file.is_file() and bkg_lc_list_file.is_file():
        # Reading lightcurves
        main_lc_list = LightcurveList.load(main_lc_list_file)
        bkg_lc_list = LightcurveList.load(bkg_lc_list_file)

        main_cr = main_lc_list.cr
        main_cr_err = main_lc_list.cr_std
        bkg_cr = bkg_lc_list.cr
        bkg_cr_err = bkg_lc_list.cr_std        

        # Subtracting background
        bkg_sub_lc_list = LightcurveList(
            [lc-np.mean(bkg.counts) for lc,bkg in zip(main_lc_list,bkg_lc_list)]
            )

        # Plotting
        bkg_sub_lc_list.plot(ax=ax,color='k',lfont=14,label='{}-{}'.\
                     format(main_en_band[0],main_en_band[1]),ndet=False)

        # Drawing vertical line per GTI
        gti = Gti.load(obs_id_dir/main_gti_name)
        gti_seg = gti>=float(tseg)
        start = main_lc_list[0].time.iloc[0]
        for g in gti_seg.stop.to_numpy():
            ax.axvline(g-start,ls='--',color='orange')
    else:
        logging.info('Main energy band lc_list file not found')
        logging.info(main_lc_list_file)
       
    # Plotting energy bands count rate
    other_crs = []
    other_crs_err = []
    for lc_list_file, en_band, col, marker in zip(lc_list_files, en_bands, colors, markers):
        low_en, high_en = en_band[0], en_band[1]
        if lc_list_file.is_file():
            lc_list = LightcurveList.load(lc_list_file)
            lc_list.plot(ax=ax,color=col,marker=marker,lfont=14,label='{}-{}'.\
                         format(low_en,high_en))
            other_crs += [lc_list.cr]
            other_crs_err += [lc_list.cr_std]
        else:
            logging.info('Single energy band lc_list not found')
            logging.info(lc_list_file)

    # I have not idea why I did this, but it works
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.grid(b=True, which='major', color='grey', linestyle='-')
    #fig.tight_layout(0.5)
    
    plot1 = std_plot_dir/'cr_per_seg_T{}_{}.jpeg'.format(tres,tseg)
    plots += [plot1]
    fig.savefig(plot1, dpi=300)

    pyplot.clf() # This is supposed to clean the figure
    plt.close(fig)
    # ------------------------------------------------------  


    # Plot2: Energy spectrum 
    # ------------------------------------------------------
    logging.info('Plotting energy spectrum')
    fig,ax = plt.subplots(figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.tight_layout()
    fig.suptitle('Energy spectrum', fontsize=14)
    
    if spec.is_file() and rsp.is_file():
        s = xspec.Spectrum(str(spec))
        s.response = str(rsp)
        
        if bkg_spec.is_file():
            s.background = str(bkg_spec)
            
        xspec.Plot.xAxis = "keV"
        #s.ignore("**-0.2 12.0-**")
        
        xspec.Plot.device = '/null'
        xspec.Plot("data")
        xspec.Plot.addCommand("rebin 3 35")
        xVals = xspec.Plot.x()
        yVals = xspec.Plot.y()
        # To get a background array, Plot.background must be set prior to plot
        xspec.Plot.background = True
        xspec.Plot("data")
        if bkg_spec.is_file():
            bkg = xspec.Plot.backgroundVals()
        # Retrieve error arrays
        xErrs = xspec.Plot.xErr()
        yErrs = xspec.Plot.yErr()
        
        ax.errorbar(xVals, yVals, yerr=yErrs, xerr=xErrs, fmt='k')
        if os.path.isfile(bkg_spec):
            ax.plot(xVals,bkg,'red',label='Background')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(b=True, which='major', color='grey', linestyle='-')
        ax.legend()
        ax.set_xlabel('Energy [keV]',fontsize=14)
        ax.set_ylabel('counts/cm$^2$/sec/keV',fontsize=14)
        
        for i,en_band in enumerate(en_bands):
            low_en = float(en_band[0])
            high_en = float(en_band[1])
            ylims = ax.get_ylim()
            rect = Rectangle((low_en,ylims[0]),high_en-low_en,
                             ylims[0]+10**(math.log10(ylims[0])+1/2),
                            color=colors[i],fill=True)
            ax.add_patch(rect)
    else:
        logging.info('Energy spectrum file not found')
        logging.info(spec)
        
    plot2 = std_plot_dir/'energy_spectrum.jpeg'
    plots += [plot2]
    fig.savefig(plot2, dpi=300)

    pyplot.clf() # This is supposed to clean the figure
    plt.close(fig) 
    # ------------------------------------------------------


    # Plot3: Full power spectrum
    # ------------------------------------------------------
    logging.info('Plotting full power spectrum')
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Full Power spectrum',fontsize=14)
    
    if main_pw_list_file.is_file():
        # Leahy power
        power_list = PowerList.load(main_pw_list_file)
        leahy = power_list.average('leahy')
        leahy_rebin = leahy.rebin(rebin)
        leahy_rebin.plot(ax=ax1)
        
        # RMS power
        if max(leahy.freq) > 3500:
            sub_poi = leahy.sub_poi(low_freq=3000)
        else:
            sub_poi = leahy.sub_poi(value=2)
            
        if os.path.isfile(bkg_spec):
            full_bkg_cr, full_bkg_cr_err = get_cr(bkg_spec,low_en=0.5,high_en=10.)
            logging.info('Background cr from bkg energy spectrum: {}'.format(full_bkg_cr))
        else:
            full_bkg_cr, full_bkg_cr_err = 0,0
        

        if os.path.isfile(bkg_lc_list_file):
            bkg_lc_list = LightcurveList.load(bkg_lc_list_file)
            full_bkg_cr_from_lc = bkg_lc_list.cr
            logging.info('Background cr from bkg lightcurve: {}'.format(full_bkg_cr_from_lc))
        
        rms = sub_poi.normalize('rms',bkg_cr=full_bkg_cr_from_lc)
        rms_rebin = rms.rebin(rebin)
        rms_rebin.plot(ax=ax2,xy=True)
    else:
        logging.info('Full Power spectrum file not found')
        logging.info(main_pw_list_file)
        
    ax1.grid(b=True, which='major', color='grey', linestyle='-')
    ax2.grid(b=True, which='major', color='grey', linestyle='-')
    fig.tight_layout(w_pad=1,rect=[0,0,1,0.9])
    
    plot3 = std_plot_dir/'full_power_spectrum_T{}_{}.jpeg'.format(tres,tseg)
    plots += [plot3]
    fig.savefig(plot3, dpi=300)

    pyplot.clf() # This is supposed to clean the figure
    plt.close(fig)
    # ------------------------------------------------------

    if len(power_list_files) != 0:
        # Plot4: different energy bands power spectra
        # ------------------------------------------------------ 
        logging.info('Plotting different energy bands power spectra')
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,5))
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        #fig.tight_layout()
        fig.suptitle('Multi-band Power spectrum',fontsize=14)  
        
        bkgs_cr, bkgs_cr_err = [],[]
        for i,pw_file in enumerate(power_list_files):
            
            low_en, high_en = en_bands[i][0],en_bands[i][1]
            
            if pw_file.is_file():
                
                # Leahy power
                power_list = PowerList.load(pw_file)
                leahy = power_list.average('leahy')
                leahy_rebin = leahy.rebin(rebin)
                leahy_rebin.plot(ax=ax1,label='{}-{}'.format(low_en,high_en),color=colors[i])
                
                # RMS power
                if max(leahy.freq) > 3500:
                    sub_poi = leahy.sub_poi(low_freq=3000)
                else:
                    sub_poi = leahy.sub_poi(value=2)

                if os.path.isfile(bkg_list_files[i]):
                    bkg_lc_list = LightcurveList.load(bkg_list_files[i])
                    bkg_cr_from_lc = bkg_lc_list.cr
                    bkg_cr_err_from_lc = bkg_lc_list.cr_std
                    
                bkgs_cr += [bkg_cr_from_lc]
                bkgs_cr_err += [bkg_cr_err_from_lc]

                rms = sub_poi.normalize('rms',bkg_cr=bkg_cr_from_lc)
                rms_rebin = rms.rebin(rebin)
                rms_rebin.plot(ax=ax2,xy=True,label='{}-{}'.format(low_en,high_en),color=colors[i])
            
            else:
                logging.info('Single power list file not found')
                logging.info(pw_file)

        ax1.grid(b=True, which='major', color='grey', linestyle='-')
        ax2.grid(b=True, which='major', color='grey', linestyle='-')
        ax1.legend(title='[keV]')
        fig.tight_layout(w_pad=1,rect=[0,0,1,0.9])

        plot4 = os.path.join(std_plot_dir,'multi_band_power_spectrum_T{}_{}.jpeg'.format(tres,tseg))
        plots += [plot4]
        fig.savefig(plot4, dpi=300)
        pyplot.clf() # This is supposed to clean the figure
        plt.close(fig)
        # ------------------------------------------------------   

    # Plot5: Full power spectra per GTI
    # ------------------------------------------------------ 
    logging.info('Plotting Full power spectra per GTI')
    if main_pw_list_file.is_file():
        power_list = PowerList.load(main_pw_list_file)
        bkg_gti_flag = False
        if bkg_lc_list_file.is_file():
            bkg_lc_list = LightcurveList.load(bkg_lc_list_file)
            bkg_gti_flag = True
        n_gtis = power_list[0].meta_data['N_GTIS']
        n_plots_pp = 3 # Gti plots per ax
        chunkss = chunks(n_gtis,n_plots_pp)

        colors2 = [item for key,item in mcolors.TABLEAU_COLORS.items()]

        for i,chunk in enumerate(chunkss):
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))

            fig.suptitle('Power spectra per GTI ({}/{})'.format(i+1,len(chunkss)), fontsize=14)
            #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            #fig.tight_layout()

            for j,gti_index in enumerate(chunk):
                local_pw_list = PowerList([pw for pw in power_list if pw.meta_data['GTI_INDEX']==gti_index])
                n_segs = len(local_pw_list)

                local_leahy = local_pw_list.average('leahy')
                local_leahy_rebin = local_leahy.rebin(rebin)
                if max(local_leahy.freq) > 3500:
                    local_sub_poi = local_leahy.sub_poi(low_freq=3000)
                else:
                    local_sub_poi = leahy.sub_poi(value=2.)

                if bkg_gti_flag:
                    bkg_gti = (bkg_lc_list[gti_index]).cr
                else:
                    bkg_gti = 0
                local_rms = local_sub_poi.normalize('rms',bkg_cr=full_bkg_cr)
                local_rms_rebin = local_rms.rebin(rebin)   

                local_leahy_rebin.plot(ax=ax1,color=colors2[j],label = f'{gti_index} ({n_segs})')
                local_rms_rebin.plot(ax=ax2,xy=True,color=colors2[j],label = f'{gti_index} ({n_segs})')
                #ax2.set_ylabel('')

            ax1.legend(title='GTI (n. segs)')
            ax1.grid(b=True, which='major', color='grey', linestyle='-')
            ax2.grid(b=True, which='major', color='grey', linestyle='-')
            fig.tight_layout(w_pad=1,rect=[0,0,1,0.9])
            
            plotx = std_plot_dir/'full_power_spectrum_{}_T{}_{}.jpeg'.\
                                 format(i,tres,tseg)
            plots += [plotx]
            fig.savefig(plotx, dpi=300)

            pyplot.clf() # This is supposed to clean the figure
            plt.close(fig)
    else:
        logging.info('Main energy band PowerList file not found')
        logging.info(main_pw_list_file)
    # ------------------------------------------------------ 

    # Extracting Information
    # ------------------------------------------------------ 

    # parent_info is defined to populate a pandas data frame with
    # info from each obs ID. This will be used for general plotting

    logging.info('Extracting info')
    parent_info = {}

    # Active detectors
    if main_lc_list_file.is_file():

        # Information from lightcurve metadata
        # -------------------------------------------------------------    
        info_dict = main_lc_list[0].meta_data['INFO_FROM_HEADER']

        for key,item in info_dict.items():
            parent_info[key] = item
        
        # This is because in the FITS file the keyword obs_ID does not
        # include the exposure obs ID
        parent_info['OBS_ID'] = str(obs_id)
        # -------------------------------------------------------------

        parent_info['N_EN_BANDS'] = len(en_bands)
    
        # Computing fractional rms
        # -----------------------------------------------------------------
        def comp_rms(power_list_file,bkg_cr):
            pw_list = PowerList.load(power_list_file)
            power = pw_list.average('leahy')
            if max(power.freq) > 3500:
                sub_poi = power.sub_poi(low_freq=3000)
            else:
                sub_poi = power.sub_poi(value=2.)
            rms = sub_poi.normalize('rms',bkg_cr=bkg_cr)
            rms,rms_err = rms.comp_frac_rms(high_freq=60)
            return rms,rms_err
        
        rms,rms_err = 0,0
        if main_pw_list_file.is_file():
            rms,rms_err = comp_rms(main_pw_list_file,full_bkg_cr_from_lc)
            parent_info['RMS'] = rms
            parent_info['RMS_ERR'] = rms_err
            
        if len(power_list_files) != 0:
            rms_bands,rms_bands_err = [],[]
            for i,power_file in enumerate(power_list_files):
                if power_file.is_file():
                    rms_band, rms_band_err = comp_rms(power_file,bkgs_cr[i])
                    rms_bands += [rms_band]
                    rms_bands_err += [rms_band_err]
                    parent_info[f'RMS{i+1}'] = rms_band
                    parent_info[f'RMS_ERR{i+1}'] = rms_band_err
        # -----------------------------------------------------------------

        
        # first_column and second_column are defined for printing purposes
        # They are still dictionaries, but the key does not matter much this
        # time, while the item consists of a 2 element list with human-
        # friendly key and description

        descr = {'TELESCOP':'Mission',
                'INSTRUME':'Instrument',
                'OBJECT':'Target',
                'OBS_ID':'Obs. ID',
                'DATE-OBS':'Start obs. time',
                'DATE-END':'Stop obs. time',
                'ONTIME':'Clean exposure [s]',
                'NAXIS2':'N. of raw events'}
        
        def truncate(n, decimals=0):
            multiplier = 10 ** decimals
            return int(n * multiplier) / multiplier
        
        ori_exp = info_dict['TSTOP']-info_dict['TSTART']
        #filt_exp = truncate((ori_exp-info_dict['ONTIME'])/ori_exp*100,1)
        filt_exp = info_dict['EXPOSURE']
        parent_info['ORI_EXP'] = ori_exp
        # This corresponds to the sum of start and stop times in the GTI
        parent_info['FILT_EXP'] = filt_exp

        if len(power_list_files) != 0:
            net_cr_hard = other_crs[1]-bkgs_cr[1]
            net_cr_soft = other_crs[0]-bkgs_cr[0]
            hr = truncate(net_cr_hard/net_cr_soft,2)
            parent_info['NET_HARD_CR'] = net_cr_hard
            parent_info['NET_SOFT_CR'] = net_cr_soft
            parent_info['HR'] = hr

        # First columns
        # -------------------------------------------------------------
        first_column = {}
        first_column['CDATE'] = ['Cre. date (this file):',my_cdate()]
        for key, item in descr.items():
            first_column[key] = [item+':',str(info_dict[key])]
        first_column['FEXP'] = ['Filtered exp. [%]:',str(filt_exp)]
        first_column['MAIN_EN_BAND'] = ['Main energy band:',
            '{}-{} [keV]'.format(main_en_band[0],main_en_band[1])]
        parent_info['MAIN_EN_BAND'] = '{}-{}'.\
            format(main_en_band[0],main_en_band[1])
        if len(power_list_files) != 0:
            for i,en_band in enumerate(en_bands):
                first_column['EN_BAND{}'.format(i+1)] = ['Energy band {}:'.\
                    format(i+1),'{}-{} [keV]'.format(en_band[0],en_band[1])]
                parent_info[f'EN_BAND{i+1}'] = '{}-{}'.\
                    format(en_band[0],en_band[1])
        # -------------------------------------------------------------

        # Second column
        # -------------------------------------------------------------
        second_column = {}
        second_column['N_GTIS'] = ['N. GTIs:',str(len(gti))]
        second_column['N_FGTIS'] = [f'N. filtered GTIs (>{tseg}):',
            str(len(gti>=tseg))]
        second_column['N_SEGS'] = ['N. of segments:',
            str(len(main_lc_list))]
        second_column['T_RES'] = ['Time resolution [s]:',
            '{}'.format(np.round(float(tres),11))]
        second_column['T_SEG'] = ['Time segment [s]:','{}'.format(tseg)]
        second_column['CR'] = ['Total count rate [c/s]:',
            '{} '.format(truncate(main_cr,1))+u'\u00B1'+\
            ' {}'.format(truncate(main_cr_err,1))]
        second_column['BKG_CR'] = ['Bkg count rate [c/s]:',
            '{} '.format(truncate(full_bkg_cr,1))+u'\u00B1'+\
            ' {}'.format(truncate(full_bkg_cr_err,1))]
        parent_info['N_GTIS'] = len(gti)
        parent_info['N_FILT_GTIS'] = len(gti>=tseg)
        parent_info['N_SEGS'] = len(main_lc_list)
        parent_info['T_RES'] = tres
        parent_info['TSEG'] = tseg
        parent_info['CR'] = main_cr
        parent_info['CR_ERR'] = main_cr_err
        parent_info['BKG_CR'] = full_bkg_cr

        if len(power_list_files) != 0:
            for i,(cr,cr_err,bkg_cr,bkg_cr_err) in \
                enumerate(zip(other_crs,other_crs_err,bkgs_cr,bkgs_cr_err)):
                second_column['CR{}'.format(i+1)] = \
                    ['Count rate band {} [c/s]:'.format(i+1),
                    '{} '.format(truncate(cr,1))+u'\u00B1'+\
                    ' {}'.format(truncate(cr_err,1))]
                second_column['BKG_CR{}'.format(i+1)] = \
                    ['Bkg count rate band {} [c/s]:'.format(i+1),
                    '{} '.format(truncate(bkg_cr,1))+u'\u00B1'+\
                    ' {}'.format(truncate(bkg_cr_err,1))]         
                parent_info[f'CR{i+1}'] = cr
                parent_info[f'CR_ERR{i+1}'] = cr_err
                parent_info[f'BKG_CR{i+1}'] = bkg_cr
                parent_info[f'BKG_CR_ERR{i+1}'] = bkg_cr_err
            second_column['HR'] = ['Hard/Soft cr ratio:',str(hr)]

        second_column['FRMS'] = ['Fractional rms (<60Hz):',
            '{} '.format(np.round(rms*100,2))+u'\u00B1'+\
            ' {} %'.format(np.round(rms_err*100,2))]

        if len(power_list_files) != 0:
            for i,(rms_band,rms_band_err,en_band) in \
                enumerate(zip(rms_bands,rms_bands_err,en_bands)):
                second_column['FRMS{}'.format(i+1)] = \
                    ['Fractional rms (<60Hz) band {}:'.format(i+1),
                    '{} '.format(np.round(rms_band*100,2))+u'\u00B1'+\
                    ' {} %'.format(np.round(rms_band_err*100,2))]
        # -------------------------------------------------------------

        # Making a single dictionary with all the information needed for
        # printing an obs_ID page
        # -------------------------------------------------------------
        cols_dict = {'COL1':first_column,'COL2':second_column,'PLOTS':plots}
            
        std_prod_dir = obs_id_dir/'std_prods'
        if not std_prod_dir.is_dir():
            print('std_prods dir does not exist, creating it')
            os.mkdir(std_prod_dir)  
        output = std_prod_dir/'info_dict_T{}_{}.pkl'.format(tres,tseg)
        with open(output, 'wb') as handle:
            pickle.dump(cols_dict, handle, pickle.HIGHEST_PROTOCOL)
        # -------------------------------------------------------------

        # Updating (or creating) global data frame with single observation
        # -------------------------------------------------------------
        parent_std_prod_dir = obs_id_dir.parent/'std_prods'
        if not parent_std_prod_dir.is_dir():
            logging.info('Parent std_prods for does not exist, creating it')
            os.mkdir(parent_std_prod_dir)
        df_name = parent_std_prod_dir/'info_data_frame_T{}_{}.pkl'.format(tres,tseg)
        if df_name.is_file():
            df = pd.read_pickle(df_name)
            df = df.append(parent_info,ignore_index=True)
            df.to_pickle(df_name)
        else:
            df = pd.DataFrame(columns=parent_info.keys())
            df = df.append(parent_info,ignore_index=True)
            df.to_pickle(df_name)
        # -------------------------------------------------------------
    else:
        first_column = {}
        second_column = {}
    
    return plots,first_column,second_column


def make_hxmt_std_prod(obs_id_dirs,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['52.6','188.4'], spec_en_ch = ['26','120'], en_bands = [],
    rebin=-30,data_dir=pathlib.Path.cwd(),
    log_name=None):
    '''
    Runs make_hxmt_std_prod_single for each obs_ID in obs_id_dirs
    
    HISTORY
    -------
    2021 05 01, Stefano Rapisarda (Uppsala), creation date
    '''

    an_dir = obs_id_dirs[0].parent

    # Logging
    if log_name is None:
        log_name = make_logger('print_std_prods',outdir=an_dir)

    for obs_id_dir in obs_id_dirs:

        if obs_id_dir.is_dir():
            make_hxmt_std_prod_single(obs_id_dir,tres=tres,tseg=tseg,
                main_en_band = main_en_band, spec_en_ch = spec_en_ch, en_bands = en_bands,
                rebin=rebin,data_dir=data_dir,
                log_name=log_name)
        else:
            logging.info('{} does not exist'.format(obs_id_dir))

    

def make_nicer_std_prods(obs_id_dirs,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['0.5','10.0'], en_bands = [['0.5','2.0'],['2.0','10.0']],
    rebin=-30,data_dir=pathlib.Path.cwd(),override=False,dpi=200):
    '''
    Runs make_nicer_std_prod_single for each obs_ID in obs_id_dirs

    PARAMETERS
    ----------
    obs_id_dirs: list or str or pathlib.Path
        List of the FULL PATH of observational ID dirs or full path of
        the dir containing all the observational ID dirs.
        These folders are supposed to contain timing data products
        (lightcurves and power spectra) and to be inside an analysis
        folder
    tres: str
        Time resolution of the timing products as it appears in their
        name. This is not used for computation, but just to identify
        the right file
    tseg: str
        Time segment of the timing products as it appears in their
        name. This is not used for computation, but just to identify
        the right file
    main_en_band: list
        2-element list containing lower and upper main energy band,
        the boundaries are specified as strings
    en_bands: list 
        n-element list containing lower and upper energy bands.
        The first two energy bands will be considered to compute the
        hardness ratio and have to be listed in ascending order of 
        energy
    rebin: str
        Rebin factor used to rebin frequency bin in the power spectra.
        If negative, logarithmic binning will be applied.
    data_dir: str or pathlib.Path()
        Full path of the dir containing data. Thi folder should contain
        obs. ID dirs containing the first products of data reduction
        (cleaned event files and energy spectra)

    HISTORY
    -------
    2021 03 18, Stefano Rapisarda (Uppsala), creation date
    2021 12 09, Stefano Rapisarda (Uppsala)
        log_name, rmf, and arf parameters removed
    '''

    if type(obs_id_dirs) in [str,type(pathlib.Path())]:
        obs_id_dirs = list_items(obs_id_dirs,itype='dir',digits=10)

    an_dir = obs_id_dirs[0].parent

    # Recording date and time
    # -----------------------------------------------------------------
    now = datetime.now()
    date = ('%d_%d_%d') % (now.year,now.month,now.day)
    time = ('%d_%d') % (now.hour,now.minute)
    # -----------------------------------------------------------------

    # Initializing logger
    # -----------------------------------------------------------------
    log_name = os.path.basename(__file__).replace('.py','')+\
        '_D{}_T{}'.format(date,time)
    make_logger(log_name,outdir=an_dir)

    mylogging = LoggingWrapper() 
    # -----------------------------------------------------------------

    # Printing info
    # -----------------------------------------------------------------
    mylogging.info(72*'*')
    mylogging.info(str_title('make_nicer_std_prods'))
    mylogging.info(72*'*'+'\n')
    mylogging.info(f'Number of observations: {len(obs_id_dirs)}')
    mylogging.info(f'Selected time resolution: {tres} [s]')
    mylogging.info(f'Selected time segment: {tseg} [s]')
    mylogging.info(f'Selected main energy band: {main_en_band[0]}-{main_en_band[1]} [keV]')
    mylogging.info(f'Other selected energy bands:')
    for i,en_band in enumerate(en_bands):
        mylogging.info(f'{i+1}) {en_band[0]}-{en_band[1]} [keV]')
    mylogging.info(f'Selected rebin factor: {rebin}')
    mylogging.info(f'Data dir: {data_dir}'+'\n')
    # -----------------------------------------------------------------


    for obs_id_dir in obs_id_dirs:

        if obs_id_dir.is_dir():
            make_nicer_std_prod_single(obs_id_dir,tres=tres,tseg=tseg,
                main_en_band = main_en_band, en_bands = en_bands,
                rebin=rebin,data_dir=data_dir,override=override,dpi=dpi)
        else:
            mylogging.info('{} does not exist'.format(obs_id_dir))

    mylogging.info('Everything is done. Goodnight!\n')

    mylogging.info(72*'*')
    mylogging.info(72*'*')
    mylogging.info(72*'*')

def make_general_plot(obs_id_dir,tres='0.0001220703125',tseg='128.0',
    main_en_band=['0.5','10.0'],time_window = 20,plt_x_dim = 8,plt_y_dim = 8,
    mission='NICER', suffix=None):
    '''
    It reads a pandas data frame containing information about all the 
    OBS_ID and makes a plot with count rate, hardness ratio, and 
    fractional RMS over time

    If obs_id is not None, a gold dot corresponding to the obs_ID time
    is plotted 

    HISTORY
    -------
    2021 03 18, Stefano Rapisarda (Uppsala), creation date
    2021 12 10, Stefano Rapisarda (Uppsala)
        Adopted LoggingWrapper approach
    '''

    def get_ylims(arr,err=None,min_spacer=16,max_spacer=8):
        diff=  np.nanmax(arr) - np.nanmin(arr)
        if err is None:
            min_y = np.nanmin(arr)-diff/min_spacer
            max_y = np.nanmax(arr)+diff/max_spacer
        else:
            min_arr = np.nanmin(arr-err)
            max_arr = np.nanmax(arr+err)
            min_y = min_arr - (max_arr-min_arr)/min_spacer
            max_y = max_arr + (max_arr-min_arr)/max_spacer
            if math.isnan(min_y) or math.isinf(min_y): 
                min_y = np.nanmin(arr)-diff/min_spacer
            if math.isnan(max_y) or math.isinf(max_y):
                max_y = np.nanmax(arr)+diff/max_spacer

        return [min_y,max_y]

    mylogging = LoggingWrapper()

    mylogging.info('*'*72)
    mylogging.info(str_title('make_general_plot'))
    mylogging.info('*'*72+'\n')
    
    if type(obs_id_dir) == str: obs_id_dir = pathlib.Path(obs_id_dir)
    an_dir = obs_id_dir.parent
    obs_id = obs_id_dir.name

    # Setting directory identifiers
    time_string = 'T{}_{}'.format(tres,tseg)
    if suffix is None:
        dir_string = 'E{}_{}_{}'.format(
            main_en_band[0],main_en_band[1],time_string)
    else:
        dir_string = 'E{}_{}_{}_{}'.format(
            main_en_band[0],main_en_band[1],time_string,suffix)

    # Reading data frame
    info_df_name = an_dir/'std_prods'/dir_string/'general_info_data_frame.pkl'
    if info_df_name.is_file():
        df = pd.read_pickle(info_df_name)
    else:
        mylogging.error('I could not finde a {} file'.format(info_df_name.name))   
    df=df.drop_duplicates(subset=['OBS_ID'])
    
    # Plotting
    # ==================================================================
        
    # General settings
    colors = ['red','blue','green','orange','brown']
    markers = ['s','^','p','H','X']
    n_en_bands = df['n_en_bands'].iloc[0]
    
    fig, axes = plt.subplots(3,1,figsize=(plt_x_dim,plt_y_dim))
    plt.subplots_adjust(hspace=0)

    # Defining time ax
    # ------------------------------------------------------------------
    start_dates = df['DATE-OBS']
    stop_dates = df['DATE-END']
    start_dates_mjd = Time(start_dates.to_list(),format='isot',scale='utc').mjd
    stop_dates_mjd = Time(stop_dates.to_list(),format='isot',scale='utc').mjd
    half_dur_mjd = (stop_dates_mjd-start_dates_mjd)/2.
    mid_dates_mjd = (start_dates_mjd+stop_dates_mjd)/2.
    start_mjd = int(np.min(start_dates_mjd))
    time = mid_dates_mjd-start_mjd
    
    obs_ids = df['OBS_ID'].to_numpy()
    if obs_id in obs_ids: 
        target_index = np.where(obs_ids == obs_id)
        time_obs_id = time[target_index]

        x_lims = [int(time_obs_id/time_window)*time_window-time_window/10,
                int(time_obs_id/time_window+1)*time_window+time_window/10*2]

        for ax in axes:
            ax.set_xlim(x_lims)
    else:
        x_lims = [np.min(time),np.max(time)]
    
    time_mask = (time >= x_lims[0]) & (time<= x_lims[1])
    # ------------------------------------------------------------------

    # Plot1, count rates versurs time
    # ------------------------------------------------------------------
    # Main energy band
    main_en_band = str(df['main_en_band'].iloc[0])
    tot_cr = df['main_cr'].to_numpy()
    bkg = df['main_bkg_cr'].to_numpy()
    if mission == 'NICER':
        n_act_det = df['n_act_det'].to_numpy()
    else:
        n_act_det = 1
    cr_err = df['main_cr_err'].to_numpy()/n_act_det
    cr = (tot_cr-bkg)/n_act_det
    
    axes[0].errorbar(time,cr,yerr=cr_err,xerr=half_dur_mjd,fmt='o',
        color='black',label=main_en_band)
    y_lims0 = get_ylims([cr[time_mask],cr_err[time_mask]],max_spacer=1.)
  
    if (not obs_id is None) and (obs_id in obs_ids):
        gold_dot, = axes[0].plot(time_obs_id,cr[target_index],'o',color='goldenrod',ms=12)
        leg2 = axes[0].legend([gold_dot],[obs_id],loc='upper left')
        axes[0].add_artist(leg2)
        
    def slice_df(df,col,index):
        array = np.array([df[col].iloc[i][index] for i in range(len(df))])
        return array

    # Other energy bands
    for e in range(n_en_bands):
        en_band = str(df['other_en_bands'].iloc[0][e])
        tot_cr = slice_df(df,'other_crs',e) 
        bkg = slice_df(df,'other_bkg_crs',e)
        cr_err = slice_df(df,'other_cr_err',e)/n_act_det
        cr = (tot_cr-bkg)/n_act_det
        axes[0].errorbar(time,cr,yerr=cr_err,xerr=half_dur_mjd,fmt=markers[e],color=colors[e],label=en_band)
        
        if (not obs_id is None) and (obs_id in obs_ids):
            axes[0].plot(time_obs_id,cr[target_index],'o',color='goldenrod',ms=12)
            #p=patches.Ellipse((time_obs_id,cr[target_index]),x_length*radius*plt_yx_ratio,y_length*radius,\
            #                                              edgecolor='goldenrod',facecolor='none',lw=2,zorder=2)
            #axes[0].add_patch(p)   
    
    if mission == 'NICER':
        axes[0].set_ylabel('Count rate [c/s/n_det]',fontsize=14)
    else:
        axes[0].set_ylabel('Count rate [c/s]',fontsize=14)
    axes[0].legend(title='[keV]',loc='upper right')
    axes[0].set_ylim(y_lims0)
    # ------------------------------------------------------------------
    
    # Plot2, hardness ratio
    # ------------------------------------------------------------------
    if n_en_bands >= 2:
        net_cr_hard = slice_df(df,'other_crs',1)-slice_df(df,'other_bkg_crs',1)
        net_cr_soft = slice_df(df,'other_crs',0)-slice_df(df,'other_bkg_crs',0)
        hr = net_cr_hard/net_cr_soft
        hr_sigma2 = (
            slice_df(df,'other_cr_err',1)**2 + slice_df(df,'other_bkg_cr_err',1)**2 +
            hr**2 * (slice_df(df,'other_cr_err',0)**2 + slice_df(df,'other_bkg_cr_err',0)**2)
            ) / net_cr_soft**2

        axes[1].errorbar(time,hr,yerr=np.sqrt(hr_sigma2),fmt='o',color='black',zorder=4)
        
        if obs_id in obs_ids:
            axes[1].plot(time_obs_id,hr[target_index],'o',color='goldenrod',ms=12) 
            
        axes[1].set_ylabel('Hardness',fontsize=14)
        #axes[1].set_ylim(get_ylims(hr[time_mask]))
        axes[1].set_ylim([0,1])
    # ------------------------------------------------------------------
    
    # Plot3, fractional rms
    # ------------------------------------------------------------------
    # Main energy band
    rms = df['main_frac_rms'].to_numpy()
    rms_err = df['main_frac_rms_err'].to_numpy()
    axes[2].errorbar(time,rms*100,rms_err*100,fmt='o',color='black')

    ylims2 = get_ylims(rms[time_mask],rms_err[time_mask])
    
    if obs_id in obs_ids:
        gold_dot, = axes[2].plot(time_obs_id,rms[target_index]*100,'o',color='goldenrod',ms=12)   
        
    # Other energy bands
    for e in range(n_en_bands):
        en_band = str(df['other_en_bands'].iloc[0][e])
        rms = slice_df(df,'other_frac_rms',e)
        rms_err = slice_df(df,'other_frac_rms_err',e)
        axes[2].errorbar(time,rms*100,rms_err*100,fmt=markers[e],color=colors[e])
        
        if obs_id in obs_ids:
            gold_dot, = axes[2].plot(time_obs_id,rms[target_index]*100,'o',color='goldenrod',ms=12)
     
    axes[2].set_ylabel('Frac. RMS [%]',fontsize=14)
    axes[2].set_xlabel('Time [MJD, {}]'.format(start_mjd),fontsize=14)
    #print(rms)
    axes[2].set_ylim([-5,70])
    #axes[2].set_ylim([r*100 for r in ylims2])
    # --------------------------------------------------------------
    
    for ax in axes: ax.grid()
        
    # Saving file
    plot_name = an_dir/obs_id/'std_plots'/'global_info.jpeg'.format(time_string)
    fig.savefig(plot_name, dpi=300)
    
    return plot_name

def old_make_nicer_std_prod_single(obs_id_dir,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['0.5','10.0'], en_bands = [['0.5','2.0'],['2.0','10.0']],
    rebin=-30,data_dir=pathlib.Path.cwd(),override=False):
    '''
    Makes plots and a dictionary with information according to user 
    settings. Standard plots and dictionary will be stored in a std_plots
    directory created inside obs_id_dir

    This function is specific for NICER reduced products. It assumes
    that products (lightcurve lists,power list, and gti for the selected 
    energy bands and time settings) are already computed. If the 
    function does not find the products, it will produce a plot anyway,
    but it will be empty.
    These products are expected to have a format E<1>_<2>_T<3>_<4>, 
    where 1 and 2 are the energy band boundaries and 3 and 4 are the 
    time resolution and time segment, respectively. 
    The first two bands of this list are soft and hard band to compute
    the hardness ratio

    PARAMETERS
    ----------
    obs_id_dir: string or pathlib.Path
        full path of the obs ID folder
    tres: string (optional)
        time resolution of the reduced products
        (default is 0.0001220703125)
    tseg: string (optional)
        time segment of the reduced products
        (default is 128.0)
    main_en_band: list (optional)
        list with low and high main energy band
        (default is ['0.5','10.0'])
    en_bands: list (optional)
        list of lists, contains low and high energy band boundaries
        for each sub (different and smaller than main) energy band
        (default is [['0.5','2.0'],['2.0','10.0']]).
        The maximum number of energy bands is five.
    rebin: int (optional)
        rebin factor for plotting the power spectra
        (default is -30)
    data_dir: string or pathlib.Path (optional)
        folder containing the energy spectrum
        (default is pathlib.Path.cwd())

    RETURNS
    -------
    (plots,first_column,second_column): tuple
        plots is a list containing the full path of produced plots
        first_column and second_column are lists of information to be 
        displayed in the first and second column of a PDF document.
        They are in the form ['Description','Value']

    HISTORY
    -------
    2020 12 ##, Stefano Rapisarda (Uppsala), creation date
    2020 03 10, Stefano Rapisarda (Uppsala)
        Cleaned up. Now it returns objects to be used directly
        in print_std_prods
    2021 09 12, Stefano Rapisarda (Uppsala)
        Removed arf, rmf, and log_name, adopted LoggingWrapper approach
    '''

    mylogging = LoggingWrapper()

    if type(obs_id_dir) == str: obs_id_dir = pathlib.Path(obs_id_dir)
    if type(data_dir) == str: data_dir = pathlib.Path(data_dir)

    obs_id = obs_id_dir.name
  
    # You can specify a maximum of five energy bands
    colors = ['red','blue','green','orange','brown']
    markers = ['s','^','p','H','X']
    if len(en_bands) > len(colors):
        raise ValueError('Energy bands cannot be more than 5')

    mylogging.info('*'*72)
    mylogging.info(str_title('make_nicer_std_prod_single'))
    mylogging.info('*'*72+'\n')


    # Making directories
    # -----------------------------------------------------------------
    root_std_plot_dir = obs_id_dir/'std_plots'
    if not root_std_plot_dir.is_dir():
        mylogging.info('root std_plots directory does not exist, creating one...')
        os.mkdir(root_std_plot_dir)
    else:
        mylogging.info('root std_plots directory already exists.')
    # -----------------------------------------------------------------
    

    # Defining names of files to read
    # -----------------------------------------------------------------
    
    # Main energy band str identifier
    main_root_str_identifier = 'E{}_{}_T{}'.\
        format(main_en_band[0],main_en_band[1],tres)
    main_str_identifier = main_root_str_identifier + f'_{tseg}'

    # Main energy band lc_list files 
    main_gti_lc_list_file = obs_id_dir/'lc_list_{}.pkl'.\
        format(main_root_str_identifier)
    main_seg_lc_list_file = obs_id_dir/'lc_list_{}.pkl'.\
        format(main_str_identifier)
    ufa_lc_list_file = obs_id_dir/'ufa_lc_list_{}.pkl'.\
        format(main_root_str_identifier)

    main_pw_list_file = obs_id_dir/'power_list_{}.pkl'.\
        format(main_str_identifier)

    # Full energy band GTI name
    main_gti_file = obs_id_dir/'gti_E{}_{}.gti'.\
        format(main_en_band[0],main_en_band[1])

    # Other energy band names
    seg_lc_list_files = []
    gti_lc_list_files = []
    power_list_files = []
    for en_band in en_bands:
        low_en, high_en = en_band[0], en_band[1]
        seg_lc_list_files += [obs_id_dir/'lc_list_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg)]
        gti_lc_list_files += [obs_id_dir/'lc_list_E{}_{}_T{}.pkl'.\
            format(low_en,high_en,tres)]   
        power_list_files += [obs_id_dir/'power_list_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg)]
    
    event_cl_dir = data_dir/obs_id/'xti'/'event_cl'
    # Energy spectrum name
    spec = event_cl_dir/f'{obs_id}_spectrum_bdc.pha'
    bkg_spec = event_cl_dir/f'{obs_id}_spectrum_bdc_bkg.pha'
    spec_3c50 = event_cl_dir/f'{obs_id}_spectrum_bdc_3C50_tot.pi'
    bkg_spec_3c50 = event_cl_dir/f'{obs_id}_spectrum_bdc_3C50_bkg.pi'

    # arf and rmf files
    arf_file = event_cl_dir/f'arf_bdc.arf'
    rmf_file = event_cl_dir/f'rmf_bdc.rmf'

    # ufa file
    ufa_evt_file = list_items(event_cl_dir,itype='file',
        include_and=[f'ni{obs_id}_0mpu7_ufa.evt'])[0]
    # -----------------------------------------------------------------

    # Making std_plot directory
    # -----------------------------------------------------------------
    std_plot_dir = root_std_plot_dir/main_str_identifier
    if not std_plot_dir.is_dir():
        mylogging.info('std_plots directory does not exist, creating one...')
        os.mkdir(std_plot_dir)
    else:
        mylogging.info('std_plots directory already exists.')
    # -----------------------------------------------------------------


    # Printing some info
    # -----------------------------------------------------------------
    mylogging.info('')
    mylogging.info('Obs ID: {}'.format(obs_id))
    mylogging.info('Settings:')
    mylogging.info('-'*72)
    mylogging.info('Selected main energy band: {}-{} keV'.\
        format(main_en_band[0],main_en_band[1]))
    for i,en_band in enumerate(en_bands):
        mylogging.info('Selected energy band {}: {}-{} keV'.\
            format(i,en_band[0],en_band[1]))        
    mylogging.info('Selected time resolution: {} s'.format(tres)) 
    mylogging.info('Selected time segment: {} s'.format(tseg)) 
    mylogging.info('-'*72)
    mylogging.info('')
    # -----------------------------------------------------------------
    
    # Plotting
    # =================================================================
    # For each plot I will create the figure anyway, to preserve the 
    # page layout. Then, if a file does not exist or something goes 
    # wrong, the figure will be empty.
    # The array "plots" will contain the full path of the names of 
    # created plots:
    # PLOT1    (plots[0]): Count rate per segment in different energy band
    # PLOT2    (plots[1]): Energy spectrum
    # PLOT(S)3 (plots[2-3]): Comparison between ufa and cl lightcurve and 
    #                       power spectra
    # PLOT4    (plots[4]): Full energy band average power spectrum
    # PLOT5    (plots[5]): Average power spectra in different energy bands
    # PLOT6    (plots[6-n]): Full energy band power spectrum per GTI


    plots = []
    plt.tight_layout()
    
    # PLOT1: count rate per GTI 
    # -----------------------------------------------------------------
    mylogging.info('Plotting count rate per GTI')
    
    # As GTIs are selected ALSO according to the selected segment
    # length, this plot depends both on tres and tseg
    plot_name = std_plot_dir/'cr_per_GTI_T{}_{}.jpeg'.\
        format(tres,tseg)
    if plot_name.is_file() and not override:
        mylogging.info('Count rate per GTI plot already exists')
        plots += [plot_name]
        # Reading GTI 
        gti = Gti.load(main_gti_file)
        gti_seg = gti>=float(tseg)
    else:
        
        fig,ax = plt.subplots(figsize=(8,5))
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'Count rate per GTI',fontsize=14)

        if main_gti_lc_list_file.is_file() and main_gti_file.is_file():

            # Reading GTI 
            gti = Gti.load(main_gti_file)
            gti_seg = gti>=float(tseg)

            fig.suptitle(f'Count rate per GTI (GTI n.: {len(gti)})',fontsize=14)

            # Loading file
            gti_lc_list = LightcurveList.load(main_gti_lc_list_file)
            gti_start = gti_lc_list[0].time.iloc[0]

            # Plotting
            label = '{}-{}'.format(main_en_band[0],main_en_band[1])
            gti_lc_list.plot(ax=ax,color='k',lfont=14,label=label,
                xbar=True,ybar=6)

            # Drawing vertical marks for each GTI
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            bottom,top = ax.get_ylim()
            for gstart,gstop,gdur in zip(gti.start,gti.stop,gti.dur):
                mid_point = (gstart+gstop)/2-gti_start
                color = 'orange'
                bottom_marker = '^'
                if gdur < float(tseg): 
                    color = 'red'
                    bottom_marker = 'o'
                #ax.axvline(g-start,ls='--',color='orange')
                ax.plot([mid_point],[bottom],
                    color=color,marker=bottom_marker)
                ax.plot([mid_point],[top],
                    color=color,marker='|')
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            del gti_lc_list

        else:
            mylogging.error('Main energy band gti_lc_list file not found')
            mylogging.error(f'({main_gti_lc_list_file})')

        # Other energy bands count rate
        for gti_lc_list_file, en_band, col, marker in \
            zip(gti_lc_list_files, en_bands, colors, markers):
            low_en, high_en = en_band[0], en_band[1]
            if gti_lc_list_file.is_file():
                lc_list = LightcurveList.load(gti_lc_list_file)
                label = '{}-{}'.format(low_en,high_en)
                lc_list.plot(ax=ax,color=col,label=label,marker=marker,
                    lfont=14,ybar=8)
                del lc_list
            else:
                mylogging.error('Single energy band gti_lc_list not found')
                mylogging.error(f'({main_gti_lc_list_file})')

            
     
        # I have not idea why I did this, but it works
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.grid(b=True, which='major', color='grey', linestyle='-')
        #fig.tight_layout(0.5)
    
        plots += [plot_name]
        fig.savefig(plot_name, dpi=300)
    # -----------------------------------------------------------------

    # PLOT1x: Count rate per segment for each GTI
    # -----------------------------------------------------------------
    mylogging.info('Plotting count rate per segment')

    plot_main_seg_cr_flag = True
    # Opening Lightcurve list files
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Main en band
    mylogging.info('Reading LightcurveLists')
    if main_seg_lc_list_file.is_file():
        main_seg_lc_list = LightcurveList.load(main_seg_lc_list_file)
        main_cr = main_seg_lc_list.cr
        main_cr_err = main_seg_lc_list.cr_std
        
        obs_start = main_seg_lc_list[0].time.iloc[0]
    else:
        mylogging.error('Main energy band seg_lc_list file not found')
        mylogging.error(f'({main_seg_lc_list_file})')
        plot_main_seg_cr_flag = False

    # Other bands
    seg_lc_lists = []
    other_crs = []
    other_crs_err = []
    plot_seg_cr_flag = True
    for seg_lc_list_file in seg_lc_list_files:
        if seg_lc_list_file.is_file():
            seg_lc_list = LightcurveList.load(seg_lc_list_file)
            seg_lc_lists += [seg_lc_list]
            other_crs += [seg_lc_list.cr]
            other_crs_err += [seg_lc_list.cr_std]
            del seg_lc_list
        else:
            mylogging.error('Single energy band lc_list not found')
            mylogging.error(f'({seg_lc_list_file})')
            plot_seg_cr_flag = False
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
    # Plotting
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for g,(gti_start,gti_stop) in enumerate(zip(gti_seg.start,gti_seg.stop)):

        mylogging.info(f'Plotting GTI {g+1}/{len(gti_seg)}')
        plot_name = std_plot_dir/'cr_per_seg_T{}_{}_{}.jpeg'.format(tres,tseg,g+1)
        if plot_name.is_file() and not override:
            mylogging.info(f'Count rate per segment plot n. {g+1} already exists')
            plots += [plot_name]
        else:
            fig,ax = plt.subplots(figsize=(8,5))
            #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle(f'Count rate per segment (GTI {g+1}/{len(gti_seg)})',
                fontsize=14)  

           # Main band
            if plot_main_seg_cr_flag:
                label = f'{main_en_band[0]}-{main_en_band[1]}'
                main_seg_lc_list.plot(ax=ax,color='k',marker='o',
                    lfont=14,label=label,ybar=4) 
                #del main_seg_lc_list

            # Other bands
            if plot_seg_cr_flag:
                for e,(en_band, col, marker) in \
                    enumerate(zip(en_bands, colors, markers)):
                    label='{}-{}'.format(en_band[0],en_band[1])
                    seg_lc_lists[e].plot(ax=ax,color=col,marker=marker,
                        lfont=14,label=label)

            ax.set_xlim([gti_start-obs_start,gti_stop-obs_start])

            # I have not idea why I did this, but it works
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.grid(b=True, which='major', color='grey', linestyle='-')
            #fig.tight_layout(0.5)

            plots += [plot_name]
            fig.savefig(plot_name, dpi=300)
    del seg_lc_lists
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++          

    # Reading parameters from main_seg_lc_list before closing it
    #if main_seg_lc_list_file.is_file():
    #    n_act_det = main_seg_lc_list[0].meta_data['N_ACT_DET']
    #    inact_det_list = main_seg_lc_list[0].meta_data['INACT_DET_LIST']
    #    info_dict = main_seg_lc_list[0].meta_data['INFO_FROM_HEADER']
    #del main_seg_lc_list
    # -----------------------------------------------------------------
    
    
    # PLOT2: Energy spectrum 
    # -----------------------------------------------------------------
    mylogging.info('Plotting energy spectrum')
    fig,ax = plt.subplots(figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.tight_layout()
    fig.suptitle('Energy spectrum', fontsize=14)
    
    # Energy spectrum created via xselect
    if spec.is_file() and rmf_file.is_file() and arf_file.is_file():
        ui.clean()
        ui.load_pha('tot',str(spec))
        ui.load_arf('tot',str(arf_file))
        ui.load_rmf('tot',str(rmf_file))
        spectrum_data = ui.get_data_plot('tot')
        ax.errorbar(spectrum_data.x,spectrum_data.y,spectrum_data.yerr,
            color='k',label='Spectrum',zorder=10)
        
        if bkg_spec.is_file():
            ui.load_bkg('tot',str(bkg_spec))
            bkg_spectrum_data = ui.get_bkg_plot('tot')
            ax.errorbar(bkg_spectrum_data.x,bkg_spectrum_data.y,bkg_spectrum_data.yerr,
                fmt='--',color='k',label='Bkg',zorder=10)
        else:
            mylogging.error('Background energy spectrum not found')
            mylogging.error(f'({bkg_spec})')
    else:
        mylogging.error('Either Energy spectrum, RMF, or ARF not found')
        mylogging.error(f'({spec})')
        mylogging.error(f'({rmf_file})')
        mylogging.error(f'({arf_file})')

    # Energy spectrum created via nibackgen3c50
    if spec_3c50.is_file() and rmf_file.is_file() and arf_file.is_file():
        ui.clean()
        ui.load_pha('3c50',str(spec_3c50))
        ui.load_arf('3c50',str(arf_file))
        ui.load_rmf('3c50',str(rmf_file)) 
        spectrum_data_3c50 = ui.get_data_plot('3c50')
        ax.errorbar(spectrum_data_3c50.x,spectrum_data_3c50.y,spectrum_data_3c50.yerr,
            color='purple',label='Spectrum (3c50)',zorder=5)

        if bkg_spec_3c50.is_file(): 
            ui.load_bkg('3c50',str(bkg_spec_3c50))
            bkg_spectrum_data_3c50 = ui.get_bkg_plot('3c50')
            ax.errorbar(bkg_spectrum_data_3c50.x,bkg_spectrum_data_3c50.y,bkg_spectrum_data_3c50.yerr,
                fmt='--',color='purple',label='Bkg (3c50)',zorder=5)
        else:
            mylogging.error('Background energy spectrum (3c50) not found')
            mylogging.error(f'({bkg_spec_3c50})')
    else:
        mylogging.error('Either Energy spectrum (3c50), RMF, or ARF not found')
        mylogging.error(f'({spec_3c50})')
        mylogging.error(f'({rmf_file})')
        mylogging.error(f'({arf_file})')

    ax.set_xlim([0.15,16])
    ax.set_xlabel('Energy [keV]',fontsize=16)
    ax.set_ylabel('Counts/Sec/keV',fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='-')
    ax.legend()

    # s = xspec.Spectrum(str(spec))
    # s.response = rmf_file
    # s.response.arf = arf_file
    
    # if bkg_spec.is_file():
    #     s.background = str(bkg_spec)
        
    # xspec.Plot.xAxis = "keV"
    # s.ignore("**-0.2 12.0-**")
    
    # xspec.Plot.device = '/null'
    # xspec.Plot("data")
    # xspec.Plot.addCommand("rebin 3 35")
    # xVals = xspec.Plot.x()
    # yVals = xspec.Plot.y()
    # # To get a background array, Plot.background must be set prior to plot
    # xspec.Plot.background = True
    # xspec.Plot("data")
    # if bkg_spec.is_file():
    #     bkg = xspec.Plot.backgroundVals()
    # # Retrieve error arrays
    # xErrs = xspec.Plot.xErr()
    # yErrs = xspec.Plot.yErr()
    
    # ax.errorbar(xVals, yVals, yerr=yErrs, xerr=xErrs, fmt='k')
    # if os.path.isfile(bkg_spec):
    #     ax.plot(xVals,bkg,'red',label='Background')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.grid(b=True, which='major', color='grey', linestyle='-')
    # ax.legend()
    # ax.set_xlabel('Energy [keV]',fontsize=14)
    # ax.set_ylabel('counts/cm$^2$/sec/keV',fontsize=14)
    
    for i,en_band in enumerate(en_bands):
        low_en = float(en_band[0])
        high_en = float(en_band[1])
        ylims = ax.get_ylim()
        rect = Rectangle((low_en,ylims[0]),high_en-low_en,
            ylims[0]+10**(math.log10(ylims[0])+1/2),
            color=colors[i],fill=True)
        ax.add_patch(rect)
        
    plot2 = root_std_plot_dir/'energy_spectrum.jpeg'
    plots += [plot2]
    fig.savefig(plot2, dpi=300)
    # -----------------------------------------------------------------
    

    # PLOT(S)3: Comparing ufa and lc data
    # -----------------------------------------------------------------
    mylogging.info('Plotting comparing plots between ufa and cl')
    # a) Lightcurve
    # *****************************************************************
    # Computing lightcurve list
    if not ufa_lc_list_file.is_file():
        mylogging.info('ufa Lightcurve does not exist. Computing it..')
        ufa_gti = Gti.read_fits(ufa_evt_file)
        ufa_event_list = Event.read_fits(ufa_evt_file).split(ufa_gti)
        print('Ufa event read')
        ufa_lcs = Lightcurve.from_event(ufa_event_list,time_res=tres,
            low_en=main_en_band[0],high_en=main_en_band[1]) 
        print('ufa lightcurve computed')
        ufa_lcs.save(ufa_lc_list_file)
        #ufa_gti.save(ufa_gti_file)
    else:
        ufa_lcs = LightcurveList.load(ufa_lc_list_file)
        #ufa_gti = Gti.load(ufa_gti_file)

    fig,ax = plt.subplots(figsize=(8,5))
    fig.suptitle(f'{tseg} s bin lightcurve',fontsize=14)
    if ufa_lc_list_file.is_file() and main_seg_lc_list_file.is_file():
        # Splitting lightcurves 
        ufa_lcs_tseg = ufa_lcs.split(tseg) >= tseg
        cl_lcs_tseg = main_seg_lc_list

        mylogging.info('Plotting ufa/cl count rate')

        ufa_lcs_tseg.plot(ax=ax,label='ufa',color='k',zero_start=False,
            ybar=False)
        cl_lcs_tseg.plot(ax=ax,label='cl',color='orange',
            ybar=False,zero_start=False)
        ax.grid(b=True, which='major', color='grey', linestyle='-')
        ax.legend(title='File')
    else:
        mylogging.error('Either the ufa or the main_lc_ist_file does not exist.')
        mylogging.error(f'({ufa_lc_list_file})')
        mylogging.error(f'({main_seg_lc_list_file})')

    plot3a = std_plot_dir/'ufa_vs_cl_lc_T{}_{}.jpeg'.format(tres,16)
    plots += [plot3a]
    fig.savefig(plot3a, dpi=300)
    # *****************************************************************

    # b) Power Spectrum
    # *****************************************************************

    # Computing power
    # (I want lightcurve of exactly 128 s)
    mylogging.info('Computing ufa/cl power spectra')
    if ufa_lc_list_file.is_file() and main_seg_lc_list_file.is_file():
        ufa_lcs_tseg = ufa_lcs.split(tseg) >= tseg
        #cl_lcs_tseg = LightcurveList.load(main_seg_lc_list_file)
        cl_lcs_tseg = main_seg_lc_list

        ufa_power_list = PowerSpectrum.from_lc(ufa_lcs_tseg)
        cl_power_list = PowerSpectrum.from_lc(cl_lcs_tseg)

        ufa_leahy = ufa_power_list.average('leahy')
        ufa_leahy_rebin = ufa_leahy.rebin(rebin)
        if max(ufa_leahy_rebin.freq) > 3500:
            ufa_sub_poi = ufa_leahy.sub_poi(low_freq=3000)
        else:
            ufa_sub_poi = ufa_leahy.sub_poi(value=2.)
        ufa_rms = ufa_sub_poi.normalize('rms')
        ufa_rms_rebin = ufa_rms.rebin(rebin)

        cl_leahy = cl_power_list.average('leahy')
        cl_leahy_rebin = cl_leahy.rebin(rebin)
        if max(cl_leahy_rebin.freq) > 3500:
            cl_sub_poi = cl_leahy.sub_poi(low_freq=3000)
        else:
            cl_sub_poi = cl_leahy.sub_poi(value=2.)
        cl_rms = cl_sub_poi.normalize('rms')
        cl_rms_rebin = cl_rms.rebin(rebin)
    else:
        mylogging.error('Either the ufa or the main_lc_ist_file does not exist.')
        mylogging.error(f'({ufa_lc_list_file})')
        mylogging.error(f'({main_seg_lc_list_file})')

    # Plotting
    mylogging.info('Plotting ufa/cl power spectra')
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    fig.suptitle('Full energy band power spectra', fontsize=14)
    if ufa_lc_list_file.is_file() and main_seg_lc_list_file.is_file():
        ufa_leahy_rebin.plot(ax=ax1,label='ufa',color='k')
        cl_leahy_rebin.plot(ax=ax1,label='cl',color='orange')
        ufa_rms_rebin.plot(ax=ax2,label='ufa',color='k',xy=True)
        cl_rms_rebin.plot(ax=ax2,label='cl',color='orange',xy=True)
        ax1.legend(title='Event file')
        ax1.grid(b=True, which='major', color='grey', linestyle='-')
        ax2.grid(b=True, which='major', color='grey', linestyle='-')
        ax.legend(title='Event file')

        fig.tight_layout(w_pad=1,rect=[0,0,1,0.98])
    else:
        mylogging.error('Either the ufa or the main_lc_ist_file does not exist.')
        mylogging.error(f'({ufa_lc_list_file})')
        mylogging.error(f'({main_seg_lc_list_file})')

    plot3b = std_plot_dir/'ufa_vs_cl_power_spectrum_T{}_{}.jpeg'.\
                            format(tres,tseg)
    plots += [plot3b]
    fig.savefig(plot3b, dpi=300)
    # *****************************************************************
    del ufa_lcs
    del ufa_lcs_tseg
    # -----------------------------------------------------------------
    

    # PLOT4: Full power spectrum
    # -----------------------------------------------------------------
    mylogging.info('Plotting full power spectrum')
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Full Power spectrum',fontsize=14)
    
    if main_pw_list_file.is_file():
        # Leahy power
        power_list = PowerList.load(main_pw_list_file)
        leahy = power_list.average('leahy')
        leahy_rebin = leahy.rebin(rebin)
        leahy_rebin.plot(ax=ax1)
        
        # RMS power
        if max(leahy.freq) > 3500:
            sub_poi = leahy.sub_poi(low_freq=3000)
        else:
            sub_poi = leahy.sub_poi(value=2)
            
        if os.path.isfile(bkg_spec):
            full_bkg_cr, full_bkg_cr_err = get_cr(bkg_spec,low_en=0.5,high_en=10.)
        else:
            full_bkg_cr, full_bkg_cr_err = 0,0
        
        rms = sub_poi.normalize('rms',bkg_cr=full_bkg_cr)
        rms_rebin = rms.rebin(rebin)
        rms_rebin.plot(ax=ax2,xy=True)
    else:
        mylogging.error('Full Power spectrum file not found')
        mylogging.error(f'({main_pw_list_file})')
        
    ax1.grid(b=True, which='major', color='grey', linestyle='-')
    ax2.grid(b=True, which='major', color='grey', linestyle='-')
    fig.tight_layout(w_pad=1,rect=[0,0,1,0.98])
    
    plot4 = std_plot_dir/'full_power_spectrum_T{}_{}.jpeg'.format(tres,tseg)
    plots += [plot4]
    fig.savefig(plot4, dpi=300)
    # -----------------------------------------------------------------

    # PLOT5: different energy bands power spectra
    # -----------------------------------------------------------------
    mylogging.info('Plotting different energy bands power spectra')
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.tight_layout()
    fig.suptitle('Multi-band Power spectrum',fontsize=14)  
    
    bkgs_cr, bkgs_cr_err = [],[]
    for i,pw_file in enumerate(power_list_files):
        
        low_en, high_en = en_bands[i][0],en_bands[i][1]
        
        if pw_file.is_file():
            
            # Leahy power
            power_list = PowerList.load(pw_file)
            leahy = power_list.average('leahy')
            leahy_rebin = leahy.rebin(rebin)
            leahy_rebin.plot(ax=ax1,label='{}-{}'.format(low_en,high_en),color=colors[i])
            
            # RMS power
            if max(leahy.freq) > 3500:
                sub_poi = leahy.sub_poi(low_freq=3000)
            else:
                sub_poi = leahy.sub_poi(value=2)
                
            if bkg_spec.is_file():
                bkg_cr, bkg_cr_err = get_cr(bkg_spec,low_en=float(low_en),high_en=float(high_en))
            else:
                bkg_cr, bkg_cr_err = 0, 0
            bkgs_cr += [bkg_cr]
            bkgs_cr_err += [bkg_cr_err]

            rms = sub_poi.normalize('rms',bkg_cr=bkg_cr)
            rms_rebin = rms.rebin(rebin)
            rms_rebin.plot(ax=ax2,xy=True,label='{}-{}'.format(low_en,high_en),color=colors[i])
        
        else:
            mylogging.error('Single power list file not found')
            mylogging.error(f'({pw_file})')

    ax1.grid(b=True, which='major', color='grey', linestyle='-')
    ax2.grid(b=True, which='major', color='grey', linestyle='-')
    ax1.legend(title='[keV]')
    fig.tight_layout(w_pad=1,rect=[0,0,1,0.98])

    plot5 = os.path.join(std_plot_dir,'multi_band_power_spectrum_T{}_{}.jpeg'.format(tres,tseg))
    plots += [plot5]
    fig.savefig(plot5, dpi=300)
    # -----------------------------------------------------------------
    
    
    # PLOT6: Full power spectra per GTI
    # ----------------------------------------------------------------- 
    mylogging.info('Plotting Full power spectra per GTI')
    if main_pw_list_file.is_file():
        power_list = PowerList.load(main_pw_list_file)
        n_gtis = power_list[0].meta_data['N_GTIS']
        n_plots_pp = 3 # Gti plots per ax
        chunkss = chunks(n_gtis,n_plots_pp)

        colors2 = [item for key,item in mcolors.TABLEAU_COLORS.items()]

        for i,chunk in enumerate(chunkss):
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))

            fig.suptitle('Power spectra per GTI ({}/{})'.format(i+1,len(chunkss)), fontsize=14)
            #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            #fig.tight_layout()

            for j,gti_index in enumerate(chunk):
                local_pw_list = PowerList([pw for pw in power_list if pw.meta_data['GTI_INDEX']==gti_index])
                n_segs = len(local_pw_list)

                local_leahy = local_pw_list.average('leahy')
                local_leahy_rebin = local_leahy.rebin(rebin)
                if max(local_leahy.freq) > 3500:
                    local_sub_poi = local_leahy.sub_poi(low_freq=3000)
                else:
                    local_sub_poi = leahy.sub_poi(value=2.)
                local_rms = local_sub_poi.normalize('rms',bkg_cr=full_bkg_cr)
                local_rms_rebin = local_rms.rebin(rebin)   

                local_leahy_rebin.plot(ax=ax1,color=colors2[j],label = f'{gti_index+1} ({n_segs})')
                local_rms_rebin.plot(ax=ax2,xy=True,color=colors2[j],label = f'{gti_index+1} ({n_segs})')
                #ax2.set_ylabel('')

            ax1.legend(title='GTI (n. segs)')
            ax1.grid(b=True, which='major', color='grey', linestyle='-')
            ax2.grid(b=True, which='major', color='grey', linestyle='-')
            fig.tight_layout(w_pad=1,rect=[0,0,1,0.98])
            
            plotx = std_plot_dir/'full_power_spectrum_{}_T{}_{}.jpeg'.\
                                 format(i,tres,tseg)
            plots += [plotx]
            fig.savefig(plotx, dpi=300)
    else:
        mylogging.error('Main energy band PowerList file not found')
        mylogging.error(f'({main_pw_list_file})')
    # ----------------------------------------------------------------- 
    
    
    # Extracting Information
    # =================================================================

    # parent_info is a dictionary defined to populate a pandas data 
    # frame with global information from each obs ID. This will be used 
    # for general plotting

    mylogging.info('Extracting info')
    parent_info = {}

    # Active detectors
    if main_seg_lc_list_file.is_file():

        # Information from lightcurve metadata
        # -------------------------------------------------------------
        n_act_det = main_seg_lc_list[0].meta_data['N_ACT_DET']
        inact_det_list = main_seg_lc_list[0].meta_data['INACT_DET_LIST']
        parent_info['N_ACT_DET'] = n_act_det
        inact_det_list_str = ''
        for el in inact_det_list: inact_det_list_str += f'{el},'
        inact_det_list_str = inact_det_list_str[:-1]

        parent_info['INACT_DET_LIST'] = inact_det_list_str
    
        info_dict = main_seg_lc_list[0].meta_data['INFO_FROM_HEADER']

        for key,item in info_dict.items():
            parent_info[key] = item
        # -------------------------------------------------------------

        parent_info['N_EN_BANDS'] = len(en_bands)
    
        # Computing fractional rms
        # --------------------------------------------------------------
        def comp_rms(power_list_file,bkg_cr):
            pw_list = PowerList.load(power_list_file)
            power = pw_list.average('leahy')
            if max(power.freq) > 3500:
                sub_poi = power.sub_poi(low_freq=3000)
            else:
                sub_poi = power.sub_poi(value=2.)
            rms = sub_poi.normalize('rms',bkg_cr=bkg_cr)
            rms,rms_err = rms.comp_frac_rms(high_freq=60)
            return rms,rms_err
        
        rms,rms_err = 0,0
        if main_pw_list_file.is_file():
            rms,rms_err = comp_rms(main_pw_list_file,full_bkg_cr)
            parent_info['RMS'] = rms
            parent_info['RMS_ERR'] = rms_err
            
        rms_bands,rms_bands_err = [],[]
        for i,power_file in enumerate(power_list_files):
            if power_file.is_file():
                rms_band, rms_band_err = comp_rms(power_file,bkgs_cr[i])
                rms_bands += [rms_band]
                rms_bands_err += [rms_band_err]
                parent_info[f'RMS{i+1}'] = rms_band
                parent_info[f'RMS_ERR{i+1}'] = rms_band_err
        # -----------------------------------------------------------------

        
        # first_column and second_column are defined for printing purposes
        # They are still dictionaries, but the key does not matter much this
        # time, while the item consists of a 2 element list with human-
        # friendly key and description

        descr = {'TELESCOP':'Mission',
                'OBJECT':'Target',
                'OBS_ID':'Obs. ID',
                'DATE-OBS':'Start obs. time',
                'DATE-END':'Stop obs. time',
                'ONTIME':'Clean exposure [s]',
                'NAXIS2':'N. of raw events'}
        
        def truncate(n, decimals=0):
            multiplier = 10 ** decimals
            return int(n * multiplier) / multiplier
        
        ori_exp = info_dict['TSTOP']-info_dict['TSTART']
        filt_exp = truncate((ori_exp-info_dict['ONTIME'])/ori_exp*100,1)
        parent_info['ORI_EXP'] = ori_exp
        parent_info['FILT_EXP'] = filt_exp

        net_cr_hard = other_crs[1]-bkgs_cr[1]
        net_cr_soft = other_crs[0]-bkgs_cr[0]
        hr = truncate(net_cr_hard/net_cr_soft,2)
        parent_info['NET_HARD_CR'] = net_cr_hard
        parent_info['NET_SOFT_CR'] = net_cr_soft
        parent_info['HR'] = hr

        # First column
        # -------------------------------------------------------------
        first_column = {}
        first_column['CDATE'] = ['Cre. date (this file):',my_cdate()]
        for key, item in descr.items():
            first_column[key] = [item+':',str(info_dict[key])]
        first_column['FEXP'] = ['Filtered exp. [%]:',str(filt_exp)]
        first_column['NACTDET'] = ['N. active det.:',
            f'{n_act_det}/{len(all_det)}']
        first_column['IDET'] = ['Inactive det.:',
            str(inact_det_list).replace('[','').replace(']','')]
        first_column['MAIN_EN_BAND'] = ['Main energy band:',
            '{} - {} [keV]'.format(main_en_band[0],main_en_band[1])]
        parent_info['MAIN_EN_BAND'] = '{} - {}'.\
            format(main_en_band[0],main_en_band[1])
        for i,en_band in enumerate(en_bands):
            first_column['EN_BAND{}'.format(i+1)] = ['Energy band {}:'.\
                format(i+1),'{} - {} [keV]'.format(en_band[0],en_band[1])]
            parent_info[f'EN_BAND{i+1}'] = '{} - {}'.\
                format(en_band[0],en_band[1])
        # -------------------------------------------------------------

        # Second column
        # -------------------------------------------------------------
        second_column = {}
        second_column['N_GTIS'] = ['N. GTIs:',str(len(gti))]
        second_column['N_FGTIS'] = [f'N. filtered GTIs (>{tseg}):',
            str(len(gti>=tseg))]
        second_column['N_SEGS'] = ['N. of segments:',
            str(len(main_seg_lc_list))]
        second_column['T_RES'] = ['Time resolution [s]:',
            '{}'.format(np.round(float(tres),11))]
        second_column['T_SEG'] = ['Time segment [s]:','{}'.format(tseg)]
        second_column['CR'] = ['Total count rate [c/s]:',
            '{} '.format(truncate(main_cr,1))+u'\u00B1'+\
            ' {}'.format(truncate(main_cr_err,1))]
        second_column['BKG_CR'] = ['Bkg count rate [c/s]:',
            '{} '.format(truncate(full_bkg_cr,1))+u'\u00B1'+\
            ' {}'.format(truncate(full_bkg_cr_err,1))]
        parent_info['N_GTIS'] = len(gti)
        parent_info['N_FILT_GTIS'] = len(gti>=tseg)
        parent_info['N_SEGS'] = len(main_seg_lc_list)
        parent_info['T_RES'] = tres
        parent_info['TSEG'] = tseg
        parent_info['CR'] = main_cr
        parent_info['CR_ERR'] = main_cr_err
        parent_info['BKG_CR'] = full_bkg_cr
        for i,(cr,cr_err,bkg_cr,bkg_cr_err) in \
            enumerate(zip(other_crs,other_crs_err,bkgs_cr,bkgs_cr_err)):
            second_column['CR{}'.format(i+1)] = \
                ['Count rate band {} [c/s]:'.format(i+1),
                '{} '.format(truncate(cr,1))+u'\u00B1'+\
                ' {}'.format(truncate(cr_err,1))]
            second_column['BKG_CR{}'.format(i+1)] = \
                ['Bkg count rate band {} [c/s]:'.format(i+1),
                '{} '.format(truncate(bkg_cr,1))+u'\u00B1'+\
                ' {}'.format(truncate(bkg_cr_err,1))]         
            parent_info[f'CR{i+1}'] = cr
            parent_info[f'CR_ERR{i+1}'] = cr_err
            parent_info[f'BKG_CR{i+1}'] = bkg_cr
            parent_info[f'BKG_CR_ERR{i+1}'] = bkg_cr_err
        second_column['HR'] = ['Hard/Soft cr ratio:',str(hr)]
        second_column['FRMS'] = ['Fractional rms (<60Hz):',
            '{} '.format(np.round(rms*100,2))+u'\u00B1'+\
            ' {} %'.format(np.round(rms_err*100,2))]
        for i,(rms_band,rms_band_err,en_band) in \
            enumerate(zip(rms_bands,rms_bands_err,en_bands)):
            second_column['FRMS{}'.format(i+1)] = \
                ['Fractional rms (<60Hz) band {}:'.format(i+1),
                '{} '.format(np.round(rms_band*100,2))+u'\u00B1'+\
                ' {} %'.format(np.round(rms_band_err*100,2))]
        # -------------------------------------------------------------

        # Making a single dictionary with all the information needed for
        # printing an obs_ID page
        # -------------------------------------------------------------
        cols_dict = {'COL1':first_column,'COL2':second_column,'PLOTS':plots}
            
        std_prod_dir = obs_id_dir/'std_prods'
        if not std_prod_dir.is_dir():
            print('std_prods dir does not exist, creating it')
            os.mkdir(std_prod_dir)  
        output = std_prod_dir/'info_dict_T{}_{}.pkl'.format(tres,tseg)
        with open(output, 'wb') as handle:
            pickle.dump(cols_dict, handle, pickle.HIGHEST_PROTOCOL)
        # -------------------------------------------------------------

        # Updating (or creating) global data frame with single observation
        # -------------------------------------------------------------
        parent_std_prod_dir = obs_id_dir.parent/'std_prods'
        if not parent_std_prod_dir.is_dir():
            mylogging.info('Parent std_prods for does not exist, creating it')
            os.mkdir(parent_std_prod_dir)
        df_name = parent_std_prod_dir/'info_data_frame_T{}_{}.pkl'.format(tres,tseg)
        if df_name.is_file():
            df = pd.read_pickle(df_name)
            df = df.append(parent_info,ignore_index=True)
            df.to_pickle(df_name)
        else:
            df = pd.DataFrame(columns=parent_info.keys())
            df = df.append(parent_info,ignore_index=True)
            df.to_pickle(df_name)
        # -------------------------------------------------------------
    else:
        mylogging.error('The main_lc_ist_file does not exist.')
        mylogging.error(f'({main_seg_lc_list_file})')
        first_column = {}
        second_column = {}

    # =================================================================
    
    return plots,first_column,second_column

def make_nicer_std_prod_single(obs_id_dir,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['0.5','10.0'], en_bands = [['0.5','2.0'],['2.0','10.0']],
    rebin=-30,data_dir=pathlib.Path.cwd(),override=True,dpi=150,suffix=None):
    '''
    Makes plots and a dictionary with information according to user 
    settings. 

    This function is specific for NICER reduced products. It assumes
    that data products (lightcurve lists, power lists, and gti) are 
    already computed. If the function does not find the products, it 
    will produce a plot anyway, but it will be empty.
    These products are expected to have a format E<1>_<2>_T<3>_<4>, 
    where 1 and 2 are the energy band boundaries and 3 and 4 are time 
    resolution and time segment, respectively. 

    PARAMETERS
    ----------
    obs_id_dir: string or pathlib.Path
        full path of the obs ID folder
    tres: string (optional)
        time resolution of the reduced products
        (default is 0.0001220703125)
    tseg: string (optional)
        time segment of the reduced products
        (default is 128.0)
    main_en_band: list (optional)
        list with low and high main energy band
        (default is ['0.5','10.0'])
    en_bands: list (optional)
        list of lists, it contains low and high energy band boundaries
        for each sub (different and smaller than main) energy band
        (default is [['0.5','2.0'],['2.0','10.0']]).
        The maximum number of energy bands is five.
    rebin: int (optional)
        rebin factor for plotting the power spectra
        (default is -30)
    data_dir: string or pathlib.Path (optional)
        folder containing the energy spectrum
        (default is pathlib.Path.cwd())
    override: boolean (optional)
        If False existing plots will NOT be overwrittern (default is
        True)
    dpi: int (optional)
        dpi of output format. Default is 200. 100 is good enough for 
        screen visualization, 300 for laser printing.
    suffix: str or None (optional)
        if not None and if std_prods/plots sub-directories already exist,
        it will make a new folder attaching the suffix at the end of the
        default sub-directory name

    RETURNS
    -------
    std_prods: dict
        Dictionary containing the full path of std_plots 
        (std_prods['plots']) and extracted data (std_prods['data'])

    TODO
    ----
    2022 03 19, Stefano Rapisarda (Uppsala)
        The override option is not very well implemented. For the plots,
        information is extracted and plotted after a conditional. If the
        plot exists, plot should NOT be performed and information should
        be read from existing products.
        I should break the code into individual functions with the option
        of returning data, plots, or both.

    HISTORY
    -------
    2020 12 ##, Stefano Rapisarda (Uppsala), creation date
    2020 03 10, Stefano Rapisarda (Uppsala)
        Cleaned up. Now it returns objects to be used directly
        in print_std_prods
    2021 09 12, Stefano Rapisarda (Uppsala)
        Removed arf, rmf, and log_name, adopted LoggingWrapper approach
    2022 03 19, Stefano Rapisarda (Uppsala)
        Big rennovation. Now the routine does just what it says: it 
        computes standard products, i.e. plots stored in jpeg and a 
        dictionary with meaningful information. Arranging this information 
        in a PDF page or in pandas dataframe will be delegated to other 
        routines.
        Efficiency has been improved, now information is read as soon as
        the corresponding data product is open. Data products and plots 
        are manually cleaned from memory after use.
    '''

    def save_and_clear(fig,plot_name,dpi=dpi):
        fig.savefig(plot_name, dpi=dpi)
        fig.clear()
        plt.close()

    mylogging = LoggingWrapper()

    if type(obs_id_dir) == str: obs_id_dir = pathlib.Path(obs_id_dir)
    if type(data_dir) == str: data_dir = pathlib.Path(data_dir)

    obs_id = obs_id_dir.name
  
    # You can specify a maximum of five energy bands
    colors = ['red','blue','green','orange','brown']
    markers = ['s','^','p','H','X']
    if len(en_bands) > len(colors):
        raise ValueError('Energy bands cannot be more than 5')

    mylogging.info('*'*72)
    mylogging.info(str_title('make_nicer_std_prod_single'))
    mylogging.info('*'*72+'\n')


    # Making directories
    # -----------------------------------------------------------------
    root_std_plot_dir = obs_id_dir/'std_plots'
    if not root_std_plot_dir.is_dir():
        mylogging.info('root std_plots directory does not exist, creating one...')
        os.mkdir(root_std_plot_dir)
    else:
        mylogging.info('root std_plots directory already exists.')

    root_std_prod_dir = obs_id_dir/'std_prods'
    if not root_std_prod_dir.is_dir():
        mylogging.info('root std_prods directory does not exist, creating one...')
        os.mkdir(root_std_prod_dir)
    else:
        mylogging.info('root std_prods directory already exists.')    
    # -----------------------------------------------------------------
    

    # Defining names of files to read
    # -----------------------------------------------------------------
    
    # Main energy band str identifier
    main_root_str_identifier = 'E{}_{}_T{}'.\
        format(main_en_band[0],main_en_band[1],tres)
    main_str_identifier = main_root_str_identifier + f'_{tseg}'

    # Main energy band lc_list files 
    main_gti_lc_list_file = obs_id_dir/'lc_list_{}.pkl'.\
        format(main_root_str_identifier)
    main_seg_lc_list_file = obs_id_dir/'lc_list_{}.pkl'.\
        format(main_str_identifier)
    ufa_lc_list_file = obs_id_dir/'ufa_lc_list_{}.pkl'.\
        format(main_root_str_identifier)

    main_pw_list_file = obs_id_dir/'power_list_{}.pkl'.\
        format(main_str_identifier)

    # Full energy band GTI name
    main_gti_file = obs_id_dir/'gti_E{}_{}.gti'.\
        format(main_en_band[0],main_en_band[1])

    # Other energy band names
    seg_lc_list_files = []
    gti_lc_list_files = []
    power_list_files = []
    for en_band in en_bands:
        low_en, high_en = en_band[0], en_band[1]
        seg_lc_list_files += [obs_id_dir/'lc_list_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg)]
        gti_lc_list_files += [obs_id_dir/'lc_list_E{}_{}_T{}.pkl'.\
            format(low_en,high_en,tres)]   
        power_list_files += [obs_id_dir/'power_list_E{}_{}_T{}_{}.pkl'.\
            format(low_en,high_en,tres,tseg)]
    
    event_cl_dir = data_dir/obs_id/'xti'/'event_cl'

    # Energy spectrum name
    spec = event_cl_dir/f'{obs_id}_spectrum_bdc.pha'
    bkg_spec = event_cl_dir/f'{obs_id}_spectrum_bdc_bkg.pha'
    spec_3c50 = event_cl_dir/f'{obs_id}_spectrum_bdc_3C50_tot.pi'
    bkg_spec_3c50 = event_cl_dir/f'{obs_id}_spectrum_bdc_3C50_bkg.pi'

    # arf and rmf files
    arf_file = event_cl_dir/f'arf_bdc.arf'
    rmf_file = event_cl_dir/f'rmf_bdc.rmf'

    # ufa file
    ufa_evt_file = list_items(event_cl_dir,itype='file',
        include_and=[f'ni{obs_id}_0mpu7_ufa.evt'])[0]
    # -----------------------------------------------------------------

    # Making std_prods and std_plots sub directories
    # -----------------------------------------------------------------
    std_plot_dir = root_std_plot_dir/main_str_identifier
    if not std_plot_dir.is_dir():
        mylogging.info('std_plots sub-directory does not exist, creating one...')
        os.mkdir(std_plot_dir)
    else:
        mylogging.info('std_plots sub-directory already exists.')
        if not suffix is None:
            mylogging.info(f'Creating std_plots sub-directory with {suffix} suffix')
            os.mkdir(std_plot_dir.parent/std_plot_dir.name+f'_{suffix}')

    std_prod_dir = root_std_prod_dir/main_str_identifier
    if not std_prod_dir.is_dir():
        mylogging.info('std_prods sub-directory does not exist, creating one...')
        os.mkdir(std_prod_dir)
    else:
        mylogging.info('std_prods sub-directory already exists.')  
        if not suffix is None:
            mylogging.info(f'Creating std_prods sub-directory with {suffix} suffix')
            os.mkdir(std_prod_dir.parent/std_prod_dir.name+f'_{suffix}')      
    # -----------------------------------------------------------------


    # Printing some info
    # -----------------------------------------------------------------
    mylogging.info('')
    mylogging.info('Obs ID: {}'.format(obs_id))
    mylogging.info('Settings:')
    mylogging.info('-'*72)
    mylogging.info('Selected main energy band: {}-{} keV'.\
        format(main_en_band[0],main_en_band[1]))
    for i,en_band in enumerate(en_bands):
        mylogging.info('Selected energy band {}: {}-{} keV'.\
            format(i,en_band[0],en_band[1]))        
    mylogging.info('Selected time resolution: {} s'.format(tres)) 
    mylogging.info('Selected time segment: {} s'.format(tseg)) 
    mylogging.info('-'*72)
    mylogging.info('')
    # -----------------------------------------------------------------
    
    # Initializing std_prods
    std_prods = {'plots':{},'data':{}}
    std_prods['plots'] = {
        'Obs. ID count rate': None,
        'Count rate per GTI': [],
        'Energy spectrum': None,
        'ufa/cl count rate per segment': None,
        'ufa/cl power spectra': None,
        'Main Power Spectrum': None,
        'Other Power Spectra': None,
        'Power Spectra per GTI': []
        }
    std_prods['data'] = {
        'main_cr' : None,
        'main_cr_err' : None,
        'other_crs' : [None for e in en_bands],
        'other_cr_err' : [None for e in en_bands],
        'main_bkg_cr' : 0,
        'main_bkg_cr_err': 0,
        'other_bkg_crs' : [0 for e in en_bands],
        'other_bkg_cr_err' : [0 for e in en_bands],
        'main_frac_rms' : None,
        'main_frac_rms_err' : None,
        'other_frac_rms' : [0 for e in en_bands],
        'other_frac_rms_err' : [0 for e in en_bands],
        'n_act_det' : None,
        'inact_det_list': None,
        'info_dict' : None,
        'n_gtis' : None,
        'n_fgtis' : None,
        'n_segs' : None,
        'n_en_bands': len(en_bands),
        'main_en_band': '{} - {}'.\
            format(main_en_band[0],main_en_band[1]),
        'other_en_bands': ['{} - {}'.format(low,high) for low,high in en_bands],
        'tres': tres,
        'tseg': tseg,
        'std_plot_dir': std_plot_dir,
        'std_prod_dir': std_prod_dir,
        'creation_date': my_cdate()
        }

    # Plotting
    # =================================================================
    # For each plot I will create the figure anyway, to preserve the 
    # std_prod page layout. Then, if a file does not exist or something 
    # goes wrong, the figure will be empty.
    # PLOT1: Count rate per GTI in different energy band
    # PLOT(S)1x : x count rate per segment in different energy bands
    # PLOT2: Energy spectrum
    # PLOT(S)3: Comparison between ufa and cl lightcurve and power spectra
    # PLOT4: Main energy band average power spectrum
    # PLOT5: Average power spectra in different energy bands
    # PLOT6x: x Main energy band power spectrum per GTI

    plt.tight_layout()
    
    # PLOT1: count rate per GTI 
    # -----------------------------------------------------------------
    mylogging.info('Plotting Obs. ID count rate')
    
    # As GTIs are selected ALSO according to the selected segment
    # length, this plot depends both on tres and tseg
    plot_name = std_plot_dir/'obs_id_cr.jpeg'
    std_prods['plots']['Obs. ID count rate'] = plot_name

    gti_seg_flag = True
    if plot_name.is_file() and not override:
        
        mylogging.info('Obs. ID count rate plot already exists')
        
        # Reading GTI 
        gti = Gti.load(main_gti_file)
        gti_seg = gti>=float(tseg)
        
        # Saving GTI data
        std_prods['data']['n_gtis'] = len(gti)
        std_prods['data']['n_fgtis'] = len(gti_seg)
    
    else:
        
        fig,ax = plt.subplots(figsize=(8,5))
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'Obs. ID count rate',fontsize=14)

        if main_gti_lc_list_file.is_file() and main_gti_file.is_file():

            # Loading GTI and saving GTI data
            gti = Gti.load(main_gti_file)
            std_prods['data']['n_gtis'] = len(gti)
            gti_seg = gti>=float(tseg)
            std_prods['data']['n_fgtis'] = len(gti_seg)

            fig.suptitle(
                f'Obs. ID count rate (total GTI: {len(gti)}, filt. GTI: {len(gti_seg)})',
                fontsize=14
                )

            # Loading file
            gti_lc_list = LightcurveList.load(main_gti_lc_list_file)
            gti_start = gti_lc_list[0].time.iloc[0]

            # Plotting
            label = '{}-{}'.format(main_en_band[0],main_en_band[1])
            gti_lc_list.plot(ax=ax,color='k',lfont=14,label=label,
                xbar=True,ybar=6)
            del gti_lc_list

            # Drawing marks for each GTI
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            bottom,top = ax.get_ylim()
            for gstart,gstop,gdur in zip(gti.start,gti.stop,gti.dur):
                
                mid_point = (gstart+gstop)/2-gti_start
                
                color = 'forestgreen'
                bottom_marker = '^'
                if gdur < float(tseg): 
                    color = 'orange'
                    bottom_marker = 'x'
                
                #ax.axvline(g-start,ls='--',color='orange')
                ax.plot([mid_point],[bottom],
                    color=color,marker=bottom_marker)
                ax.plot([mid_point],[top],
                    color=color,marker='|')
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # Other energy bands count rate
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            for gti_lc_list_file, en_band, col, marker in \
                zip(gti_lc_list_files, en_bands, colors, markers):
                
                low_en, high_en = en_band[0], en_band[1]
                
                if gti_lc_list_file.is_file():
                    lc_list = LightcurveList.load(gti_lc_list_file)
                    label = '{}-{}'.format(low_en,high_en)
                    lc_list.plot(ax=ax,color=col,label=label,marker=marker,
                        lfont=14,ybar=8)
                    del lc_list
                else:
                    mylogging.error('Single energy band gti_lc_list not found')
                    mylogging.error(f'({gti_lc_list_file})') 
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # I have not idea why I did this, but it works
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.grid(b=True, which='major', color='grey', linestyle='-')
            
            save_and_clear(fig,plot_name)   

        else:

            gti_seg_flag = False
            # If there is not main_gti_lc_list file, the plot will be 
            # empty
            save_and_clear(fig,plot_name)   
            mylogging.error('Main energy band gti_lc_list file not found')
            mylogging.error(f'({main_gti_lc_list_file})')           
    # -----------------------------------------------------------------


    # PLOT1x: Count rate per segment for each GTI
    # -----------------------------------------------------------------
    mylogging.info('Plotting count rate per segment')

    # Opening Lightcurve list files
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # Main en band
    mylogging.info('Reading LightcurveList(s)')

    plot_main_seg_cr_flag = True
    if main_seg_lc_list_file.is_file():

        main_seg_lc_list = LightcurveList.load(main_seg_lc_list_file)
        
        # Reading data
        std_prods['data']['main_cr'] = main_seg_lc_list.cr
        std_prods['data']['main_cr_err'] = main_seg_lc_list.cr_std
        std_prods['data']['n_act_det'] = main_seg_lc_list[0].meta_data['N_ACT_DET']
        std_prods['data']['inact_det_list'] = main_seg_lc_list[0].meta_data['INACT_DET_LIST']
        std_prods['data']['info_dict'] = main_seg_lc_list[0].meta_data['INFO_FROM_HEADER']
        std_prods['data']['n_segs'] = str(len(main_seg_lc_list))

        obs_start = main_seg_lc_list[0].time.iloc[0]

    else:

        mylogging.error('Main energy band seg_lc_list file not found')
        mylogging.error(f'({main_seg_lc_list_file})')
        plot_main_seg_cr_flag = False

    # Other bands
    seg_lc_lists = []
    plot_seg_cr_flag = True
    for e,seg_lc_list_file in enumerate(seg_lc_list_files):
        if seg_lc_list_file.is_file():

            seg_lc_list = LightcurveList.load(seg_lc_list_file)
            seg_lc_lists += [seg_lc_list]

            # Reading data
            std_prods['data']['other_crs'][e] = seg_lc_list.cr
            std_prods['data']['other_cr_err'][e] = seg_lc_list.cr_std
            del seg_lc_list

        else:

            mylogging.error('Single energy band lc_list not found')
            mylogging.error(f'({seg_lc_list_file})')
            plot_seg_cr_flag = False
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
    # Plotting
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if gti_seg_flag:
        for g,(gti_start,gti_stop) in enumerate(zip(gti_seg.start,gti_seg.stop)):

            mylogging.info(f'Plotting GTI {g+1}/{len(gti_seg)}')

            plot_name = std_plot_dir/'cr_per_gti_{}.jpeg'.format(g+1)
            std_prods['plots']['Count rate per GTI'] += [plot_name]
            
            if plot_name.is_file() and not override:
                mylogging.info(f'Count rate per GTI plot n. {g+1} already exists')
            else:
                fig,ax = plt.subplots(figsize=(8,5))
                #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.suptitle(f'Count rate per GTI (GTI {g+1}/{len(gti_seg)})',
                    fontsize=14)  

                # Main band
                if plot_main_seg_cr_flag:
                    label = f'{main_en_band[0]}-{main_en_band[1]}'
                    main_seg_lc_list.plot(ax=ax,color='k',marker='o',
                        lfont=14,label=label,ybar=4) 

                # Other bands
                if plot_seg_cr_flag:
                    for e,(en_band, col, marker) in \
                        enumerate(zip(en_bands, colors, markers)):
                        label='{}-{}'.format(en_band[0],en_band[1])
                        seg_lc_lists[e].plot(ax=ax,color=col,marker=marker,
                            lfont=14,label=label)

                ax.set_xlim([gti_start-obs_start,gti_stop-obs_start])

                # I have not idea why I did this, but it works
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                ax.grid(b=True, which='major', color='grey', linestyle='-')
                #fig.tight_layout(0.5)

                save_and_clear(fig,plot_name)
        
        # Cleaning up
        if plot_main_seg_cr_flag: del main_seg_lc_list
        if plot_seg_cr_flag: del seg_lc_lists
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++          
    # ------------------------------------------------------------------
    

    # PLOT2: Energy spectrum 
    # ------------------------------------------------------------------
    mylogging.info('Plotting energy spectrum')

    plot_name = root_std_plot_dir/'energy_spectrum.jpeg'
    std_prods['plots']['Energy spectrum'] = plot_name
    fig,ax = plt.subplots(figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Energy spectrum', fontsize=14)
    
    # Energy spectrum created via xselect
    if spec.is_file() and rmf_file.is_file() and arf_file.is_file():
        ui.clean()
        ui.load_pha('tot',str(spec))
        ui.load_arf('tot',str(arf_file))
        ui.load_rmf('tot',str(rmf_file))
        spectrum_data = ui.get_data_plot('tot')
        ax.errorbar(spectrum_data.x,spectrum_data.y,spectrum_data.yerr,
            color='k',label='Spectrum',zorder=10)
        
        if bkg_spec.is_file():
            ui.load_bkg('tot',str(bkg_spec))
            bkg_spectrum_data = ui.get_bkg_plot('tot')
            ax.errorbar(bkg_spectrum_data.x,bkg_spectrum_data.y,bkg_spectrum_data.yerr,
                fmt='--',color='k',label='Bkg',zorder=10)
            
            # Reading background count rate
            main_bkg_cr, main_bkg_cr_err = get_cr(
                bkg_spec,low_en=main_en_band[0],high_en=main_en_band[1]
                )
            std_prods['data']['main_bkg_cr'] = main_bkg_cr
            std_prods['data']['main_bkg_cr_err'] = main_bkg_cr_err
            del main_bkg_cr, main_bkg_cr_err

            # Reading other en bands count rate
            for e,(low_en,high_en) in enumerate(en_bands):
                bkg_cr, bkg_cr_err = get_cr(
                    bkg_spec,low_en=float(low_en),high_en=float(high_en)
                    )
                std_prods['data']['other_bkg_crs'][e] = bkg_cr
                std_prods['data']['other_bkg_cr_err'][e] = bkg_cr_err
                del bkg_cr, bkg_cr_err
        else:
            mylogging.error('Background energy spectrum not found')
            mylogging.error(f'({bkg_spec})')
    
    else:

        mylogging.error('Either Energy spectrum, RMF, or ARF not found')
        mylogging.error(f'({spec})')
        mylogging.error(f'({rmf_file})')
        mylogging.error(f'({arf_file})')

    # Energy spectrum created via nibackgen3c50
    if spec_3c50.is_file() and rmf_file.is_file() and arf_file.is_file():
        ui.clean()
        ui.load_pha('3c50',str(spec_3c50))
        ui.load_arf('3c50',str(arf_file))
        ui.load_rmf('3c50',str(rmf_file)) 
        spectrum_data_3c50 = ui.get_data_plot('3c50')
        ax.errorbar(spectrum_data_3c50.x,spectrum_data_3c50.y,spectrum_data_3c50.yerr,
            color='purple',label='Spectrum (3c50)',zorder=5)

        if bkg_spec_3c50.is_file(): 
            ui.load_bkg('3c50',str(bkg_spec_3c50))
            bkg_spectrum_data_3c50 = ui.get_bkg_plot('3c50')
            ax.errorbar(bkg_spectrum_data_3c50.x,bkg_spectrum_data_3c50.y,bkg_spectrum_data_3c50.yerr,
                fmt='--',color='purple',label='Bkg (3c50)',zorder=5)
        else:
            mylogging.error('Background energy spectrum (3c50) not found')
            mylogging.error(f'({bkg_spec_3c50})')
    else:
        mylogging.error('Either Energy spectrum (3c50), RMF, or ARF not found')
        mylogging.error(f'({spec_3c50})')
        mylogging.error(f'({rmf_file})')
        mylogging.error(f'({arf_file})')

    ax.set_xlim([0.15,16])
    ax.set_xlabel('Energy [keV]',fontsize=16)
    ax.set_ylabel('Counts/Sec/keV',fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(b=True, which='major', color='grey', linestyle='-')
    ax.legend()

    # Plotting selected energy bands references   
    for i,en_band in enumerate(en_bands):
        low_en = float(en_band[0])
        high_en = float(en_band[1])
        ylims = ax.get_ylim()
        rect = Rectangle((low_en,ylims[0]),high_en-low_en,
            ylims[0]+10**(math.log10(ylims[0])+1/2),
            color=colors[i],fill=True)
        ax.add_patch(rect)
        
    save_and_clear(fig,plot_name)
    # -----------------------------------------------------------------
    

    # PLOT(S)3: Comparing ufa and lc data
    # -----------------------------------------------------------------
    
    mylogging.info('Plotting comparing plots between ufa and cl')
    
    # a) Lightcurve
    # *****************************************************************
    
    # Computing (or loading) ufa lightcurve list
    if not ufa_lc_list_file.is_file():
        mylogging.info('ufa Lightcurve does not exist. Computing it..')
        
        ufa_gti = Gti.read_fits(ufa_evt_file)
        ufa_event_list = Event.read_fits(ufa_evt_file).split(ufa_gti)
        ufa_lcs = Lightcurve.from_event(ufa_event_list,time_res=tres,
            low_en=main_en_band[0],high_en=main_en_band[1]) 
        
        mylogging.info('Saving ufa Lightcurve')
        ufa_lcs.save(ufa_lc_list_file)

    else:
        ufa_lcs = LightcurveList.load(ufa_lc_list_file)

    fig,ax = plt.subplots(figsize=(8,5))
    fig.suptitle(f'{tseg} s bin lightcurve',fontsize=14)
    plot_name = std_plot_dir/'ufa_vs_cl_lc.jpeg'
    std_prods['plots']['ufa/cl count rate per segment'] = plot_name
        
    if ufa_lc_list_file.is_file() and main_seg_lc_list_file.is_file() \
        and main_gti_file.is_file():
        
        # Splitting lightcurve into specified segments
        # (the >= sign is because if a Lightcurve in a Lightcurve list 
        # has a duration smaller than tseg, then the split method would
        # return that shorter-than-tseg lightcurve. >= tseg, therefore,
        # will ensure that all Lightcurves in ufa_lcs_tseg are longer 
        # than tseg)
        ufa_lcs_tseg = ufa_lcs.split(tseg) >= tseg

        cl_lcs_tseg = Lightcurve.load(main_seg_lc_list_file)
        gti = Gti.load(main_gti_file)
        gti_seg = gti>=float(tseg)

        mylogging.info('Plotting ufa/cl count rate')

        ufa_lcs_tseg.plot(ax=ax,label='ufa',color='k',zero_start=False,
            ybar=False)
        cl_lcs_tseg.plot(ax=ax,label='cl',color='orange',zero_start=False,
            ybar=False) 

        # Printing GTI bars
        for stop in gti_seg.stop:
            ax.axvline(stop,ymin=0,ymax=1,ls='--',color='forestgreen')

        ax.grid(b=True, which='major', color='grey', linestyle='-')
        ax.legend(title='File')

    else:
        mylogging.error('Either the ufa or the main_lc_ist_file does not exist.')
        mylogging.error(f'({ufa_lc_list_file})')
        mylogging.error(f'({main_seg_lc_list_file})')

    save_and_clear(fig,plot_name)
    # *****************************************************************

    # b) Power Spectrum
    # *****************************************************************

    plot_name = std_plot_dir/'ufa_vs_cl_power_spectrum.jpeg'
    std_prods['plots']['ufa/cl power spectra'] = plot_name
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    fig.suptitle('Full energy band power spectra', fontsize=14)
    
    if ufa_lc_list_file.is_file() and main_pw_list_file.is_file():
        
        # Computing power
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        mylogging.info('Computing ufa/cl power spectra')
        
        ufa_power_list = PowerSpectrum.from_lc(ufa_lcs_tseg)  
        cl_power_list = PowerList.load(main_pw_list_file)

        ufa_leahy = ufa_power_list.average('leahy')
        ufa_leahy_rebin = ufa_leahy.rebin(rebin)
        if max(ufa_leahy_rebin.freq) > 3500:
            ufa_sub_poi = ufa_leahy.sub_poi(low_freq=3000)
        else:
            ufa_sub_poi = ufa_leahy.sub_poi(value=2.)
        ufa_rms = ufa_sub_poi.normalize('rms')
        ufa_rms_rebin = ufa_rms.rebin(rebin)

        cl_leahy = cl_power_list.average('leahy')
        cl_leahy_rebin = cl_leahy.rebin(rebin)
        if max(cl_leahy_rebin.freq) > 3500:
            cl_sub_poi = cl_leahy.sub_poi(low_freq=3000)
        else:
            cl_sub_poi = cl_leahy.sub_poi(value=2.)
        cl_rms = cl_sub_poi.normalize('rms')
        cl_rms_rebin = cl_rms.rebin(rebin)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Plotting
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        mylogging.info('Plotting ufa/cl power spectra')

        ufa_leahy_rebin.plot(ax=ax1,label='ufa',color='k')
        cl_leahy_rebin.plot(ax=ax1,label='cl',color='orange')
        ufa_rms_rebin.plot(ax=ax2,label='ufa',color='k',xy=True)
        cl_rms_rebin.plot(ax=ax2,label='cl',color='orange',xy=True)
        ax1.legend(title='Event file')
        ax1.grid(b=True, which='major', color='grey', linestyle='-')
        ax2.grid(b=True, which='major', color='grey', linestyle='-')
        ax.legend(title='Event file')

        fig.tight_layout(w_pad=1,rect=[0,0,1,0.98])
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
        del ufa_lcs,ufa_lcs_tseg,ufa_power_list,cl_power_list
        del ufa_leahy,ufa_leahy_rebin,ufa_sub_poi,ufa_rms,ufa_rms_rebin
        del cl_leahy,cl_leahy_rebin,cl_sub_poi,cl_rms,cl_rms_rebin
    
    else:
    
        mylogging.error('Either the ufa or the main_lc_ist_file does not exist.')
        mylogging.error(f'({ufa_lc_list_file})')
        mylogging.error(f'({main_seg_lc_list_file})')

    save_and_clear(fig,plot_name)
    # ******************************************************************
    # ------------------------------------------------------------------
    

    # PLOT4: Full power spectrum
    # ------------------------------------------------------------------
    mylogging.info('Plotting full power spectrum')
    
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Full Power spectrum',fontsize=14)
    plot_name = std_plot_dir/'main_power_spectrum.jpeg' 
    std_prods['plots']['Main Power Spectrum'] = plot_name
           
    if main_pw_list_file.is_file():
        # Leahy power
        power_list = PowerList.load(main_pw_list_file)
        leahy = power_list.average('leahy')
        leahy_rebin = leahy.rebin(rebin)
        leahy_rebin.plot(ax=ax1)
        
        # RMS power
        if max(leahy.freq) > 3500:
            sub_poi = leahy.sub_poi(low_freq=3000)
        else:
            sub_poi = leahy.sub_poi(value=2)
        
        rms_power = sub_poi.normalize('rms',
            bkg_cr=std_prods['data']['main_bkg_cr'])
        rms_rebin = rms_power.rebin(rebin)
        rms_rebin.plot(ax=ax2,xy=True)

        # Extracting information
        rms, rms_err = rms_power.comp_frac_rms(high_freq=60) 
        std_prods['data']['main_frac_rms'] = rms
        std_prods['data']['main_frac_rms_err'] = rms_err

        del power_list, leahy, leahy_rebin, sub_poi, rms_power,\
            rms_rebin, rms, rms_err
    else:
        mylogging.error('Main Power spectrum file not found')
        mylogging.error(f'({main_pw_list_file})')
        
    ax1.grid(b=True, which='major', color='grey', linestyle='-')
    ax2.grid(b=True, which='major', color='grey', linestyle='-')
    fig.tight_layout(w_pad=1,rect=[0,0,1,0.98])
    save_and_clear(fig,plot_name)
    # -----------------------------------------------------------------


    # PLOT5: different energy bands power spectra
    # -----------------------------------------------------------------
    mylogging.info('Plotting different energy bands power spectra')
    
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,5))
    fig.suptitle('Multi-band Power spectrum',fontsize=14)  
    plot_name = std_plot_dir/'multi_band_power_spectrum.jpeg'
    std_prods['plots']['Other Power Spectra'] = plot_name
        
    for i,pw_file in enumerate(power_list_files):
        
        low_en, high_en = en_bands[i][0],en_bands[i][1]
        
        if pw_file.is_file():
            
            # Leahy power
            power_list = PowerList.load(pw_file)
            leahy = power_list.average('leahy')
            leahy_rebin = leahy.rebin(rebin)
            leahy_rebin.plot(ax=ax1,label='{}-{}'.\
                format(low_en,high_en),color=colors[i])
            
            # RMS power
            if max(leahy.freq) > 3500:
                sub_poi = leahy.sub_poi(low_freq=3000)
            else:
                sub_poi = leahy.sub_poi(value=2)

            rms_power = sub_poi.normalize('rms',
                bkg_cr=std_prods['data']['other_bkg_crs'][i])
            rms_rebin = rms_power.rebin(rebin)
            rms_rebin.plot(ax=ax2,xy=True,label='{}-{}'.format(low_en,high_en),color=colors[i])
            
            # Extracting information
            rms, rms_err = rms_power.comp_frac_rms(high_freq=60)
            std_prods['data']['other_frac_rms'][i] = rms
            std_prods['data']['other_frac_rms_err'][i] = rms_err

            del power_list, leahy, leahy_rebin, sub_poi, rms_power,\
                rms_rebin, rms, rms_err
        else:
            mylogging.error('Single power list file not found')
            mylogging.error(f'({pw_file})')

    ax1.grid(b=True, which='major', color='grey', linestyle='-')
    ax2.grid(b=True, which='major', color='grey', linestyle='-')
    ax1.legend(title='[keV]')
    fig.tight_layout(w_pad=1,rect=[0,0,1,0.98])
    save_and_clear(fig,plot_name)
    # -----------------------------------------------------------------
    
    
    # PLOT6: Full power spectra per GTI
    # ----------------------------------------------------------------- 
    mylogging.info('Plotting Full power spectra per GTI')
    
    if main_pw_list_file.is_file():
        
        power_list = PowerList.load(main_pw_list_file)
        
        n_gtis = power_list[0].meta_data['N_GTIS']
        n_plots_pp = 3 # Gti plots per ax
        chunkss = chunks(n_gtis,n_plots_pp)

        colors2 = [item for key,item in mcolors.TABLEAU_COLORS.items()]

        for i,chunk in enumerate(chunkss):
            
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
            fig.suptitle('Power spectra per GTI ({}/{})'.\
                format(i+1,len(chunkss)), fontsize=14)
            plot_name = std_plot_dir/'full_power_spectrum_per_gti_{}.jpeg'.\
                format(i)
            std_prods['plots']['Power Spectra per GTI'] += [plot_name]
            
            for j,gti_index in enumerate(chunk):
                
                local_pw_list = PowerList([pw for pw in power_list if pw.meta_data['GTI_INDEX']==gti_index])
                n_segs = len(local_pw_list)

                local_leahy = local_pw_list.average('leahy')
                local_leahy_rebin = local_leahy.rebin(rebin)
                if max(local_leahy.freq) > 3500:
                    local_sub_poi = local_leahy.sub_poi(low_freq=3000)
                else:
                    local_sub_poi = leahy.sub_poi(value=2.)
                local_rms = local_sub_poi.normalize('rms',
                    bkg_cr=std_prods['data']['main_bkg_cr'])
                local_rms_rebin = local_rms.rebin(rebin)   

                local_leahy_rebin.plot(ax=ax1,color=colors2[j],label = f'{gti_index+1} ({n_segs})')
                local_rms_rebin.plot(ax=ax2,xy=True,color=colors2[j],label = f'{gti_index+1} ({n_segs})')
                
                del local_pw_list, local_leahy, local_leahy_rebin,\
                    local_sub_poi, local_rms, local_rms_rebin

            ax1.legend(title='GTI (n. segs)')
            ax1.grid(b=True, which='major', color='grey', linestyle='-')
            ax2.grid(b=True, which='major', color='grey', linestyle='-')
            fig.tight_layout(w_pad=1,rect=[0,0,1,0.98])
        
            save_and_clear(fig,plot_name)
    else:
        mylogging.error('Main energy band PowerList file not found')
        mylogging.error(f'({main_pw_list_file})')
    # ----------------------------------------------------------------- 
    
    # Saving standard produts
    mylogging.info('Saving std_prods dictionary')
    dict_name = std_prod_dir/'std_prods.pkl'
    with open(dict_name, 'wb') as output:
        pickle.dump(std_prods, output, pickle.HIGHEST_PROTOCOL)

    mylogging.info('*'*72)
    mylogging.info(str_title(f'exiting make_std_prod_single'))
    mylogging.info('*'*72+'\n')    

    return dict_name

def print_std_prods(obs_id_dirs,main_en_band=['0.5','10.0'],
    tres='0.0001220703125',tseg='128.0',suffix=None,add_plots=[]):

    def read_bookmark(reader):
        outlines = reader.outlines
        
        parent_title = outlines[0]['/Title']
        parent_page = outlines[0]['/Page']
        titles = []
        pages = []
        if len(outlines) > 1:
            for i in range(len(outlines[1])):
                titles += [outlines[1][i]['/Title']]
                pages  += [outlines[1][i]['/Page']]
        return [parent_title,parent_page],[titles,pages]

    mylogging = LoggingWrapper()

    time_string = 'T{}_{}'.format(tres,tseg)
    if suffix is None:    
        dir_string = 'E{}_{}_{}'.format(
            main_en_band[0],main_en_band[1],time_string
            )
    else:
        dir_string = 'E{}_{}_{}_{}'.format(
            main_en_band[0],main_en_band[1],time_string,suffix
            )        

    first_obs_id_dir = obs_id_dirs[0]
    if type(first_obs_id_dir) == str:
        first_obs_id_dir = pathlib.Path(first_obs_id_dir)
    an_dir = first_obs_id_dir.parent

    pdf_files = []
    for o,obs_id_dir in enumerate(obs_id_dirs):

        if isinstance(obs_id_dir,str):
            obs_id_dir = pathlib.Path(obs_id_dir)
        
        # Extracting obs. ID
        obs_id = obs_id_dir.name

        mylogging.info('Printing std prods for obs. ID {} ({}/{})'.\
            format(obs_id,o+1,len(obs_id_dirs)))

        # Checking existance of std_prods dir inside the obs_id_dir
        std_prod_dir = obs_id_dir/'std_prods'/dir_string
        if not std_prod_dir.is_dir():
            mylogging.info('std_prods dir does not exist, skipping obs.')
            mylogging.info(std_prod_dir)
            continue

        try:
            pdf_file = print_std_prod_single(
                obs_id_dir,
                main_en_band=main_en_band,
                tres=tres,tseg=tseg,suffix=suffix
                )
        except:
            mylogging.error(f'Something went wrong with obs. ID {obs_id}')
            continue

        pdf_files += [pdf_file]

    mylogging.info('Making total PDF')

    current_page = 0
    merger = PdfFileMerger()

    if add_plots != []:
        mylogging.info('Adding additional plots')
        for plot_to_add in add_plots:
            reader = PdfFileReader(plot_to_add[0])
            parent, children = read_bookmark(reader)
            parent_bookmark = merger.addBookmark(parent[0],parent[1]+current_page)
            for title,page in zip(children[0],children[1]):
                merger.addBookmark(title,page+current_page,parent=parent_bookmark)
            merger.append(reader,bookmark=plot_to_add[1]+current_page)
            current_page += reader.getNumPages()

    mylogging.info('Merging pdf files')
    for pdf_file in pdf_files:
        if pdf_file.is_file():
            with open(pdf_file,'rb') as infile:
                reader = PdfFileReader(infile)
                parent, children = read_bookmark(reader)
                parent_bookmark = merger.addBookmark(parent[0],parent[1]+current_page)
                for title,page in zip(children[0],children[1]):
                    merger.addBookmark(title,page+current_page,parent=parent_bookmark)
                merger.append(PdfFileReader(infile))
                current_page += reader.getNumPages()

    # Saving file
    output_file = an_dir/'std_prods'/dir_string/'std_prods.pdf'

    merger.write(str(output_file))

def make_general_info(obs_id_dirs,main_en_band=['0.5','10.0'],
    tres='0.0001220703125',tseg='128.0',suffix=None,clean=True):

    mylogging = LoggingWrapper()

    mylogging.info('*'*72)
    mylogging.info(str_title('make general info'))
    mylogging.info('*'*72)

    # Setting directory identifiers
    time_string = 'T{}_{}'.format(tres,tseg)
    if suffix is None:
        dir_string = 'E{}_{}_{}'.format(
            main_en_band[0],main_en_band[1],time_string)
    else:
        dir_string = 'E{}_{}_{}_{}'.format(
            main_en_band[0],main_en_band[1],time_string,suffix)
    
    # Setting directories
    an_dir = obs_id_dirs[0].parent
    if (an_dir/'std_prods').is_dir():
        mylogging.info('general_std_prod_dir already exists')
    else:
        mylogging.info('Creating general_std_prod_dir')
        os.mkdir(an_dir/'std_prods')
    specific_std_prod_dir = an_dir/'std_prods'/dir_string
    if specific_std_prod_dir.is_dir():
        mylogging.info('specific_std_prod_dir dir already exists')
    else:
        mylogging.info('Creating specific_std_prod_dir dir')
        os.mkdir(specific_std_prod_dir)

    df_name = specific_std_prod_dir/'general_info_data_frame.pkl'
    first_good = True
    o = 0
    for obs_id_dir in obs_id_dirs:

        obs_id_full_info = {}

        # Setting directories
        obs_id = obs_id_dir.name
        an_dir = obs_id_dir.parent
        std_prod_dir = obs_id_dir/'std_prods'/dir_string
        std_prod_file = std_prod_dir/f'std_prods.pkl'

        mylogging.info(f'Reading Obs. ID {obs_id} ({o}/{len(obs_id_dirs)})')
        o += 1

        if std_prod_file.is_file():
            
            with open(std_prod_file,'rb') as infile:
                std_prods = pickle.load(infile)
            
            data = std_prods['data']
            for key,item in data.items():
                if key == 'info_dict':
                    if item is None:
                        obs_id_full_info[key] = False
                        obs_id_full_info['OBS_ID'] = obs_id 
                    else:
                        obs_id_full_info[key] = True
                        for keykey,itemitem in item.items():
                            obs_id_full_info[keykey] = itemitem
                else:
                    obs_id_full_info[key] = item

            if first_good:
                df = pd.DataFrame(columns=obs_id_full_info.keys())
                first_good = False
            else:
                df = pd.read_pickle(df_name)
            df = df.append(obs_id_full_info,ignore_index=True)
            df.to_pickle(df_name) 
        
        else:
            mylogging.warning('std_prods file not found')    

    if clean:
        mylogging.info('Cleaning data frame')
        df = pd.read_pickle(df_name)
        cleaned_df = df[~df['DATE-OBS'].isna()]     
        dirty_df = df[df['DATE-OBS'].isna()]
        mylogging.info(f'Removed {len(dirty_df)} rows')        
        cleaned_df.to_pickle(df_name)
        dirty_df.to_pickle(obs_id_dir/'std_prods'/dir_string/'bad_obs_id.pkl')

    return df_name

def make_info_columns(std_prods):
    '''
    It extracts information from std_prods returning data formatted into
    two columns to be used in a PDF page object

    HISTORY
    -------
    2022 03 19, Stefano Rapisarda, Uppsala (creation date)
    '''

    def truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    data = std_prods['data']
    info = data['info_dict']

    # Computing exposition and filtered exposure
    ori_exp = info['TSTOP']-info['TSTART']
    filt_exp = truncate((ori_exp-info['ONTIME'])/ori_exp*100,1)

    # Formatting inact_det_list
    inact_det_list = data['inact_det_list']
    if len(inact_det_list) > 6:
        inact_det_list_to_print = '!!! > 6 ({}) !!!'.\
            format(len(inact_det_list))
    else:
        inact_det_list_to_print = ''
        for det in inact_det_list:
            inact_det_list_to_print += f'{det}, '
        inact_det_list_to_print = inact_det_list_to_print[:-2]

    # Computing hardness ratio with corresponding error
    net_cr_hard = data['other_crs'][1]-data['other_bkg_crs'][1]
    net_cr_soft = data['other_crs'][0]-data['other_bkg_crs'][0]
    hr = net_cr_hard/net_cr_soft
    hr_sigma2 = (
        data['other_cr_err'][1]**2 + data['other_bkg_cr_err'][1]**2 +
        hr**2 * (data['other_cr_err'][0]**2 + data['other_bkg_cr_err'][0]**2)
        ) / net_cr_soft**2

    # Initializing columns
    col1 = {
        'Cre. date (this file):': data['creation_date'],
        'Mission:'              : info['TELESCOP'],
        'Target:'               : info['OBJECT'],
        'Obs. ID:'              : info['OBS_ID'],
        'Start obs. time:'      : info['DATE-OBS'],
        'Stop  obs. time:'      : info['DATE-END'],
        'Clean exposure [s]:'   : str(info['ONTIME']),
        'N. raw events:'        : str(info['NAXIS2']),
        'Filtered exp. [%]:'    : str(filt_exp),
        'N. act. det.:'         : str(data['n_act_det']),
        'Inactive det. list:'   : inact_det_list_to_print,
        'Main energy band:'     : '{} [keV]'.format(data['main_en_band'])
        }
    for e,band in enumerate(data['other_en_bands']):
        col1[f'Energy band {e+1}:'] = f'{band} [keV]'

    col2 = {
        'N. GTIs:'                  : str(data['n_gtis']),
        'N. filtered GTIs (>{}):'.format(data['tseg'])  : str(data['n_fgtis']),
        'N. of segments:'           : str(data['n_segs']),
        'Time resolution [s]:'      : str(data['tres']),
        'Time segment [s]:'         : str(data['tseg']),
        'Main count rate [c/s]:'    : '{} '.\
            format(truncate(data['main_cr'],1))+u'\u00B1'+\
            ' {}'.format(truncate(data['main_cr_err'],1)),
        'Main bkg count rate [c/s]:': '{} '.\
            format(truncate(data['main_bkg_cr'],1))+u'\u00B1'+\
            ' {}'.format(truncate(data['main_bkg_cr_err'],1))
        }
    for e,(cr,cr_err,bkg,bkg_err) in \
        enumerate(zip(data['other_crs'],data['other_cr_err'],
                    data['other_bkg_crs'],data['other_bkg_cr_err'])):
        key1 = f'Count rate b{e+1} [c/s]:'
        key2 = f'Bkg count rate b{e+1} [c/s]:'
        item1 = '{} '.format(truncate(cr,1))+u'\u00B1'+\
            ' {}'.format(truncate(cr_err,1))
        item2 = '{} '.format(truncate(bkg,1))+u'\u00B1'+\
            ' {}'.format(truncate(bkg_err,1))
        col2[key1] = item1
        col2[key2] = item2
    col2['Hard(b2)/Soft(b1) ratio:'] = '{} '.\
        format(truncate(hr,2))+u'\u00B1'+\
        ' {}'.format(truncate(math.sqrt(hr_sigma2),2))   
    col2['Main frac. rms (<60Hz) [%]:'] = '{} '.\
        format(np.round(data['main_frac_rms']*100,2))+u'\u00B1'+ \
        ' {}'.format(np.round(data['main_frac_rms_err']*100,2))
    for e,(rms,rms_err) in \
        enumerate(zip(data['other_frac_rms'],data['other_frac_rms_err'])):
        key = f'Frac. rms (<60Hz) b{e+1} [%]'
        item = '{} '.format(np.round(rms*100,2))+u'\u00B1'+ \
        ' {}'.format(np.round(rms_err*100,2))
        col2[key] = item

    return col1, col2

def print_std_prod_single(obs_id_dir,main_en_band=['0.5','10.0'],
    tres='0.0001220703125',tseg='128.0',suffix=None):
    '''
    It creates PDF pages containing information and plots
    '''

    if type(obs_id_dir) is str: obs_id_dir = pathlib.Path(obs_id_dir)

    mylogging = LoggingWrapper()

    mylogging.info('*'*72)
    mylogging.info(str_title('print_std_prod_single'))
    mylogging.info('*'*72)

    # Setting directory identifiers
    time_string = 'T{}_{}'.format(tres,tseg)
    if suffix is None:
        dir_string = 'E{}_{}_{}'.format(
            main_en_band[0],main_en_band[1],time_string)
    else:
        dir_string = 'E{}_{}_{}_{}'.format(
            main_en_band[0],main_en_band[1],time_string,suffix)
    
    # Setting directories
    obs_id = obs_id_dir.name
    an_dir = obs_id_dir.parent
    std_prod_dir = obs_id_dir/'std_prods'/dir_string

    pages = []

    # First PDF page, data and general plot 
    # ------------------------------------------------------------------   
    page_name = std_prod_dir/'general_info.pdf'
    # Extracting information into two columns
    std_prod_file = std_prod_dir/f'std_prods.pkl'
    if std_prod_file.is_file():
        # Reading std_prods dictionary
        with open(std_prod_file,'rb') as infile:
            std_prods = pickle.load(infile)
        col1, col2 = make_info_columns(std_prods)
        plots = std_prods['plots']

    df_file = an_dir/'std_prods'/dir_string/f'general_info_data_frame.pkl'
    if df_file.is_file():
        gplot = make_general_plot(obs_id_dir,tres=tres,tseg=tseg)
    
    pdf = pdf_page(margins=[10,10,10,10])
    pdf.add_page()
    mark_index = 0
    pages += [[page_name,'general info',mark_index]]
    mark_index += 1
    if std_prod_file.is_file():
        pdf.print_key_items(info=col1,grid=[2,2,5,5],sel='11',conv=0.28)
        pdf.print_key_items(info=col2,grid=[2,2,5,5],sel='12',conv=0.28)
    if df_file.is_file():
        coors = pdf.get_grid_coors(grid=[3,1,5,5],sel='21',margins=[5,0,0,0])
        pdf.image(str(gplot),x=coors[0],y=coors[1],w=coors[2]-coors[0])
    pdf.output(page_name,'F')
    # ------------------------------------------------------------------ 

    # Second PDF page, count rate and energy spectrum
    # ------------------------------------------------------------------
    page_name = std_prod_dir/'cr_and_spectrum.pdf'
    pdf = pdf_page(margins=[10,10,10,10])
    pdf.add_page()
    pages += [[page_name,'cr and spectrum',mark_index]]
    mark_index += 1
    if std_prod_file.is_file():
        coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='11',margins=[5,0,0,0])
        pdf.image(str(plots['Obs. ID count rate']),x=coors[0],y=coors[1],w=coors[2]-coors[0])
        coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='21',margins=[5,0,0,0])
        pdf.image(str(plots['Energy spectrum']),x=coors[0],y=coors[1],w=coors[2]-coors[0])
    pdf.output(page_name,'F')
    # ------------------------------------------------------------------

    # Third PDF page, count rate per GTI
    # ------------------------------------------------------------------
    page_name = std_prod_dir/'cr_per_gti.pdf'
    pages += [[page_name,'cr per GTI',mark_index]]  
    pdf = pdf_page(margins=[10,10,10,10])
    if std_prod_file.is_file():
        for p,plot in enumerate(plots['Count rate per GTI']):
            row = 2
            if p%2 == 0:
                row = 1
                pdf.add_page()
                mark_index += 1
            coors = pdf.get_grid_coors(grid=[2,1,5,5],sel=f'{row}1',margins=[5,0,0,0])
            pdf.image(str(plot),x=coors[0],y=coors[1],w=coors[2]-coors[0])
    pdf.output(page_name,'F')
    # ------------------------------------------------------------------
  
    # Forth PDF page, comparison between ufa and cl
    # ------------------------------------------------------------------
    page_name = std_prod_dir/'ufa_cl.pdf'
    pages += [[page_name,'ufa/cl comparison',mark_index]]
    mark_index += 1
    pdf = pdf_page(margins=[10,10,10,10])
    pdf.add_page()
    if std_prod_file.is_file():
        coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='11',margins=[5,0,0,0])
        pdf.image(str(plots['ufa/cl count rate per segment']),x=coors[0],y=coors[1],w=coors[2]-coors[0])
        coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='21',margins=[5,0,0,0])
        pdf.image(str(plots['ufa/cl power spectra']),x=coors[0],y=coors[1],w=coors[2]-coors[0])
    pdf.output(page_name,'F')
    # ------------------------------------------------------------------

    # Fifth PDF page, average power spectra
    # ------------------------------------------------------------------
    page_name = std_prod_dir/'power_spectra.pdf'
    pages += [[page_name,'Average Power Spectra',mark_index]]
    mark_index += 1
    pdf = pdf_page(margins=[10,10,10,10])
    pdf.add_page()
    if std_prod_file.is_file():
        coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='11',margins=[5,0,0,0])
        pdf.image(str(plots['Main Power Spectrum']),x=coors[0],y=coors[1],w=coors[2]-coors[0])
        coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='21',margins=[5,0,0,0])
        pdf.image(str(plots['Other Power Spectra']),x=coors[0],y=coors[1],w=coors[2]-coors[0])
    pdf.output(page_name,'F')
    # ------------------------------------------------------------------

    # Sixth PDF page, power spectra per GTI
    # ------------------------------------------------------------------
    page_name = std_prod_dir/'gti_power_spectra.pdf'
    pages += [[page_name,'Power Spectra per GTI',mark_index]]   
    pdf = pdf_page(margins=[10,10,10,10])
    if std_prod_file.is_file():
        for p,plot in enumerate(plots['Power Spectra per GTI']):
            row = 2
            if p%2 == 0:
                row = 1
                pdf.add_page()
                mark_index += 1
            coors = pdf.get_grid_coors(grid=[2,1,5,5],sel=f'{row}1',margins=[5,0,0,0])
            pdf.image(str(plot),x=coors[0],y=coors[1],w=coors[2]-coors[0])
    pdf.output(page_name,'F')
    # ------------------------------------------------------------------

    # Merging all the pages and assigning bookmarks
    # ------------------------------------------------------------------
    output_file = std_prod_dir/f'{obs_id}_std_prods.pdf'
    merger = PdfFileMerger()
    for p,(page,title,index) in enumerate(pages):
        with open(page,'rb') as infile:
            merger.append(PdfFileReader(infile))
            if p == 0:
                parent = merger.addBookmark(obs_id,0)
            merger.addBookmark(title,index,parent=parent)
    merger.write(str(output_file))
    # ------------------------------------------------------------------

    mylogging.info('*'*72)
    mylogging.info(str_title('quitting print_std_prod'))
    mylogging.info('*'*72)

    return output_file




