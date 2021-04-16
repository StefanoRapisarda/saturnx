import os
import math
from os import path
import pathlib
import logging
import pickle
import numpy as np
import pandas as pd

from astropy.time import Time

from PyPDF2 import PdfFileMerger, PdfFileReader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import xspec

from kronos.core.gti import Gti
from kronos.core.lightcurve import LightcurveList
from kronos.core.power import PowerList
from kronos.utils.logging import make_logger
from kronos.utils.generic import chunks, my_cdate, is_number
from kronos.utils.pdf import pdf_page
from kronos.functions.nicer_functions import get_counts, all_det

def make_nicer_std_prod(obs_id_dirs,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['0.5','10.0'], en_bands = [['0.5','2.0'],['2.0','10.0']],
    rebin=-30,data_dir=pathlib.Path.cwd(),
    rmf='/Users/xizg0003/AstroBoy/caldb/data/nicer/xti/cpf/rmf/nixtiref20170601v002.rmf',
    arf='/Users/xizg0003/AstroBoy/caldb/data/nicer/xti/cpf/arf/nixtiaveonaxis20170601v004.arf',
    log_name=None):
    '''
    Runs make_nicer_std_prod_single for each obs_ID in obs_id_dirs

    HISTORY
    -------
    2021 03 18, Stefano Rapisarda (Uppsala), creation date
    '''

    an_dir = obs_id_dirs[0].parent

    # Logging
    if log_name is None:
        log_name = make_logger('print_std_prods',outdir=an_dir)

    for obs_id_dir in obs_id_dirs:

        if obs_id_dir.is_dir():
            make_nicer_std_prod_single(obs_id_dir,tres=tres,tseg=tseg,
                main_en_band = main_en_band, en_bands = en_bands,
                rebin=rebin,data_dir=data_dir,
                rmf=rmf,arf=arf,log_name=log_name)
        else:
            logging.info('{} does not exist'.format(obs_id_dir))


def make_nicer_general_plot(an_dir,tres='0.0001220703125',tseg='128.0',
    time_window = 20,plt_x_dim = 8,plt_y_dim = 8,
    obs_id = None,log_name=None):
    '''
    It reads a pandas data frame containing information about all the 
    OBS_ID and makes a plot with count rate, hardness ratio, and 
    fractional RMS over time

    If obs_id is not None, a gold dot corresponding to the obs_ID time
    is plotted 

    HISTORY
    -------
    2021 03 18, Stefano Rapisarda (Uppsala), creation date
    '''
    
    time_string = 'T{}_{}'.format(tres,tseg)
    
    if type(an_dir) == str: an_dir = pathlib.Path(an_dir)
    obs_id_dir = an_dir/obs_id

    # Logging
    if log_name is None:
        log_name = make_logger('make_nicer_general_plot',outdir=obs_id_dir)

    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'make_nicer_general_plot','*'*24))
    logging.info('*'*72)

    # Making general plot folder
    std_plot_dir = an_dir/'std_plots'
    if not std_plot_dir.is_dir():
        logging.info('std_plot folder does not exist. Creating it')
        os.mkdir(std_plot_dir)

    # Reading info
    info_df_name = an_dir/'std_prods'/'info_data_frame_{}.pkl'.format(time_string)
    if info_df_name.is_file():
        df = pd.read_pickle(info_df_name)
        
    df=df.drop_duplicates(subset=['OBS_ID'])
        
    # Plotting
    # =======================================
    
    # General settings
    colors = ['red','blue','green','orange','brown']
    markers = ['s','^','p','H','X']
    n_en_bands = df['N_EN_BANDS'].iloc[0]
    
    fig, axes = plt.subplots(3,1,figsize=(plt_x_dim,plt_y_dim))
    plt.subplots_adjust(hspace=0)
    
    # Defining time ax
    start_dates = df['DATE-OBS']
    stop_dates = df['DATE-END']
    start_dates_mjd = Time(start_dates.to_list(),format='isot',scale='utc').mjd
    stop_dates_mjd = Time(stop_dates.to_list(),format='isot',scale='utc').mjd
    half_dur_mjd = (stop_dates_mjd-start_dates_mjd)/2.
    mid_dates_mjd = (start_dates_mjd+stop_dates_mjd)/2.
    start_mjd = int(np.min(start_dates_mjd))
    time = mid_dates_mjd-start_mjd
    
    obs_ids = df['OBS_ID'].to_numpy()
    if (not obs_id is None) and (obs_id in obs_ids): 
        target_index = np.where(obs_ids == obs_id)
        time_obs_id = time[target_index]

        for ax in axes:
            ax.set_xlim(int(time_obs_id/time_window)*time_window-time_window/10,\
                        int(time_obs_id/time_window+1)*time_window+time_window/10*2)
    
    # Plot1, count rates versurs time
    # --------------------------------------------------------------
    # Main energy band
    main_en_band = str(df['MAIN_EN_BAND'].iloc[0])
    tot_cr = df.CR.to_numpy()
    bkg = df['BKG_CR'].to_numpy()
    n_act_det = df['N_ACT_DET'].to_numpy()
    cr_err = df['CR_ERR'].to_numpy()/n_act_det
    cr = (tot_cr-bkg)/n_act_det
    axes[0].errorbar(time,cr,yerr=cr_err,xerr=half_dur_mjd,fmt='o',color='black',label=main_en_band)
   
    if (not obs_id is None) and (obs_id in obs_ids):
        gold_dot, = axes[0].plot(time_obs_id,cr[target_index],'o',color='goldenrod',ms=12)
        leg2 = axes[0].legend([gold_dot],[obs_id],loc='upper left')
        axes[0].add_artist(leg2)
        
    # Other energy bands
    for e in range(n_en_bands):
        en_band = str(df[f'EN_BAND{e+1}'].iloc[0])
        tot_cr = df[f'CR{e+1}'].to_numpy()  
        bkg = df[f'BKG_CR{e+1}'].to_numpy()
        cr_err = df[f'CR_ERR{e+1}'].to_numpy()/n_act_det
        cr = (tot_cr-bkg)/n_act_det
        axes[0].errorbar(time,cr,yerr=cr_err,xerr=half_dur_mjd,fmt=markers[e],color=colors[e],label=en_band)
        
        if (not obs_id is None) and (obs_id in obs_ids):
            axes[0].plot(time_obs_id,cr[target_index],'o',color='goldenrod',ms=12)
            #p=patches.Ellipse((time_obs_id,cr[target_index]),x_length*radius*plt_yx_ratio,y_length*radius,\
            #                                              edgecolor='goldenrod',facecolor='none',lw=2,zorder=2)
            #axes[0].add_patch(p)   
    
    axes[0].set_ylabel('Count rate [c/s/n_det]',fontsize=14)
    axes[0].legend(title='[keV]',loc='upper right')
    # --------------------------------------------------------------
    
    # Plot2, hardness ratio
    # --------------------------------------------------------------
    hr = df.HR.to_numpy()
    axes[1].plot(time,hr,'o',color='black',zorder=4)
    
    if (not obs_id is None) and (obs_id in obs_ids):
        axes[1].plot(time_obs_id,hr[target_index],'o',color='goldenrod',ms=12) 
        
    axes[1].set_ylabel('Hardness',fontsize=14)
    # --------------------------------------------------------------
    
    # Plot3, fractional rms
    # --------------------------------------------------------------
    # Main energy band
    rms = df.RMS.to_numpy()
    rms_err = df['RMS_ERR'].to_numpy()
    axes[2].errorbar(time,rms*100,rms_err*100,fmt='o',color='black')
    
    if (not obs_id is None) and (obs_id in obs_ids):
        gold_dot, = axes[2].plot(time_obs_id,rms[target_index]*100,'o',color='goldenrod',ms=12)   
        
    # Other energy bands
    for e in range(n_en_bands):
        en_band = str(df[f'EN_BAND{e+1}'].iloc[0]) 
        rms = df[f'RMS{e+1}'].to_numpy()
        rms_err = df[f'RMS_ERR{e+1}'].to_numpy()
        axes[2].errorbar(time,rms*100,rms_err*100,fmt=markers[e],color=colors[e])
        
        if (not obs_id is None) and (obs_id in obs_ids):
            gold_dot, = axes[2].plot(time_obs_id,rms[target_index]*100,'o',color='goldenrod',ms=12)
     
    axes[2].set_ylabel('Frac. RMS [%]',fontsize=14)
    axes[2].set_xlabel('Time [MJD, {}]'.format(start_mjd),fontsize=14)
    # --------------------------------------------------------------
    
    for ax in axes: ax.grid()
        
    # Saving file
    plot_name = an_dir/obs_id/'std_plots'/'global_info_{}.jpeg'.format(time_string)
    fig.savefig(plot_name, dpi=300)
        
    logging.shutdown() 
    
    return plot_name

def make_nicer_std_prod_single(obs_id_dir,tres='0.0001220703125',tseg='128.0',
    main_en_band = ['0.5','10.0'], en_bands = [['0.5','2.0'],['2.0','10.0']],
    rebin=-30,data_dir=pathlib.Path.cwd(),
    rmf='/Users/xizg0003/AstroBoy/caldb/data/nicer/xti/cpf/rmf/nixtiref20170601v002.rmf',
    arf='/Users/xizg0003/AstroBoy/caldb/data/nicer/xti/cpf/arf/nixtiaveonaxis20170601v004.arf',
    log_name=None):
    '''
    Save plots and a dictionary with information according with user 
    settings

    This function is specific for NICER reduced products. It assumes
    that products (lightcurve lists and power list for the selected 
    energy bands and time settings) are already computed. If the 
    function does not find the products, it will produce a plot anyway,
    but it will be empty.
    These products are expected to have a format E<1>_<2>_T<3>_<4>, 
    where 1 and 2 are the energy band boundary and 3 and 4 are time
    resolution and time segment, respectively. 
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
        list of lists, containes low and high energy band boundaries
        for each sub (different and smaller than main) energy band
        (default is [['0.5','2.0'],['2.0','10.0']])
    rebin: int (optional)
        rebin factor for plotting the power spectra
        (default is -30)
    data_dir: string or pathlib.Path (optional)
        folder containing the energy spectrum
        (default is pathlib.Path.cwd())
    rmf: string (optional)
        full path of the response matrix (for plotting with xspec)
    arf: string (optional)
        full path of the ancillary response file (for plotting with 
        xspec)
    log_name: string or None (optional)
        name of the log file
        (default is None)

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
    '''
    
    colors = ['red','blue','green','orange','brown']
    markers = ['s','^','p','H','X']
    if len(en_bands) > len(colors):
        raise ValueError('Energy bands cannot be more than 5')

    if type(obs_id_dir) == str: obs_id_dir = pathlib.Path(obs_id_dir)
    if type(data_dir) == str: data_dir = pathlib.Path(data_dir)

    # Logging
    if log_name is None:
        log_name = make_logger('make_std_prod_single',outdir=obs_id_dir)

    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'make_nicer_std_prod_single','*'*24))
    logging.info('*'*72)
    
    # Making folders 
    # -----------------------------------------------------
    obs_id = obs_id_dir.name
    
    std_plot_dir = obs_id_dir/'std_plots'
    if not std_plot_dir.is_dir():
        logging.info('std_plots does not exist, creating one...')
        os.mkdir(std_plot_dir)
    else:
        logging.info('std_plots already exists.')
    # -----------------------------------------------------
    
    # Defining names of files to read
    # -----------------------------------------------------
    
    # Main files (Full energy band or energy band to highlight)
    main_prod_name = 'E{}_{}_T{}_{}'.format(main_en_band[0],main_en_band[1],tres,tseg)
    
    main_gti_name = 'gti_E{}_{}.gti'.format(main_en_band[0],main_en_band[1])
    main_lc_list_file = obs_id_dir/'lc_list_{}.pkl'.format(main_prod_name)
    main_pw_list_file = obs_id_dir/'power_list_{}.pkl'.format(main_prod_name)
    
    # Other energy bands
    lc_list_files = []
    power_list_files = []
    for en_band in en_bands:
        low_en, high_en = en_band[0], en_band[1]
        lc_list_files += [obs_id_dir/'lc_list_E{}_{}_T{}_{}.pkl'.\
                                       format(low_en,high_en,tres,tseg)]
        power_list_files += [obs_id_dir/'power_list_E{}_{}_T{}_{}.pkl'.\
                                        format(low_en,high_en,tres,tseg)]
    
    # Energy spectra
    spec = data_dir/obs_id/'xti'/'event_cl'/(str(obs_id)+'_tot_spec.pi')
    bkg_spec = data_dir/obs_id/'xti'/'event_cl'/(str(obs_id)+'_bkg_spec.pi')
    # ------------------------------------------------------

    # Printing some info
    # -----------------------------------------------------------------
    logging.info('')
    logging.info('Obs ID: {}'.format(obs_id))
    logging.info('Settings:')
    logging.info('-'*60)
    logging.info('Selected main energy band: {}-{} keV'.\
        format(main_en_band[0],main_en_band[1]))
    for i,en_band in enumerate(en_bands):
        logging.info('Selected energy band {}: {}-{} keV'.\
            format(i,en_band[0],en_band[1]))        
    logging.info('Selected time resolution: {} s'.format(tres)) 
    logging.info('Selected time segment: {} s'.format(tseg)) 
    logging.info('Log file name: {}'.format(log_name))
    logging.info('-'*60)
    logging.info('')
    # -----------------------------------------------------------------
    
    # Plotting
    # =======================================================
    
    plots = []
    plt.tight_layout()
    # I create the figure anyway, then, if the file does not exists, 
    # the figure will be empty
    
    
    # Plot1: count rate per segment
    # ------------------------------------------------------
    logging.info('Plotting count rate per segment')
    fig,ax = plt.subplots(figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Count rate per segment',fontsize=14)
    
    # Full energy band count rate
    if main_lc_list_file.is_file():
        # Plotting lightcurve
        main_lc_list = LightcurveList.load(main_lc_list_file)
        main_lc_list.plot(ax=ax,color='k',lfont=14,label='{}-{}'.\
                     format(main_en_band[0],main_en_band[1]))
        main_cr = main_lc_list.cr
        main_cr_err = main_lc_list.cr_std
        
        # Drawing vertical line per GTI
        gti = Gti.load(obs_id_dir/main_gti_name)
        gti_seg = gti>=float(tseg)
        start = main_lc_list[0].time.iloc[0]
        for g in gti_seg.stop.to_numpy():
            ax.axvline(g-start,ls='--',color='orange')
    else:
        logging.info('Main energy band lc_list file not found')
        logging.info(main_lc_list_file)
       
    # Other energy bands count rate
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
    # ------------------------------------------------------
    
    
    # Plot2: Energy spectrum 
    # ------------------------------------------------------
    logging.info('Plotting energy spectrum')
    fig,ax = plt.subplots(figsize=(8,5))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.tight_layout()
    fig.suptitle('Energy spectrum', fontsize=14)
    
    if spec.is_file():
        s = xspec.Spectrum(str(spec))
        s.response = rmf
        s.response.arf = arf
        
        if bkg_spec.is_file():
            s.background = str(bkg_spec)
            
        xspec.Plot.xAxis = "keV"
        s.ignore("**-0.2 12.0-**")
        
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
            full_bkg_cr, full_bkg_cr_err = get_counts(bkg_spec,low_en=0.5,high_en=10.)
        else:
            full_bkg_cr, full_bkg_cr_err = 0,0
        
        rms = sub_poi.normalize('rms',bkg_cr=full_bkg_cr)
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
    # ------------------------------------------------------
    
    
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
                
            if bkg_spec.is_file():
                bkg_cr, bkg_cr_err = get_counts(bkg_spec,low_en=float(low_en),high_en=float(high_en))
            else:
                bkg_cr, bkg_cr_err = 0, 0
            bkgs_cr += [bkg_cr]
            bkgs_cr_err += [bkg_cr_err]

            rms = sub_poi.normalize('rms',bkg_cr=bkg_cr)
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
    # ------------------------------------------------------    
    
    
    # Plot5: Full power spectra per GTI
    # ------------------------------------------------------ 
    logging.info('Plotting Full power spectra per GTI')
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
        n_act_det = main_lc_list[0].meta_data['N_ACT_DET']
        inact_det_list = main_lc_list[0].meta_data['INACT_DET_LIST']
        parent_info['N_ACT_DET'] = n_act_det
        inact_det_list_str = ''
        for el in inact_det_list: inact_det_list_str += f'{el},'
        inact_det_list_str = inact_det_list_str[:-1]

        parent_info['INACT_DET_LIST'] = inact_det_list_str
    
        info_dict = main_lc_list[0].meta_data['INFO_FROM_HEADER']

        for key,item in info_dict.items():
            parent_info[key] = item
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

        # First columns
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
            '{}-{} [keV]'.format(main_en_band[0],main_en_band[1])]
        parent_info['MAIN_EN_BAND'] = '{}-{}'.\
            format(main_en_band[0],main_en_band[1])
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


def print_std_prod(obs_id_dirs,tres='0.0001220703125',tseg='128.0',
    add_plot=[],log_name=None):

    time_string = 'T{}_{}'.format(tres,tseg)

    first_obs_id_dir = obs_id_dirs[0]
    if type(first_obs_id_dir) == str:
        first_obs_id_dir = pathlib.Path(first_obs_id_dir)
    an_dir = first_obs_id_dir.parent

    # Logging
    if log_name is None:
        log_name = make_logger('print_std_prods',outdir=an_dir)

    merger = PdfFileMerger()
    if add_plot != []:
        for plot_to_add in add_plot:
            merger.append(PdfFileReader(plot_to_add[0]),bookmark=plot_to_add[1])

    for obs_id_dir in obs_id_dirs:

        if isinstance(obs_id_dir,str):
            obs_id_dir = pathlib.Path(obs_id_dir)
        
        obs_id = obs_id_dir.name
        std_prod_dir = obs_id_dir/'std_prods'
        if not std_prod_dir.is_dir():
            logging.info('std_prods dir does not exist, skipping obs.')
            logging.info(std_prod_dir)
            continue
        pdf_file = std_prod_dir/'{}_std_prods_{}.pdf'.format(obs_id,time_string)

        if pdf_file.is_file():

            with open(pdf_file,'rb') as infile:
                merger.append(PdfFileReader(infile),bookmark=obs_id)

    # Saving file
    output_file = an_dir/'std_prods'/'std_prods_{}.pdf'.format(time_string)

    merger.write(str(output_file))


def print_std_prod_single(an_dir,obs_id,col1,col2,plots,
    tres='0.0001220703125',tseg='128.0',log_name=None):
    '''
    '''

    time_string = 'T{}_{}'.format(tres,tseg)
    
    if type(an_dir) == str: an_dir = pathlib.Path(an_dir)
    obs_id_dir = an_dir/obs_id

    # Logging
    if log_name is None:
        log_name = make_logger('print_std_prods',outdir=obs_id_dir)

    logging.info('*'*72)
    logging.info('{:24}{:^24}{:24}'.format('*'*24,'print_std_prod','*'*24))
    logging.info('*'*72)
    
    if type(col1) == list:
        info1 = {i[0]:i[1] for i in col1}
    elif type(col1) == dict:
        if type(col1[list(col1.keys())[0]]) == list:
            info1 = {item[0]:item[1] for key,item in col1.items()}
        else:
            info1 = col1
    if type(col2) == list:
        info2 = {i[0]:i[1] for i in col2}
    elif type(col2) == dict:
        if type(col2[list(col2.keys())[0]]) == list:
            info2 = {item[0]:item[1] for key,item in col2.items()}
        else:
            info2 = col2

    # Saving
    std_prod_dir = obs_id_dir/'std_prods'
    if not std_prod_dir.is_dir():
        logging.info('std_prods dir does not exist, creating it')
        os.mkdir(std_prod_dir)
    output_file = std_prod_dir/'{}_std_prods_{}.pdf'.format(obs_id,time_string)

    pdf = pdf_page(margins=[10,10,10,10])
    pdf.add_page()
    pdf.print_key_items(info=info1,grid=[2,2,5,5],sel='11',conv=0.28)
    pdf.print_key_items(info=info2,grid=[2,2,5,5],sel='12',conv=0.28)
    coors = pdf.get_grid_coors(grid=[3,1,5,5],sel='21',margins=[5,0,0,0])
    pdf.image(str(plots[0]),x=coors[0],y=coors[1],w=coors[2]-coors[0])
    for i in range(1,len(plots)):
        row=2
        if i%2 != 0: 
            pdf.add_page()
            row=1 
        coors = pdf.get_grid_coors(grid=[2,1,5,5],sel=f'{row}1',margins=[5,0,0,0])
        pdf.image(str(plots[i]),x=coors[0],y=coors[1],w=coors[2]-coors[0])
    pdf.output(output_file,'F')

    return output_file