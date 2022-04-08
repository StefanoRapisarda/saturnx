import pandas as pd 
import numpy as np
import time
import os
from timing.utilities import *
from timing.event import *
from timing.gti import *
from astropy.io.fits import getdata,getheader,getval

import lmfit
from lmfit import Model,Parameters
from lmfit.model import save_modelresult,load_modelresult
from lmfit.models import LorentzianModel

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def select_src_bkg(event,win_half_size=30,distance=5.5,plot=False):

    # Determining edges of the CCD
    max_px = np.max(event['detx'])
    min_px = np.min(event['detx'])

    # Source peak will be identified in the histogram
    bin_array_for_hist = np.arange(min_px-0.5,max_px+1+0.5,1)
    bin_array = np.arange(min_px,max_px+1,1)
    hist,dummy = np.histogram(event['detx'],bin_array_for_hist)

    # The pixel with maximum emission will be the initial value for the fit
    max_i = np.argmax(hist)
    in_center = bin_array[max_i]

    # Find with a fit the peak and the wifth of the source
    model = LorentzianModel()
    params = model.make_params()
    params.add('amplitude',value=np.max(hist),min=0,max=np.inf)
    params.add('center',value=in_center,min=min_px,max=max_px)
    result = model.fit(hist,params,x=bin_array) 

    center = result.params['center'].value
    fwhm = result.params['fwhm'].value
    height = result.params['height'].value

    # Defining src and bkg windows according to fit results
    src_win_min = int(center)-win_half_size
    src_win_max = int(center)+win_half_size 
    print('src win size',int(src_win_max-src_win_min))

    bkg_win = []
    # Case1: there is space from both sizes
    if ((max_px-src_win_max) > 35) and ((src_win_min-min_px) > 35):
        tot_win_size = 0
        print('left and right')
        
        bkg_win += [[src_win_max+distance*fwhm,src_win_max+distance*fwhm+win_half_size]]
        for i in range(int(src_win_max+distance*fwhm),int(src_win_max+distance*fwhm+win_half_size)):
            tot_win_size += 1
        
        bkg_win += [[src_win_min-distance*fwhm-win_half_size,src_win_min-distance*fwhm]]
        for i in range(int(src_win_max-distance*fwhm-win_half_size),int(src_win_max-distance*fwhm)):
            tot_win_size += 1    

    # Case2: there is space only on the left
    elif ((src_win_min-min_px) > 70):
        tot_win_size = 0
        print('left')
        
        bkg_win += [[src_win_min-distance*fwhm-win_half_size*2,src_win_min-distance*fwhm]]
        for i in range(int(src_win_max-distance*fwhm-win_half_size*2),int(src_win_min-distance*fwhm)):
            tot_win_size += 1  
            
    # Case3: there is space only on the right
    elif ((max_px-src_win_max) > 70):
        tot_win_size = 0
        print('right')
        
        bkg_win += [[src_win_max+distance*fwhm,src_win_max+distance*fwhm+win_half_size*2]]
        for i in range(int(src_win_max+distance*fwhm),int(src_win_max+distance*fwhm+win_half_size*2)):
            tot_win_size += 1 
        
    print('bkg_win_size',tot_win_size)

    event_src = event.filter(f'(detx >= {src_win_min}) & (detx < {src_win_max})')
    expr = ''
    for i in range(len(bkg_win)):
        win = bkg_win[i]
        if i == len(bkg_win)-1:
            expr += '(detx >= {}) & (detx < {})'.format(win[0],win[1])
        else:
            expr += '(detx >= {}) & (detx < {}) &'.format(win[0],win[1])
    event_bkg = event.filter(expr)

    total_counts = len(event)
    src_counts = len(event_src)
    bkg_counts = len(event_bkg)

    src_frac = src_counts/total_counts
    bkg_frac = bkg_counts/total_counts

    #print(src_frac,bkg_frac)

    if src_frac < 0.9: print('WARNING: source fraction is smaller than 90%')
    if bkg_frac > 0.05: print('WARNING: background fraction is larger than 5%')

    if plot:
        hist_src,dummy = np.histogram(event_src['detx'],bin_array_for_hist)
        hist_bkg,dummy = np.histogram(event_bkg['detx'],bin_array_for_hist)

        fig,ax = plt.subplots(figsize=(6,6))
        plt.plot(bin_array,result.best_fit,label='fit')
        plt.plot(bin_array,hist,'o-k',ms=2,label='data')
        src_mask = (bin_array > src_win_min) & (bin_array < src_win_max)

        plt.plot(bin_array[hist_src>0],hist_src[hist_src>0],'ob',label='src')
        plt.plot(bin_array[hist_bkg>0],hist_bkg[hist_bkg>0],'or',label='bkg')

        src_rects = [Rectangle((src_win_min,0),src_win_max-src_win_min,height/3.)]
        src_pc = PatchCollection(src_rects,facecolor='green',alpha=0.5,edgecolor='None')
        ax.add_collection(src_pc)
        
        bkg_rects = []
        for win in bkg_win:
            #print(win[0],win[1])
            bkg_rects = [Rectangle((win[0],0),win[1]-win[0],height/4)]
            bkg_pc = PatchCollection(bkg_rects,facecolor='red',alpha=0.5,edgecolor='None')
            ax.add_collection(bkg_pc)

        plt.grid()       

    return event_src,event_bkg

