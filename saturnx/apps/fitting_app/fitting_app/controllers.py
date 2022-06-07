from nntplib import NNTPDataError
import os
import sys
import glob
import pathlib
import numpy as np
from tkinter import filedialog
import tkinter as tk

import matplotlib.pyplot as plt


sys.path.append('/Volumes/Samsung_T5/saturnx')
import saturnx as sx

def eval_gti_str(gti_str):
        if ',' in gti_str:
            div = gti_str.split(',')
        else:
            div = [gti_str]

        gti_indices = []
        for d in div:
            if '-' in d:
                start = int(d.split('-')[0])
                stop  = int(d.split('-')[1]) + 1
                gti_indices += [i for i in range(start,stop)]
            else:
                gti_indices += [int(d)]

        return sorted(list(set(gti_indices)))

class Controller:

    def __init__(self,model,view,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self._model = model
        self._view = view

        # Control variables
        # (To check what it is currently plotted on canvas)
        self._fit_on_canvas = False

        self._n_segs = tk.StringVar()
        self._n_segs.set('')
        self._power = None # Loaded data
        self._power_list = None # Loaded data
        self._to_plot = None # Data on canvas

        self._var_change_flag = False

        self._plot_vars = {
            'data_dir':[
                self._view._input_dir_box._input_dir,
                '/Volumes/BigBoy/NICER_data/MAXI_J1820+070/analysis/qpoCB_transition',
                '/Volumes/BigBoy/NICER_data/MAXI_J1820+070/analysis/qpoCB_transition'],
            'file':[
                self._view._file_box._file,
                '',''],
            'gti_str':[
                self._view._gti_box._gti_sel_string,
                '',''],
            'sub_poi_flag':[
                self._view._poi_box._poi_flag,
                0,0],
            'poi_freq':[
                self._view._poi_box._low_freq,
                3000,3000],
            'poi_value':[
                self._view._poi_box._poi_level,
                0,0],
            'rebin_str':[
                self._view._rebin_box._rebin_factor,
                '-30','-30'],
            'xy_flag':[
                self._view._norm_box._xy_flag,
                0,0],
            'norm':[
                self._view._norm_box._norm,
                'Leahy','Leahy'],
            'bkg':[
                self._view._norm_box._bkg,
                0,0],
            }

        self._init_inputs()
        self._trace_variables()
        self._configure_buttons()
        self._bind_entries()

    def _init_inputs(self):
        self._view._input_dir_box._input_dir.set(self._model['plot_fields']._data_dir)

    def _trace_variables(self):
        # File menu
        self._view._file_box._file.trace_add('write',self._load_data)

        for key,item in self._plot_vars.items():
            if not key in ['data_dir','file']:
                item[0].trace_add('write',self._check_variables) 

    def _configure_buttons(self):
        # Command buttons
        self._view._plot_button.configure(command=self._plot)
        self._view._fit_button.configure(command=self._fit)
        self._view._reset_button.configure(command=self._reset)

        # Poisson buttons
        self._view._poi_box._est_poi_button.configure(command=self._est_poi)

        # Input dir
        self._view._input_dir_box._set_dir_button.configure(command=self._set_dir)

    def _bind_entries(self):
        self._view._input_dir_box._dir_entry.bind(
            '<Return>',lambda event, flag=False: self._set_dir(flag)
            )

    def _check_variables(self,var,index,mode):
        self._check_variable_change()
        if self._var_change_flag:
            self._view._plot_button.config(bg='red')
        else:
            button = tk.Button(None)
            self._view._plot_button.config(bg=button.cget('bg')) 


    def _check_variable_change(self):
        print('-'*72)
        print('Checking changes')
        self._var_change_flag = False
        self._filter_gti = False
        for key,item in self._plot_vars.items():
            if not key in ['data_dir','file','poi_freq']:
                if item[0].get() != item[1]:
                    self._var_change_flag = True
                    print(f'{key} changed')
                    if key == 'gti_str':
                        self._filter_gti = True
        print('-'*72)

    def _update_variables(self):
        print('Updating variables')
        for key,item in self._plot_vars.items():
            if not key in ['data_dir','file']:
                new_var = item[0].get()
                if key == 'poi_value':
                    if new_var < 0: new_var=0
                if isinstance(new_var,str): new_var = new_var.strip()
                item[1] = new_var    
                item[0].set(item[1]) 
        self._var_change_flag = False
        self._check_variables(None,None,None)   

    def _update_file_list(self):
        '''
        Triggered when data dir is selected 
        '''
        print('Updating file list')
        menu = self._view._file_box._file_menu
        menu["menu"].delete(0,'end')
        files = glob.glob(self._plot_vars['data_dir'][1]+'/*.pkl')
        if len(files) != 0:
            for item in files:
                menu["menu"].add_command(
                    label=item,
                    command=lambda value=item:self._view._file_box._file.set(value)
                )

    def _load_data(self,var,index,mode):
        '''
        Triggered when selecting file
        '''
        # Cleaning canvas
        self._view._plot_area._ax.clear()
        self._data_on_canvas = False
        self._fit_on_canvas = False

        selected_file = self._view._file_box._file.get()
        data_dir = self._view._input_dir_box._input_dir.get()
        full_path = os.path.join(data_dir,selected_file)

        if os.path.isfile(full_path):
            print('Loading data...')

            if 'LIST' in selected_file.upper():
                self._power_list = sx.PowerList.load(full_path)
                n_gti = self._power_list[0].meta_data['N_GTIS']
                self._view._file_box._box.configure(
                    text='File (GTIs ({}): 0-{})'.format(n_gti,n_gti-1)
                    )
                self._view._gti_box._enable()
            else:
                self._power = sx.PowerSpectrum.load(full_path)
                self._view._file_box._box.configure(text='File')
                self._view._gti_box._disable()

            # Initializing info like tres, texp, fres, and n_seg
            print('Done!') 
            
    def _plot(self):
        print('Plotting')
        self._update_variables()
        self._view._plot_area._ax.clear()

        if self._power_list is not None and self._filter_gti:
            gti_str = self._view._gti_box._gti_sel_string.get()
            if gti_str != '':
                gti_indices = eval_gti_str(gti_str)
                power_list = sx.PowerList(
                    [pw for pw in self._power_list if pw.meta_data['GTI_INDEX'] in gti_indices]
                )
            else:
                power_list = self._power_list
            self._power = power_list.average(norm=self._plot_vars['norm'][1])
            
        if self._power is not None:

            power = self._power

            if self._plot_vars['norm'][1].strip().upper() != 'NONE':
                print('Normalizing')
                power = power.normalize(
                    norm=self._plot_vars['norm'][1].strip(),
                    bkg_cr = self._plot_vars['bkg'][1]
                    )

            if self._plot_vars['sub_poi_flag'][1]:
                print('Subtracting Poisson')
                power = power.sub_poi(
                    low_freq = self._plot_vars['poi_freq'][1]
                )

            rebin_str = self._plot_vars['rebin_str'][1]
            if rebin_str != '':
                print('Rebinning')
                if ',' in rebin_str:
                    rmp = rebin_str.split(',')
                    for rf in rmp:
                        rf = rf.strip()
                        power = power.rebin(float(rf))
                else:
                    power = power.rebin(float(rebin_str))

            self._to_plot = power
            self._to_plot.plot(
                ax=self._view._plot_area._ax,lfont=12,marker='',
                xy=self._view._norm_box._xy_flag.get()
                )
            self._view._plot_area._canvas.draw()
            self._view._plot_area._canvas.mpl_connect('motion_notify_event',self._update_cursor)
        else:
            print('Power not loaded')

    def _update_cursor(self,event):
        x = event.xdata
        y = event.ydata
        if x is not None:
            self._view._plot_area._x_pos.configure(text=str(np.round(x,6)))
        else:
            self._view._plot_area._x_pos.configure(text='Out of frame')
        if y is not None:
            self._view._plot_area._y_pos.configure(text=str(np.round(y,6)))
        else:
            self._view._plot_area._y_pos.configure(text='Out of frame')

    def _fit(self):
        print('Clicking fit')

    def _reset(self):
        print('Clicking reset')
        self._view._plot_area._ax.clear()
        self._view._plot_area._canvas.draw()
        self._power = None
        self._power_list = None
        self._to_plot = None
        self._filter_gti = False
        for key,item in self._plot_vars.items():
            if not key in ['data_dir','file']:
                item[0].set(item[2])
        self._view._norm_box._xy_flag.set(0)
        self._view._poi_box._poi_flag.set(0)
        self._update_variables()

    def _est_poi(self):
        print('Clicking est poi')
        low_freq = float(self._plot_vars['poi_freq'][1])
        mask = self._power.freq>=low_freq
        value = self._power.power[mask].mean()
        self._view._poi_box._poi_level.set(value)

    def _set_dir(self, choice_flag=True):
        if choice_flag:
            data_dir = filedialog.askdirectory(
                initialdir=self._model['plot_fields']._data_dir,
                title='Select folder for data products')
            self._view._input_dir_box._input_dir.set(data_dir)

        self._view._file_box._reset()
        self._update_variables()
        self._update_file_list()



    