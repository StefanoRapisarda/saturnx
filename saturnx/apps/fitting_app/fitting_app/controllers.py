from nntplib import NNTPDataError
import os
import sys
import glob
import random 
import pathlib
import numpy as np
from tkinter import filedialog
import tkinter as tk

import matplotlib
import matplotlib.pyplot as plt

from sherpa.data import Data1D
from sherpa.stats import LeastSq,Chi2,Chi2DataVar
from sherpa.optmethods import LevMar,MonCar, GridSearch
from sherpa.estmethods import Confidence, Covariance
from sherpa.fit import Fit
from sherpa.models.basic import Gauss1D,PowLaw1D,Const1D

from .views import FitView, FitResultView
from .utils import init_sherpa_model

sys.path.append('/Volumes/Samsung_T5/saturnx')
import saturnx as sx
from saturnx.fitting.sherpa_custom_models import *




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

    def __init__(self,model,view,root,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self._model = model
        self._view = view
        self._root = root

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

        # Fitting
        self._index = 1
        self._canvas_connected = False
        self._fit_funcs_dict = {}
        self._total_fit_func_dict = {}
        self._fit_func_colors = [
            matplotlib.colors.rgb2hex(col) for col in plt.get_cmap('tab10').colors
            ]
        self._model_func_dict = {
            'fmax_lorentzian':Fmax_lorentzian1D,
            'f0_lorentzian':F0_lorentzian1D,
            'constant':Const1D,
            'gaussian':Gauss1D,
            'power_law':PowLaw1D
        }
        self._model_func_list = [name for name in self._model_func_dict.keys()]

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
        # --------------------------------------------------------------

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
        new_window = tk.Toplevel(self._root)
        self._fit_window = FitView(new_window,controller=self)
        self._fit_window.grid(row=0,column=0,sticky='snew')

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

    # ------------------------------------------------------------------

    def _add_func(self):
        print('Adding function')

        all_colors = self._fit_func_colors.copy()
        # Remove from the list of random color existing colors
        for key,item in self._fit_funcs_dict.items():
            all_colors.remove(item['color'])
        # Selecting a random color
        col = random.choice(all_colors)   

        self._fit_funcs_dict[self._index] = {
            'name':self._fit_window._fit_function_box._fit_func.get(),
            'color':col
            }

        # Populating the fitting function box
        self._fit_window._fit_function_box._fit_func_listbox.insert(
            tk.END,
            f'{self._index}) {self._fit_window._fit_function_box._fit_func.get()}'
        )

        # Assigning a color to text corresponding to the just 
        # added function
        self._fit_window._fit_function_box._fit_func_listbox.itemconfig(
            self._index-1,{'fg':col})   

        # Incrementing fit function index
        self._index += 1

        self._disconnect_canvas()

    def _del_func(self):
        print('Deleting function')

        # Selected items can be more than 1
        if self._fit_funcs_dict != {}:

            listbox = self._fit_window._fit_function_box._fit_func_listbox

            sel = listbox.curselection()

            # Removing selected items from listbox, plot, and dictionary
            # Selection index MUST be reversed because self._fit_func_listbox 
            # index changes every time you remove an item
            for index in sel[::-1]:

                listbox.delete(index) 

                # Removing selected items from plot
                if 'plots' in self._fit_funcs_dict[index+1].keys():
                    self._view._plot_area._ax.lines.remove(
                        self._fit_funcs_dict[index+1]['plots']
                        )

                # Removing selected items from self._fit_func_dict
                del self._fit_funcs_dict[index+1]

            # Resetting index of the remaining items
            items = listbox.get(0,tk.END)
            if len(items) != 0:
                self._reset_index()

                if self._to_plot is not None:
                    self._plot_fit_func()
                    self._print_par_value()

            self._disconnect_canvas()

        else:

            # Just to be sure
            self._index = 1

    def _reset_index(self):
        '''
        Resets the index of the fitting function according to the 
        number of fitting functions in the fitting function listbox
        '''

        listbox = self._fit_window._fit_function_box._fit_func_listbox
        items = listbox.get(0,tk.END)

        # Reading and storing old info
        old_func_info = []
        old_items = []
        for i in range(len(items)):
            item = items[i]
            old_items += [item.split(')')[1].strip()]
            old_index = int(item.split(')')[0].strip())
            old_func_info += [self._fit_funcs_dict[old_index]]
        
        # Cleaning listbox and self._fit_funcs_dict
        listbox.delete(0,tk.END)
        self._fit_funcs_dict = {}
        
        # Re-writing old items with new index
        for i in range(len(items)):
            listbox.insert(tk.END,str(i+1)+') '+old_items[i])
            listbox.itemconfig(i,{'fg':old_func_info[i]['color']})
            self._fit_funcs_dict[i+1] = old_func_info[i]

        # Resetting index
        self._index = len(items) + 1

    def _activate_draw_function(self,event):
        listbox = self._fit_window._fit_function_box._fit_func_listbox
        sel_items = listbox.get(0,tk.END)
        if len(sel_items) != 0:

            print('Draw function activated')

            sel = listbox.curselection()
            if not sel:
                self._disconnect_canvas()
            else:
                self._connect_canvas()

                # In case of (accidental) multiple selection, it will be 
                # considered always the first one
                self._sel_index = int(sel[0])

    def _connect_canvas(self):
        if not self._canvas_connected:
            print('Connecting canvas')
            self._cidclick = self._view._plot_area._canvas.mpl_connect('button_press_event',self._on_click)
            self._cidscroll = self._view._plot_area._canvas.mpl_connect('scroll_event',self._on_roll)
            self._canvas_connected = True 

    def _disconnect_canvas(self):
        if self._canvas_connected:  
            print('Disconnecting canvas')
            self._view._plot_area._canvas.mpl_disconnect(self._cidclick)
            self._view._plot_area._canvas.mpl_disconnect(self._cidscroll)
            self._canvas_connected = False           

    def _on_click(self,event):
        print('Clicking')

        if not event.dblclick:
            # Left click or right click
            if event.button in [1,3]:

                # Position of the cursor
                self._xpos = event.xdata
                self._ypos = event.ydata

                if (self._xpos != None) and (self._ypos != None):

                    # Fetching name of the currently selected function
                    name = self._fit_funcs_dict[self._sel_index+1]['name']

                    # The following printing options depend on the 
                    # fitting functions
                    if name == 'fmax_lorentzian':
                        # Choosing standard value if not existing
                        if not 'par_values' in self._fit_funcs_dict[self._sel_index+1].keys():
                            q = 10
                            status = [False,False,False]
                        else:
                            q = self._fit_funcs_dict[self._sel_index+1]['par_values'][1]
                            status = self._fit_funcs_dict[self._sel_index+1]['frozen']
                            
                        delta = self._xpos/np.sqrt(1+4*q**2)
                        amplitude = delta*(np.pi/2 + np.arctan(2*q))*self._ypos
                        fmax = np.sqrt(self._xpos**2+delta**2)

                        if self._view._norm_box._xy_flag.get():
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [amplitude/self._xpos,q,fmax]
                        else:
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [amplitude,q,fmax]  

                        self._fit_funcs_dict[self._sel_index+1]['frozen'] = status    
                        self._fit_funcs_dict[self._sel_index+1]['par_names'] = \
                            [par.name for par in self._model_func_dict[name]().pars]

                    elif name=='constant': 

                        if not 'par_values' in self._fit_funcs_dict[self._sel_index+1].keys():
                            status = [False]
                        else:
                            status = self._fit_funcs_dict[self._sel_index+1]['frozen']

                        amplitude = self._ypos
                        self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [amplitude]                       
                        self._fit_funcs_dict[self._sel_index+1]['frozen'] = status   
                        self._fit_funcs_dict[self._sel_index+1]['par_names'] = \
                            [par.name for par in self._model_func_dict[name]().pars] 

                    elif name=='gaussian':

                        if not 'par_values' in self._fit_funcs_dict[self._sel_index+1].keys():
                            fwhm = 1
                            status = [False,False,False]
                        else:
                            fwhm = self._fit_funcs_dict[self._sel_index+1]['par_values'][0]
                            status = self._fit_funcs_dict[self._sel_index+1]['frozen']

                        amplitude = self._ypos
                        pos = self._xpos

                        if self._view._norm_box._xy_flag.get():
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [fwhm,pos,amplitude/self._xpos]
                        else:
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [fwhm,pos,amplitude]                       
                        self._fit_funcs_dict[self._sel_index+1]['frozen'] = status   
                        self._fit_funcs_dict[self._sel_index+1]['par_names'] = \
                            [par.name for par in self._model_func_dict[name]().pars]    

                    elif name=='power law':

                        if not 'par_values' in self._fit_funcs_dict[self._sel_index+1].keys():
                            gamma = 1
                            status = [False,True,False]
                        else:
                            gamma = self._fit_funcs_dict[self._sel_index+1]['par_values'][0]
                            status = self._fit_funcs_dict[self._sel_index+1]['frozen']

                        amplitude = self._ypos
                        ref = self._xpos

                        if self._view._norm_box._xy_flag.get():
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [gamma,ref,amplitude/self._xpos]
                        else:
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [gamma,ref,amplitude]                       
                        self._fit_funcs_dict[self._sel_index+1]['frozen'] = status   
                        self._fit_funcs_dict[self._sel_index+1]['par_names'] = \
                            [par.name for par in self._model_func_dict[name]().pars] 

                    # Plotting                                   
                    self._plot_fit_func()

    def _on_roll(self,event):
        name = self._fit_funcs_dict[self._sel_index+1]['name']
        if name=='fmax_lorentzian':
            q = self._fit_funcs_dict[self._sel_index+1]['par_values'][1]
            if q > 1:
                step = 1
            else:
                step = 0.1
            if event.button == 'up':
                q -= step
                if q <= 0: q = 0.
                self._fit_funcs_dict[self._sel_index+1]['par_values'][1] = q
                self._plot_fit_func()
            elif event.button == 'down':              
                q += step
                self._fit_funcs_dict[self._sel_index+1]['par_values'][1] = q
                self._plot_fit_func()

        elif name=='gaussian':
            fwhm = self._fit_funcs_dict[self._sel_index+1]['par_values'][0]
            if fwhm > 1:
                step = 1
            else:
                step = 0.1
            if event.button == 'up':
                fwhm -= step
                if fwhm <= 0: fwhm = 0.
                self._fit_funcs_dict[self._sel_index+1]['par_values'][0] = fwhm
                self._plot_fit_func()
            elif event.button == 'down':              
                fwhm += step
                self._fit_funcs_dict[self._sel_index+1]['par_values'][0] = fwhm
                self._plot_fit_func()            

        elif name=='power law':
            gamma = self._fit_funcs_dict[self._sel_index+1]['par_values'][0]

            step = 0.1
            if event.button == 'up':
                gamma -= step
                self._fit_funcs_dict[self._sel_index+1]['par_values'][0] = gamma
                self._plot_fit_func()
            elif event.button == 'down':              
                gamma += step
                self._fit_funcs_dict[self._sel_index+1]['par_values'][0] = gamma
                self._plot_fit_func()  


    def _plot_fit_func(self):

        # Getting the current x axis
        x = self._to_plot.freq
        x = x[x>0]

        if (not x is None) and (self._fit_funcs_dict != {}):
            counter = 0

            # Initializing total function
            psum = np.zeros(len(x))
            for key,value in self._fit_funcs_dict.items():

                if 'par_values' in value.keys():
                    # This remove previous plot of the function
                    if 'plots' in value.keys():
                        self._view._plot_area._ax.lines.remove(value['plots'])

                    # Computing function plot
                    col = value['color']
                    pars = value['par_values']
                    func_name = self._model_func_dict[value['name']]
                    func = init_sherpa_model(func_name,parvals=pars)
                    y = func(x)

                    # Plotting
                    ylim = self._view._plot_area._ax.set_ylim()
                    xlim = self._view._plot_area._ax.set_xlim()
                    if not self._view._norm_box._xy_flag.get():
                        line, = self._view._plot_area._ax.plot(x,y,'--',\
                            color = col,label=str(key))
                        psum += y
                    else:
                        line, = self._view._plot_area._ax.plot(x,y*x,'--',\
                            color = col,label=str(key))
                        psum += y*x
                    self._view._plot_area._ax.set_ylim(ylim)
                    self._view._plot_area._ax.set_xlim(xlim)
                    self._fit_funcs_dict[key]['plots'] = line
                    counter +=1

            # Plotting the full function
            if 'plot' in self._total_fit_func_dict.keys():
                self._view._plot_area._ax.lines.remove(self._total_fit_func_dict['plot'])
            if counter > 1:
                allp, = self._view._plot_area._ax.plot(x,psum,'r-')
                self._total_fit_func_dict['plot'] = allp

            self._view._plot_area._canvas.draw()

            # Print corresponding parameter values
            self._print_par_value() 

    def _print_par_value(self):
        '''
        Prints model parameters stored in self._fit_funcs_dict in the 
        parameter listbox
        '''

        # Cleaning par listbox
        listbox = self._fit_window._fit_parameters_box._fit_pars_listbox
        listbox.delete(0,tk.END)

        # Writing function pars only if plotted
        for key,value in self._fit_funcs_dict.items():

            err_flag = False
            if 'errors' in value.keys(): err_flag = True
            if 'plots' in value.keys():
                n_pars = len(value['par_values'])
                for i in range(n_pars):
                    if err_flag:
                        print('Printing parameters with errors')
                        # Minus sign is included in negative error
                        line = '{:2}) {:>5} = {:6.4} ({:4}) + {:6.6} {:7.6}'.\
                            format(key,value['par_names'][i],
                                float(value['par_values'][i]),
                                ('froz' if value['frozen'][i] else 'free'),
                                (0. if value['frozen'][i] else value['errors'][i][0]),
                                (0. if value['frozen'][i] else value['errors'][i][1]) )
                    else:
                        line = '{:2}) {:>5} = {:6.4} ({:4})'.\
                            format(key,value['par_names'][i],
                            float(value['par_values'][i]),
                            ('froz' if value['frozen'][i] else 'free'))
                    listbox.insert(tk.END,line) 
                    listbox.itemconfig(tk.END,{'fg':value['color']})          

    def _comp_fit(self):

        if not self._fit_on_canvas:
            new_window = tk.Toplevel(self._root)
            self._fit_result_window = FitResultView(new_window,controller=self)
            self._fit_result_window.grid(row=0,column=0,sticky='snew')

        # Preparing data
        # -------------------------------------------------------------
        freq = self._to_plot.freq
        y = self._to_plot.power
        yerr = self._to_plot.spower
        x = freq[freq>0]
        y = y[freq>0]
        yerr = yerr[freq>0]

        self._data_to_fit = Data1D('power_spectrum',x,y,staterror=yerr)

        freq_range = self._fit_window._freq_range_box._freq_range.get()
        if ',' in freq_range:
            ranges = freq_range.split(',')
            for range in ranges:
                range = range.strip()
                start = range.split('-')[0]
                stop  = range.split('-')[1]
                self._data_to_fit.notice(start,stop)
        else:
            start = freq_range.split('-')[0].strip()
            stop = freq_range.split('-')[1].strip()
            self._data_to_fit.notice(start,stop)
        # -------------------------------------------------------------
    
        self._build_model()

        self._stat = Chi2()        
        self._method = LevMar()
        self._fit = Fit(self._data_to_fit,self._model, 
                        stat=self._stat, method=self._method)
        self._fit_result = self._fit.fit()
        
        print('Fitting results')
        print('='*80)
        print('Model')
        print('-'*80)
        print(self._model)
        print('-'*80+'\n')
        print('-'*80)
        print(self._fit_result)
        print('-'*80)
        print('='*80)      

        self._update_fit_funcs()
        self._plot_model_func()
        if not self._fit_on_canvas:
            self._plot_fit()
            self._fit_on_canvas = True
        else:
            self._update_fit_plot()
        self._update_info()

    def _update_fit_funcs(self):
        print('Updating fit functions')
        for key, value in self._fit_funcs_dict.items():
            if 'plots' in value.keys():
                self._fit_funcs_dict[key]['par_values']=[]
                for par in self._model.pars:
                    if str(key) == par.fullname.split('.')[0]:
                        self._fit_funcs_dict[key]['par_values'] += [par.val] 

    def _update_fit_plot(self):

        # NOTE: Using the get_x() method you can extract the masked data
        y_data  = self._data_to_fit.get_y(filter=True)
        y_err   = self._data_to_fit.get_staterror(filter=True)
        y_model = self._data_to_fit.eval_model_to_fit(self._model)

        res  = (y_data-y_model)/y_err
        chi2 = self._stat.calc_chisqr(self._data_to_fit,self._model)

        self._line1.set_ydata(res)
        self._line2.set_ydata(chi2)    
        self._view._plot_area._canvas.draw()        

    def _plot_fit(self):

        # NOTE: Using the get_x() method you can extract the masked data
        x_data  = self._data_to_fit.get_x(filter=True)
        y_data  = self._data_to_fit.get_y(filter=True)
        y_err   = self._data_to_fit.get_staterror(filter=True)
        y_model = self._data_to_fit.eval_model_to_fit(self._model)

        res  = (y_data-y_model)/y_err
        chi2 = self._stat.calc_chisqr(self._data_to_fit,self._model)

        self._line1,=self._controller._ax1.plot(x_data,res,'-r')
        self._line2,=self._controller._ax2.plot(x_data,chi2,'-r')

        # Residuals
        maxr = np.max(abs(res))
        self._view._plot_area._ax1.set_xscale('log')
        self._view._plot_area._ax1.set_ylim([-maxr-maxr/3,maxr+maxr/3])
        self._view._plot_area._ax1.grid()
        self._view._plot_area._ax1.set_ylabel('Res. [(model-data)/err]',fontsize=12)
        self._view._plot_area._ax1.set_title('').set_visible(False)

        self._view._plot_area._ax1bis = self._view._plot_area._ax1.twinx()
        self._view._plot_area._to_plot.plot(ax=self._view._plot_area._ax1bis,\
            alpha=0.3,lfont=12,xy=self._view._plot_area._xy_flag.get())
        self._view._plot_area._ax1bis.set_ylabel('')
        self._view._plot_area._ax1bis.grid(False)
        self._view._plot_area._ax1bis.tick_params(axis='both',which='both',length=0)
        self._view._plot_area._ax1bis.set_yscale('log')  
        self._view._plot_area._ax1bis.set_yticklabels([])     
            
        # Contribution to chi2
        self._view._plot_area._ax2.set_ylabel('$\chi^2$',fontsize=12)
        self._view._plot_area._ax2.set_xscale('log')
        self._view._plot_area._ax2.set_xlabel('Frequency [ Hz]',fontsize=12)
        self._view._plot_area._ax2.grid()
        self._view._plot_area._ax2.set_title('').set_visible(False)
        self._view._plot_area._ax2.yaxis.set_label_position('left')
        self._view._plot_area._ax2.yaxis.tick_right()

        self._view._plot_area._ax2bis = self._view._plot_area._ax2.twinx()
        self._view._plot_area._to_plot.plot(ax=self._view._plot_area._ax2bis,\
            alpha=0.3,lfont=12,xy=self._view._plot_area._xy_flag.get())
        self._view._plot_area._ax2bis.set_ylabel('')
        self._view._plot_area._ax2bis.grid(False)
        self._view._plot_area._ax2bis.tick_params(axis='both',which='both',length=0)
        self._view._plot_area._ax2bis.set_yscale('log')  
        self._view._plot_area._ax2bis.set_yticklabels([])     
 
        if self._first_fit: 
            self._controller._canvas2.draw()
        self._controller._canvas2.mpl_connect(\
            'motion_notify_event',self._controller._update_cursor)