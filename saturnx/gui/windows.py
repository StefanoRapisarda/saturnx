
import matplotlib 
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

# Fitting data
import lmfit
from lmfit import Model,Parameters
from lmfit.model import save_modelresult,load_modelresult

from astropy.modeling import models, fitting
#from saba import SherpaFitter
#from astropy.modeling.fitting import SherpaFitter

from sherpa.data import Data1D
from sherpa.stats import LeastSq,Chi2,Chi2DataVar
from sherpa.optmethods import LevMar,MonCar, GridSearch
from sherpa.estmethods import Confidence, Covariance
from sherpa.fit import Fit
from sherpa.utils import calc_ftest
from sherpa.models.basic import Gauss1D,PowLaw1D,Const1D

import random
import uuid

import os
import pandas as pd
import numpy as np
import pickle
import random

from tkinter import ttk
import tkinter as tk
from tkinter import filedialog

import glob

from functools import partial

import sys
sys.path.append('/Volumes/Samsung_T5/saturnx')

from saturnx.fitting.fitting_functions import lorentzian
from saturnx.utils.rxte_functions import list_modes
from saturnx.utils.generic import plt_color
from saturnx.gui.tabs import FittingTab, TimingTab
from saturnx.fitting.astropy_custom_models import *
from saturnx.fitting.sherpa_custom_models import *
from saturnx.utils.pdf import pdf_page

__all__ = ['MakePowerWin','LogWindow','TestButton','RxteModes',
            'FitWindow_sherpa','PlotFitWindow']

def init_sherpa_model(sherpa_model,name=None,
        parvals=None,frozen=None,mins=None,maxs=None):
    '''
    Function to initialize sherpa model

    PARAMETERS
    ----------
    model: Sherpa model (not an istance)
    ''' 

    if not name is None:
        model = sherpa_model(name)
    else:
        model = sherpa_model()

    for i,par in enumerate(model.pars):
        if not parvals is None: par.val = parvals[i]
        if not frozen  is None: par.frozen = frozen[i]
        if not mins    is None: par.min = mins[i]
        if not maxs    is None: par.max = maxs[i]

    return model

def make_sherpa_result_dict(sherpa_result):
    '''
    Makes a dictionary containing information from fit result
    '''

    result_dict = {}
    lines=sherpa_result.__str__().split('\n')
    for line in lines:
        key = line.split('=')[0].strip()
        item = line.split('=')[1].strip()
        result_dict[key]=item
    return result_dict

def print_fit_results(stat_dict,fit_pars,plot1,plot2,output_name):
    stat_dict1 = {key:stat_dict[key] for key in list(stat_dict.keys())[:8]}
    stat_dict2 = {key:stat_dict[key] for key in list(stat_dict.keys())[8:]}

    # Preparing model parameters arrays
    par_names = []
    par_values = []
    par_perror = []
    par_nerror = []
    for key,item in fit_pars.items():
        for name,val,status,error in zip(item['par_names'],item['par_values'],item['frozen'],item['errors']):
            frozen = ('(frozen)' if status else '(free)')
            par_names += ['{:2}) {:>6}{:>8}'.format(key,frozen,name)]
            par_values += [f'{val:>20.6}']
            if error[0] == np.NaN:
                par_perror += ['+{:>20}'.format('NaN')]
            else:
                par_perror += [f'+{error[0]:>20.6}']
            if error[1] == np.NaN:
                par_nerror += ['-{:>20}'.format('NaN')]            
            else:
                par_nerror += ['-'+f'{error[1]:>20.6}'.replace('-','')]   
    par_info = [par_names,par_values,par_perror,par_nerror]

    pdf = pdf_page(margins=[10,10,10,10])
    pdf.add_page()

    # Printing fit statistics
    pdf.print_key_items(title='Fit statistics',info=stat_dict1,grid=[2,2,5,5],sel='11')
    pdf.print_key_items(title=' ',info=stat_dict2,grid=[4,2,5,5],sel='12')
    
    # Print parameters
    for i in range(1,5):
        coors = pdf.get_grid_coors(grid=[4,4,5,5],sel='2'+str(i))
        if i == 1:
            title = 'Fitting parameters'
        else:
            title = ' '
        pdf.print_column(title=title,rows=par_info[i-1],xy=(coors[0],coors[1]))
        
    # Plotting plots
    pdf.add_page()
    coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='11',margins=[10,0,0,0])
    pdf.image(plot1,x=coors[0],y=coors[1],h=coors[3]-coors[1])
    coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='21')
    pdf.image(plot2,x=coors[0],y=coors[1],h=coors[3]-coors[1])

    pdf.output(output_name,'F')

class MakePowerWin(tk.Tk):
    '''
    Window for easily select parameters to compute power spectra.

    It is a top level tkinter widget and inherit the properties of 
    tkinter.Tk.
    The window is intended to be used by another script, i.e. once
    the user set up all the variables (obs_IDs, fourier analysis
    parameters, energy bands, etc), the script will read the variables,
    feed them to the user defined function to compute fourier products,
     and print operation logs on another window.

    HISTORY
    -------
    2020 11 05, Stefano Rapisarda, Uppsala (Sweden)
        creation date. This is a second version of a previous window 
        that I've written around April-May 2020. I simplified the 
        previous version a lot and added new features to get the life
        of future Stefano easier. 
    2020 11 18, Stefano Rapisarda (Uppsala)
        - output suffix added (to give to your output a unique identi-
          fier);
        - added mission box, it autimatically updates when choosing the
          data folder;
        - added new RXTE data mode window, it displays the available
          obs modes per obs ID and it allows selection;
        - added button window for functionality testing.
    2020 12 05, Stefano Rapisarda (Uppsala)
        added panel options. The previous right frame is now in a 
        timing panel, the other panels are still to fill.
    '''

    def __init__(self,*args,**kwargs):
        '''
        Define two main frames (left and write), populate them with 
        widgets, and initialize variables.
        '''
 
        super().__init__(*args,**kwargs)

        #self.configure(bg='burlywood3')

        self._obs_id = ''

        #For testing bindings
        #self._new_window(TestButton)
        #self._test_button['command'] = self._read_boxes

        # Common style for main section labels 
        # s.configure('Red.TLabelframe.Label', font=('courier', 15, 'bold'))
        # s.configure('Red.TLabelframe.Label', foreground ='red')
        # s.configure('Red.TLabelframe.Label', background='blue')
        s = ttk.Style(self)
        s.configure('Black.TLabelframe.Label',
                    font=('times', 16, 'bold'))
        self._head_style = 'Black.TLabelframe'

        # Frame for obs ID list
        left_frame = ttk.Frame(self)
        left_frame.grid(row=0,column=0,padx=5,pady=5,sticky='nswe') 
        self._init_left_frame(left_frame)

        # Frame for settings
        right_frame = ttk.Frame(self)
        right_frame.grid(row=0,column=1,padx=5,pady=5,sticky='nswe') 
        self._init_right_frame(right_frame)  

    def _init_left_frame(self,frame):
        '''
        Populate the left frame with widgets, most importantly a 
        listbox to host a list of observational IDs
        '''

        # Init variables
        self._pickle_files = ()

        frame.grid_columnconfigure(0,weight=1)
        frame.grid_rowconfigure(1,weight=1)

        # Mission box
        box1 = ttk.LabelFrame(frame,text='Mission',
                style=self._head_style)
        box1.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        box1.grid_columnconfigure(0,weight=1)
        self._mission = tk.StringVar()
        self._mission.set(' ')
        # With this, every time self._mission is changed, either from 
        # the OptionMenu or by another method, self._mission_updated 
        # is called
        self._mission.trace_add('write', self._mission_updated)
        self._missions = ('NICER','RXTE','Swift','NuStar','HXMT')
        menu = tk.OptionMenu(box1, self._mission, *self._missions)
        menu.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')

        # Group of widget to put as label for box
        label_frame = tk.Frame(frame)
        label = ttk.Label(label_frame,text='Obs. IDs  ',
        font=('times', 16, 'bold'))
        label.grid(column=0,row=0,sticky='nswe')
        load_list = ttk.Button(label_frame,text='Load',width=4,
                             command=lambda: self._load_obs_ids(0))
        load_list.grid(column=1,row=0,sticky='nswe')  

        box2 = ttk.LabelFrame(frame,labelwidget=label_frame)
        box2.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        box2.grid_rowconfigure(0,weight=1)
        box2.grid_columnconfigure(0,weight=1)

        self._obs_id_box = tk.Listbox(box2, selectmode='extended')
        self._obs_id_box.grid(column=0,row=0,padx=5,pady=5,
                              sticky='nswe')
        self._obs_id_box.bind('<ButtonRelease-1>',self._click_on_obs_id)
        self._obs_id_box.bind('<Down>',self._click_on_obs_id)
        self._obs_id_box.bind('<Up>',self._click_on_obs_id)

    def _mission_updated(self,var,indx,mode):
        mission = self._mission.get().strip()

        if mission.upper() == 'RXTE':
            self._timing_tab._rxte_called = True
            for child in self._timing_tab._frame2.winfo_children():
                if hasattr(child,'to_disable'):
                    if child.to_disable:
                        child.configure(state='disabled')
            self._new_window(RxteModes)
        else:
            for child in self._timing_tab._frame2.winfo_children():
                if hasattr(child,'to_disable'):
                    if child.to_disable and child['state']=='disabled':
                        child.configure(state='normal')
            if self._timing_tab._rxte_called: 
                self.new.destroy()
                self._timing_tab._rxte_called = False

    def _click_on_obs_id(self,event):
        '''
        This function is (or should be) called every time the user 
        select a single or multiple items inside the Listbox
        self._obs_id_box
        '''

        # Listing pkl files inside a potential reduced product
        # folder
        sel = self._obs_id_box.curselection()
        if len(sel) == 0:
            self._obs_id = ''
            target_dir = self._fitting_tab._output_dir2.get()
            pickle_files = sorted(glob.glob('{}/*.pkl'.format(target_dir)))
            self._pickle_files = tuple(os.path.basename(pf) for pf in\
                pickle_files if 'power' in pf)
        elif len(sel) == 1:
            self._obs_id = self._obs_id_box.get(sel)
            target_dir = os.path.join(self._fitting_tab._output_dir2.get(),self._obs_id)
            pickle_files = sorted(glob.glob('{}/*.pkl'.format(target_dir)))
            self._pickle_files = tuple(os.path.basename(pf) for pf in\
                pickle_files if 'power' in pf)
        else:
            self._pickle_files = ()
        self._fitting_tab._update_file_menu()
        

        mission = self._mission.get().strip()

        if mission.upper() == 'RXTE':

            # Cleaning Treeview
            for child in self._mode_box.get_children():
                self._mode_box.delete(child)

            data_dir = self._input_dir.get()

            sel = self._obs_id_box.curselection()
            if len(sel) == 0:
                sel_obs_ids = sorted(self._obs_id_box.get(0,tk.END))
            else:
                sel_obs_ids = sorted([self._obs_id_box.get(s) for s in sel])

            info_dict = {}
            data_dir = self._input_dir.get()
            for obs_id in sel_obs_ids:
                obs_id_dir = os.path.join(data_dir,obs_id)
                info_dict[obs_id] = list_modes(obs_id_dir)

            modes = []
            for key, item in info_dict.items():
                for key2, item2 in item.items():
                    if not item2 is None:
                        dur = sum([stop-start for stop,start in \
                            zip(item2['mode_stop'],item2['mode_start'])])
                        modes += [item2['modename']]
                        line = [item2['modename'],dur,item2['modefile']]
                        self._mode_box.insert('','end',text='',values=line)

            if len(sel_obs_ids)>1:
                for child in self._mode_box.get_children():
                    self._mode_box.delete(child)
                modes = set(modes)
                for mode in modes:
                    line = [mode,'-','-']
                    self._mode_box.insert('','end',text='',values=line)

    def _load_obs_ids(self,value):
        '''
        Populates _obs_id_box with obs IDs, folders with names selected
        according to current mission. If no mission is selected, than
        all the folders in value will be listed.
        '''

        for mission in self._missions:
            if mission.upper() in self._timing_tab._input_dir.get().upper():
                self._mission.set(mission)
                break     

        self._obs_id_box.delete(0,tk.END)

        if type(value)==int:
            file_name = filedialog.askopenfilename(initialdir=os.getcwd(),
            title = 'Select a obs ID list file')
            if '.txt' in file_name:
                with open(file_name,'r') as infile:
                    lines = infile.readlines()
                obs_ids = sorted([line.strip() for line in lines])
            elif '.pkl' in file_name:
                with open(file_name,'rb') as infile:
                    obs_ids = pickle.load(infile)
        elif type(value)==str:
            if value != '':
                print('Loading obs IDs from {}'.format(value))
                dirs = next(os.walk(value))[1]
                # !!! RXTE obs IDs have - in their names
                mission = self._mission.get()
                if mission == 'RXTE' or mission == 'NICER':
                    obs_ids = sorted([d for d in dirs if d.replace('-','').isdigit()])
                elif mission == 'HXMT':
                    obs_ids = sorted([d for d in dirs if d[1:].replace('-','').isdigit()])
                else:
                    obs_ids = sorted([d for d in dirs])

        for obs_id in obs_ids: 
            self._obs_id_box.insert(tk.END,obs_id)


    def _test_change_mission(self):
        self._mission.set('Test Mission')

    def _init_right_frame(self,frame):
        '''
        Populate the right frame with widgets.

        '''

        tabControl = ttk.Notebook(frame)
        diagnosis_tab = ttk.Frame(tabControl)
        timing_tab = ttk.Frame(tabControl)
        plotting_tab = ttk.Frame(tabControl)
        fitting_tab = ttk.Frame(tabControl)
        reduction_tab = ttk.Frame(tabControl)
        tabControl.add(diagnosis_tab,text='Diagnosis')
        tabControl.add(reduction_tab,text='Data Reduction')
        tabControl.add(timing_tab, text='Timing')
        tabControl.add(plotting_tab,text='Plotting')
        tabControl.add(fitting_tab,text='Fitting')

        tabControl.grid(column=0,row=0,sticky='nswe')
        #tabControl.pack(expand=1, fill="both")

        self._current_tab = ''
        tabControl.bind('<<NotebookTabChanged>>',self._handle_tab_changed)

        self._timing_tab = TimingTab(timing_tab,self)
        #self._init_fitting_tab(fitting_tab)
        self._fitting_tab = FittingTab(fitting_tab,self,FitWindow_sherpa)

    def _handle_tab_changed(self,event):
        selection = event.widget.select()
        self._current_tab = event.widget.tab(selection,'text')  

    def _new_window(self, newWindow):
        self.new = tk.Toplevel()
        newWindow(self.new, self)       

class LogWindow:
    def __init__(self,parent,controller):
        self._controller = controller
        self._parent = parent

        frame = tk.Frame(self._parent)
        frame.pack()

        self._controller._logs = tk.Text(frame,width=120,height=40)
        self._controller._logs.grid(row=0,column=0,sticky='nswe')

class TestButton:
    def __init__(self,parent,controller):
        self._controller = controller
        self._parent = parent

        frame = tk.Frame(self._parent)
        frame.pack()

        self._controller._test_button = ttk.Button(frame, text='TEST')
        self._controller._test_button.grid(column=0,row=0,sticky='nswe')

class RxteModes:
    def __init__(self,parent,controller):
        self.parent = parent
        self.controller = controller

        main_frame = ttk.LabelFrame(self.parent,text='Obs Modes',
            style = self.controller._head_style)
        main_frame.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        columns = ['mode identifier','duration [s]','file name']
        self.controller._mode_box = ttk.Treeview(main_frame,columns=columns,
                            show='headings')
        self.controller._mode_box.grid(column=0,row=0,sticky='nswe')
        for column in columns:
            self.controller._mode_box.column(column)
            self.controller._mode_box.heading(column,text=column,
            command = lambda _col=column: \
                self._sort_modes(self.controller._mode_box, _col, False))

        #columns = ['mode identifier','file name','duration']
        #for i, column in enumerate(columns):
        #    self._mode_box.heading(str(i+1),text=column)

    def _sort_modes(self,tree,col,reverse):
        l = [(tree.set(k,col),k) for k in tree.get_children('')]
        l.sort(reverse=reverse)

        for index, (val,k) in enumerate(l):
            tree.move(k,'',index)

        tree.heading(col, command=lambda: self._sort_modes(tree,col,not reverse))   

class FitWindow_mpl:
    '''
    Window called by the fitting tab

    HISTORY
    -------
    2020 12 10, Stefano Rapisarda (Uppsala), creation date
        This was independently created in April, here I cleaned it up
        and I incorporate the window in MakePowerWin (specifically the
        timing tab).

    TODO:
    - Implement other fitting functions 
    '''
    def __init__(self,parent,controller):
        # Controller is the timing tab widget
        self._controller = controller
        self._parent = parent
        self._parent.title = 'Fit window'

        s = ttk.Style()
        s.configure('Black.TLabelframe.Label',
                    font=('times', 16, 'bold'))
        self._head_style = 'Black.TLabelframe'

        # self._fit_funcs_dict has an integer as key. Integer is assigned
        # according to the position of the fitting function in the
        # fit_funcs_listbox
        # For each fitting function, the dict item is another dictionary
        # with keys name, color, par_name (list), par_value (list), 
        # par_status (list), plots (containing the line plotted in the
        # canvas)
        self._fit_funcs_dict = {}
        # total_fit_func does not have any other specific function than 
        # storing the plot of the full function (the sum of different
        # model component)
        self._total_fit_func_dict = {}

        self._index = 1
        self._first_fit = True
        self._canvas_connected = False
        self._func_list = {'lorentzian':lorentzian}

        # Main frame
        frame = tk.Frame(self._parent)
        frame.grid(column=0,row=0,padx=5,pady=5)
        self._populate_main_frame(frame)

    def _populate_main_frame(self,frame):

        width = 200

        # Fit frequency boxes
        freq_frame = ttk.LabelFrame(frame,text='Frequency boundaries',\
            width=width,height=50)
        freq_frame.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._populate_freq_frame(freq_frame)

        # Fitting functions options and drawing them on the plot
        fit_func_frame = ttk.LabelFrame(frame,\
            text='Fitting functions',width=width)
        fit_func_frame.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        self._populate_fit_func_frame(fit_func_frame)

        # Fitting function parameters
        fit_pars_frame = ttk.LabelFrame(frame,\
            text='Fitting parameters',width=width)
        fit_pars_frame.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')
        self._populate_fit_pars_frame(fit_pars_frame)

    def _populate_freq_frame(self,frame):
        '''
        Sets the two buttons to select start and stop frequency for fit
        '''
        self._start_fit_freq = tk.DoubleVar()
        self._start_fit_freq.set(0)
        start_freq_entry = tk.Entry(frame, \
            textvar=self._start_fit_freq, width=10)
        start_freq_entry.grid(column=0,row=0,padx=5,pady=5,sticky='w')

        self._stop_fit_freq = tk.DoubleVar()
        self._stop_fit_freq.set(100)
        stop_freq_entry = tk.Entry(frame, \
            textvar=self._stop_fit_freq, width=10)
        stop_freq_entry.grid(column=2,row=0,padx=5,pady=5,sticky='w')

        dummy1 = tk.Label(frame,text='-')
        dummy1.grid(column=1,row=0,padx=5,pady=5,sticky='w')
        dummy2 = tk.Label(frame,text='[Hz]')
        dummy2.grid(column=3,row=0,padx=5,pady=5,sticky='w')

    def _populate_fit_func_frame(self,frame):

        frame.grid_columnconfigure(0,weight=1)

        # Left box (fitting function list Box)
        # -------------------------------------------------------------
        left_frame = tk.Frame(frame)
        left_frame.grid(column=0,row=0,sticky='nswe')

        self._fit_func_listbox = tk.Listbox(left_frame,\
            selectmode='multiple',height=12)
        self._fit_func_listbox.grid(column=0,row=1,\
            padx=5,pady=5,sticky='nsew')

        # Draw and hold radio button
        radio_frame = tk.Frame(left_frame)
        radio_frame.grid(column=0,row=2,sticky='nsew')

        self._v = tk.IntVar()
        draw_radio = tk.Radiobutton(radio_frame, text='DRAW',\
            variable = self._v, value = 1,\
            command=self._activate_draw_func)
        draw_radio.grid(column=0,row=0,padx=5,pady=5,sticky='nsew')
        hold_radio = tk.Radiobutton(radio_frame, text='HOLD',\
            variable = self._v, value = 0, \
            command=self._hold_func)
        hold_radio.grid(column=1,row=0,padx=5,pady=5,sticky='ensw')
        hold_radio.select()  
        # -------------------------------------------------------------

        # Right box
        # -------------------------------------------------------------
        right_frame = tk.Frame(frame)
        right_frame.grid(column=1,row=0,sticky='nswe')
        
        # Fitting function menu
        self._fit_func = tk.StringVar()
        fit_funcs = tuple([i for i in self._func_list.keys()])
        fit_func_box = ttk.OptionMenu(right_frame,\
            self._fit_func,*fit_funcs)
        fit_func_box.grid(column=0,row=0, columnspan=2,\
            sticky='w',padx=5,pady=5)

        # Add and delete buttons
        add_button = ttk.Button(right_frame, text='ADD', \
            command=self._clickAdd)
        add_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        del_button = ttk.Button(right_frame, text='DEL', \
            command=self._clickDel)
        del_button.grid(column=1,row=1,padx=5,pady=5,sticky='e') 

        # Fit and clear button
        fit_button = ttk.Button(right_frame, text='FIT', \
            command=self._clickFit)
        fit_button.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')        
        clear_button = ttk.Button(right_frame, text='CLEAR', \
            command=self._clear)
        clear_button.grid(column=1,row=2,padx=5,pady=5,sticky='nsew')   

        # Save and load buttons
        save_frame = ttk.LabelFrame(right_frame,text='output name')
        save_frame.grid(column=0,row=3,columnspan=2,padx=5,pady=5,sticky='nswe')
        self._output_name = tk.StringVar()
        name_entry = tk.Entry(save_frame,textvariable=self._output_name)
        name_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        save_button = ttk.Button(save_frame, text='SAVE', \
            command=self._save_fit)
        save_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')        
        load_button = ttk.Button(save_frame, text='LOAD', \
            command=self._load_fit)
        load_button.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')   
        # -------------------------------------------------------------

    def _clickAdd(self):
        # Add a fitting function to the dictionary of fitting 
        # functions and update the ditting function box
        # If it is the first call, seld._index = 1

        # To avoid a new added function has the same color of the just
        # deleted one
        colors = plt_color(hex=True)
        for key,item in self._fit_funcs_dict.items():
            colors.remove(item['color'])
        col = colors[0]

        self._fit_funcs_dict[self._index] = {'name':self._fit_func.get(),'color':col}
        self._fit_func_listbox.insert(tk.END,\
            str(self._index)+') '+self._fit_func.get())
        # Give a color to text in the color box corresponding to the 
        # just added function
        self._fit_func_listbox.itemconfig(self._index-1,{'fg':col})
        self._index += 1
        
        # Disconnect the canvas when adding a new function
        self._v.set(0)
        if self._canvas_connected:
            self._disconnect()

    def _clickDel(self):
        # Selected items can be more than 1
        sel = self._fit_func_listbox.curselection()
        #print(sel)
    
        # Aelection index MUST be reversed because self._fit_func_listbox 
        # index changes every time you remove an item
        for index in sel[::-1]:
            #print('Deleting function index',index)

            # Removing selected items from listbox
            self._fit_func_listbox.delete(index) 

            # Removing selected items from plot
            if 'plots' in self._fit_funcs_dict[index+1].keys():
                self._controller._ax.lines.remove(self._fit_funcs_dict[index+1]['plots'])
    
            # Removing selected items from self._fit_func_dict
            del self._fit_funcs_dict[index+1]

        # Resetting index of the remaining items
        items = self._fit_func_listbox.get(0,tk.END)
        if len(items) != 0:
            self._reset_index()
            if not self._controller._to_plot is None:
                self._plot_func()
                self._print_par_value()

       # Disconnect the canvas when deleting a new function
        self._v.set(0)
        if self._canvas_connected:
            self._disconnect()

        #for key,value in self._fit_funcs_dict.items():
        #    print(key,value)
        #print('-'*80)
        print('Index at the end of deleting',self._index)

    def _populate_fit_pars_frame(self,frame):
        
        # Left box
        # -------------------------------------------------------------
        left_frame = tk.Frame(frame)
        left_frame.grid(column=0,row=0,sticky='nswe')

        self._fit_pars_listbox = tk.Listbox(left_frame, selectmode='multiple')
        self._fit_pars_listbox .grid(column=0,row=0,padx=5,pady=5,\
            sticky='nsew')
        # -------------------------------------------------------------

        # Right box
        # -------------------------------------------------------------
        right_frame = tk.Frame(frame)
        right_frame.grid(column=1,row=0,sticky='nswe')

        self._par_val = tk.StringVar()
        reset_entry = tk.Entry(right_frame, textvariable=self._par_val, \
            width=10)
        reset_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nsew')  

        reset_button = ttk.Button(right_frame, text='RESET', \
            command=lambda: self._reset_par_value())
        reset_button.grid(column=1,row=0,padx=5,pady=5,sticky='nsew')        

        self._vII = tk.IntVar()
        freeze_radio = tk.Radiobutton(right_frame, text='FREEZE',\
            variable = self._vII, value = 1, \
            command=lambda: self._reset_par_value(freeze=True))
        freeze_radio.grid(column=0,row=1,padx=5,pady=5,sticky='senw')
        free_radio = tk.Radiobutton(right_frame, text='FREE',
            variable = self._vII, value = 0, \
            command=lambda: self._reset_par_value())
        free_radio.grid(column=1,row=1,padx=7,pady=7,sticky='senw')
        free_radio.select() 
        # -------------------------------------------------------------

    def _freeze_par(self):
        par_val = float(self._par_val.get())        

    def _reset_par_value(self,freeze=False):
        # First getting the value then the selected function and parameter
        sel = self._fit_pars_listbox.curselection()
        items = self._fit_pars_listbox.get(0,tk.END)
        for index in sel[::-1]:
            item = items[index]
            key = int(item.split(')')[0].split('L')[1])
            par_name = item.split('=')[0].strip().split(')')[1].strip()
            pars = self._fit_funcs_dict[key]['par_name']

            for i in range(len(pars)):
                if par_name == self._fit_funcs_dict[key]['par_name'][i]:

                    if self._par_val.get() == '':
                        par_val = self._fit_funcs_dict[key]['par_value'][i]
                    else:
                        par_val = float(self._par_val.get())

                    self._fit_funcs_dict[key]['par_value'][i] = par_val  
                    if freeze:
                        self._fit_funcs_dict[key]['par_status'][i] = False
                    else:
                        self._fit_funcs_dict[key]['par_status'][i] = True
        self._print_par_value()
        self._plot_func()

    def _print_par_value(self):
        # Cleaning par listbox
        self._fit_pars_listbox.delete(0,tk.END)

        # Writing function pars only if plotted
        for key,value in self._fit_funcs_dict.items():

            if 'plots' in value.keys():
                n_pars = len(value['par_value'])
                for i in range(n_pars):
                    line = 'L{:2}) {:4} = {:6.4} ({})'.\
                        format(key,value['par_name'][i],
                               float(value['par_value'][i]),
                                ('free' if value['par_status'][i] else 'frozen'))
                    self._fit_pars_listbox.insert(tk.END,line) 
                    self._fit_pars_listbox.itemconfig(tk.END,{'fg':value['color']})                 



    
    def _clear(self):
        # Deleting plots (if existing)
        items = self._fit_pars_listbox.get(0,tk.END)
        if len(items) != 0:
            print('Cleaning ')
            for key,item in self._fit_funcs_dict.items():
                print('Cleaning parameter',key)
                self._controller._ax.lines.remove(self._fit_funcs_dict[key]['plots'])
        if 'plot' in self._total_fit_func_dict.keys():
            self._controller._ax.lines.remove(self._total_fit_func_dict['plot'])
        self._controller._canvas.draw()
        self._controller._ax1.clear()
        self._controller._ax1bis.clear()
        self._controller._ax2.clear()
        self._controller._ax2bis.clear()
        self._controller._canvas2.draw()

        # Deleting boxes
        self._fit_func_listbox.delete(0,tk.END)
        self._fit_pars_listbox.delete(0,tk.END)
        self._controller._fit_info_box.delete(0,tk.END)
        
        # Resetting variables
        self._first_fit = True
        self._index = 1
        self._fit_funcs_dict = {}
        self._total_fit_func_dict = {}



    def _save_fit(self):
        fit_dir = os.path.join(self._controller._output_dir2.get(),\
            self._controller._obs_id,'fits')
        os.system(f'mkdir {fit_dir}')
        save_modelresult(self._fit_result,\
            os.path.join(fit_dir,self._output_name.get()+'.sav'))

    def _load_fit(self):
        pass

    def _reset_index(self):
        items = self._fit_func_listbox.get(0,tk.END)

        # Reading and storing old info
        old_func_info = []
        old_items = []
        for i in range(len(items)):
            item = items[i]
            old_items += [item.split(')')[1].strip()]
            old_index = int(item.split(')')[0].strip())
            old_func_info += [self._fit_funcs_dict[old_index]]
        
        # Cleaning listbox and self._fit_funcs_dict
        self._fit_func_listbox.delete(0,tk.END)
        self._fit_funcs_dict = {}
        
        # Re-writing old items with new index
        for i in range(len(items)):
            self._fit_func_listbox.insert(tk.END,str(i+1)+') '+old_items[i])
            self._fit_func_listbox.itemconfig(i,{'fg':old_func_info[i]['color']})
            self._fit_funcs_dict[i+1] = old_func_info[i]

        # Resetting index
        print('len(items) + 1',len(items) + 1)
        self._index = len(items) + 1

    def _close(self):
        self._parent.destroy()

    def _activate_draw_func(self):
        items = self._fit_func_listbox.get(0,tk.END)
        # There must be at least a function to draw
        #print(len(items),self._controller._to_plot)
        if len(items) != 0 and not self._controller._to_plot is None:

            print('Draw function activated')

            # Connecting the canvas
            if not self._canvas_connected:
                self._connect()

            sel = self._fit_func_listbox.curselection()
            # If nothing is selected, select the first one
            if not sel:
                sel = (0,)
            # In case of (accidental) multiple selection, it will be 
            # considered always the first one
            self._sel_index = int(sel[0])

    def _hold_func(self):
        if self._canvas_connected:
            self._disconnect()

        
    def _plot_func(self,reset=False):

        x = self._controller._to_plot.freq
        x = x[x>0]
        if not x is None:
            counter = 0
            psum = np.zeros(len(x))
            for key,value in self._fit_funcs_dict.items():

                if 'par_value' in value.keys():

                    # This remove previous plot of the function
                    if 'plots' in value.keys():
                        self._controller._ax.lines.remove(value['plots'])

                    # Computing function plot
                    col = value['color']
                    pars = value['par_value']
                    func = self._func_list[value['name']]
                    y = func(x,*pars)

                    # Plotting
                    ylim = self._controller._ax.set_ylim()
                    xlim = self._controller._ax.set_xlim()
                    if self._controller._norm.get() == 'Leahy':
                        lor, = self._controller._ax.plot(x,y,'--',color = col)
                        psum += y
                    elif self._controller._norm.get() == 'RMS':
                        lor, = self._controller._ax.plot(x,y*x,'--',color = col)
                        psum += y*x
                    self._controller._ax.set_ylim(ylim)
                    self._controller._ax.set_xlim(xlim)
                    self._fit_funcs_dict[key]['plots'] = lor
                    counter +=1

            # Replotting full function
            if 'plot' in self._total_fit_func_dict.keys():
                self._controller._ax.lines.remove(self._total_fit_func_dict['plot'])
            if counter > 1:
                allp, = self._controller._ax.plot(x,psum,'r-')
                self._total_fit_func_dict['plot'] = allp

            self._controller._canvas.draw()
            self._print_par_value()

    def _connect(self):
        if not self._canvas_connected:
            print('Canvas is connected')
            self._cidclick = self._controller._canvas.mpl_connect('button_press_event',self._on_click)
            self._cidscroll = self._controller._canvas.mpl_connect('scroll_event',self._on_roll)
            self._canvas_connected = True

    def _disconnect(self):
        if self._canvas_connected:
            print('Canvas is diconnected')
            self._controller._canvas.mpl_disconnect(self._cidclick)
            self._controller._canvas.mpl_disconnect(self._cidscroll)
            self._canvas_connected = False

    def _on_click(self,event):
        if not event.dblclick:
            # Left click or right lick
            if event.button == 1 or event.button == 3:
                # Position of the cursor
                self._xpos = event.xdata
                self._ypos = event.ydata

                if (self._xpos != None) and (self._ypos != None):

                    # Choosing standard value if not existing
                    if not 'par_value' in self._fit_funcs_dict[self._sel_index+1].keys():
                        q = 10
                    else:
                        q = self._fit_funcs_dict[self._sel_index+1]['par_value'][1]

                    # Computing amp according to cursort position
                    delta = self._xpos/np.sqrt(1+4*q**2)
                    r2 = np.pi/2 + np.arctan(2*q)
                    amp = self._ypos*r2*delta
                    if self._controller._xy_flag:
                        self._fit_funcs_dict[self._sel_index+1]['par_value'] = \
                            [amp/self._xpos,q,self._xpos]
                    else:
                         self._fit_funcs_dict[self._sel_index+1]['par_value'] = \
                            [amp,q,self._xpos]                       
                    self._fit_funcs_dict[self._sel_index+1]['par_status'] = \
                        [True,True,True]    
                    self._fit_funcs_dict[self._sel_index+1]['par_name'] = \
                        ['amp','q','freq']    

                    # Plotting                                   
                    self._plot_func()

            if event.button == 2:
                self._disconnect()
                self._v.set(0)

    def _on_roll(self,event):
        q = self._fit_funcs_dict[self._sel_index+1]['par_value'][1]
        if q > 1:
            step = 1
        else:
            step = 0.1
        if event.button == 'up':
            q -= step
            if q <= 0: q = 0.
            self._fit_funcs_dict[self._sel_index+1]['par_value'][1] = q
            self._plot_func()
        elif event.button == 'down':              
            q += step
            self._fit_funcs_dict[self._sel_index+1]['par_value'][1] = q
            self._plot_func()

            
    def _clickFit(self):  

        if self._first_fit:
            self._controller._new_child_window(PlotFitWindow)

        self._hold_func()

        freq = self._controller._to_plot.freq
        y = self._controller._to_plot.power
        yerr = self._controller._to_plot.spower
        x = freq[freq>0]
        y = y[freq>0]
        yerr = yerr[freq>0]

        self._fit_mask = (x> self._start_fit_freq.get()) & (x<= self._stop_fit_freq.get())
        self._build_model()
        #init = self.model.eval(self.fit_pars,x=x[self.fit_mask])
        self._fit_result = self._model.fit(y[self._fit_mask],\
            self._fit_pars,x=x[self._fit_mask],\
            weights=1./(yerr[self._fit_mask]),mthod='leastsq')
        #self.comps = self.controller.fit_result.eval_components(x=x[self.fit_mask])
        self._update_fit_funcs()
        self._plot_func()
        if self._first_fit:
            self._plot_fit()
            self._first_fit = False
        else:
            self._update_fit_plot()
        self._update_info()

    def _update_fit_funcs(self):
        for key, value in self._fit_funcs_dict.items():
            if 'plots' in value.keys():
                par_names = value['par_name']
                n_pars = len(par_names)

                for i in range(n_pars):
                    par_name = 'L{}_{}'.format(key,par_names[i])
                    self._fit_funcs_dict[key]['par_value'][i] = \
                        self._fit_result.best_values[par_name]

    def _update_fit_plot(self):
        freq = self._controller._to_plot.freq
        y = self._controller._to_plot.power
        yerr = self._controller._to_plot.spower
        x = freq[freq>0]
        y = y[freq>0]
        yerr = yerr[freq>0]

        self._line1.set_ydata(self._fit_result.best_fit-y[self._fit_mask])
        self._line2.set_ydata((self._fit_result.best_fit-y[self._fit_mask])**2/yerr[self._fit_mask]**2/self._fit_result.nfree)    
        self._controller._canvas2.draw()


    def _plot_fit(self):
        freq = self._controller._to_plot.freq
        y = self._controller._to_plot.power
        yerr = self._controller._to_plot.spower
        x = freq[freq>0]
        y = y[freq>0]
        yerr = yerr[freq>0]

        self._line1,=self._controller._ax1.plot(
            x[self._fit_mask],\
             (self._fit_result.best_fit-y[self._fit_mask]),'-r'
            )
        self._line2,=self._controller._ax2.plot(
            x[self._fit_mask],\
             (self._fit_result.best_fit-y[self._fit_mask])**2/yerr[self._fit_mask]**2/self._fit_result.nfree,'-r'
            )

        # Residuals
        maxr = np.max(abs(self._fit_result.best_fit-y[self._fit_mask]))
        #ax.plot(x[fit_mask],result.best_fit-y,'r')
        self._controller._ax1.set_xscale('log')
        self._controller._ax1.set_ylim([-maxr-maxr/3,maxr+maxr/3])
        self._controller._ax1.grid()
        #self._controller._ax1.set_xlabel('Frequency [ Hz]')
        self._controller._ax1.set_ylabel('Residuals [model-data]',fontsize=12)
        self._controller._ax1.set_title('').set_visible(False)

        self._controller._ax1bis = self._controller._ax1.twinx()
        self._controller._to_plot.plot(ax=self._controller._ax1bis,\
            alpha=0.3,lfont=12,xy=self._controller._xy_flag,marker='')
        self._controller._ax1bis.set_ylabel('')
        self._controller._ax1bis.grid(False)
        self._controller._ax1bis.tick_params(axis='both',which='both',length=0)
        self._controller._ax1bis.set_yscale('log')  
        self._controller._ax1bis.set_yticklabels([])     
            

        # Contribution to chi2
        self._controller._ax2.set_ylabel('Contribution to $\chi^2$',fontsize=12)
        self._controller._ax2.set_xscale('log')
        self._controller._ax2.set_xlabel('Frequency [ Hz]',fontsize=12)
        self._controller._ax2.grid()
        self._controller._ax2.set_title('').set_visible(False)
        self._controller._ax2.yaxis.set_label_position('left')
        self._controller._ax2.yaxis.tick_right()

        self._controller._ax2bis = self._controller._ax2.twinx()
        self._controller._to_plot.plot(ax=self._controller._ax2bis,\
            alpha=0.3,lfont=12,xy=self._controller._xy_flag,marker='')
        self._controller._ax2bis.set_ylabel('')
        self._controller._ax2bis.grid(False)
        self._controller._ax2bis.tick_params(axis='both',which='both',length=0)
        self._controller._ax2bis.set_yscale('log')  
        self._controller._ax2bis.set_yticklabels([])     
 
        if self._first_fit: self._controller._canvas2.draw()

    def _build_model(self):
        first = True
        for key, value in self._fit_funcs_dict.items():
            print(key)
            if 'plots' in value.keys():
                par_names = value['par_name']
                n_pars = len(par_names)
                func = self._func_list[value['name']]

                print('This one',key,func)
                
                tmp_model = Model(func,prefix='L{}_'.format(key))
                if first:
                    first = False
                    self._fit_pars = tmp_model.make_params()
                    self._model = tmp_model
                else:
                    self._fit_pars.update(tmp_model.make_params())
                    self._model += tmp_model

                par_label = par_names

                for i in range(n_pars):
                    par_val = value['par_value'][i]
                    status = value['par_status'][i]
                    self._fit_pars['L{}_{}'.\
                        format(key,par_label[i])].\
                            set(value=par_val,vary=status,min=0)

    def _update_info(self):
        self._controller._report = lmfit.fit_report(self._fit_result).split('\n')
        for line in self._controller._report:
            self._controller._fit_info_box.insert(tk.END,line)
        if not self._first_fit:
            self._controller._fit_info_box.insert(tk.END,'='*70+'\n')

class FitWindow_astropy:
    '''
    Window called by the fitting tab

    HISTORY
    -------
    2020 12 10, Stefano Rapisarda (Uppsala), creation date
        This was independently created in April, here I cleaned it up
        and I incorporate the window in MakePowerWin (specifically the
        timing tab).

    TODO:
    - Implement other fitting functions 
    '''
    def __init__(self,parent,controller):
        # Controller is the timing tab widget
        self._controller = controller
        self._parent = parent
        self._parent.title = 'Fit window'

        s = ttk.Style()
        s.configure('Black.TLabelframe.Label',
                    font=('times', 16, 'bold'))
        self._head_style = 'Black.TLabelframe'

        # self._fit_funcs_dict has an integer as key. Integer is assigned
        # according to the position of the fitting function in the
        # fit_funcs_listbox
        # For each fitting function, the dict item is another dictionary
        # with keys name, color, par_name (list), par_value (list), 
        # par_status (list), plots (containing the line plotted in the
        # canvas)
        self._fit_funcs_dict = {}
        # total_fit_func does not have any other specific function than 
        # storing the plot of the full function (the sum of different
        # model component)
        self._total_fit_func_dict = {}

        self._index = 1
        self._first_fit = True
        self._canvas_connected = False
        # Listing 1D models
        self._func_list = {'fmax_lorentzian':Fmax_lorentzian1D,
                           'f0_lorentzian':F0_lorentzian1D}

        # Main frame
        frame = tk.Frame(self._parent)
        frame.grid(column=0,row=0,padx=5,pady=5)
        self._populate_main_frame(frame)

    def _populate_main_frame(self,frame):

        width = 200

        # Fit frequency boxes
        freq_frame = ttk.LabelFrame(frame,text='Frequency boundaries',\
            width=width,height=50)
        freq_frame.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._populate_freq_frame(freq_frame)

        # Fitting functions options and drawing them on the plot
        fit_func_frame = ttk.LabelFrame(frame,\
            text='Fitting functions',width=width)
        fit_func_frame.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        self._populate_fit_func_frame(fit_func_frame)

        # Fitting function parameters
        fit_pars_frame = ttk.LabelFrame(frame,\
            text='Fitting parameters',width=width)
        fit_pars_frame.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')
        self._populate_fit_pars_frame(fit_pars_frame)

    def _populate_freq_frame(self,frame):
        '''
        Sets the two buttons to select start and stop frequency for fit
        '''
        self._start_fit_freq = tk.DoubleVar()
        self._start_fit_freq.set(0)
        start_freq_entry = tk.Entry(frame, \
            textvar=self._start_fit_freq, width=10)
        start_freq_entry.grid(column=0,row=0,padx=5,pady=5,sticky='w')

        self._stop_fit_freq = tk.DoubleVar()
        self._stop_fit_freq.set(100)
        stop_freq_entry = tk.Entry(frame, \
            textvar=self._stop_fit_freq, width=10)
        stop_freq_entry.grid(column=2,row=0,padx=5,pady=5,sticky='w')

        dummy1 = tk.Label(frame,text='-')
        dummy1.grid(column=1,row=0,padx=5,pady=5,sticky='w')
        dummy2 = tk.Label(frame,text='[Hz]')
        dummy2.grid(column=3,row=0,padx=5,pady=5,sticky='w')

    def _populate_fit_func_frame(self,frame):

        frame.grid_columnconfigure(0,weight=1)

        # Left box (fitting function list Box)
        # -------------------------------------------------------------
        left_frame = tk.Frame(frame)
        left_frame.grid(column=0,row=0,sticky='nswe')

        self._fit_func_listbox = tk.Listbox(left_frame,\
            selectmode='multiple',height=12)
        self._fit_func_listbox.grid(column=0,row=1,\
            padx=5,pady=5,sticky='nsew')

        # Draw and hold radio button
        radio_frame = tk.Frame(left_frame)
        radio_frame.grid(column=0,row=2,sticky='nsew')

        self._v = tk.IntVar()
        draw_radio = tk.Radiobutton(radio_frame, text='DRAW',\
            variable = self._v, value = 1,\
            command=self._activate_draw_func)
        draw_radio.grid(column=0,row=0,padx=5,pady=5,sticky='nsew')
        hold_radio = tk.Radiobutton(radio_frame, text='HOLD',\
            variable = self._v, value = 0, \
            command=self._hold_func)
        hold_radio.grid(column=1,row=0,padx=5,pady=5,sticky='ensw')
        hold_radio.select()  
        # -------------------------------------------------------------

        # Right box
        # -------------------------------------------------------------
        right_frame = tk.Frame(frame)
        right_frame.grid(column=1,row=0,sticky='nswe')
        
        # Fitting function menu
        self._fit_func = tk.StringVar()
        fit_funcs = tuple([i for i in self._func_list.keys()])
        fit_func_box = ttk.OptionMenu(right_frame,\
            self._fit_func,*fit_funcs)
        fit_func_box.grid(column=0,row=0, columnspan=2,\
            sticky='w',padx=5,pady=5)

        # Add and delete buttons
        add_button = ttk.Button(right_frame, text='ADD', \
            command=self._clickAdd)
        add_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        del_button = ttk.Button(right_frame, text='DEL', \
            command=self._clickDel)
        del_button.grid(column=1,row=1,padx=5,pady=5,sticky='e') 

        # Fit and clear button
        fit_button = ttk.Button(right_frame, text='FIT', \
            command=self._clickFit)
        fit_button.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')        
        clear_button = ttk.Button(right_frame, text='CLEAR', \
            command=self._clear)
        clear_button.grid(column=1,row=2,padx=5,pady=5,sticky='nsew')   

        # Save and load buttons
        save_frame = ttk.LabelFrame(right_frame,text='output name')
        save_frame.grid(column=0,row=3,columnspan=2,padx=5,pady=5,sticky='nswe')
        self._output_name = tk.StringVar()
        name_entry = tk.Entry(save_frame,textvariable=self._output_name)
        name_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        save_button = ttk.Button(save_frame, text='SAVE', \
            command=self._save_fit)
        save_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')        
        load_button = ttk.Button(save_frame, text='LOAD', \
            command=self._load_fit)
        load_button.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')   
        # -------------------------------------------------------------

    def _clickAdd(self):
        # Add a fitting function to the dictionary of fitting 
        # functions and update the ditting function box
        # If it is the first call, seld._index = 1

        # To avoid a new added function has the same color of the just
        # deleted one
        colors = plt_color(hex=True)
        for key,item in self._fit_funcs_dict.items():
            colors.remove(item['color'])
        col = colors[0]

        self._fit_funcs_dict[self._index] = {'name':self._fit_func.get(),'color':col}
        self._fit_func_listbox.insert(tk.END,\
            str(self._index)+') '+self._fit_func.get())
        # Give a color to text in the color box corresponding to the 
        # just added function
        self._fit_func_listbox.itemconfig(self._index-1,{'fg':col})
        self._index += 1
        
        # Disconnect the canvas when adding a new function
        self._v.set(0)
        if self._canvas_connected:
            self._disconnect()

    def _clickDel(self):
        # Selected items can be more than 1
        sel = self._fit_func_listbox.curselection()
        #print(sel)
    
        # Aelection index MUST be reversed because self._fit_func_listbox 
        # index changes every time you remove an item
        for index in sel[::-1]:
            #print('Deleting function index',index)

            # Removing selected items from listbox
            self._fit_func_listbox.delete(index) 

            # Removing selected items from plot
            if 'plots' in self._fit_funcs_dict[index+1].keys():
                self._controller._ax.lines.remove(self._fit_funcs_dict[index+1]['plots'])
    
            # Removing selected items from self._fit_func_dict
            del self._fit_funcs_dict[index+1]

        # Resetting index of the remaining items
        items = self._fit_func_listbox.get(0,tk.END)
        if len(items) != 0:
            self._reset_index()
            if not self._controller._to_plot is None:
                self._plot_func()
                self._print_par_value()

       # Disconnect the canvas when deleting a new function
        self._v.set(0)
        if self._canvas_connected:
            self._disconnect()

        #for key,value in self._fit_funcs_dict.items():
        #    print(key,value)
        #print('-'*80)
        print('Index at the end of deleting',self._index)

    def _populate_fit_pars_frame(self,frame):
        
        # Top (parameter) box
        # -------------------------------------------------------------
        top_frame = tk.Frame(frame)
        top_frame.grid(column=0,row=0,sticky='nswe')

        self._fit_pars_listbox = tk.Listbox(top_frame, selectmode='multiple')
        self._fit_pars_listbox .grid(column=0,row=0,padx=5,pady=5,\
            sticky='nsew')
        # -------------------------------------------------------------

        # Bottom (button) box
        # -------------------------------------------------------------
        bot_frame = tk.Frame(frame)
        bot_frame.grid(column=0,row=1,sticky='nswe')

        self._par_val = tk.StringVar()
        reset_entry = tk.Entry(bot_frame, textvariable=self._par_val, \
            width=10)
        reset_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nsew')  

        reset_button = ttk.Button(bot_frame, text='RESET', \
            command=lambda: self._reset_par_value())
        reset_button.grid(column=1,row=0,padx=5,pady=5,sticky='nsew')        


        self._vII = tk.IntVar()
        freeze_radio = tk.Radiobutton(bot_frame, text='FREEZE',\
            variable = self._vII, value = 1, \
            command=lambda: self._reset_par_value(freeze=True))
        freeze_radio.grid(column=2,row=0,padx=5,pady=5,sticky='senw')
        free_radio = tk.Radiobutton(bot_frame, text='FREE',
            variable = self._vII, value = 0, \
            command=lambda: self._reset_par_value())
        free_radio.grid(column=3,row=0,padx=7,pady=7,sticky='senw')
        free_radio.select() 
        # -------------------------------------------------------------

    def _freeze_par(self):
        par_val = float(self._par_val.get())        

    def _reset_par_value(self,freeze=False):
        # First getting the value then the selected function and parameter
        sel = self._fit_pars_listbox.curselection()
        items = self._fit_pars_listbox.get(0,tk.END)
        for index in sel[::-1]:
            item = items[index]
            key = int(item.split(')')[0].split('L')[1])
            par_name = item.split('=')[0].strip().split(')')[1].strip()
            pars = self._fit_funcs_dict[key]['par_name']

            for i in range(len(pars)):
                if par_name == self._fit_funcs_dict[key]['par_name'][i]:

                    if self._par_val.get() == '':
                        par_val = self._fit_funcs_dict[key]['par_value'][i]
                    else:
                        par_val = float(self._par_val.get())

                    self._fit_funcs_dict[key]['par_value'][i] = par_val  
                    if freeze:
                        self._fit_funcs_dict[key]['par_status'][i] = False
                    else:
                        self._fit_funcs_dict[key]['par_status'][i] = True
        self._print_par_value()
        self._plot_func()

    def _print_par_value(self):
        # Cleaning par listbox
        self._fit_pars_listbox.delete(0,tk.END)

        # Writing function pars only if plotted
        for key,value in self._fit_funcs_dict.items():

            if 'plots' in value.keys():
                n_pars = len(value['par_value'])
                for i in range(n_pars):
                    line = 'L{:2}) {:4} = {:6.4} ({})'.\
                        format(key,value['par_name'][i],
                               float(value['par_value'][i]),
                                ('free' if value['par_status'][i] else 'frozen'))
                    self._fit_pars_listbox.insert(tk.END,line) 
                    self._fit_pars_listbox.itemconfig(tk.END,{'fg':value['color']})                 



    
    def _clear(self):
        # Deleting plots (if existing)
        items = self._fit_pars_listbox.get(0,tk.END)
        if len(items) != 0:
            print('Cleaning ')
            for key,item in self._fit_funcs_dict.items():
                print('Cleaning parameter',key)
                self._controller._ax.lines.remove(self._fit_funcs_dict[key]['plots'])
        if 'plot' in self._total_fit_func_dict.keys():
            self._controller._ax.lines.remove(self._total_fit_func_dict['plot'])
        self._controller._canvas.draw()
        self._controller._ax1.clear()
        self._controller._ax1bis.clear()
        self._controller._ax2.clear()
        self._controller._ax2bis.clear()
        self._controller._canvas2.draw()

        # Deleting boxes
        self._fit_func_listbox.delete(0,tk.END)
        self._fit_pars_listbox.delete(0,tk.END)
        self._controller._fit_info_box.delete(0,tk.END)
        
        # Resetting variables
        self._first_fit = True
        self._index = 1
        self._fit_funcs_dict = {}
        self._total_fit_func_dict = {}



    def _save_fit(self):
        fit_dir = os.path.join(self._controller._controller._output_dir.get(),\
            'analysis',self._controller._controller._obs_id,'fits')
        os.system(f'mkdir {fit_dir}')
        save_modelresult(self._fit_result,\
            os.path.join(fit_dir,self._output_name.get()+'.sav'))

    def _load_fit(self):
        pass

    def _reset_index(self):
        items = self._fit_func_listbox.get(0,tk.END)

        # Reading and storing old info
        old_func_info = []
        old_items = []
        for i in range(len(items)):
            item = items[i]
            old_items += [item.split(')')[1].strip()]
            old_index = int(item.split(')')[0].strip())
            old_func_info += [self._fit_funcs_dict[old_index]]
        
        # Cleaning listbox and self._fit_funcs_dict
        self._fit_func_listbox.delete(0,tk.END)
        self._fit_funcs_dict = {}
        
        # Re-writing old items with new index
        for i in range(len(items)):
            self._fit_func_listbox.insert(tk.END,str(i+1)+') '+old_items[i])
            self._fit_func_listbox.itemconfig(i,{'fg':old_func_info[i]['color']})
            self._fit_funcs_dict[i+1] = old_func_info[i]

        # Resetting index
        print('len(items) + 1',len(items) + 1)
        self._index = len(items) + 1

    def _close(self):
        self._parent.destroy()

    def _activate_draw_func(self):
        items = self._fit_func_listbox.get(0,tk.END)
        # There must be at least a function to draw
        #print(len(items),self._controller._to_plot)
        if len(items) != 0 and not self._controller._to_plot is None:

            print('Draw function activated')

            # Connecting the canvas
            if not self._canvas_connected:
                self._connect()

            sel = self._fit_func_listbox.curselection()
            # If nothing is selected, select the first one
            if not sel:
                sel = (0,)
            # In case of (accidental) multiple selection, it will be 
            # considered always the first one
            self._sel_index = int(sel[0])

    def _hold_func(self):
        if self._canvas_connected:
            self._disconnect()

        
    def _plot_func(self,reset=False):

        x = self._controller._to_plot.freq
        x = x[x>0]
        if not x is None:
            counter = 0
            psum = np.zeros(len(x))
            for key,value in self._fit_funcs_dict.items():

                if 'par_value' in value.keys():

                    # This remove previous plot of the function
                    if 'plots' in value.keys():
                        self._controller._ax.lines.remove(value['plots'])

                    # Computing function plot
                    col = value['color']
                    pars = tuple(value['par_value'])
                    print('plotting function',value['name'])
                    print('<--------->>>>')
                    print(pars)
                    func = self._func_list[value['name']](*pars)
                    print(func)
                    y = func(x)

                    # Plotting
                    ylim = self._controller._ax.set_ylim()
                    xlim = self._controller._ax.set_xlim()
                    print('xy flag status',self._controller._xy_flag.get())
                    if not self._controller._xy_flag.get():
                        lor, = self._controller._ax.plot(x,y,'--',color = col)
                        psum += y
                    else:
                        lor, = self._controller._ax.plot(x,y*x,'--',color = col)
                        psum += y*x
                    self._controller._ax.set_ylim(ylim)
                    self._controller._ax.set_xlim(xlim)
                    self._fit_funcs_dict[key]['plots'] = lor
                    counter +=1

            # Replotting full function
            if 'plot' in self._total_fit_func_dict.keys():
                self._controller._ax.lines.remove(self._total_fit_func_dict['plot'])
            if counter > 1:
                allp, = self._controller._ax.plot(x,psum,'r-')
                self._total_fit_func_dict['plot'] = allp

            self._controller._canvas.draw()
            self._print_par_value()

    def _connect(self):
        if not self._canvas_connected:
            print('Canvas is connected')
            self._cidclick = self._controller._canvas.mpl_connect('button_press_event',self._on_click)
            self._cidscroll = self._controller._canvas.mpl_connect('scroll_event',self._on_roll)
            self._canvas_connected = True

    def _disconnect(self):
        if self._canvas_connected:
            print('Canvas is diconnected')
            self._controller._canvas.mpl_disconnect(self._cidclick)
            self._controller._canvas.mpl_disconnect(self._cidscroll)
            self._canvas_connected = False

    def _on_click(self,event):
        if not event.dblclick:
            # Left click or right lick
            if event.button == 1 or event.button == 3:
                # Position of the cursor
                self._xpos = event.xdata
                self._ypos = event.ydata

                if (self._xpos != None) and (self._ypos != None):

                    name = self._fit_funcs_dict[self._sel_index+1]['name']

                    if name=='fmax_lorentzian':
                        # Choosing standard value if not existing
                        if not 'par_value' in self._fit_funcs_dict[self._sel_index+1].keys():
                            q = 10
                        else:
                            q = self._fit_funcs_dict[self._sel_index+1]['par_value'][1]
                        delta = self._xpos/np.sqrt(1+4*q**2)
                        amplitude = delta*(np.pi/2 + np.arctan(2*q))*self._ypos
                        fmax = np.sqrt(self._xpos**2+delta**2)

                        if self._controller._xy_flag.get():
                            self._fit_funcs_dict[self._sel_index+1]['par_value'] = \
                                [amplitude/self._xpos,q,fmax]
                        else:
                            self._fit_funcs_dict[self._sel_index+1]['par_value'] = \
                                [amplitude,q,fmax]                       
                        self._fit_funcs_dict[self._sel_index+1]['par_status'] = \
                            [False,False,False]    
                        self._fit_funcs_dict[self._sel_index+1]['par_name'] = \
                            self._func_list[name].param_names   

                          

                    # Plotting                                   
                    self._plot_func()

            if event.button == 2:
                self._disconnect()
                self._v.set(0)

    def _on_roll(self,event):
        name = self._fit_funcs_dict[self._sel_index+1]['name']

        if name=='fmax_lorentzian':
            q = self._fit_funcs_dict[self._sel_index+1]['par_value'][1]
            if q > 1:
                step = 1
            else:
                step = 0.1
            if event.button == 'up':
                q -= step
                if q <= 0: q = 0.
                self._fit_funcs_dict[self._sel_index+1]['par_value'][1] = q
                self._plot_func()
            elif event.button == 'down':              
                q += step
                self._fit_funcs_dict[self._sel_index+1]['par_value'][1] = q
                self._plot_func()



    def _clickFit(self):  

        if self._first_fit:
            self._controller._new_child_window(PlotFitWindow)

        self._hold_func()

        freq = self._controller._to_plot.freq
        y = self._controller._to_plot.power
        yerr = self._controller._to_plot.spower
        x = freq[freq>0]
        y = y[freq>0]
        yerr = yerr[freq>0]

        self._fit_mask = (x> self._start_fit_freq.get()) & (x<= self._stop_fit_freq.get())
        self._build_model()

        fit = fitting.LevMarLSQFitter()
        #fit = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='confidence')
        print('='*80)
        print('This is the model right before fitting')
        print(self._model)
        print('='*80)
        self._fit_result = fit(self._model,
            x[self._fit_mask],y[self._fit_mask],err=yerr[self._fit_mask])
        print(fit.fit_info['message'])
        print('Model AFTER fitting')
        print(self._fit_result)

        self._update_fit_funcs()
        self._plot_func()
        if self._first_fit:
            self._plot_fit()
            self._first_fit = False
        else:
            self._update_fit_plot()
        self._update_info()

    def _update_fit_funcs(self):
        index = 0
        for key, value in self._fit_funcs_dict.items():
            print('Updating function',key)
            if 'plots' in value.keys():
                par_names = value['par_name']
                n_pars = len(par_names)

                for i in range(n_pars):
                    print('Updating parameter',i)
                    print(self._fit_funcs_dict[key]['par_value'][i],'---->',self._fit_result.parameters[index])
                    self._fit_funcs_dict[key]['par_value'][i] = \
                        self._fit_result.parameters[index]
                    index+=1

        print('$'*80)
        print('Updated fit funcs')
        print(self._fit_funcs_dict)

    def _update_fit_plot(self):
        freq = self._controller._to_plot.freq
        y = self._controller._to_plot.power
        yerr = self._controller._to_plot.spower
        x = freq[freq>0]
        y = y[freq>0]
        yerr = yerr[freq>0]

        n_free_pars = 0
        for key,value in self._fit_result.fixed.items():
            if value == False: n_free_pars+=1

        self._line1.set_ydata(self._fit_result(x[self._fit_mask])-y[self._fit_mask])
        self._line2.set_ydata((self._fit_result(x[self._fit_mask])-y[self._fit_mask])**2/yerr[self._fit_mask]**2/n_free_pars)    
        self._controller._canvas2.draw()


    def _plot_fit(self):
        freq = self._controller._to_plot.freq
        y = self._controller._to_plot.power
        yerr = self._controller._to_plot.spower
        x = freq[freq>0]
        y = y[freq>0]
        yerr = yerr[freq>0]

        n_free_pars = 0
        for key,value in self._fit_result.fixed.items():
            if value == False: n_free_pars+=1

        self._line1,=self._controller._ax1.plot(
            x[self._fit_mask],\
             (self._fit_result(x[self._fit_mask])-y[self._fit_mask]),'-r'
            )
        self._line2,=self._controller._ax2.plot(
            x[self._fit_mask],\
             (self._fit_result(x[self._fit_mask])-y[self._fit_mask])**2/yerr[self._fit_mask]**2/n_free_pars,'-r'
            )

        # Residuals
        maxr = np.max(abs(self._fit_result(x[self._fit_mask])-y[self._fit_mask]))
        #ax.plot(x[fit_mask],result.best_fit-y,'r')
        self._controller._ax1.set_xscale('log')
        self._controller._ax1.set_ylim([-maxr-maxr/3,maxr+maxr/3])
        self._controller._ax1.grid()
        #self._controller._ax1.set_xlabel('Frequency [ Hz]')
        self._controller._ax1.set_ylabel('Residuals [model-data]',fontsize=12)
        self._controller._ax1.set_title('').set_visible(False)

        self._controller._ax1bis = self._controller._ax1.twinx()
        self._controller._to_plot.plot(ax=self._controller._ax1bis,\
            alpha=0.3,lfont=12,xy=self._controller._xy_flag.get())
        self._controller._ax1bis.set_ylabel('')
        self._controller._ax1bis.grid(False)
        self._controller._ax1bis.tick_params(axis='both',which='both',length=0)
        self._controller._ax1bis.set_yscale('log')  
        self._controller._ax1bis.set_yticklabels([])     
            

        # Contribution to chi2
        self._controller._ax2.set_ylabel('Contribution to $\chi^2$',fontsize=12)
        self._controller._ax2.set_xscale('log')
        self._controller._ax2.set_xlabel('Frequency [ Hz]',fontsize=12)
        self._controller._ax2.grid()
        self._controller._ax2.set_title('').set_visible(False)
        self._controller._ax2.yaxis.set_label_position('left')
        self._controller._ax2.yaxis.tick_right()

        self._controller._ax2bis = self._controller._ax2.twinx()
        self._controller._to_plot.plot(ax=self._controller._ax2bis,\
            alpha=0.3,lfont=12,xy=self._controller._xy_flag.get())
        self._controller._ax2bis.set_ylabel('')
        self._controller._ax2bis.grid(False)
        self._controller._ax2bis.tick_params(axis='both',which='both',length=0)
        self._controller._ax2bis.set_yscale('log')  
        self._controller._ax2bis.set_yticklabels([])     
 
        if self._first_fit: self._controller._canvas2.draw()

    def _build_model(self):
        print('Building model')
        print('-'*70)
        print('Composite model of {} functions'.format(len(self._fit_funcs_dict.keys())))
        first = True
        for key, value in self._fit_funcs_dict.items():
            print('Function n.',key)
            if 'plots' in value.keys():
                print('Function is plotted on the screen')
                par_names = value['par_name']
                par_values = value['par_value']
                par_status = value['par_status']
                func = self._func_list[value['name']]

                # Maybe should I do the same for the parameter values?
                fixed_dict = {}
                bounds_dict = {}
                par_dict = {}
                for par,status,value in zip(par_names,par_status,par_values):
                    fixed_dict[par] = not status
                    bounds_dict[par] = (0,None)
                    par_dict[par] = value
                
                # first tracks the first iteration
                if first:
                    first = False
                    self._model = func(**par_dict,fixed=fixed_dict,bounds=bounds_dict)
                else:
                    self._model += func(**par_dict,fixed=fixed_dict,bounds=bounds_dict)
                print(par_values,fixed_dict)
        print()
        print('Initial fitting MODEL')
        print(self._model)
        print('-'*70)

    def _update_info(self):
        self._controller._report = lmfit.fit_report(self._fit_result).split('\n')
        for line in self._controller._report:
            self._controller._fit_info_box.insert(tk.END,line)
        if not self._first_fit:
            self._controller._fit_info_box.insert(tk.END,'='*70+'\n')

class FitWindow_sherpa:
    '''
    Window called by the fitting tab

    HISTORY
    -------
    2020 12 10, Stefano Rapisarda (Uppsala), creation date
        This was independently created in April, here I cleaned it up
        and I incorporate the window in MakePowerWin (specifically the
        timing tab).
    2020 01 20, Stefano Rapisarda (Uppsala), modifications
        Methods have been modified in order to use sherpa as fitting routine

    NOTES
    -----
    2020 01 20, Stefano Rapisarda (Uppsala)
        The philosophy of this routine is trying to keep it relatively 
        open to different fitting implementations (astropy, sherpa, ml, 
        whatever). For this reason parameters names, values, and other 
        numbers are always stored in a dictionary. Then, the values are 
        fed to one or another fit implementation with an appropriate 
        function. 
        This is the main philosophy, but it has not been achieved yet. 

    TODO:
    - Implement other fitting functions 
    '''

    def __init__(self,parent,controller,window):
        # Controller is the timing tab widget
        # Parent is the top level window
        self._controller = controller
        self._parent = parent
        self._window = window
        self._parent.title = 'Fit window'

        #s = ttk.Style()
        #s.configure('Black.TLabelframe.Label',
        #            font=('times', 16, 'bold'))
        #self._head_style = 'Black.TLabelframe'

        # Initializing variables
        # -------------------------------------------------------------

        # NOTE:
        # self._fit_funcs_dict has an integer as key. Integer is 
        # assigned according to the position of the fitting function in
        # the fit_funcs_listbox (addition time)
        # For each key, and so fitting function, the dict item is 
        # another dictionary name, color, par_names (list), par_values 
        # (list), frozen (list), plots (containing the line plotted in 
        # the canvas).
        # If you want to modify the fitting method in the future, just
        # modify the function the feed this dictionary in the fit
        # implementation
         
        self._fit_funcs_dict = {}

        # total_fit_func does not have any other specific function than 
        # storing the plot of the full function (the sum of different
        # model component)
        self._total_fit_func_dict = {}

        # Controlling variables
        self._index = 1
        self._first_fit = True
        self._canvas_connected = False

        # Listing 1D models
        # Add new models here
        self._func_list = {'fmax_lorentzian':Fmax_lorentzian1D,
                           'f0_lorentzian':F0_lorentzian1D,
                           'constant':Const1D,
                           'gaussian':Gauss1D,
                           'power law':PowLaw1D}
        # -------------------------------------------------------------

        # Main frame
        frame = tk.Frame(self._parent)
        frame.grid(column=0,row=0,padx=5,pady=5)
        self._populate_main_frame(frame)

    def _populate_main_frame(self,frame):

        frame.columnconfigure(0,weight=1)

        # Fit frequency boxes
        freq_frame = ttk.LabelFrame(frame,text='Frequency boundaries',\
            height=50)
        freq_frame.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._populate_freq_frame(freq_frame)

        # Fitting functions options and drawing them on the plot
        fit_func_frame = ttk.LabelFrame(frame,\
            text='Fitting functions')
        fit_func_frame.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        self._populate_fit_func_frame(fit_func_frame)

        # Fitting function parameters
        fit_pars_frame = ttk.LabelFrame(frame,\
            text='Fitting parameters')
        fit_pars_frame.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')
        self._populate_fit_pars_frame(fit_pars_frame)

    def _populate_freq_frame(self,frame):
        '''
        Sets the two buttons to select start and stop frequency for fit
        '''
        self._start_fit_freq = tk.DoubleVar()
        self._start_fit_freq.set(0)
        start_freq_entry = tk.Entry(frame, \
            textvar=self._start_fit_freq, width=10)
        start_freq_entry.grid(column=0,row=0,padx=5,pady=5,sticky='w')

        self._stop_fit_freq = tk.DoubleVar()
        self._stop_fit_freq.set(100)
        stop_freq_entry = tk.Entry(frame, \
            textvar=self._stop_fit_freq, width=10)
        stop_freq_entry.grid(column=2,row=0,padx=5,pady=5,sticky='w')

        dummy1 = tk.Label(frame,text='-')
        dummy1.grid(column=1,row=0,padx=5,pady=5,sticky='w')
        dummy2 = tk.Label(frame,text='[Hz]')
        dummy2.grid(column=3,row=0,padx=5,pady=5,sticky='w')

    def _populate_fit_func_frame(self,frame):

        frame.grid_columnconfigure(0,weight=1)

        # Left box (fitting function list Box)
        # -------------------------------------------------------------
        left_frame = tk.Frame(frame)
        left_frame.grid(column=0,row=0,sticky='nswe')

        self._fit_func_listbox = tk.Listbox(left_frame,\
            selectmode='multiple',height=18)
        self._fit_func_listbox.grid(column=0,row=1,\
            padx=5,pady=5,sticky='nsew')
        # -------------------------------------------------------------

        # Right box 
        # (function menu, buttons, save panel)
        # -------------------------------------------------------------
        right_frame = tk.Frame(frame)
        right_frame.grid(column=1,row=0,sticky='nswe')
        
        # Fitting function menu
        self._fit_func = tk.StringVar()
        fit_funcs = tuple(['']+[i for i in self._func_list.keys()])
        fit_func_box = ttk.OptionMenu(right_frame,\
            self._fit_func,*fit_funcs)
        fit_func_box.grid(column=0,row=0, columnspan=2,\
            sticky='we',padx=5,pady=5)

        # Draw and hold radio button
        radio_frame = tk.Frame(right_frame)
        radio_frame.grid(column=0,row=1,columnspan=2,sticky='nsew')

        self._v = tk.IntVar()
        draw_radio = tk.Radiobutton(radio_frame, text='DRAW',\
            variable = self._v, value = 1,\
            command=self._activate_draw_func)
        draw_radio.grid(column=0,row=0,padx=5,pady=2,sticky='nsew')
        hold_radio = tk.Radiobutton(radio_frame, text='HOLD',\
            variable = self._v, value = 0, \
            command=self._hold_func)
        hold_radio.grid(column=1,row=0,padx=5,pady=2,sticky='ensw')
        hold_radio.select()  

        # Add and delete buttons
        add_button = ttk.Button(right_frame, text='ADD', \
            command=self._clickAdd)
        add_button.grid(column=0,row=2,padx=5,pady=2,sticky='nswe')
        del_button = ttk.Button(right_frame, text='DEL', \
            command=self._clickDel)
        del_button.grid(column=1,row=2,padx=5,pady=2,sticky='nswe') 

        # Fit and clear button
        print('Fit and clear buttons')
        fit_button = ttk.Button(right_frame, text='FIT', \
            command=self._clickFit)
        fit_button.grid(column=0,row=3,padx=5,pady=2,sticky='nswe')        
        clear_button = ttk.Button(right_frame, text='CLEAR', \
            command=self._clear)
        clear_button.grid(column=1,row=3,padx=5,pady=2,sticky='nsew')   

        # Compute errors
        error_frame = ttk.LabelFrame(right_frame,text='Compute error')
        error_frame.grid(column=0,row=4,columnspan=2,padx=5,pady=5,sticky='nswe')
        error_frame.columnconfigure(0,weight=1)
        error_frame.columnconfigure(1,weight=1)

        error_button = ttk.Button(error_frame,text='ERRORS',\
            command=self._comp_errors)
        error_button.grid(column=0,row=0,padx=5,pady=2,sticky='nsew') 
        sigma_frame = tk.Frame(error_frame)
        sigma_frame.grid(column=1,row=0,sticky='nswe')
        #sigma_letter = tk.Label(sigma_frame,text=u'\u03c3')
        sigma_letter = tk.Label(sigma_frame,text='sigma')
        sigma_letter.grid(column=0,row=0,sticky='nswe')
        self._error_sigma = tk.DoubleVar()
        self._error_sigma.set(1.0)
        sigma_entry = tk.Entry(sigma_frame,textvariable=self._error_sigma,
            width=8)
        sigma_entry.grid(column=1,row=0,sticky='nswe')

        # Save and load buttons
        print('save and load buttons')
        save_frame = ttk.LabelFrame(right_frame,text='Fit output name')
        save_frame.grid(column=0,row=5,columnspan=2,padx=5,pady=5,sticky='nswe')

        self._output_name = tk.StringVar()
        name_entry = tk.Entry(save_frame,textvariable=self._output_name)
        name_entry.grid(column=0,row=0,padx=5, pady=5, sticky='nswe')
        save_button = ttk.Button(save_frame, text='SAVE', \
            command=self._save_fit)
        save_button.grid(column=0,row=1,padx=5,sticky='nswe')        
        load_button = ttk.Button(save_frame, text='LOAD', \
            command=self._load_fit)
        load_button.grid(column=0,row=2,padx=5,sticky='nswe')   
        # -------------------------------------------------------------

    def _clickAdd(self):
        '''
        Add a fitting function to the dictionary of fitting 
        functions and update the fitting function box
        If it is the first call, self._index = 1
        
        Called when the button ADD is clicked.
        '''

        # Choose random color
        colors = plt_color(hex=True)
        # Remove from the list of random color existing colors
        for key,item in self._fit_funcs_dict.items():
            colors.remove(item['color'])
        # Selecting a random color
        col = random.choice(colors)

        # Initializing self._fit_funcs_dict
        self._fit_funcs_dict[self._index] = {'name':self._fit_func.get(),'color':col}

        # Populating the fitting function box
        self._fit_func_listbox.insert(tk.END,\
            str(self._index)+') '+self._fit_func.get())

        # Assigning a color to text corresponding to the just 
        # added function
        self._fit_func_listbox.itemconfig(self._index-1,{'fg':col})
        
        # Incrementing fit function index
        self._index += 1
        
        # Disconnect the canvas when adding a new function
        self._v.set(0)
        if self._canvas_connected:
            self._disconnect()

    def _clickDel(self):
        '''
        Delete the selected fit function from the fitting function 
        listbox and the self._fit_func_dict dictionary
        
        Called when the button DEL is clicked
        '''

        # If the dictionary is already empty, no need to delete items
        if self._fit_funcs_dict != {}:

            # Selected items can be more than 1
            sel = self._fit_func_listbox.curselection()
    
            # Removing selected items from listbox, plot, and dictionary
            # Selection index MUST be reversed because self._fit_func_listbox 
            # index changes every time you remove an item
            for index in sel[::-1]:

                self._fit_func_listbox.delete(index) 

                # Removing selected items from plot
                if 'plots' in self._fit_funcs_dict[index+1].keys():
                    self._controller._ax.lines.remove(self._fit_funcs_dict[index+1]['plots'])
        
                # Removing selected items from self._fit_func_dict
                del self._fit_funcs_dict[index+1]

            # Resetting index of the remaining items
            items = self._fit_func_listbox.get(0,tk.END)
            if len(items) != 0:
                self._reset_index()

                # Replotting remaining functions and corresponding 
                # parameters
                if not self._controller._to_plot is None:
                    self._plot_func()
                    self._print_par_value()

            # Disconnect the canvas when deleting a new function
            self._v.set(0)
            if self._canvas_connected:
                self._disconnect()

        else:

            # Just to be sure
            self._index = 1


    def _populate_fit_pars_frame(self,frame):

        frame.columnconfigure(0,weight=1)
        
        # Top (parameter) box
        # -------------------------------------------------------------
        top_frame = tk.Frame(frame)
        top_frame.grid(column=0,row=0,sticky='nswe')
        top_frame.columnconfigure(0,weight=1)

        self._fit_pars_listbox = tk.Listbox(top_frame, selectmode='multiple')
        self._fit_pars_listbox .grid(column=0,row=0,padx=5,pady=5,\
            sticky='nsew')
        # -------------------------------------------------------------

        # Bottom (button) box
        # -------------------------------------------------------------
        bot_frame = tk.Frame(frame)
        bot_frame.grid(column=0,row=1,sticky='nswe')

        self._par_val = tk.StringVar()
        reset_entry = tk.Entry(bot_frame, textvariable=self._par_val, \
            width=10)
        reset_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nsew')  

        reset_button = ttk.Button(bot_frame, text='RESET', \
            command=lambda: self._reset_par_value())
        reset_button.grid(column=1,row=0,padx=5,pady=5,sticky='nsew')        


        self._vII = tk.IntVar()
        freeze_radio = tk.Radiobutton(bot_frame, text='FREEZE',\
            variable = self._vII, value = 0, \
            command=lambda: self._reset_par_value(freeze=True))
        freeze_radio.grid(column=2,row=0,padx=5,pady=5,sticky='senw')
        free_radio = tk.Radiobutton(bot_frame, text='FREE',
            variable = self._vII, value = 1, \
            command=lambda: self._reset_par_value(freeze=False))
        free_radio.grid(column=3,row=0,padx=7,pady=7,sticky='senw')
        free_radio.select() 
        # ------------------------------------------------------------- 

    def _reset_par_value(self,freeze=False):
        '''
        Called when either the freeze/free radio button is selected or
        the RESET button is clicked
        '''

        # Getting the value of the selected function and parameter
        sel = self._fit_pars_listbox.curselection()
        items = self._fit_pars_listbox.get(0,tk.END)
        for index in sel[::-1]:
            item = items[index]
            # Lines in the par box are in the format:
            # key) par_name = value (status)
            key = int(item.split(')')[0])
            par_name = item.split()[1]
            pars = self._fit_funcs_dict[key]['par_names']

            for i in range(len(pars)):
                if par_name == self._fit_funcs_dict[key]['par_names'][i]:

                    if self._par_val.get() == '':
                        par_val = self._fit_funcs_dict[key]['par_values'][i]
                    else:
                        par_val = float(self._par_val.get())

                    self._fit_funcs_dict[key]['par_values'][i] = par_val  
                    if freeze:
                        self._fit_funcs_dict[key]['frozen'][i] = True
                    else:
                        self._fit_funcs_dict[key]['frozen'][i] = False
        
        self._print_par_value()
        self._plot_func()

    def _print_par_value(self):
        '''
        Prints model parameters stored in self._fit_funcs_dict in the 
        parameter listbox
        '''

        # Cleaning par listbox
        self._fit_pars_listbox.delete(0,tk.END)

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
                    self._fit_pars_listbox.insert(tk.END,line) 
                    self._fit_pars_listbox.itemconfig(tk.END,{'fg':value['color']})                 

    def _clear(self):
        '''
        Clear all the model related objects:
        - the dictionary self._fit_funcs_dict;
        - the listbox self._fit_func_listbox;
        - the listbox self._fit_pars_listbox;
        - plot lines

        Called when the button CLEAR is clicked
        '''

        # Deleting plots (if existing)
        if self._fit_funcs_dict != {}:
            for key,item in self._fit_funcs_dict.items():
                self._controller._ax.lines.remove(self._fit_funcs_dict[key]['plots'])
        if 'plot' in self._total_fit_func_dict.keys():
            self._controller._ax.lines.remove(self._total_fit_func_dict['plot'])

        self._controller._canvas.draw()
        self._controller._ax1.clear()
        self._controller._ax1bis.clear()
        self._controller._ax2.clear()
        self._controller._ax2bis.clear()
        self._controller._canvas2.draw()

        # Deleting boxes
        self._fit_func_listbox.delete(0,tk.END)
        self._fit_pars_listbox.delete(0,tk.END)
        self._controller._fit_info_box.delete(0,tk.END)
        
        # Resetting variables
        self._first_fit = True
        self._index = 1
        self._fit_funcs_dict = {}
        self._total_fit_func_dict = {}

    def _save_fit(self):
        fit_dir = os.path.join(self._controller._output_dir2.get(),\
            self._window._obs_id,'fits')
        os.system(f'mkdir {fit_dir}')
        output_file_name = os.path.join(fit_dir,self._output_name.get())

        # Saving fit plots  
        self._controller._chi_fig.savefig(output_file_name+'_chi2.jpeg', dpi=300)
        
        self._controller._ax.legend(title='Model comp.')
        self._controller._fig.savefig(output_file_name+'_fit.jpeg', dpi=300)

        # Saving model result dictionary
        result_dict=make_sherpa_result_dict(self._fit_result)
        del result_dict['parnames']
        del result_dict['parvals']
        fit_dict=make_sherpa_result_dict(self._fit)
        estmethod_dict=make_sherpa_result_dict(self._fit.estmethod)
        result_dict['model']=fit_dict['model']
        result_dict['estmethod']=fit_dict['estmethod']
        result_dict['sigma_error']=estmethod_dict['sigma']

        with open(output_file_name+'_fit_stat.pkl','wb') as outfile:
            pickle.dump(result_dict,outfile)

        # Saving model pars dictionary
        with open(output_file_name+'_fit_pars.pkl','wb') as outfile:
            pickle.dump(self._fit_funcs_dict,outfile) 

        # Make pdf page        
        print_fit_results(result_dict,self._fit_funcs_dict,
                        output_file_name+'_fit.jpeg',
                        output_file_name+'_chi2.jpeg',
                        output_file_name+'_fit_results.pdf')


    def _load_fit(self):
        # To implement
        pass

    def _reset_index(self):
        '''
        Resets the index of the fitting function according to the 
        number of fitting functions in the fitting function listbox
        '''

        items = self._fit_func_listbox.get(0,tk.END)

        # Reading and storing old info
        old_func_info = []
        old_items = []
        for i in range(len(items)):
            item = items[i]
            old_items += [item.split(')')[1].strip()]
            old_index = int(item.split(')')[0].strip())
            old_func_info += [self._fit_funcs_dict[old_index]]
        
        # Cleaning listbox and self._fit_funcs_dict
        self._fit_func_listbox.delete(0,tk.END)
        self._fit_funcs_dict = {}
        
        # Re-writing old items with new index
        for i in range(len(items)):
            self._fit_func_listbox.insert(tk.END,str(i+1)+') '+old_items[i])
            self._fit_func_listbox.itemconfig(i,{'fg':old_func_info[i]['color']})
            self._fit_funcs_dict[i+1] = old_func_info[i]

        # Resetting index
        self._index = len(items) + 1

    def _close(self):
        self._parent.destroy()

    def _activate_draw_func(self):
        # There must be at least a function to draw
        items = self._fit_func_listbox.get(0,tk.END)
        if len(items) != 0 and not self._controller._to_plot is None:

            print('Draw function activated')

            # Connecting the canvas
            if not self._canvas_connected:
                self._connect()

            sel = self._fit_func_listbox.curselection()
            # If nothing is selected, select the first one
            if not sel:
                sel = (0,)
            # In case of (accidental) multiple selection, it will be 
            # considered always the first one
            self._sel_index = int(sel[0])

    def _hold_func(self):
        self._v.set(0)
        if self._canvas_connected:
            self._disconnect()

        
    def _plot_func(self,reset=False):
        '''
        Plots all the fitting functions stored in self._fit_funcs_dict
        and their sum. Models will be plottes on the entire x axis,
        frequency limits will not affect the plot.
        '''

        # Getting the current x axis
        x = self._controller._to_plot.freq
        x = x[x>0]

        if (not x is None) and (self._fit_funcs_dict != {}):
            counter = 0
            # Initializing total function
            psum = np.zeros(len(x))
            for key,value in self._fit_funcs_dict.items():

                if 'par_values' in value.keys():

                    # This remove previous plot of the function
                    if 'plots' in value.keys():
                        self._controller._ax.lines.remove(value['plots'])

                    # Computing function plot
                    col = value['color']
                    pars = value['par_values']
                    func_name = self._func_list[value['name']]
                    func = init_sherpa_model(func_name,parvals=pars)
                    y = func(x)

                    # Plotting
                    ylim = self._controller._ax.set_ylim()
                    xlim = self._controller._ax.set_xlim()
                    if not self._controller._xy_flag.get():
                        line, = self._controller._ax.plot(x,y,'--',\
                            color = col,label=str(key))
                        psum += y
                    else:
                        line, = self._controller._ax.plot(x,y*x,'--',\
                            color = col,label=str(key))
                        psum += y*x
                    self._controller._ax.set_ylim(ylim)
                    self._controller._ax.set_xlim(xlim)
                    self._fit_funcs_dict[key]['plots'] = line
                    counter +=1

            # Plotting the full function
            if 'plot' in self._total_fit_func_dict.keys():
                self._controller._ax.lines.remove(self._total_fit_func_dict['plot'])
            if counter > 1:
                allp, = self._controller._ax.plot(x,psum,'r-')
                self._total_fit_func_dict['plot'] = allp

            self._controller._canvas.draw()

            # Print corresponding parameter values
            self._print_par_value()

    def _connect(self):
        if not self._canvas_connected:
            print('Canvas is connected')
            self._cidclick = self._controller._canvas.mpl_connect('button_press_event',self._on_click)
            self._cidscroll = self._controller._canvas.mpl_connect('scroll_event',self._on_roll)
            self._canvas_connected = True

    def _disconnect(self):
        if self._canvas_connected:
            print('Canvas is diconnected')
            self._controller._canvas.mpl_disconnect(self._cidclick)
            self._controller._canvas.mpl_disconnect(self._cidscroll)
            self._canvas_connected = False

    def _on_click(self,event):
        '''


        Called when canvas is connected and mouse click inside the plot
        '''

        if not event.dblclick:
            # Left click or right lick
            if event.button == 1 or event.button == 3:
                # Position of the cursor
                self._xpos = event.xdata
                self._ypos = event.ydata

                if (self._xpos != None) and (self._ypos != None):

                    name = self._fit_funcs_dict[self._sel_index+1]['name']

                    if name=='fmax_lorentzian':
                        # Choosing standard value if not existing
                        if not 'par_values' in self._fit_funcs_dict[self._sel_index+1].keys():
                            q = 10
                            status = [False,False,False]
                        else:
                            q = self._fit_funcs_dict[self._sel_index+1]['par_values'][1]
                            status = self._fit_funcs_dict[self._sel_index+1]['frozen']
                        #delta = self._xpos/np.sqrt(1+4*q**2)
                        delta = self._xpos/2/q
                        amplitude = delta*(np.pi/2 + np.arctan(2*q))*self._ypos
                        fmax = np.sqrt(self._xpos**2+delta**2)

                        if self._controller._xy_flag.get():
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [amplitude/self._xpos,q,fmax]
                        else:
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [amplitude,q,fmax]                       
                        self._fit_funcs_dict[self._sel_index+1]['frozen'] = status    
                        self._fit_funcs_dict[self._sel_index+1]['par_names'] = \
                            [par.name for par in self._func_list[name]().pars]    

                    elif name=='constant': 

                        if not 'par_values' in self._fit_funcs_dict[self._sel_index+1].keys():
                            status = [False]
                        else:
                            status = self._fit_funcs_dict[self._sel_index+1]['frozen']

                        amplitude = self._ypos
                        print('Plotting const amplitude',amplitude)
                        self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [amplitude]                       
                        self._fit_funcs_dict[self._sel_index+1]['frozen'] = status   
                        self._fit_funcs_dict[self._sel_index+1]['par_names'] = \
                            [par.name for par in self._func_list[name]().pars] 

                    elif name=='gaussian':

                        if not 'par_values' in self._fit_funcs_dict[self._sel_index+1].keys():
                            fwhm = 1
                            status = [False,False,False]
                        else:
                            fwhm = self._fit_funcs_dict[self._sel_index+1]['par_values'][0]
                            status = self._fit_funcs_dict[self._sel_index+1]['frozen']

                        amplitude = self._ypos
                        pos = self._xpos

                        if self._controller._xy_flag.get():
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [fwhm,pos,amplitude/self._xpos]
                        else:
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [fwhm,pos,amplitude]                       
                        self._fit_funcs_dict[self._sel_index+1]['frozen'] = status   
                        self._fit_funcs_dict[self._sel_index+1]['par_names'] = \
                            [par.name for par in self._func_list[name]().pars]    

                    elif name=='power law':

                        if not 'par_values' in self._fit_funcs_dict[self._sel_index+1].keys():
                            gamma = 1
                            status = [False,True,False]
                        else:
                            gamma = self._fit_funcs_dict[self._sel_index+1]['par_values'][0]
                            status = self._fit_funcs_dict[self._sel_index+1]['frozen']

                        amplitude = self._ypos
                        ref = self._xpos

                        if self._controller._xy_flag.get():
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [gamma,ref,amplitude/self._xpos]
                        else:
                            self._fit_funcs_dict[self._sel_index+1]['par_values'] = \
                                [gamma,ref,amplitude]                       
                        self._fit_funcs_dict[self._sel_index+1]['frozen'] = status   
                        self._fit_funcs_dict[self._sel_index+1]['par_names'] = \
                            [par.name for par in self._func_list[name]().pars] 


                    # Plotting                                   
                    self._plot_func()

            if event.button == 2:
                self._disconnect()
                self._v.set(0)

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
                self._plot_func()
            elif event.button == 'down':              
                q += step
                self._fit_funcs_dict[self._sel_index+1]['par_values'][1] = q
                self._plot_func()

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
                self._plot_func()
            elif event.button == 'down':              
                fwhm += step
                self._fit_funcs_dict[self._sel_index+1]['par_values'][0] = fwhm
                self._plot_func()            

        elif name=='power law':
            gamma = self._fit_funcs_dict[self._sel_index+1]['par_values'][0]

            step = 0.1
            if event.button == 'up':
                gamma -= step
                self._fit_funcs_dict[self._sel_index+1]['par_values'][0] = gamma
                self._plot_func()
            elif event.button == 'down':              
                gamma += step
                self._fit_funcs_dict[self._sel_index+1]['par_values'][0] = gamma
                self._plot_func()  

            
    def _clickFit(self):  

        if self._first_fit:
            print('Clicking fit')
            self._controller._new_child_window(PlotFitWindow)
            print('After window')

        self._hold_func()

        # Preparing data
        # -------------------------------------------------------------
        freq = self._controller._to_plot.freq
        y = self._controller._to_plot.power
        yerr = self._controller._to_plot.spower
        x = freq[freq>0]
        y = y[freq>0]
        yerr = yerr[freq>0]

        min_x = self._start_fit_freq.get()
        max_x = self._stop_fit_freq.get()

        self._data_to_fit = Data1D('power_spectrum',x,y,staterror=yerr)
        self._data_to_fit.notice(min_x,max_x)
        # -------------------------------------------------------------

        self._build_model()

        # TODO: making stat and method optional
        self._stat = Chi2()
        self._method = LevMar()
        self._fit = Fit(self._data_to_fit,self._model, 
                        stat=self._stat, method=self._method)
        self._fit_result = self._fit.fit()
        # To allow the plot window to fetch fit results
        self._controller._fit_result = self._fit_result

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
        self._plot_func()
        if self._first_fit:
            self._plot_fit()
            self._first_fit = False
        else:
            self._update_fit_plot()
        self._update_info()

    def _comp_errors(self):
        if not self._first_fit:
            # TODO: make this an option in the future
            self._fit.estmethod = Covariance()
            self._fit.estmethod.sigma = self._error_sigma.get()
            self._errors = self._fit.est_errors()
            print(self._errors)

            for key0,item in self._fit_funcs_dict.items():
                errors = [[0.,0.] for j in range(len(item['par_names']))]
                for j,name0 in enumerate(item['par_names']):
                    for i,full_name in enumerate(self._errors.parnames):
                        key = full_name.split('.')[0]
                        name = full_name.split('.')[1]
                        if str(key0) == key and name0 == name:                        
                            plus = self._errors.parmaxes[i]
                            minus = self._errors.parmins[i]
                            if not plus is None:
                                plus = np.round(plus,6)
                            else: 
                                plus = np.NaN
                            if not minus is None:
                                minus = np.round(minus,6)
                            else:
                                minus=np.NaN
                            errors[j] = [plus,minus]
                self._fit_funcs_dict[key0]['errors']=errors

            self._print_par_value()
                

    def _update_fit_funcs(self):
        for key, value in self._fit_funcs_dict.items():
            print('Updating function',key)
            if 'plots' in value.keys():

                self._fit_funcs_dict[key]['par_values']=[]
                for par in self._model.pars:
                    if str(key) == par.fullname.split('.')[0]:
                        self._fit_funcs_dict[key]['par_values'] += [par.val]

        print('Updated fit funcs')
        print(self._fit_funcs_dict)

    def _update_fit_plot(self):

        # NOTE: Using the get_x() method you can extract the masked data
        y_data  = self._data_to_fit.get_y(filter=True)
        y_err   = self._data_to_fit.get_staterror(filter=True)
        y_model = self._data_to_fit.eval_model_to_fit(self._model)

        res  = (y_data-y_model)/y_err
        chi2 = self._stat.calc_chisqr(self._data_to_fit,self._model)

        self._line1.set_ydata(res)
        self._line2.set_ydata(chi2)    
        self._controller._canvas2.draw()


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
        self._controller._ax1.set_xscale('log')
        self._controller._ax1.set_ylim([-maxr-maxr/3,maxr+maxr/3])
        self._controller._ax1.grid()
        self._controller._ax1.set_ylabel('Res. [(model-data)/err]',fontsize=12)
        self._controller._ax1.set_title('').set_visible(False)

        self._controller._ax1bis = self._controller._ax1.twinx()
        self._controller._to_plot.plot(ax=self._controller._ax1bis,\
            alpha=0.3,lfont=12,xy=self._controller._xy_flag.get())
        self._controller._ax1bis.set_ylabel('')
        self._controller._ax1bis.grid(False)
        self._controller._ax1bis.tick_params(axis='both',which='both',length=0)
        self._controller._ax1bis.set_yscale('log')  
        self._controller._ax1bis.set_yticklabels([])     
            
        # Contribution to chi2
        self._controller._ax2.set_ylabel('$\chi^2$',fontsize=12)
        self._controller._ax2.set_xscale('log')
        self._controller._ax2.set_xlabel('Frequency [ Hz]',fontsize=12)
        self._controller._ax2.grid()
        self._controller._ax2.set_title('').set_visible(False)
        self._controller._ax2.yaxis.set_label_position('left')
        self._controller._ax2.yaxis.tick_right()

        self._controller._ax2bis = self._controller._ax2.twinx()
        self._controller._to_plot.plot(ax=self._controller._ax2bis,\
            alpha=0.3,lfont=12,xy=self._controller._xy_flag.get())
        self._controller._ax2bis.set_ylabel('')
        self._controller._ax2bis.grid(False)
        self._controller._ax2bis.tick_params(axis='both',which='both',length=0)
        self._controller._ax2bis.set_yscale('log')  
        self._controller._ax2bis.set_yticklabels([])     
 
        if self._first_fit: 
            self._controller._canvas2.draw()
        self._controller._canvas2.mpl_connect(\
            'motion_notify_event',self._controller._update_cursor)

    def _build_model(self):
        print('Building model')
        first = True
        for key, value in self._fit_funcs_dict.items():
            print('Building model component n.',key)
            if 'plots' in value.keys():
                
                # Initializing model accortding to values stored in
                # self._fit_funcs_dict
                func=init_sherpa_model(sherpa_model=self._func_list[value['name']],
                    name=str(key),
                    parvals=value['par_values'],
                    frozen=value['frozen'])
                
                # first tracks the first iteration
                if first:
                    first = False        
                    self._model = func
                else:
                    self._model += func

    def _update_info(self):
        self._controller._report = self._fit_result.__str__().split('\n')
        for line in self._controller._report:
            self._controller._fit_info_box.insert(tk.END,line)
        if not self._first_fit:
            self._controller._fit_info_box.insert(tk.END,'='*70+'\n')


class PlotFitWindow:
    def __init__(self,parent,controller):
        self._controller = controller
        self._parent = parent

        s = ttk.Style()
        s.configure('Black.TLabelframe.Label',
                    font=('times', 16, 'bold'))
        self._head_style = 'Black.TLabelframe'

        # Initializaing plot frame
        # -------------------------------------------------------------
        plot_frame = ttk.LabelFrame(self._parent,text='Residual',
            style=self._head_style)
        plot_frame.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        fig = Figure(figsize=(6.5,5),dpi=100)
        #gs = fig.add_gridspec(2,1)
        #gs.tight_layout(fig)
        self._controller._ax1 = fig.add_subplot(211)
        self._controller._ax2 = fig.add_subplot(212,sharex=self._controller._ax1)  
        fig.tight_layout(w_pad=1,rect=[0.1,0.05,1.0,1.])
        fig.align_ylabels([self._controller._ax1,self._controller._ax2])

        self._controller._ax1.get_shared_x_axes().\
            join(self._controller._ax1, self._controller._ax2)

        self._controller._canvas2 = FigureCanvasTkAgg(fig,\
            master = plot_frame)
        self._controller._canvas2.draw()
        self._controller._canvas2.get_tk_widget().\
            grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._controller._canvas2.draw()
        self._controller._canvas2.mpl_connect(\
            'motion_notify_event',self._update_cursor)
        self._controller._chi_fig = fig

        coor_frame = tk.Frame(plot_frame)
        coor_frame.grid(column=0,row=1,pady=5,sticky='nswe')
        coor_frame.grid_columnconfigure(0,weight=1)
        coor_frame.grid_columnconfigure(1,weight=1)
        coor_frame.grid_columnconfigure(2,weight=1)
        coor_frame.grid_columnconfigure(3,weight=1)
        labelx = tk.Label(coor_frame,text='x coor: ')
        labelx.grid(column=0,row=0,pady=5,padx=5,sticky='nswe')
        self._x_pos = tk.Label(coor_frame,text=' ')
        self._x_pos.grid(column=1,row=0,pady=5,padx=5,sticky='nswe')
        labely = tk.Label(coor_frame,text='y coor: ')
        labely.grid(column=2,row=0,pady=5,padx=5,sticky='nswe')
        self._y_pos = tk.Label(coor_frame,text=' ')
        self._y_pos.grid(column=3,row=0,pady=5,padx=5,sticky='nswe')
        # -------------------------------------------------------------

        # F-test frame
        # -------------------------------------------------------------
        entry_width = 8
        entry_width2 = 4
        button_width = 5

        ftest_frame = ttk.LabelFrame(self._parent,text='F-test',
            style=self._head_style)
        ftest_frame.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')

        self._chi2_1 = tk.DoubleVar()
        self._dof_1  = tk.DoubleVar()
        chi2_1_label = tk.Label(ftest_frame,text='chi2/dof (simple)')
        chi2_1_label.grid(column=0,row=0,sticky='nswe')
        chi2_1_entry = tk.Entry(ftest_frame,textvar=self._chi2_1,
            width=entry_width)
        chi2_1_entry.grid(column=1,row=0,sticky='nswe')
        dof_1_entry = tk.Entry(ftest_frame,textvar=self._dof_1,
            width=entry_width2)
        dof_1_entry.grid(column=2,row=0,sticky='nswe')
        get_button_1 = ttk.Button(ftest_frame,text='GET',\
            command=lambda:self._get_chi_dof(1),width=button_width)
        get_button_1.grid(column=3,row=0,sticky='nswe')


        self._chi2_2 = tk.DoubleVar()
        self._dof_2  = tk.DoubleVar()
        chi2_2_label = tk.Label(ftest_frame,text='chi2/dof (complex)')
        chi2_2_label.grid(column=4,row=0,sticky='nsew')
        chi2_2_entry = tk.Entry(ftest_frame,textvar=self._chi2_2,
            width=entry_width)
        chi2_2_entry.grid(column=5,row=0,sticky='nswe')
        dof_2_entry = tk.Entry(ftest_frame,textvar=self._dof_2,
            width=entry_width2)
        dof_2_entry.grid(column=6,row=0,sticky='nswe')
        get_button_2 = ttk.Button(ftest_frame,text='GET',\
            command=lambda:self._get_chi_dof(2),width=button_width)
        get_button_2.grid(column=7,row=0,sticky='nswe')

        stat_frame = tk.Frame(ftest_frame)
        stat_frame.grid(column=0,row=1,columnspan=8,sticky='nsew')
        stat_frame.grid_columnconfigure(0,weight=2)
        stat_frame.grid_columnconfigure(1,weight=2)
        stat_frame.grid_columnconfigure(2,weight=2)
        stat_frame.grid_columnconfigure(3,weight=2)   
        stat_frame.grid_columnconfigure(4,weight=1)  
        labelf = tk.Label(stat_frame,text='chi2 ratio: ')
        labelf.grid(column=0,row=0,pady=5,padx=5,sticky='nswe')
        self._f = tk.Label(stat_frame,text=' ')
        self._f.grid(column=1,row=0,pady=5,padx=5,sticky='nswe')
        labelstat = tk.Label(stat_frame,text='p-value: ')
        labelstat.grid(column=2,row=0,pady=5,padx=5,sticky='nswe')
        self._fstat = tk.Label(stat_frame,text=' ')
        self._fstat.grid(column=3,row=0,pady=5,padx=5,sticky='nswe')

        comp_button = ttk.Button(stat_frame,text='COMP',
            command=self._f_test)
        comp_button.grid(column=4,row=0,pady=5,padx=5,sticky='nswe')
        # -------------------------------------------------------------

        info_frame = ttk.LabelFrame(self._parent,text='Fit output',
            style=self._head_style)
        info_frame.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')
        self._f_test = ''

        self._controller._fit_info_box = tk.Listbox(self._parent, selectmode='multiple')
        self._controller._fit_info_box.grid(column=0,row=2,padx=5,pady=5,sticky='nsew')

    def _update_cursor(self,event):
        self._x_pos.configure(text=str(np.round(event.xdata,4)))
        self._y_pos.configure(text=str(np.round(event.ydata,4)))

    def _f_test(self):
        f = self._chi2_1.get()/self._chi2_2.get()*self._dof_2.get()/self._dof_1.get()
        p = calc_ftest(self._dof_1.get(),self._chi2_1.get(),self._dof_2.get(),self._chi2_2.get())
        self._f.configure(text=str(f))
        self._fstat.configure(text=str(p))

    def _get_chi_dof(self,opt):
        chi2 = self._controller._fit_result.statval
        dof  = self._controller._fit_result.dof
        if opt == 1:
            self._chi2_1.set(chi2)
            self._dof_1.set(dof)
        elif opt == 2:
            self._chi2_2.set(chi2)
            self._dof_2.set(dof)            

if __name__ == '__main__':
    #win = tk.Tk()
    #app=PlotFitWindow(win,win)
    app = MakePowerWin()
    app.mainloop()