from functools import partial

import os

import numpy as np
import pickle

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import sys
sys.path.append('/Volumes/Samsung_T5/kronos')
import kronos as kr
from kronos.functions.rxte_functions import list_modes
from kronos.gui.tabs import *
from kronos.core.power import *
from kronos.fitting.fitting_functions import *
from kronos.functions.my_functions import plt_color

# Fitting data
import lmfit
from lmfit import Model,Parameters
from lmfit.model import save_modelresult,load_modelresult

import random
import uuid

#from ..functions.rxte_functions import list_modes

#from ..scripts.makers import make_power

def my_tk_eval(variable):
    '''
    Evaluate a mathematical expression embedded in a string or a 
    Tkinter variable

    PARAMETERS
    ----------
    variable: string or tkinter variable

    RETURN
    ------
    numerical result of expression or 0 in case of error

    HISTORY
    -------
    2020 04 00, Stefano Rapisarda, Uppsala (Sweden)
        creation date
    '''

    if 'tkinter' in str(type(variable)):
        expression = variable.get()
    else:
        expression = variable
    if not isinstance(expression,str): 
        return 0

    try:
        result = eval(expression)
        return result
    except NameError as e:
        print(e)
        return 0

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

        tk.Tk.__init__(self,*args,**kwargs)   
        self.configure(bg='burlywood3')

        #For testing bindings
        #self._new_window(TestButton)
        #self._test_button['command'] = self._read_boxes

        # Common style for main section labels 
        # s.configure('Red.TLabelframe.Label', font=('courier', 15, 'bold'))
        # s.configure('Red.TLabelframe.Label', foreground ='red')
        # s.configure('Red.TLabelframe.Label', background='blue')
        s = ttk.Style()
        s.configure('Black.TLabelframe.Label',
                    font=('times', 16, 'bold'))
        self._head_style = 'Black.TLabelframe'

        # Frame for obs ID list
        left_frame = tk.Frame(self)
        left_frame.grid(row=0,column=0,padx=5,pady=5,sticky='nswe') 
        self._init_left_frame(left_frame)

        # Frame for settings
        right_frame = tk.Frame(self)
        right_frame.grid(row=0,column=1,padx=5,pady=5,sticky='nswe') 
        self._init_right_frame(right_frame)  

        # Initializing variables
        self._init_var()  

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
        missions = ('NICER','RXTE','Swift','NuStar','HXMT')
        menu = tk.OptionMenu(box1, self._mission, *missions)
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
            self._rxte_called = True
            for child in self._frame2.winfo_children():
                if hasattr(child,'to_disable'):
                    if child.to_disable:
                        child.configure(state='disabled')
            self._new_window(RxteModes)
        else:
            for child in self._frame2.winfo_children():
                if hasattr(child,'to_disable'):
                    if child.to_disable and child['state']=='disabled':
                        child.configure(state='normal')
            if self._rxte_called: 
                self.new.destroy()
                self._rxte_called = False

    def _click_on_obs_id(self,event):
        '''
        This function is (or should be) called every time the user 
        select a single or multiple items inside the Listbox
        self._obs_id_box
        '''

        # Listing pkl files inside a potential reduced product
        # folder
        sel = self._obs_id_box.curselection()
        if len(sel) == 1:
            self._obs_id = self._obs_id_box.get(sel)
            target_dir = os.path.join(self._output_dir.get(),\
                'analysis',self._obs_id)
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

        self._init_timing_tab(timing_tab)
        #self._init_fitting_tab(fitting_tab)
        self._fitting_tab = FittingTab(fitting_tab,self)

        common_frame = ttk.LabelFrame(frame,text='Reduced products directory',
            style=self._head_style)
        common_frame.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        common_frame.grid_columnconfigure(0,weight=1)
        self._output_dir = tk.StringVar()
        out_entry = tk.Entry(common_frame,textvariable=self._output_dir)
        out_entry.grid(column=0,row=0,sticky='nswe')
        out_button = ttk.Button(common_frame,text='SET',command=self._sel_output_dir)
        out_button.grid(column=1,row=0,sticky='nswe')
        load_obs_id_button2 = ttk.Button(common_frame,text='Load IDs.',
            command=lambda: self._load_obs_ids(self._output_dir.get()+'/analysis'))
        load_obs_id_button2.grid(column=2,row=0,sticky='nswe')

    def _init_timing_tab(self,frame):
        '''
        Populate the timing

        There are 4 panels:
            - Fourier analysis parameters;
            - Other seggings (energy range and GTI min dur);
            - Comments;
            - Buttons to trigger function and stop operations. 
        '''
        

        # Main Boxes
        # -------------------------------------------------------------
        label = 'Timing settings (press key after modifying any of '\
                'the fields)'
        upper_box = ttk.LabelFrame(frame,text=label,
                                   style=self._head_style)
        upper_box.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')

        mid_box = ttk.LabelFrame(frame,text='Other settings',
                                 style=self._head_style)
        mid_box.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')  

        mid_box2 = ttk.LabelFrame(frame,text='Comments',
                                  style=self._head_style)
        mid_box2.grid(column=0,row=3,padx=5,pady=5,sticky='nswe')      

        mid_box3 = ttk.LabelFrame(frame,text='Data to reduce',
                                  style=self._head_style)
        mid_box3.grid(column=0,row=4,padx=5,pady=5,sticky='nswe')              

        low_box = tk.Frame(frame)
        low_box.grid(column=0,row=5,padx=5,pady=5,sticky='nswe')  
        # -------------------------------------------------------------

        # Populating Fourier parameters box
        self._init_timing_boxes(upper_box)

        # Populating other settings box
        self._init_other_settings(mid_box)

        # Populating comment box
        self._init_comment_box(mid_box2)

        # Populating directory box
        self._init_dir_box(mid_box3)

        # Populating execution buttons
        self._init_buttons(low_box)                

    def _init_timing_boxes(self,frame):
        '''
        Generate entries and buttons for Fourier parameters

        - 6 fields in a cycle (tres, tseg, fres, nyqf, ntbs, nfbs);
        - a checkbox to fix the number of bins;
        - a listbox to organize selected modes.
        '''

        labels = [['Time resolution [s]','Time Segment [s]'],
            ['Frequency resolution [Hz]','Nyquist frequency [Hz]'],
                            ['Time bins','Frequency bins (>0)']]
        keys = [['tres','tseg'],
                ['fres','nyqf'],
                ['ntbs','nfbs']]

        # In order to create different entries and variables in a 
        # cycle, you need to save such variables in a container.
        # This container is usually a list, in my case I used a 
        # fictionary, in order to access the variables by an identifier
        self._fourier_pars = {}
        self._fix_bins = tk.IntVar()
        self._entries = {}
        for row in range(len(labels)):
            for col in range(len(labels[row])):

                # Container
                box = ttk.LabelFrame(frame,text=labels[row][col])
                box.grid(column=col,row=row,padx=5,pady=5,sticky='nswe')
                
                # StringVar and NOT DoubleVar to allow 1./512 syntax
                var = tk.StringVar()
                
                entry = tk.Entry(box,textvariable=var,width=30)
                entry.grid(column=0,row=0,sticky='nswe')
                entry.bind('<Key>',
                            partial(self._modifying_var,keys[row][col]))
                entry.bind('<Return>',
                            partial(self._update_var,keys[row][col]))
                self._entries[keys[row][col]]=entry
 
                self._fourier_pars[keys[row][col]] = var

        # To fix the bins
        checkbox = tk.Checkbutton(frame,
                                  text='Fix time and frequency bins',
                                  var=self._fix_bins)
        checkbox.grid(column=0,row=3,pady=5,sticky='nswe')

        # Selected modes box and buttons
        # -------------------------------------------------------------

        # Group of widgets to put as label for the box
        label_frame = tk.Frame(frame)
        label = tk.Label(label_frame,text='Selected modes',bg='grey92')
        label.grid(column=0,row=0,sticky='nswe')
        load_tset = ttk.Button(label_frame,text='Load File',
                               command=self._load_modes)
        load_tset.grid(column=1,row=0,sticky='nswe')

        # Box for everything
        modes_box = ttk.LabelFrame(frame,labelwidget=label_frame)
        modes_box.grid(column=0,row=4,sticky='nswe',columnspan=2)
        modes_box.grid_columnconfigure(1,weight=1)
        
        # Buttons
        plus = ttk.Button(modes_box,text='+',width=1,
                          command=self._add_tmode)
        plus.grid(column=0,row=0,sticky='nswe')
        minus = ttk.Button(modes_box,text='-',width=1,
                           command=self._del_tmode)
        minus.grid(column=0,row=1,sticky='nswe')  

        # Mode panel      
        self._fourier_mode_box = tk.Listbox(modes_box,
            selectmode='multiple',height=5,width=28)
        self._fourier_mode_box.grid(column=1,row=0,padx=5,pady=5,
                                 sticky='nswe',rowspan=2)
        # -------------------------------------------------------------

    def _modifying_var(self,var,event):
        '''
        Change the color of the modified var box to red untule the return 
        button is pressed
        '''

        self._entries[var].configure(background='red')


    def _update_var(self,var,event):
        '''
        Update Fourier analysis parameters when one of them changes

        To compute all the six parameters, you need two out of three
        parameters among tres, tseg, and ntbs. I choose to always 
        compute everything from tres and ntbs. According to the last
        modified parameter and the fixed bins checkbot, the function
        compute tres and ntbs, then it calls _compute_fourier_pars, the
        function that actually perform the computation.
        '''

        print('Updating vars',var,'fix_bins',self._fix_bins.get())
        if var == 'tres':
            tres = my_tk_eval(self._fourier_pars['tres'])
            self._fourier_pars['tres'].set(tres)
            if not self._fix_bins.get():
                n = my_tk_eval(self._fourier_pars['tseg'])/tres
                self._fourier_pars['ntbs'].set(n)

        if var == 'tseg':
            tseg = my_tk_eval(self._fourier_pars['tseg'])
            self._fourier_pars['tseg'].set(tseg)
            if self._fix_bins.get():
                tres = tseg/my_tk_eval(self._fourier_pars['ntbs'])
                self._fourier_pars['tres'].set(tres)
            else:
                n = tseg/my_tk_eval(self._fourier_pars['tres'])
                self._fourier_pars['ntbs'].set(n)

        if var == 'fres':
            fres = my_tk_eval(self._fourier_pars['fres'])
            self._fourier_pars['fres'].set(fres)
            if self._fix_bins.get():
                tres = 1./my_tk_eval(self._fourier_pars['ntbs'])/fres
                self._fourier_pars['tres'].set(tres)
            else:
                n = 1./fres/my_tk_eval(self._fourier_pars['tres'])
                self._fourier_pars['ntbs'].set(n)

        if var == 'nyqf':
            nyqf = my_tk_eval(self._fourier_pars['nyqf'])
            self._fourier_pars['nyqf'].set(nyqf)
            if self._fix_bins.get():
                tres = 1./2./nyqf
            else:
                n = 2.*nyqf*my_tk_eval(self._fourier_pars['tseg'])
                self._fourier_pars['ntbs'].set(n)

        if var == 'ntbs':
            n = my_tk_eval(self._fourier_pars['ntbs'])
            self._fourier_pars['ntbs'].set(n)
            if self._fix_bins.get():
                tres = my_tk_eval(self._fourier_pars['tseg'])/n
                self._fourier_pars['tres'].set(tres)  

        self._compute_fourier_pars() 

        # Restoring the original color of the modified var box
        self._entries[var].configure(background='#ffffff')             
                

    def _compute_fourier_pars(self):
        '''
        Compute all the Fourier analysis parameters from time 
        resolution (tres) and number of time bins (n)
        '''

        tres = my_tk_eval(self._fourier_pars['tres'])
        n = my_tk_eval(self._fourier_pars['ntbs'])

        self._fourier_pars['tseg'].set(tres*n)
        self._fourier_pars['fres'].set(1./tres/n)
        self._fourier_pars['nyqf'].set(1./2./tres)
        self._fourier_pars['nfbs'].set(n//2)

        self._gti_dur.set(self._fourier_pars['tseg'].get())

    def _stop(self):
        '''
        Stop all the running process
        '''
        self.destroy()

    def _load_modes(self):
        pass

    def _load_en_bands(self):
        pass

    def _load_obs_ids(self,value):

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
                print('Loading obs IDs from {}'.format(ValueError))
                dirs = next(os.walk(value))[1]
                # !!! RXTE obs IDs have - in their names
                obs_ids = sorted([d for d in dirs if d.replace('-','').isdigit()])

        for obs_id in obs_ids: 
            self._obs_id_box.insert(tk.END,obs_id)


    def _load_obs_dir(self):
        pass


    def _init_other_settings(self,frame):
        '''
        Generate entries and buttons for other settings

        - energy range entries;
        - minimum GTI duration;
        - split event check buttons;
        - energy range selection panel and button.
        '''

        frame.grid_columnconfigure(0,weight=1)
        frame.grid_columnconfigure(1,weight=1)
        
        left_frame = tk.Frame(frame)
        left_frame.grid(column=0,row=0,sticky='nswe')
        right_frame = tk.Frame(frame)
        right_frame.grid(column=1,row=0,sticky='nswe')        

        # Energy selection
        en_width=6
        self._low_en = tk.DoubleVar()
        self._high_en = tk.DoubleVar()
        box1 = ttk.LabelFrame(left_frame,text='Energy range [keV]')
        box1.grid(column=0,row=0,padx=5,pady=5,
                  columnspan=2,sticky='nswe')
        low_en_entry = tk.Entry(box1,textvariable=self._low_en,
                                width=en_width)
        low_en_entry.grid(column=0,row=0,sticky='nswe')
        separator = tk.Label(box1,text='-')
        separator.grid(column=1,row=0,sticky='nswe')
        high_en_entry = tk.Entry(box1,textvariable=self._high_en,
                                 width=en_width)
        high_en_entry.grid(column=2,row=0,sticky='nswe')
        plus = ttk.Button(box1,text='+',width=1,
                          command=self._add_energy)
        plus.grid(column=3,row=0,sticky='nswe')
        minus = ttk.Button(box1,text='-',width=1,
                           command=self._del_energy)
        minus.grid(column=4,row=0,sticky='nswe')
        arrow = tk.Label(box1,text='  ======>  ')
        arrow.grid(column=5,row=0,sticky='nswe')        

        # GTI duration
        self._gti_dur = tk.DoubleVar()
        box2 = ttk.LabelFrame(left_frame,text='GTI min. duration [s]')
        box2.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        gti_entry = tk.Entry(box2,textvariable=self._gti_dur,width=15)
        gti_entry.grid(column=0,row=0,sticky='nswe')

        # Split event checkbox
        self._split_event = tk.IntVar()
        box3 = ttk.LabelFrame(left_frame,text='Split event')
        box3.grid(column=1,row=1,padx=5,pady=5,sticky='nswe')
        button = tk.Checkbutton(box3,variable=self._split_event,
                                text='(checked = YES)')
        button.grid(column=0,row=0,pady=3,sticky='nswe')

        # Group of widget to put as label for box4
        label_frame = tk.Frame(frame)
        label = tk.Label(label_frame,text='Selected Energy bands',
                         bg='grey92')
        label.grid(column=0,row=0,sticky='nswe')
        load_en = ttk.Button(label_frame,text='Load',
                             command=self._load_en_bands)
        load_en.grid(column=1,row=0,sticky='nswe')

        # Energy band box
        box4 = ttk.LabelFrame(right_frame,labelwidget=label_frame)
        box4.grid(column=0,row=0,padx=5,pady=5,rowspan=3,sticky='nswe')
        self._en_band_box = tk.Listbox(box4, selectmode='multiple',
                                       height=5,width=28)
        self._en_band_box.grid(column=0,row=0,padx=5,pady=5,
                               sticky='nswe')


    def _init_comment_box(self,frame):
        frame.grid_columnconfigure(0,weight=1)
        self._comment_box = tk.Text(frame,height=5)
        self._comment_box.grid(column=0,row=0,sticky='nswe')

    def _init_dir_box(self,frame):

        frame.grid_columnconfigure(1,weight=1)
        #frame.grid_columnconfigure(2,weight=3)
        #frame.grid_columnconfigure(3,weight=3)

        in_dir = tk.Label(frame,text='Data Dir')
        in_dir.grid(column=0,row=0,sticky='nswe')
        self._input_dir = tk.StringVar()
        in_entry = tk.Entry(frame,textvariable=self._input_dir)
        in_entry.grid(column=1,row=0,sticky='nswe')
        in_button = ttk.Button(frame,text='SET',command=self._sel_input_dir)
        in_button.grid(column=2,row=0,sticky='nswe')
        load_obs_id_button1 = ttk.Button(frame,text='Load IDs.',
            command=lambda: self._load_obs_ids(self._input_dir.get()))
        load_obs_id_button1.grid(column=3,row=0,sticky='nswe')

        self._frame2 = tk.Frame(frame)
        self._frame2.grid(column=0,row=2,columnspan=4,sticky='nswe')
        
        ext = tk.Label(self._frame2,text='Event file ext:')
        ext.grid(column=0,row=0,sticky='nswe')
        ext.to_disable=True
        self._event_ext = tk.StringVar()
        event_entry = tk.Entry(self._frame2,textvariable=self._event_ext,
                        width=5)
        event_entry.grid(column=1,row=0,sticky='nswe')
        event_entry.to_disable=True
        
        identifier = tk.Label(self._frame2,text='Event file identifier:')
        identifier.grid(column=2,row=0,sticky='nswe')
        identifier.to_disable=True
        self._event_str = tk.StringVar()
        identifier_entry = tk.Entry(self._frame2,textvariable=self._event_str,
                        width=9)
        identifier_entry.grid(column=3,row=0,sticky='nswe')
        identifier_entry.to_disable=True

        suffix = tk.Label(self._frame2,text='Output suffix:')
        suffix.grid(column=4,row=0,sticky='nswe')
        self._output_suffix = tk.StringVar()
        self._output_suffix.set('')
        suffix_entry = tk.Entry(self._frame2, textvariable=self._output_suffix,
                        width=14)
        suffix_entry.grid(column=5,row=0,sticky='nswe')

    def _init_buttons(self,frame):
        frame.grid_columnconfigure(0,weight=1)
        frame.grid_columnconfigure(1,weight=1)

        self._button1 = tk.Button(frame,text='Compute Power',
                            height=2)
        self._button1.grid(column=0,row=0,padx=5,pady=5,sticky='nesw')

        self._button2 = tk.Button(frame,text='Compute Lightcurve',
                            height=2)
        self._button2.grid(column=1,row=0,padx=5,pady=5,sticky='nesw')        

        button3 = tk.Button(frame,text='STOP!',
                            height=2,        
                            command=self._stop)
        button3.grid(column=2,row=0,padx=5,pady=5,sticky='nswe')  

    def _add_tmode(self):
        mode = 'tres: {}, time seg: {}, time bins: {}'.format(
            self._fourier_pars['tres'].get(),
            self._fourier_pars['tseg'].get(),
            self._fourier_pars['ntbs'].get())
        items = self._fourier_mode_box.get(0,tk.END)
        if not mode in items: 
            self._fourier_mode_box.insert(tk.END,mode)
            self._sort_tmodes()

    def _del_tmode(self):
        sel = self._fourier_mode_box.curselection()
        for index in sel[::-1]:
            self._fourier_mode_box.delete(index)
            self._sort_tmodes()

    def _sort_tmodes(self):       
        items = list(self._fourier_mode_box.get(0,tk.END))
        self._fourier_mode_box.delete(0,tk.END)
        #print('sorting modes',items)
        sorted_items = sorted(items, 
        key = lambda x: (x.split(',')[0].split(':')[1],
                         x.split(',')[1].split(':')[1],
                         x.split(',')[2].split(':')[1])
        )
        for item in sorted_items:
            self._fourier_mode_box.insert(tk.END,item)             

    def _add_energy(self):
        low_en = self._low_en.get()
        high_en = self._high_en.get()
        entry = '{}-{}'.format(low_en,high_en)
        items = self._en_band_box.get(0,tk.END)
        if not entry in items: 
            self._en_band_box.insert(tk.END,entry)
            self._sort_energy()

    def _del_energy(self):
        sel = self._en_band_box.curselection()
        for index in sel[::-1]:
            self._en_band_box.delete(index)
            self._sort_energy()

    def _sort_energy(self):       
        items = list(self._en_band_box.get(0,tk.END))
        self._en_band_box.delete(0,tk.END)
        sorted_items = sorted(items, 
        key = lambda x: (x.split('-')[0],x.split('-')[1]))
        for item in sorted_items:
            self._en_band_box.insert(tk.END,item)            

    def _new_window(self, newWindow):
        self.new = tk.Toplevel(self)
        newWindow(self.new, self)  

    def _sel_input_dir(self):
        indir = filedialog.askdirectory(initialdir=os.getcwd(),
              title='Select the folder containing obs ID directories')
        self._input_dir.set(indir)

        missions = ['RXTE','NICER','Swift','NuStar','HXMT']
        for mission in missions:
            if mission.upper() in indir.upper():
                self._mission.set(mission)
                break

    def _sel_output_dir(self):
        outdir = filedialog.askdirectory(initialdir=os.getcwd(),
              title='Select folder for data products')
        self._output_dir.set(outdir)

        

    def _read_boxes(self):
        '''
        Read obs ID, fourier parameters, energy bands, and comment 
        boxes.
        '''

        # Reading (selected) obs IDs
        sel = self._obs_id_box.curselection()
        if len(sel) == 0:
            self._obs_ids = sorted(self._obs_id_box.get(0,tk.END))
        else:
            self._obs_ids = sorted([self._obs_id_box.get(s) for s in sel])
        print('Obs Ids')
        print(self._obs_ids)

        # Reading (selected) fourier modes
        sel = self._fourier_mode_box.curselection()
        if len(sel) == 0:
            self._fmodes = self._fourier_mode_box.get(0,tk.END)
        else:
            self._fmodes = [self._fourier_mode_box.get(s) for s in sel]  
        print('Fourier modes')
        print(self._fmodes)

        # Reading (selected) energy bands
        sel = self._en_band_box.curselection()   
        if len(sel) == 0:
            self._en_bands = self._en_band_box.get(0,tk.END)
        else:
            self._en_bands = [self._en_band_box.get(s) for s in sel]  
        print('Energy bands')
        print(self._en_bands)

        # Reading comments
        self._comments = self._comment_box.get("1.0",tk.END)
        print('Comments')
        print(self._comments)

        # Reading RXTE modes
        mission = self._mission.get().strip()

        if mission.upper() == 'RXTE':
            sel = self._mode_box.selection()
            self._modes = []
            for s in sel:
                self._modes += [self._mode_box.item(s)['values'][0]]
            print('RXTE selected modes')
            print(self._modes)

    def _init_var(self):
        self._rxte_called = False

        self._fourier_pars['tres'].set(my_tk_eval('1./8192'))
        self._fourier_pars['tseg'].set('128')
        self._fourier_pars['fres'].set(my_tk_eval('1./128'))
        self._fourier_pars['nyqf'].set(my_tk_eval('8192./2'))
        self._fourier_pars['ntbs'].set(int(my_tk_eval('128*8192')))
        self._fourier_pars['nfbs'].set(int(my_tk_eval('128*8192/2-1')))

        self._fix_bins.set(1)

        self._low_en.set('0.5')
        self._high_en.set('10.0')
        self._gti_dur.set(self._fourier_pars['tseg'].get())

        # For comodity
        self._event_ext.set('evt')
        self._event_str.set('bc_bdc')
        self._input_dir.set('/Volumes/Seagate/NICER_data/Cygnus_X1')
        self._output_dir.set('/Volumes/Transcend/NICER/Cyg_X1')
        
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

class FitWindow:
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
            width=width,height=50,style=self._head_style)
        freq_frame.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._populate_freq_frame(freq_frame)

        # Fitting functions options and drawing them on the plot
        fit_func_frame = ttk.LabelFrame(frame,\
            text='Fitting functions',width=width,style=self._head_style)
        fit_func_frame.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        self._populate_fit_func_frame(fit_func_frame)

        # Fitting function parameters
        fit_pars_frame = ttk.LabelFrame(frame,\
            text='Fitting parameters',width=width,style=self._head_style)
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
            alpha=0.3,lfont=12,xy=self._controller._xy_flag)
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
            alpha=0.3,lfont=12,xy=self._controller._xy_flag)
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

class PlotFitWindow:
    def __init__(self,parent,controller):
        self._controller = controller
        self._parent = parent

        s = ttk.Style()
        s.configure('Black.TLabelframe.Label',
                    font=('times', 16, 'bold'))
        self._head_style = 'Black.TLabelframe'

        plot_frame = ttk.LabelFrame(self._parent,text='Residual',
            style=self._head_style)
        plot_frame.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        fig = Figure(figsize=(6.3,5),dpi=100)
        #gs = fig.add_gridspec(2,1)
        #gs.tight_layout(fig)
        self._controller._ax1 = fig.add_subplot(211)
        self._controller._ax2 = fig.add_subplot(212)  
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

        info_frame = ttk.LabelFrame(self._parent,text='Fit output',
            style=self._head_style)
        info_frame.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')

        self._controller._fit_info_box = tk.Listbox(self._parent, selectmode='multiple')
        self._controller._fit_info_box.grid(column=0,row=1,padx=5,pady=5,sticky='nsew')

    def _update_cursor(self,event):
        print('Updating cursor')
        self._x_pos.configure(text=str(event.xdata))
        self._y_pos.configure(text=str(event.ydata))

if __name__ == '__main__':
    app = MakePowerWin()
    app.mainloop()