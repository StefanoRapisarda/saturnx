import matplotlib 
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from saturnx.core.power import PowerSpectrum,PowerList

# Fitting data
import lmfit
from lmfit import Model,Parameters
from lmfit.model import save_modelresult,load_modelresult

import numpy as np

import os
import pandas as pd

from tkinter import ttk
import tkinter as tk
from tkinter import filedialog

import glob

from functools import partial

__all__ = ['FittingTab']

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

class TimingTab:
    def __init__(self,frame,parent):
        self._parent = parent

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
        self._upper_box = ttk.LabelFrame(frame,text=label,
                                   style=self._parent._head_style)
        self._upper_box.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')

        mid_box = ttk.LabelFrame(frame,text='Other settings',
                                 style=self._parent._head_style)
        mid_box.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')  

        #mid_box2 = ttk.LabelFrame(frame,text='Comments',
        #                          style=self._head_style)
        #mid_box2.grid(column=0,row=3,padx=5,pady=5,sticky='nswe')      

        mid_box2 = ttk.LabelFrame(frame,text='Data to reduce',
                                  style=self._parent._head_style)
        mid_box2.grid(column=0,row=3,padx=5,pady=5,sticky='nswe')   

        mid_box3 = ttk.LabelFrame(frame,text='Reduced products directory',
                                  style=self._parent._head_style)
        mid_box3.grid(column=0,row=4,padx=5,pady=5,sticky='nswe')             

        low_box = tk.Frame(frame)
        low_box.grid(column=0,row=5,padx=5,pady=5,sticky='nswe')  
        # -------------------------------------------------------------

        # Populating Fourier parameters box
        self._init_timing_boxes(self._upper_box)

        # Populating other settings box
        self._init_other_settings(mid_box)

        # Populating comment box
        #self._init_comment_box(mid_box2)

        # Populating directory box
        self._init_dir_box(mid_box2)

        # Populating output directory box
        self._init_output_dir_box(mid_box3)

        # Populating execution buttons
        self._init_buttons(low_box)  

        # Initializing variables
        self._init_var() 

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
            selectmode='multiple',height=4,width=28)
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
                                       height=4,width=28)
        self._en_band_box.grid(column=0,row=0,padx=5,pady=5,
                               sticky='nswe')

    def _load_obs_dir(self):
        pass

    def _init_comment_box(self,frame):
        frame.grid_columnconfigure(0,weight=1)
        self._comment_box = tk.Text(frame,height=5)
        self._comment_box.grid(column=0,row=0,sticky='nswe') 

    def _init_dir_box(self,frame):
        '''
        Containes inut dir, file extension, file identifier,
        and output suffix
        '''

        frame.grid_columnconfigure(1,weight=1)
        #frame.grid_columnconfigure(2,weight=3)
        #frame.grid_columnconfigure(3,weight=3)

        # Input dir Entry
        in_dir = tk.Label(frame,text='Data Dir')
        in_dir.grid(column=0,row=0,sticky='nswe')
        self._input_dir = tk.StringVar() # <---- VAR
        in_entry = tk.Entry(frame,textvariable=self._input_dir)
        in_entry.grid(column=1,row=0,sticky='nswe')
       
        # SET button
        in_button = ttk.Button(frame,text='SET',command=self._sel_input_dir)
        in_button.grid(column=2,row=0,sticky='nswe')
    
        # Load obs_IDs button
        load_obs_id_button1 = ttk.Button(frame,text='Load obs_IDs.',
            command=lambda: self._parent._load_obs_ids(self._input_dir.get()))
        load_obs_id_button1.grid(column=3,row=0,sticky='nswe')

        # Sub frame for file ext, file identifier and output suffix
        self._frame2 = tk.Frame(frame)
        self._frame2.grid(column=0,row=2,columnspan=4,sticky='nswe')
        
        # File ext
        ext = tk.Label(self._frame2,text='File ext:')
        ext.grid(column=0,row=0,sticky='nswe')
        ext.to_disable=True
        self._event_ext = tk.StringVar() # <---- VAR
        event_entry = tk.Entry(self._frame2,textvariable=self._event_ext,
                        width=5)
        event_entry.grid(column=1,row=0,sticky='nswe')
        event_entry.to_disable=True
        
        # File identifier
        identifier = tk.Label(self._frame2,text='File identifier:')
        identifier.grid(column=2,row=0,sticky='nswe')
        identifier.to_disable=True
        self._event_str = tk.StringVar() # <---- VAR
        identifier_entry = tk.Entry(self._frame2,textvariable=self._event_str,
                        width=9)
        identifier_entry.grid(column=3,row=0,sticky='nswe')
        identifier_entry.to_disable=True

        # Output suffix
        suffix = tk.Label(self._frame2,text='Output suffix:')
        suffix.grid(column=4,row=0,sticky='nswe')
        self._output_suffix = tk.StringVar() # <---- VAR
        self._output_suffix.set('')
        suffix_entry = tk.Entry(self._frame2, textvariable=self._output_suffix,
                        width=14)
        suffix_entry.grid(column=5,row=0,sticky='nswe')

    def _init_output_dir_box(self,frame):
        '''
        Reduced products directory box
        '''
        frame.grid_columnconfigure(0,weight=1)

        self._output_dir = tk.StringVar() # <---- VAR
        out_entry = tk.Entry(frame,textvariable=self._output_dir)
        out_entry.grid(column=0,row=0,sticky='nswe')

        label = tk.Label(frame,text='/analysis/')
        label.grid(column=1,row=0,sticky='nswe')

        out_button = ttk.Button(frame,text='SET',command=self._sel_output_dir)
        out_button.grid(column=2,row=0,sticky='nswe')


    def _init_buttons(self,frame):
        '''
        Initialize buttons and checkboxes at the end of the timing tab
        '''
        frame.grid_columnconfigure(0,weight=4)
        frame.grid_columnconfigure(1,weight=1)

        self._comp_lc = tk.IntVar()
        self._read_lc = tk.IntVar()
        self._read_lc.trace_add('write', self._click_read_lc)
        self._comp_pow = tk.IntVar()
        self._override = tk.IntVar()

        check_frame = tk.Frame(frame)
        check_frame.grid(column=0,row=0,padx=5,pady=5,sticky='nesw')
        
        comp_light_check = tk.Checkbutton(check_frame,text='Compute Lightcurves',
            var = self._comp_lc)
        comp_light_check.grid(column=0,row=0,sticky='nswe')
        
        read_light_check = tk.Checkbutton(check_frame,
            text='Read Lightcurves',var = self._read_lc)
        read_light_check.grid(column=1,row=0,sticky='nswe')
        
        comp_pow_check = tk.Checkbutton(check_frame,text='Compute PowerSpectra',
            var = self._comp_pow)
        comp_pow_check.grid(column=2,row=0,sticky='nswe')

        override_check = tk.Checkbutton(check_frame,text='Override',
            var = self._override)
        override_check.grid(column=3,row=0,sticky='nswe')

        self._comp_button = tk.Button(check_frame,text='Compute')
        self._comp_button.grid(column=0,row=1,padx=5,pady=5,
            columnspan=4,sticky='nesw')

        #self._button1 = tk.Button(frame,text='Compute Power',
        #                    height=2)
        #self._button1.grid(column=0,row=0,padx=5,pady=5,sticky='nesw')

        #self._button2 = tk.Button(frame,text='Compute Lightcurve',
        #                    height=2)
        #self._button2.grid(column=1,row=0,padx=5,pady=5,sticky='nesw')        

        stop_button = tk.Button(frame,text='STOP!',        
                            command=self._stop)
        stop_button.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')

    def _click_read_lc(self,var,indx,mode):
        flag = self._read_lc.get()
        if flag:
            status = 'disable'
        else:
            status = 'normal'
        for child in self._upper_box.winfo_children():
            if child.winfo_children():
                for grandchild in child.winfo_children():
                    grandchild.configure(state=status)
            else:
                child.configure(state=status)


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

        for mission in self._parent._missions:
            if mission.upper() in indir.upper():
                self._parent._mission.set(mission)
                break

    def _sel_output_dir(self):
        outdir = filedialog.askdirectory(initialdir=os.getcwd(),
              title='Select folder for data products')
        self._output_dir.set(outdir)

    def _read_boxes(self):
        '''
        Read obs ID, fourier parameters, energy bands, and comment 
        boxes.
        It stores the variables in attributes of the parent widget,
        that should be the main window
        '''

        # Reading (selected) obs IDs
        sel = self._parent._obs_id_box.curselection()
        if len(sel) == 0:
            self._parent._obs_ids = sorted(self._parent._obs_id_box.get(0,tk.END))
        else:
            self._parent._obs_ids = sorted([self._parent._obs_id_box.get(s) for s in sel])
        #print('Obs Ids')
        #print(self._obs_ids)

        # Reading (selected) fourier modes
        sel = self._fourier_mode_box.curselection()
        if len(sel) == 0:
            self._parent._fmodes = self._fourier_mode_box.get(0,tk.END)
        else:
            self._parent._fmodes = [self._fourier_mode_box.get(s) for s in sel]  
        #print('Fourier modes')
        #print(self._fmodes)

        # Reading (selected) energy bands
        sel = self._en_band_box.curselection()   
        if len(sel) == 0:
            self._parent._en_bands = self._en_band_box.get(0,tk.END)
        else:
            self._parent._en_bands = [self._en_band_box.get(s) for s in sel]  
        #print('Energy bands')
        #print(self._en_bands)

        # Reading comments
        #self._comments = self._comment_box.get("1.0",tk.END)
        #print('Comments')
        #print(self._comments)

        # Reading RXTE modes
        mission = self._parent._mission.get().strip()

        if mission.upper() == 'RXTE':
            sel = self._parent._mode_box.selection()
            self._modes = []
            for s in sel:
                self.parent__modes += [self._parent._mode_box.item(s)['values'][0]]
            #print('RXTE selected modes')
            #print(self._modes)

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


class FittingTab:
    def __init__(self, frame, parent, fit_window):
        self._parent = parent
        self._fit_window = fit_window

        # When self._to_plot is None, nothing has been plotted yet
        self._to_plot = None
        self._first_plot = True
        self._leahy = None

        # Frames
        # -------------------------------------------------------------
        # Panel with settings
        self._frame1 = tk.Frame(frame)
        #self._frame1 = ttk.LabelFrame(frame,text='Plot buttons',
        #    style=self._parent._head_style)
        self._frame1.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        # Panel with plotting area
        #label_frame = tk.Frame(frame)
        #label_frame.grid(column=0,row=0,sticky='nswe')
        #label = tk.Label(label_frame,text='To Fit',
        #    fg='black',font='times 16 bold')
        #label.grid(column=0,row=0,sticky='nswe')
        #fit_button = ttk.Button(label_frame,width=60,text='FIT',
        #    command=self._fit)
        #fit_button.grid(column=0,row=0,sticky='nswe')

        #self._frame2 = ttk.LabelFrame(frame,labelwidget=label_frame)
        self._frame2 = tk.Frame(frame)
        self._frame2.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')

        # Panel with input directory
        #frame3 = ttk.LabelFrame(frame,text='Reduced products directory',
        #                          style=self._parent._head_style)
        frame3 = tk.Frame(frame)
        frame3.grid(column=0,row=4,padx=5,pady=5,sticky='nswe')  

        #self._frame3 = ttk.LabelFrame(self._parent,text='Comments',
        #    style=self._parent._head_style)
        #self._frame3.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')
        # -------------------------------------------------------------

        # Populating frames
        # -------------------------------------------------------------
        self._init_buttons(self._frame1)
        self._init_plot_area(self._frame2)
        self._init_output_dir(frame3)
        # -------------------------------------------------------------

    def _init_buttons(self,frame):

        row1 = tk.Frame(frame)
        row1.grid(column=0,row=0,sticky='nswe')
        row2 = tk.Frame(frame)
        row2.grid(column=0,row=1,sticky='nswe')

        # File scrollbox
        # -------------------------------------------------------------
        box11 = ttk.LabelFrame(row1,text='File')
        box11.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        box11.tag = 1
        self._file = tk.StringVar() # <--- PowerSpectrum or PowerList file
        self._file.set(' ')
        # The number of GTIs should be updated depending on the
        # selected file
        self._file.trace_add('write',self._update_label)
        if len(self._parent._pickle_files) != 0:
            self._file.set(self._parent._pickle_files[0])
        self._file_menu = ttk.OptionMenu(box11, self._file, \
            *self._parent._pickle_files)
        self._file_menu.config(width=23)
        self._file_menu.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        # -------------------------------------------------------------

        # GTI index box
        # -------------------------------------------------------------
        box12 = ttk.LabelFrame(row1,text='GTI selection')  
        box12.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')
        # This will be initialized whhen self._file is changed
        self._gti_sel_string = tk.StringVar()
        self._gti_sel_string.set('')
        gti_entry = tk.Entry(box12, textvariable=self._gti_sel_string,width=13)
        gti_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        gti_entry.to_change = 1
        # -------------------------------------------------------------


        # PLOT button
        #plot_frame = tk.Frame(row1)
        #plot_frame.grid(column=2,row=0,padx=5,pady=5,sticky='nswe')
        #plot_frame.grid_rowconfigure(0,weight=1)
        #plot_frame.grid_rowconfigure(1,weight=1)
        #plot_frame.grid_columnconfigure(0,weight=1)
        plot_button = ttk.Button(row1,text='PLOT',
            command=lambda:self._plot())
        plot_button.grid(column=2,row=0,padx=5,pady=5,sticky='nswe')
        plot_button = ttk.Button(row1,text='FIT',
            command=self._fit)
        plot_button.grid(column=3,row=0,padx=5,pady=5,sticky='nswe')

        # Poisson buttons
        # -------------------------------------------------------------
        box21 = ttk.LabelFrame(row2, text='Poisson level')
        box21.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        est_poi_button = tk.Button(box21,text='Estimate',
            command=self._estimate_poi)
        est_poi_button.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        
        self._poi_level = tk.DoubleVar()
        self._poi_level.set(0)
        poi_level_entry = tk.Entry(box21,textvariable=self._poi_level,
            width=5)
        poi_level_entry.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')
        
        self._low_freq = tk.DoubleVar()
        self._low_freq.set(3000)
        freq_label1 = tk.Label(box21,text='>')
        freq_label1.grid(column=2,row=0,padx=5,pady=5,sticky='nswe')
        freq_entry = tk.Entry(box21,textvariable=self._low_freq,
            width=5)
        freq_entry.grid(column=3,row=0,padx=5,pady=5,sticky='nswe')
        freq_label2 = tk.Label(box21,text='Hz')
        freq_label2.grid(column=4,row=0,padx=5,pady=5,sticky='nswe')
        
        sub_poi_button = tk.Button(box21,text='Subtract Poi',
            command=lambda: self._update_plot('','',''))
        sub_poi_button.grid(column=0,row=1,columnspan=5,padx=5,pady=5,
            sticky='nswe')
        # -------------------------------------------------------------

        # Rebin buttons
        # -------------------------------------------------------------
        box22 = ttk.LabelFrame(row2, text='Rebinning')
        box22.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')  
        self._rebin_factor = tk.StringVar()
        self._rebin_factor.set(-30)
        rebin_entry = tk.Entry(box22,textvariable=self._rebin_factor,
            width=9)
        rebin_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        rebin_button = tk.Button(box22,text='Rebin',
            command=lambda: self._update_plot('','',''))
        rebin_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        # -------------------------------------------------------------

        # Normalization
        # -------------------------------------------------------------
        box23 = ttk.LabelFrame(row2, text='Normalization')
        box23.grid(column=2,row=0,padx=5,pady=5,sticky='nswe') 

        self._xy_flag = tk.IntVar()
        self._xy_flag.trace_add('write', self._update_plot)
        xy_checkbox = tk.Checkbutton(box23,text='X Vs. XY',
            var=self._xy_flag)
        xy_checkbox.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        self._norm = tk.StringVar()
        self._norm.set('Leahy')
        self._norm.trace_add('write', self._update_plot)
        norm_menu = ttk.OptionMenu(box23,self._norm,*tuple(['','Leahy','RMS']))
        norm_menu.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')

        bkg_label = tk.Label(box23,text='BKG [c/s]')
        bkg_label.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        self._bkg = tk.DoubleVar()
        self._bkg.set(0)
        bkg_entry = tk.Entry(box23,textvariable=self._bkg,width=10)
        bkg_entry.grid(column=1,row=1,padx=5,pady=5,sticky='nswe')
        # -------------------------------------------------------------

    def _update_label(self,var,indx,mode):
        '''
        Updates the label of the File Option menu and the GTI entry

        It checks all the children of the button frame (frame1). It 
        looks for children that have the attribute tag (File Option 
        Menu label) and to_change (GTI entry).
        If the file is a power_list, the label is updated with the 
        number of GTIs and available GTIs, if it is a Power Spectrum
        the label is the default and the GTI option is disabled
        '''

        if self._parent._current_tab == 'Fitting':

            selection = self._file.get()
            print('selecting',selection)

            data_file = os.path.join(self._output_dir2.get(),
                                    self._parent._obs_id,
                                    selection)

            if 'LIST' in selection.upper():
                self._power_list = PowerList.load(data_file)
                n_gti = self._power_list[0].meta_data['N_GTIS']
                for child in self._frame1.winfo_children():
                    for gchild in child.winfo_children():

                        for ggchild in gchild.winfo_children():
                            if hasattr(ggchild,'to_change') and ggchild['state']=='disabled':
                                ggchild.configure(state='normal')                  

                        if hasattr(gchild,'tag'):
                            gchild.configure(text='File (GTIs ({}): 0-{})'.\
                                format(n_gti,n_gti-1))

            else:
                self._power = PowerSpectrum.load(data_file)
                for child in self._frame1.winfo_children():
                    for gchild in child.winfo_children():

                        for ggchild in gchild.winfo_children():
                            if hasattr(ggchild,'to_change'):
                                ggchild.configure(state='disabled')

                        if hasattr(gchild,'tag'):
                            gchild.configure(text='File')

    def _update_file_menu(self):
        # Reset var and delete all old options
        self._file.set('')
        self._file_menu["menu"].delete(0, "end")

        # Inserting new choices
        for f in self._parent._pickle_files:
            self._file_menu["menu"].add_command(label=f, 
                command=tk._setit(self._file,f))

    def _plot(self):
        if self._first_plot:
            self._ax.clear()
            self._first_plot = False
        else:
            self._ax.clear()
            self._poi_level.set(0)
            self._rebin_factor.set(-30)
            self._xy_flag.set(0)
            self._bkg.set(0)
            self._norm.set('Leahy')
            self._first_plot = True

        if 'LIST' in self._file.get().upper():
            if self._gti_sel_string.get() != '':     
                gti_indices = self.eval_gti_str(self._gti_sel_string.get())
                power_list = PowerList([p for p in self._power_list \
                    if p.meta_data['GTI_INDEX'] in gti_indices])
                power = power_list.average('leahy')
            else:
                power = self._power_list.average('leahy')
            self._leahy = power
        else:
            self._leahy = self._power.normalize('leahy')

        self._to_plot = self._leahy.rebin(-30)
        self._to_plot.plot(ax=self._ax,lfont=14,marker='')
        self._canvas.draw()
        self._canvas.mpl_connect('motion_notify_event',self._update_cursor)

    def eval_gti_str(self,gti_str):
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


    def _update_plot(self,var,indx,mode):

        poi_level = self._poi_level.get()
        bkg = self._bkg.get()
        norm = self._norm.get()

        poi = self._leahy.sub_poi(value=poi_level)

        if norm == 'RMS':
            to_rebin = poi.normalize('rms',bkg_cr=bkg)
        elif norm == 'Leahy':
            to_rebin = poi

        rebin_str = self._rebin_factor.get()
        first = True
        if ',' in rebin_str:
            rfcs = rebin_str.split(',')
            for rfc in rfcs:
                if first:
                    rebin = to_rebin.rebin(eval(rfc))
                    first = False
                else:
                    rebin = rebin.rebin(eval(rfc))
        else:
            rebin = to_rebin.rebin(eval(rebin_str))

        self._to_plot = rebin
        self._ax.clear()
        if self._xy_flag.get():
            self._to_plot.plot(ax=self._ax,xy=True,lfont=14,marker='')
        else:
            self._to_plot.plot(ax=self._ax,lfont=14,marker='')
        self._canvas.draw()

    def _reset_plot(self):
        self._poi_level.set(0)
        self._norm.set('Leahy')
        self._xy_flag.set(0)
        self._rebin_factor.set('-30')
        self._update_plot('','','')

    def _estimate_poi(self):
        if not self._leahy is None:
            low_freq = float(self._low_freq.get())
            mask = self._leahy.freq>=low_freq
            value = self._leahy.power[mask].mean()
            self._poi_level.set(value)

    def _init_plot_area(self,frame):

        self._fig = Figure(figsize=(6.3,4.8),dpi=100) 
        self._ax = self._fig.add_subplot() 

        self._canvas = FigureCanvasTkAgg(self._fig, master = frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(column=0,row=0,sticky='nswe')

        coor_frame = tk.Frame(frame)
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

        #fit_button = tk.Button(frame,text='FIT',command = self._fit,
        #height=2)
        #fit_button.grid(column=0,row=2,pady=5,sticky='nswe')

    def _init_output_dir(self,frame):
        frame.grid_columnconfigure(0,weight=1)
        self._output_dir2 = tk.StringVar()
        out_entry = tk.Entry(frame,textvariable=self._output_dir2)
        out_entry.grid(column=0,row=0,sticky='nswe')
        out_button = ttk.Button(frame,text='SET',command=self._sel_output_dir)
        out_button.grid(column=1,row=0,sticky='nswe')
        load_obs_id_button2 = ttk.Button(frame,text='Load IDs.',
            command=lambda: self._parent._load_obs_ids(self._output_dir2.get()))
        load_obs_id_button2.grid(column=2,row=0,sticky='nswe') 

    def _sel_output_dir(self):
        outdir = filedialog.askdirectory(initialdir=os.getcwd(),
              title='Select folder for data products')
        self._output_dir2.set(outdir)    
        for mission in self._parent._missions:
            if mission.upper() in outdir.upper():
                self._parent._mission.set(mission)  
        self._parent._click_on_obs_id('dummy')

    def _update_cursor(self,event):
        self._x_pos.configure(text=str(np.round(event.xdata,6)))
        self._y_pos.configure(text=str(np.round(event.ydata,6)))

    def _fit(self):
        self._new_window(self._fit_window)

    def _new_window(self, newWindow):
        self.new = tk.Toplevel(self._parent)
        newWindow(self.new, self, self._parent)  

    def _new_child_window(self, newWindow):
        new = tk.Toplevel(self._parent)
        newWindow(new, self)




